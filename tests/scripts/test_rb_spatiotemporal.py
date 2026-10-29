"""RB experiment composition, batch accounting and forecast integration."""

import pytest
import torch
from hydra import compose, initialize_config_dir

from autocast.nn.spatiotemporal_vit import SpatiotemporalViTBackbone
from autocast.processors.diffusion import DiffusionProcessor
from autocast.processors.flow_matching import FlowMatchingProcessor
from autocast.processors.spatiotemporal_vit import (
    SpatiotemporalViTMaskedWindowProcessor,
)
from autocast.scripts.setup import (
    _build_processor,
    setup_epd_model,
    setup_processor_model,
)
from autocast.types import Batch, EncodedBatch


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize(
    ("experiment", "preset"),
    [
        ("fm_vit_large", "flow_matching_spatiotemporal_vit"),
        (
            "fm_vit_large_masked_window",
            "flow_matching_masked_window_spatiotemporal_vit",
        ),
        ("diffusion_vit_large_lola", "diffusion_spatiotemporal_vit"),
    ],
)
def test_rb_experiments_accept_opt_in_backbone(experiment, preset, REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="processor",
            overrides=[
                f"local_experiment=the_well/rayleigh_benard/{experiment}",
                f"processor@model.processor={preset}",
                "backbone@model.processor.backbone=spatiotemporal_vit",
            ],
        )
    # These experiments retain scalar patch_size=1 and, for diffusion,
    # explicit rope/rpb/window_size values from the original configuration.
    cfg.model.processor.backbone.hid_channels = 8
    cfg.model.processor.backbone.hid_blocks = 1
    cfg.model.processor.backbone.mod_features = 4
    cfg.model.processor.backbone.attention_heads = 2
    processor = _build_processor(
        cfg.model,
        {
            "in_channels": 2,
            "out_channels": 2,
            "n_channels_out": 2,
            "n_steps_input": 1,
            "n_steps_output": 4,
            "spatial_resolution": (4, 2),
        },
        global_cond_channels=2,
    )
    assert isinstance(processor, (FlowMatchingProcessor, DiffusionProcessor))
    backbone = (
        processor.flow_matching_model
        if isinstance(processor, FlowMatchingProcessor)
        else processor.denoiser.backbone
    )
    assert isinstance(backbone, SpatiotemporalViTBackbone)
    assert backbone.patch_size == (1, 1, 1)
    assert backbone.conditioning == ("mask" if "masked" in preset else "frames")


@pytest.mark.parametrize(
    ("method", "batch_size", "members"), [("fm", 64, 1), ("crps", 8, 8)]
)
def test_rb_spatiotemporal_training_protocol(method, batch_size, members, REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="processor",
            overrides=[
                f"local_experiment=the_well/rayleigh_benard/{method}_spatiotemporal_latent",
                "trainer.max_epochs=16",  # Timing must retain the full LR horizon.
            ],
        )
    assert cfg.optimizer.scheduler_interval == "epoch"
    assert cfg.optimizer.cosine_epochs == 4096
    assert cfg.optimizer.learning_rate == 1e-4
    assert cfg.trainer.devices == 4
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.limit_train_batches == 64
    assert cfg.trainer.gradient_clip_val == 1.0
    assert cfg.model.train_loss_log_every_n_steps == cfg.trainer.log_every_n_steps == 64
    grad_norm = next(
        callback
        for callback in cfg.trainer.callbacks
        if callback._target_ == "autocast.callbacks.grad_norm.GradNormCallback"
    )
    assert grad_norm.every_n_steps == 64
    assert cfg.datamodule.batch_size == batch_size
    assert cfg.model.get("n_members", 1) == members
    assert batch_size * members * cfg.trainer.devices == 256
    backbone = cfg.model.processor.backbone
    assert (backbone.hid_channels, backbone.hid_blocks, backbone.attention_heads) == (
        1024,
        16,
        8,
    )
    assert list(backbone.patch_size) == [1, 1, 1]
    assert backbone.conditioning == "mask"
    assert backbone.use_precomputed_modulation == (method == "crps")
    assert cfg.model.processor.context_mask._partial_
    assert cfg.model.processor.context_mask.lmbda == 1.0
    assert cfg.model.processor.context_mask.rho == 1.0
    assert cfg.model.processor.context_mask.atleast == 1
    if method == "crps":
        assert cfg.model.loss_func is None
        assert cfg.model.processor.n_members == members
        assert cfg.model.processor.loss_func._target_ == (
            "autocast.losses.ensemble.AlphaFairCRPSLoss"
        )
        assert cfg.model.processor.loss_func.reduction == "none"
    else:
        assert not cfg.model.processor.loss_on_observed


@pytest.mark.parametrize("method", ["fm", "crps"])
def test_rb_masked_configs_train_and_forecast_through_existing_setup(
    method, REPO_ROOT, dummy_datamodule
):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="processor",
            overrides=[
                f"local_experiment=the_well/rayleigh_benard/{method}_spatiotemporal_latent"
            ],
        )
    backbone = cfg.model.processor.backbone
    backbone.hid_channels, backbone.hid_blocks = 8, 1
    backbone.attention_heads, backbone.mod_features = 2, 4
    backbone.dropout = 0.0
    if method == "crps":
        cfg.model.n_members = 3
    else:
        cfg.model.processor.flow_ode_steps = 2
    batch = EncodedBatch(
        encoded_inputs=torch.full((2, 1, 2, 2, 1), 100.0),
        encoded_output_fields=torch.zeros(2, 4, 2, 2, 1),
        global_cond=torch.tensor([[2.0], [4.0]]),
        encoded_info={},
    )
    stats = {
        "channel_count": 1,
        "n_steps_input": 1,
        "n_steps_output": 4,
        "input_shape": batch.encoded_inputs.shape,
        "example_batch": batch,
    }
    model = setup_processor_model(cfg, stats, dummy_datamodule)
    assert model.train_loss_log_every_n_steps == 64
    core = (
        model.processor.backbone
        if method == "crps"
        else model.processor.flow_matching_model
    )
    assert isinstance(core, SpatiotemporalViTBackbone)
    assert core.n_steps_input == core.n_steps_output == 5
    loss = model.loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    pred = model(batch.encoded_inputs, batch.global_cond)
    expected_shape = (
        (*batch.encoded_output_fields.shape, 3)
        if method == "crps"
        else batch.encoded_output_fields.shape
    )
    assert pred.shape == expected_shape
    assert torch.isfinite(pred).all()
    if method == "crps":
        assert not torch.equal(pred[..., 0], pred[..., 1])


def test_rb_ambient_spatiotemporal_protocol(REPO_ROOT):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_ambient",
                "trainer.max_epochs=16",
            ],
        )
        latent = compose(
            config_name="processor",
            overrides=[
                "local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_latent",
                "trainer.max_epochs=16",
            ],
        )
    assert cfg.optimizer == latent.optimizer
    assert cfg.trainer == latent.trainer
    assert cfg.model.val_metrics == latent.model.val_metrics
    for key in ("_target_", "context_mask", "loss_func", "n_noise_input_channels"):
        assert cfg.model.processor[key] == latent.model.processor[key]
    assert cfg.datamodule._target_ == "autocast.data.datamodule.TheWellDataModule"
    assert "data_path" not in cfg.datamodule  # No cached-latent loader settings.
    assert cfg.datamodule.use_normalization
    assert cfg.datamodule.normalization_path.endswith("rayleigh_benard/stats.yaml")
    assert list(cfg.datamodule.transform.scalar_names) == ["Rayleigh", "Prandtl"]
    assert cfg.model.encoder._target_ == "autocast.encoders.identity.IdentityEncoder"
    assert cfg.model.decoder._target_ == "autocast.decoders.identity.IdentityDecoder"
    assert cfg.autoencoder_checkpoint is None
    assert cfg.model.train_in_latent_space  # Delegate to masked processor loss.
    assert cfg.model.loss_func is None
    assert cfg.datamodule.batch_size == cfg.model.n_members == 8
    assert cfg.datamodule.batch_size * cfg.trainer.devices == 32
    assert 32 * cfg.model.processor.n_members == 256
    assert cfg.trainer.accumulate_grad_batches == 1
    assert cfg.trainer.limit_train_batches == 64
    assert cfg.optimizer.cosine_epochs == 4096
    assert cfg.optimizer.learning_rate == 1e-4
    assert cfg.eval.rollout_start == cfg.eval.n_members == 16
    backbone = cfg.model.processor.backbone
    assert (backbone.hid_channels, backbone.hid_blocks, backbone.attention_heads) == (
        2048,
        16,
        16,
    )
    assert list(backbone.patch_size) == [1, 16, 16]
    assert backbone.checkpointing
    with torch.device("meta"):
        processor = _build_processor(
            cfg.model,
            {
                "in_channels": 4,
                "out_channels": 4,
                "n_channels_out": 4,
                "n_steps_input": 1,
                "n_steps_output": 4,
                "spatial_resolution": (512, 128),
            },
            global_cond_channels=6,
        )
    assert isinstance(processor, SpatiotemporalViTMaskedWindowProcessor)
    assert processor.backbone.vit.in_proj.in_features == 2 * 4 * 16 * 16
    assert len(processor.backbone.vit.blocks) == 16


def test_rb_ambient_masked_loss_and_forecast_keep_physical_grid(
    REPO_ROOT, dummy_datamodule
):
    with initialize_config_dir(
        str(REPO_ROOT / "src/autocast/configs"), version_base=None
    ):
        cfg = compose(
            config_name="encoder_processor_decoder",
            overrides=[
                "local_experiment=the_well/rayleigh_benard/crps_spatiotemporal_ambient"
            ],
        )
    # Keep the pixel patch/grid and real setup path; shrink only the trunk.
    backbone = cfg.model.processor.backbone
    backbone.hid_channels, backbone.hid_blocks = 8, 1
    backbone.attention_heads, backbone.mod_features = 2, 4
    backbone.dropout = 0.0
    cfg.model.n_members = 3
    batch = Batch(
        input_fields=torch.randn(1, 1, 512, 128, 4),
        output_fields=torch.randn(1, 4, 512, 128, 4),
        constant_scalars=torch.tensor([[2.0, 4.0]]),
        constant_fields=None,
        boundary_conditions=torch.zeros(1, 2, 2),
    )
    model = setup_epd_model(
        cfg,
        {
            "channel_count": 4,
            "n_steps_input": 1,
            "n_steps_output": 4,
            "example_batch": batch,
        },
        dummy_datamodule,
    )
    encoded = model.encoder_decoder.encoder.encode_batch(batch)
    torch.testing.assert_close(encoded.encoded_inputs, batch.input_fields)
    torch.testing.assert_close(encoded.encoded_output_fields, batch.output_fields)
    assert encoded.global_cond is not None
    torch.testing.assert_close(encoded.global_cond[:, :2], batch.constant_scalars)
    assert encoded.global_cond.shape == (1, 6)
    processor = model.processor
    assert isinstance(processor, SpatiotemporalViTMaskedWindowProcessor)
    shapes = []
    hook = processor.backbone.vit.blocks[0].register_forward_pre_hook(
        lambda _, args: shapes.append(args[0].shape)
    )
    loss, pred = model.loss(batch)
    assert pred is None  # Masked processor owns the training ensemble/loss.
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with torch.no_grad():
        pred = model(batch)
    hook.remove()
    assert shapes
    assert all(shape == (3, 1280, 8) for shape in shapes)
    assert pred.shape == (1, 4, 512, 128, 4, 3)
    assert torch.isfinite(pred).all()
    assert not torch.equal(pred[..., 0], pred[..., 1])
