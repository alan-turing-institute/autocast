"""Base class for temporal backbone architectures."""

from abc import ABC, abstractmethod

from azula.nn.embedding import SineEncoding
from einops import rearrange
from torch import nn

from autocast.nn.temporal_modules import (
    TemporalAttention,
    TemporalConvNet,
)
from autocast.types import Tensor, TensorBTSC


class TemporalBackboneBase(nn.Module, ABC):
    """Base class for temporal backbone architectures.

    Provides common functionality for:
    - Time embedding for diffusion timesteps
    - Temporal processing method selection and initialization
    - Shared temporal processing application
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        n_steps_output: int,
        n_steps_input: int,
        global_cond_channels: int | None,
        include_global_cond: bool,
        mod_features: int = 256,
        temporal_method: str = "none",
        temporal_attention_heads: int = 8,
        temporal_attention_hidden_dim: int = 64,
        # TCN parameters
        tcn_kernel_size: int = 3,
        tcn_num_layers: int = 2,
        use_precomputed_modulation: bool = False,
        include_time_embedding: bool = True,
        include_loss_weight: bool = False,
    ):
        """Initialize Temporal Backbone Base.

        Args:
            in_channels: Number of input channels per timestep
            out_channels: Number of output channels per timestep
            cond_channels: Number of conditioning channels per timestep
            n_steps_output: Number of output timesteps to predict
            n_steps_input: Number of input timesteps for conditioning
            mod_features: Dimension for time embedding (diffusion timestep)
            global_cond_channels: Dimension for optional conditioning/modulation
            include_global_cond: Whether to include global conditioning
            temporal_method: Method for temporal processing. Options:
                - "attention": Multi-head self-attention over time
                - "tcn": Temporal convolutional network
                - "none": No temporal processing (identity)
            temporal_attention_heads: Number of heads for attention methods
            temporal_attention_hidden_dim: Hidden dimension for attention methods
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers
            use_precomputed_modulation: If True, callers pass precomputed
                modulation vectors of shape ``(B, mod_features)`` directly as
                ``t`` and no SineEncoding-based embedding is registered.
            include_time_embedding: If False, skip the time-embedding
                registration entirely. Used by one-step processors (e.g.
                drifting) that have no integration time ``t`` to embed. When
                False, callers may pass ``t=None`` to ``forward``.
            include_loss_weight: If True, register an embedding for a scalar
                loss-mixing weight ``w`` (the convex form ``w=1/(1+lambda)``)
                that is added to the modulation vector, so a single model can
                be conditioned on lambda and swept at inference. Default
                False leaves all existing backbones byte-identical (no new
                parameters), which is DDP-safe. Forwarded by the U-Net and
                multi-layer-perceptron backbones; the vision-transformer
                backbone does not expose it yet.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.temporal_method = temporal_method
        self.n_steps_output = n_steps_output
        self.n_steps_input = n_steps_input
        self.mod_features = mod_features
        self.use_precomputed_modulation = use_precomputed_modulation
        self.include_time_embedding = include_time_embedding
        self.include_loss_weight = include_loss_weight

        # Validate global conditioning configuration
        if include_global_cond and (
            global_cond_channels is None or global_cond_channels <= 0
        ):
            msg = "`include_global_cond` is True but global_cond_channels <= 0"
            raise ValueError(msg)
        self.global_cond_channels = global_cond_channels
        self.include_global_cond = include_global_cond

        # Time embedding for scalar diffusion timesteps. Some models pass
        # precomputed modulation vectors directly and should not register
        # unused embedding parameters under strict DDP. One-step processors
        # (drifting) opt out entirely via ``include_time_embedding=False``.
        self.time_embedding = (
            nn.Sequential(
                SineEncoding(mod_features),
                nn.Linear(mod_features, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, mod_features),
            )
            if include_time_embedding and not use_precomputed_modulation
            else None
        )

        self.global_cond_embedding = (
            nn.Sequential(
                nn.Linear(global_cond_channels, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, mod_features),
            )
            if global_cond_channels is not None
            and global_cond_channels > 0
            and include_global_cond
            else None
        )

        # Scalar loss-mixing-weight embedding. Mirrors the global-cond
        # embedding but takes the single bounded weight w=1/(1+lambda) in [0,1].
        # Registered only when include_loss_weight is True, so default backbones
        # gain no parameters and stay byte-identical / DDP-safe.
        self.lambda_embedding = (
            nn.Sequential(
                nn.Linear(1, mod_features),
                nn.SiLU(),
                nn.Linear(mod_features, mod_features),
            )
            if include_loss_weight
            else None
        )

        # Initialize temporal processing modules
        self.temporal_proc_input = self._create_temporal_module(
            channels=in_channels,
            temporal_method=temporal_method,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

        self.temporal_proc_cond = self._create_temporal_module(
            channels=cond_channels,
            temporal_method=temporal_method,
            temporal_attention_heads=temporal_attention_heads,
            temporal_attention_hidden_dim=temporal_attention_hidden_dim,
            tcn_kernel_size=tcn_kernel_size,
            tcn_num_layers=tcn_num_layers,
        )

    def _create_temporal_module(
        self,
        channels: int,
        temporal_method: str,
        temporal_attention_heads: int,
        temporal_attention_hidden_dim: int,
        tcn_kernel_size: int,
        tcn_num_layers: int,
    ) -> nn.Module:
        """Create temporal processing module based on method selection.

        Args:
            channels: Number of channels for this module
            temporal_method: Method name
            temporal_attention_heads: Number of heads for attention
            temporal_attention_hidden_dim: Hidden dimension for attention
            tcn_kernel_size: Kernel size for TCN
            tcn_num_layers: Number of TCN layers

        Returns:
            Temporal processing module
        """
        if temporal_method == "attention":
            return TemporalAttention(
                channels=channels,
                attention_heads=temporal_attention_heads,
                hidden_dim=temporal_attention_hidden_dim,
            )
        if temporal_method == "tcn":
            return TemporalConvNet(
                channels=channels,
                kernel_size=tcn_kernel_size,
                num_layers=tcn_num_layers,
            )
        if temporal_method == "none":
            return nn.Identity()

        raise ValueError(
            f"Unknown temporal_method: {temporal_method}. "
            f"Choose from: attention, tcn, none"
        )

    def apply_temporal_processing(
        self, x_t: TensorBTSC, cond: TensorBTSC | None
    ) -> tuple[TensorBTSC, TensorBTSC | None]:
        """Apply temporal processing to input and conditioning.

        Args:
            x_t: Input tensor (B, T, W, H, C)
            cond: Conditioning tensor (B, T_cond, W, H, C), or None

        Returns:
            Tuple of (processed_input, processed_cond)
        """
        x_t_temporal = self.temporal_proc_input(x_t)
        cond_temporal = self.temporal_proc_cond(cond) if cond is not None else None
        return x_t_temporal, cond_temporal

    @abstractmethod
    def _build_backbone(self, **kwargs) -> nn.Module:
        """Build the underlying backbone architecture (UNet, ViT, etc.).

        This method should be implemented by subclasses to instantiate
        their specific backbone architecture.

        Returns:
            The backbone module (e.g., UNet or ViT)
        """

    @property
    @abstractmethod
    def backbone(self) -> nn.Module:
        """Return the backbone module.

        Subclasses should define this as a property that returns
        their backbone (e.g., self.unet or self.vit).
        """

    def forward(
        self,
        x_t: TensorBTSC,
        t: Tensor | None,
        cond: TensorBTSC,
        global_cond: Tensor | None = None,
        loss_weight: Tensor | None = None,
    ) -> TensorBTSC:
        """Forward pass of the temporal backbone.

        Args:
            x_t: Noisy data (B, T, W, H, C) - spatial dims before channels
            t: Diffusion modulation input. Either:
                - scalar timesteps with shape (B,), which are embedded via SineEncoding
                - precomputed modulation vectors with shape (B, D), where D=mod_features
                - ``None``, when the backbone was built with
                  ``include_time_embedding=False`` (one-step processors)
            cond: Conditioning input (B, T_cond, W, H, C)
            global_cond: Optional global conditioning/modulation vector (B, D)
            loss_weight: Optional scalar loss-mixing weight ``w`` per sample,
                shape ``(B,)`` or ``(B, 1)``. Only consumed when the backbone was
                built with ``include_loss_weight=True``; embedded and added to the
                modulation vector so the network can condition on lambda.

        Returns:
            Denoised output (B, T, W, H, C)
        """
        # Build modulation embedding. ``t`` may be None when
        # ``include_time_embedding=False`` (one-step processors).
        if t is None:
            if self.include_time_embedding:
                msg = (
                    "Backbone was built with include_time_embedding=True "
                    "but received t=None."
                )
                raise ValueError(msg)
            # AdaLN/FiLM consumers still need a (B, mod_features) tensor;
            # zeros are the neutral identity shift, leaving whichever other
            # modulators are enabled (global conditioning, the loss weight) to
            # carry the signal -- or none at all, if neither is.
            t_emb = x_t.new_zeros((x_t.shape[0], self.mod_features))
        elif t.ndim == 2 and t.shape[-1] == self.mod_features:
            t_emb = t
        else:
            if self.time_embedding is None:
                reason = (
                    "was built with include_time_embedding=False"
                    if not self.include_time_embedding
                    else "uses precomputed modulation vectors"
                )
                msg = (
                    f"Backbone {reason}, so it cannot embed scalar timesteps; "
                    f"received t with shape {tuple(t.shape)}."
                )
                raise ValueError(msg)
            t_emb = self.time_embedding(t)

        # Combine with global conditioning embedding if provided
        if self.global_cond_embedding is not None:
            if global_cond is None:
                msg = "Model init with global_cond_channels but no global_cond provided"
                raise ValueError(msg)
            t_emb = t_emb + self.global_cond_embedding(global_cond)

        # Combine with the loss-mixing-weight embedding if conditioning on lambda.
        if self.lambda_embedding is None:
            if loss_weight is not None:
                msg = (
                    "loss_weight was passed to forward() but the backbone was "
                    "built with include_loss_weight=False, so it would be "
                    "silently ignored and the model would train unconditioned."
                )
                raise ValueError(msg)
        else:
            if loss_weight is None:
                msg = (
                    "Backbone was built with include_loss_weight=True but no "
                    "loss_weight was provided to forward()."
                )
                raise ValueError(msg)
            # ``reshape(-1, 1)`` also accepts a 0-d scalar weight; the device is
            # pinned because a caller may build the weight on the CPU.
            w = loss_weight.reshape(-1, 1).to(device=t_emb.device, dtype=t_emb.dtype)
            t_emb = t_emb + self.lambda_embedding(w)

        # Apply temporal processing
        x_t_temporal, cond_temporal = self.apply_temporal_processing(x_t, cond)

        # Convert to channels-first format: (B, T, W, H, C) -> (B, T*C, W, H)
        x_t_cf = rearrange(x_t_temporal, "b t w h c -> b (t c) w h")
        x_cond_cf = (
            rearrange(cond_temporal, "b t w h c -> b (t c) w h")
            if cond_temporal is not None
            else None
        )

        # Backbone forward: (B, T*C, W, H) -> (B, T*out_channels, W, H)
        output = self.backbone(x=x_t_cf, mod=t_emb, cond=x_cond_cf)

        # Convert back to channels-last format:
        # (B, T*self.out_channels, W, H) -> (B, T, W, H, self.out_channels)
        return rearrange(
            output,
            "b (t c) w h -> b t w h c",
            t=self.n_steps_output,
            c=self.out_channels,
        )
