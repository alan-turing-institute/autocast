# Spatial and spatiotemporal ViT backbones

Both backbones share the unchanged `TemporalBackboneBase`. It supplies the
sequence interface, modulation and optional temporal preprocessing. The
concrete backbones choose the token layout and attention mechanism:

| Backbone | Transformer attention |
| --- | --- |
| `SpatialViTBackbone` | Spatial patches, with frames bundled into channels |
| `SpatiotemporalViTBackbone` | Joint spatiotemporal tokens in every transformer block |

For the spatial backbone, `temporal_method=attention` or `tcn` processes the
field and context timelines separately before bundling. This happens once
per network evaluation, before the spatial transformer blocks. The default
is `none`. `AzulaViTProcessor` bundles observed frames before the backbone,
so that wrapper's temporal preprocessing sees only one effective time step.
The spatiotemporal backbone instead preserves the frame axis within attention.

Existing imports and Hydra targets remain valid through identity aliases:

| Existing name | Canonical name |
| --- | --- |
| `TemporalViTBackbone` | `SpatialViTBackbone` |

Existing spatial presets and checkpoint layouts remain compatible. Renaming
does not require retraining; switching between the two architectures does.
Printed class names use the canonical names, including when an object was
created through a legacy alias.

The pre-merge `autocast.nn.spacetime_vit` and
`autocast.processors.spacetime_vit` imports and `spacetime_vit` preset spellings
are temporary aliases for the Isambard timing runs. Remove these aliases after
collecting the timing results and migrating any retained configs/checkpoints.
The established `TemporalViTBackbone` alias remains supported.

## Joint spatiotemporal backbone

`SpatiotemporalViTBackbone` matches the transformer architecture in
[LOLA's pinned implementation](https://github.com/francois-rozet/lola/blob/21a4354b327e6e5ee06da5075ba3bd1dd88c61f1/lola/nn/vit.py):
the default preset uses width 1024, 16 blocks, eight heads, SiLU FFNs with
expansion four, dropout 0.05, query/key RMS normalization, absolute sinusoidal
positions plus learned three-axis RoPE, and four AdaLN outputs including
the initial-token skip. Azula supplies patching, attention and FFN components.
Matching the transformer does not
prescribe an objective, mask distribution, training schedule or sampler.

Fields use `(B,T,H,W,C)` layout with matching context/forecast channel
counts. Encoders must preserve the time axis. Patches are either a scalar
spatial size `p` (meaning `[1,p,p]`) or explicit `[time,height,width]`, with
temporal patch size one. A full window produces
`T * (H / patch_height) * (W / patch_width)` tokens.
Explicit `rope=true`, `rpb=false` and `window_size=null` settings are
accepted; other values are rejected.

| Conditioning mode | Inputs | Output |
| --- | --- | --- |
| `frames` | Forecast slots and separate clean context, joined along time | Forecast slice |
| `mask` | Full window and observation mask | Full window |

The observation mask is an input feature, not a causal attention mask.
Processors own noise, clamping, losses and sampling.

## Processor presets

Select a new processor preset using the existing configuration override:

| Processor preset | Objective/conditioning |
| --- | --- |
| `flow_matching_spatiotemporal_vit` | Existing FM objective, fixed clean context |
| `flow_matching_masked_window_spatiotemporal_vit` | Existing full-window FM objective with sampled context masks |
| `diffusion_spatiotemporal_vit` | Existing diffusion objective and sampler |
| `spatiotemporal_vit` | One-pass stochastic generator, suitable for the existing ensemble model and CRPS loss |
| `spatiotemporal_vit_masked_window` | One-pass masked-window ensembles with a configurable unreduced score (default alpha-fair CRPS) |

For example, add `processor@model.processor=flow_matching_spatiotemporal_vit`
to a processor training command. An experiment that explicitly overrides
`backbone@model.processor.backbone` must also select `spatiotemporal_vit` there.
The stochastic generator preset additionally needs
`model.processor.backbone.use_precomputed_modulation=true` if an experiment
replaces its backbone defaults.

## Stochastic ensembles

`SpatiotemporalViTProcessor` fills future slots with zeros and samples one
independent global Gaussian noise vector per member. If
`n_noise_input_channels` differs from `backbone.mod_features`, a learned
linear projection maps between them; otherwise the projection is an
identity. The projected noise is added to the physical-parameter embedding
before each block's shared modulation head. No unobserved target values
enter the generator. The defaults use 1024 noise dimensions and 256
modulation features; these are configurable independently of the score.

With this fixed-context processor, configure the CRPS score and member
count through the ensemble model's `loss_func` and `n_members`.

## Masked training

`SpatiotemporalViTMaskedWindowProcessor` owns its training member expansion:
it samples one context mask per real example, reveals only those states,
and shares that context across independently sampled members. Its configured
score must return one value per target element (`reduction: none`). Scores
are averaged over unobserved elements within each example, then across the
batch. Set `model.loss_func: null` so the existing ensemble model delegates
to `processor.loss`. Set `model.processor.n_members` for training and
`model.n_members` for inference ensembles.
At inference, `map` observes only the supplied input frames and returns the
forecast slice. Random training masks never change that rollout interface.

Mask sampling is an injectable callable, independent of the backbone and
objective. Both masked processors accept `context_mask`, which receives
the full window and returns a boolean observation mask broadcastable to
that window. The reusable `autocast.processors.masking.random_context_mask`
samples prefix/suffix masks with configurable Poisson rate `lmbda`, prefix
probability `rho` and minimum context `atleast`. Context lengths use modulo
wrapping and always leave a forecast frame.

The masked FM processor retains its full-window loss by default. Set
`loss_on_observed=false` to score only unobserved elements, averaging within
each example before averaging the batch. The masked ensemble processor
always uses this forecast-only reduction with its configured score.

## Verification and example experiment

`tests/nn/fixtures/lola_spacetime.json` contains forward values and input
gradients generated independently from the pinned upstream global-attention
code. Its recipe fixes all weights, inputs and modulation; the test accounts
for the transposed RoPE parameter storage in Azula. Tests also cover
parameter counts, processor presets, ensemble gradients and both
checkpointing modes.

The [Rayleigh-Benard example](https://github.com/alan-turing-institute/autocast/blob/main/slurm_scripts/comparison/the_well/rayleigh_benard/README.md)
composes these library components into an FM/CRPS comparison. Its configs,
LOLA and retrofitting protocol differences, batch accounting and measured
runtimes are documented alongside the experiment submitters.
