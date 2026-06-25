# MultiEncoder and AttentionMixer 

## 1. Multi-Encoder

Each encoder in `MultiEncoder.encoders` produces:

- Input-like tensor per level: `(B, T, S1, S2, ..., C_in_i)`, where `S1, S2, ...` are spatial dimensions in the ambient space
- Encoded latent per level: `(B, T, L1_i, L2_i, ..., C_i)`, where `L1_i, L2_i, ...` and `C_i` are latent spatial dimensions and latent channels, respectively, for encoder `i`.

So we obtain:

- `List[(B, T, L1_i, L2_i, ..., C_i)]` for `i = 1:D`, where `D` is the number of datasets/encoders. Note that each encoder can potentially have different latent dimensions and different number of channels.

## 2. AttentionMixer

The latent embeddings need to be passed to the `AttentionMixer`, a trasnformer architecture which needs to operate on data of shape `(batch, n_fidelity_levels, transformer_dim)`, computing attention on vectors of size `transformer_dim` (transformer embedding dimension) over `n_fidelity_levels`. 
In order to do so:
- `n_fidelity_levels`, we consider as a different fidelity level each channel of each dataset and we compute attention over latent embedding of each. Therefore we `n_fidelity_leves = sum C_i`.
- `transformer_dim`, the AntentionMixer (=transformer) need to compute attention over vectors of the same size. Therefore we first flatten spatial dimesnion `(L1_i, L2_i, ...) -> (L1_i*L2_i*...)` and then we expand (or reduce) to `transformer_dim` by linear projection which is either provided by the user or is taken as the maximum flattent latent spatial dimensions (i.e., `max {L1_i*L2_i*...}`).
- all remaining dimension `(B,T)` are flattened over the first dimension. If multiple masks are considered, also masks are flattend over first dimension.

In practice:

For each encoder output `i`, we flatten the latent dimension first

1. `(B, T, L1_i, L2_i, ..., C_i) -> (B, T, L_i_flat, C_i)`, where `L_i_flat = L1_i * L2_i * ...` is flattened latent size for encoder `i`. 
2. Transpose to move channels into token axis:
   - `(B, T, L_i_flat, C_i) -> (B, T, C_i, L_i_flat)`
3. apply per-level linear projection on the last axis `Linear(L_i_flat -> transformer_dim)`:
   - `(B, T, C_i, L_i_flat) -> (B, T, C_i, transformer_dim)`

Therefore we get:
`List[(B, T, C_i, transformer_dim)]`

Then we

4. Concatenate all levels along channel-token axis:
- `List[(B, T, C_i, transformer_dim)] -> (B, T, sum_C, transformer_dim)`


5. Flatten all leading dimensions except the last two for attention:
   - `(B, T, sum_C, transformer_dim) -> (B*T, sum_C, transformer_dim)`

**Multiple Masks Handling:**

If multiple masks are applied (e.g., for missing data ablations or combinatorial masking), the data and masks are both expanded to include a mask axis (let's call it $M$ for the number of mask combinations). For each mask combination, the data is repeated (not just the mask), so that for each sample in the batch, you have a copy of the data for each mask scenario.

**Shape transformation:**

- Original data shape: $(B, T, \ldots)$
- With $M$ masks: data is expanded to $(B, T, M, \ldots)$
- Before passing to AttentionMixer, all leading dimensions $(B, T, M)$ are flattened into a single batch axis: $(B \times T \times M, \ldots)$

**Explanation:**

This means that for each mask scenario, the model sees the same data but with a different mask applied. The batch size for the transformer is effectively multiplied by the number of mask combinations. This allows the transformer to process all mask scenarios in parallel, and the output can later be unflattened to recover the $(B, T, M, \ldots)$ structure if needed.

**Mask flattening:**
The mask tensor is expanded and flattened in exactly the same way as the data, so that each data sample aligns with its corresponding mask scenario.

**Summary:**
- Data and masks are repeated for each mask scenario.
- All leading dimensions (including mask) are flattened into the batch axis for attention.

This ensures that each mask scenario is processed independently, but efficiently, in a single forward pass.

This is exactly the expected `AttentionMixer` input shape:
- `(batch, n_fidelity_levels, transformer_dim)`

where here `Batch = B*T` (or `B*T*M` when using mask ensembles), `n_fidelity_levels = sum_C`.

## 3. After AttentionMixer

Attention output (no mask case):

- `(B*T, sum_C, transformer_dim) -> (B, T, sum_C, transformer_dim)`

Attention output (masked case with `M` scenarios):

- The mixer operates on `(B*T*M, sum_C, transformer_dim)`.
- After mixing, the implementation keeps mask scenarios as independent batch items for decoding:
   `(B, T, M, sum_C, transformer_dim) -> (B*M, T, sum_C, transformer_dim)`.
- So the decoder sees shape `(B*M, T, L1, L2, ..., C)` after projection/unflattening.

Then project back from transformer width to a target latent flat size:

- `Linear(transformer_dim -> max_i L_i_flat)` and then `(B, T, sum_C, transformer_dim) -> (B, T, sum_C, L_target_flat)`
- Transpose back to latent-last layout: `(B, T, sum_C, L_target_flat) -> (B, T, L_target_flat, sum_C)`
- Unflatten to target latent spatial shape (currently chosen from the encoder with largest flattened latent size):  `(B, T, L_target_flat, sum_C) -> (B, T, L1_target, L2_target, ..., sum_C)`. In this way, data are restored to their originala dimension (pre-attention mixer), such that AttentionMixer is not modifying their shape, which is consistent wether AttentionMixer is use dor not. 

## 4. Masked attention 

`mask` is expected as `TensorDBM` with shape `(D, B, M)`, where `D` is the number of datasets, `B` is batch size, and `M` is the number of masking scenarios.
For the 2-level case and `M=3`, the scenarios are:

- LF1 only available: `[False, True]`
- LF2 only available: `[True, False]`
- both available: `[False, False]`

where `False` means available and `True` means masked (missing).
Note that we assume masking is applied to all channels of a given dataset, that is either all channels are avialable or all are masked.

The code maps dataset-level masks to channel-token masks:

1. `(D, B, M) -> (B, M, D)`
2. repeat each dataset mask `C_i` times and get `(B, M, sum_C)`
4. broadcast over extra batch dims and flatten to `(B*T*M, sum_C)`

Data is similarly expanded to include `M` and flattened to match attention call.

Important semantic detail: in `AttentionMixer` / `MultiheadAttention`, `True` in `levels_mask` means masked (missing) token.




## Compact summary

- Encode per level: `List[(B, T, L1_i, L2_i, ..., C_i)]`
- Flatten latent space: `List[(B, T, L_i_flat, C_i)]`
- Project flattened latent axis to transformer width: `List[(B, T, C_i, transformer_dim)]`
- Concatenate levels: `(B, T, sum_C, transformer_dim)`
- Attention input: `(B*T, sum_C, transformer_dim)` or `(B*T*M, sum_C, transformer_dim)` with mask ensembles
- Project back to latent flat size, transpose, and unflatten:
   `(B, T, L1_target, L2_target, ..., sum_C)` (no mask)
   or `(B*M, T, L1_target, L2_target, ..., sum_C)` (masked with ensembles)

