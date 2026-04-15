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
- all remaining dimension `(B,T)` are flattened over the first dimension

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

This is exactly the expected `AttentionMixer` input shape:
- `(batch, n_fidelity_levels, transformer_dim)`

where here `Batch = B*T`, `n_fidelity_levels = sum_C`.