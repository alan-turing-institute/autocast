#!/usr/bin/env bash
set -euo pipefail

AUTOCAST_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

WORKDIR=$(mktemp -d)
echo "Working directory: $WORKDIR"
echo "Using autocast from: $AUTOCAST_DIR"

# Clone autosim; use local checkout for autocast
git clone --depth 1 https://github.com/alan-turing-institute/autosim.git "$WORKDIR/autosim"

# Install dependencies
(cd "$WORKDIR/autosim" && uv sync)
(cd "$AUTOCAST_DIR" && uv sync)

# --- 1. Simulate data ---
echo "=== Simulating data ==="
(cd "$WORKDIR/autosim" && uv run autosim \
    simulator=spatiotemporal/advection_diffusion \
    simulator.n=16 \
    seed=42 \
    simulator.T=1.0 simulator.dt=0.1 \
    dataset.n_train=10 dataset.n_valid=3 dataset.n_test=3 \
    dataset.output_dir="$WORKDIR/ad_data")

# --- 2. Train autoencoder ---
echo "=== Training autoencoder ==="
(cd "$AUTOCAST_DIR" && uv run autocast ae \
    --workdir "$WORKDIR/ae_output" \
    ++datamodule.data_path="$WORKDIR/ad_data" \
    ++trainer.max_epochs=10)

# --- 3. Cache latents ---
echo "=== Caching latents ==="
(cd "$AUTOCAST_DIR" && uv run autocast cache-latents \
    --workdir "$WORKDIR/ae_output" \
    --output-dir "$WORKDIR/ae_output/cached_latents" \
    ++datamodule.data_path="$WORKDIR/ad_data" \
    ++autoencoder_checkpoint="$WORKDIR/ae_output/autoencoder.ckpt")

# --- 4. Train processor ---
echo "=== Training processor ==="
(cd "$AUTOCAST_DIR" && uv run autocast processor \
    --workdir "$WORKDIR/proc_output" \
    datamodule=cached_latents \
    ++datamodule.data_path="$WORKDIR/ae_output/cached_latents" \
    ++trainer.max_epochs=10)

# --- 5. Train full encoder-processor-decoder ---
echo "=== Training full EPD ==="
(cd "$AUTOCAST_DIR" && uv run autocast epd \
    --workdir "$WORKDIR/full_epd_output" \
    ++datamodule.data_path="$WORKDIR/ad_data" \
    processor@model.processor=flow_matching \
    ++trainer.max_epochs=10 \
    encoder@model.encoder=dc_deep_256_v2 \
    decoder@model.decoder=dc_deep_256_v2 \
    ++autoencoder_checkpoint="$WORKDIR/ae_output/autoencoder.ckpt")

# --- 6. Evaluate the EPD model ---
echo "=== Evaluating EPD model ==="
(cd "$AUTOCAST_DIR" && uv run autocast eval \
    --workdir "$WORKDIR/full_epd_output")

echo "=== Walkthrough complete ==="
echo "All outputs are in: $WORKDIR"
