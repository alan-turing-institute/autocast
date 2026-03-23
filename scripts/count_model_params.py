#!/usr/bin/env python3
"""Count parameters for different processor configurations.

This script instantiates processor models with various configurations
and reports the total number of parameters for comparison.

Usage:
    python scripts/count_model_params.py
    python scripts/count_model_params.py --processor fno
    python scripts/count_model_params.py --processor unet
    python scripts/count_model_params.py --processor azulaunet
    python scripts/count_model_params.py --detailed
"""

import argparse
from typing import Any

from azula.noise import VPSchedule
from torch import nn

from autocast.processors.diffusion import DiffusionProcessor
from autocast.processors.fno import FNOProcessor
from autocast.processors.unet import AzulaUNetProcessor, UNetProcessor
from autocast.processors.vit import AViT


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Parameters
    ----------
    model : nn.Module
        The model to count parameters for.

    Returns
    -------
    dict[str, int]
        Dictionary with 'total', 'trainable', and 'non_trainable' counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def format_number(num: int) -> str:
    """Format large numbers with K, M, B suffixes.

    Parameters
    ----------
    num : int
        Number to format.

    Returns
    -------
    str
        Formatted string (e.g., "1.5M", "234K").
    """
    if num >= 1e9:
        return f"{num / 1e9:.2f}B"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def compare_fno_configs(detailed: bool = False):
    """Compare parameter counts for different FNO configurations."""
    print("\n" + "=" * 80)
    print("FNO PROCESSOR - Parameter Comparison")
    print("=" * 80)

    # Common parameters
    in_channels = 2
    out_channels = 2

    configs = [
        {
            "hidden_channels": 32,
            "n_modes": [16, 16],
            "n_layers": 4,
            "name": "Tiny (32 ch, 16 modes, 4 layers)",
        },
        {
            "hidden_channels": 64,
            "n_modes": [16, 16],
            "n_layers": 4,
            "name": "Small (64 ch, 16 modes, 4 layers)",
        },
        {
            "hidden_channels": 64,
            "n_modes": [32, 32],
            "n_layers": 4,
            "name": "Small-HiRes (64 ch, 32 modes, 4 layers)",
        },
        {
            "hidden_channels": 128,
            "n_modes": [16, 16],
            "n_layers": 4,
            "name": "Medium (128 ch, 16 modes, 4 layers)",
        },
        {
            "hidden_channels": 128,
            "n_modes": [32, 32],
            "n_layers": 4,
            "name": "Medium-HiRes (128 ch, 32 modes, 4 layers)",
        },
        {
            "hidden_channels": 128,
            "n_modes": [32, 32],
            "n_layers": 6,
            "name": "Medium-Deep (128 ch, 32 modes, 6 layers)",
        },
        {
            "hidden_channels": 256,
            "n_modes": [16, 16],
            "n_layers": 8,
            "name": "Large (256 ch, 16 modes, 8 layers)",
        },
        {
            "hidden_channels": 256,
            "n_modes": [32, 32],
            "n_layers": 8,
            "name": "XLarge (256 ch, 32 modes, 8 layers)",
        },
    ]

    results = []
    for config in configs:
        processor = FNOProcessor(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=config["n_modes"],
            hidden_channels=config["hidden_channels"],
            n_layers=config["n_layers"],
        )
        params = count_parameters(processor)
        results.append((config["name"], params, config))

    # Print table
    print(f"\n{'Configuration':<45} {'Total Params':<15}")
    print("-" * 75)
    for name, params, config in results:
        print(f"{name:<45} {format_number(params['total']):<15}")

    if detailed:
        print("\nDetailed breakdown:")
        for name, params, config in results:
            print(f"\n{name}:")
            print(f"  Hidden channels: {config['hidden_channels']}")
            print(f"  N modes:         {config['n_modes']}")
            print(f"  N layers:        {config['n_layers']}")
            print(f"  Total:           {params['total']:,}")
            print(f"  Trainable:       {params['trainable']:,}")
            print(f"  Non-trainable:   {params['non_trainable']:,}")


def compare_vit_configs(detailed: bool = False):
    """Compare parameter counts for different ViT configurations."""
    from autocast.processors.vit import AViTProcessor

    print("\n" + "=" * 80)
    print("VISION TRANSFORMER (ViT) PROCESSOR - Parameter Comparison")
    print("=" * 80)

    # Common parameters
    in_channels = 2
    out_channels = 2
    spatial_resolution = (64, 64)

    configs = [
        {
            "hidden_dim": 192,
            "num_heads": 3,
            "n_layers": 4,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "Tiny (192 dim, 4 layers)",
        },
        {
            "hidden_dim": 256,
            "num_heads": 4,
            "n_layers": 6,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "Small (256 dim, 6 layers)",
        },
        {
            "hidden_dim": 384,
            "num_heads": 6,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "Medium (384 dim, 8 layers)",
        },
        {
            "hidden_dim": 384,
            "num_heads": 6,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": 1024,
            "name": "Medium-Noise (384 dim, 8 layers, noise=1024)",
        },
        {
            "hidden_dim": 384,
            "num_heads": 6,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": 2048,
            "name": "Medium-Noise (384 dim, 8 layers, noise=2048)",
        },
        {
            "hidden_dim": 384,
            "num_heads": 6,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": 4096,
            "name": "Medium-Noise (384 dim, 8 layers, noise=4096)",
        },
        {
            "hidden_dim": 768,
            "num_heads": 12,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "Large (768 dim, 8 layers)",
        },
        {
            "hidden_dim": 768,
            "num_heads": 12,
            "n_layers": 8,
            "patch_size": 8,
            "n_noise_channels": 4096,
            "name": "Large-Noise (768 dim, 8 layers, noise=4096)",
        },
        {
            "hidden_dim": 768,
            "num_heads": 12,
            "n_layers": 12,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "Large-Deep (768 dim, 12 layers)",
        },
        {
            "hidden_dim": 1024,
            "num_heads": 16,
            "n_layers": 12,
            "patch_size": 8,
            "n_noise_channels": None,
            "name": "XLarge (1024 dim, 12 layers)",
        },
    ]

    results = []
    for config in configs:
        try:
            processor = AViTProcessor(
                in_channels=in_channels,
                out_channels=out_channels,
                spatial_resolution=spatial_resolution,
                hidden_dim=config["hidden_dim"],
                num_heads=config["num_heads"],
                n_layers=config["n_layers"],
                patch_size=config["patch_size"],
                n_noise_channels=config["n_noise_channels"],
            )
            params = count_parameters(processor)
            results.append((config["name"], params, config))
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue

    # Print table
    print(f"\n{'Configuration':<50} {'Total Params':<15}")
    print("-" * 65)
    for name, params, config in results:
        print(f"{name:<50} {format_number(params['total']):<15}")

    if detailed:
        print("\nDetailed breakdown:")
        for name, params, config in results:
            print(f"\n{name}:")
            print(f"  Hidden dim:       {config['hidden_dim']}")
            print(f"  Num heads:        {config['num_heads']}")
            print(f"  N layers:         {config['n_layers']}")
            print(f"  Patch size:       {config['patch_size']}")
            print(f"  N noise channels: {config['n_noise_channels']}")
            print(f"  Total:            {params['total']:,}")
            print(f"  Trainable:        {params['trainable']:,}")
            print(f"  Non-trainable:    {params['non_trainable']:,}")


def compare_diffusion_configs(detailed: bool = False):
    """Compare parameter counts for Diffusion with ViT backbone."""
    from autocast.nn.vit import TemporalViTBackbone

    print("\n" + "=" * 80)
    print("DIFFUSION PROCESSOR (with ViT backbone) - Parameter Comparison")
    print("=" * 80)

    # Common parameters
    in_channels = 2
    out_channels = 2
    schedule = VPSchedule()

    configs = [
        {
            "hid_channels": 192,
            "attention_heads": 3,
            "hid_blocks": 4,
            "patch_size": 4,
            "name": "Tiny (192 dim, 4 layers, ps=4)",
        },
        {
            "hid_channels": 256,
            "attention_heads": 4,
            "hid_blocks": 6,
            "patch_size": 4,
            "name": "Small (256 dim, 6 layers, ps=4)",
        },
        {
            "hid_channels": 384,
            "attention_heads": 6,
            "hid_blocks": 8,
            "patch_size": 1,
            "name": "Medium-Fine (384 dim, 8 layers, ps=1)",
        },
        {
            "hid_channels": 384,
            "attention_heads": 6,
            "hid_blocks": 8,
            "patch_size": 4,
            "name": "Medium (384 dim, 8 layers, ps=4)",
        },
        {
            "hid_channels": 384,
            "attention_heads": 6,
            "hid_blocks": 8,
            "patch_size": 8,
            "name": "Medium-Coarse (384 dim, 8 layers, ps=8)",
        },
        {
            "hid_channels": 768,
            "attention_heads": 12,
            "hid_blocks": 8,
            "patch_size": 4,
            "name": "Large (768 dim, 8 layers, ps=4)",
        },
        {
            "hid_channels": 768,
            "attention_heads": 12,
            "hid_blocks": 12,
            "patch_size": 4,
            "name": "Large-Deep (768 dim, 12 layers, ps=4)",
        },
        {
            "hid_channels": 1024,
            "attention_heads": 16,
            "hid_blocks": 12,
            "patch_size": 4,
            "name": "XLarge (1024 dim, 12 layers, ps=4)",
        },
    ]

    results = []
    for config in configs:
        try:
            # Create ViT backbone for diffusion
            backbone = TemporalViTBackbone(
                in_channels=in_channels,
                out_channels=out_channels,
                cond_channels=in_channels,
                n_steps_output=4,
                n_steps_input=1,
                include_global_cond=False,
                global_cond_channels=None,
                mod_features=256,
                hid_channels=config["hid_channels"],
                attention_heads=config["attention_heads"],
                hid_blocks=config["hid_blocks"],
                patch_size=config["patch_size"],
                spatial=2,
            )

            processor = DiffusionProcessor(
                backbone=backbone,
                schedule=schedule,
                n_steps_output=4,
                n_channels_out=out_channels,
            )
            params = count_parameters(processor)
            results.append((config["name"], params, config))
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue

    # Print table
    print(f"\n{'Configuration':<45} {'Total Params':<15}")
    print("-" * 75)
    for name, params, config in results:
        print(f"{name:<45} {format_number(params['total']):<15}")

    if detailed:
        print("\nDetailed breakdown:")
        for name, params, config in results:
            print(f"\n{name}:")
            print(f"  Hid channels:     {config['hid_channels']}")
            print(f"  Attention heads:  {config['attention_heads']}")
            print(f"  Hid blocks:       {config['hid_blocks']}")
            print(f"  Patch size:       {config['patch_size']}")
            print(f"  Total:            {params['total']:,}")
            print(f"  Trainable:        {params['trainable']:,}")
            print(f"  Non-trainable:    {params['non_trainable']:,}")


def compare_unet_configs(detailed: bool = False):
    """Compare parameter counts for different Classic UNet configurations."""
    print("\n" + "=" * 80)
    print("CLASSIC UNET PROCESSOR - Parameter Comparison")
    print("=" * 80)

    # Common parameters
    in_channels = 2
    out_channels = 2
    spatial_resolution = (64, 64)
    n_spatial_dims = 2

    configs = [
        {
            "init_features": 16,
            "name": "Tiny (16 init features)",
        },
        {
            "init_features": 32,
            "name": "Small (32 init features)",
        },
        {
            "init_features": 48,
            "name": "Medium (48 init features)",
        },
        {
            "init_features": 64,
            "name": "Large (64 init features)",
        },
        {
            "init_features": 96,
            "name": "XLarge (96 init features)",
        },
        {
            "init_features": 128,
            "name": "XXLarge (128 init features)",
        },
    ]

    results = []
    for config in configs:
        try:
            processor = UNetProcessor(
                in_channels=in_channels,
                out_channels=out_channels,
                spatial_resolution=spatial_resolution,
                n_spatial_dims=n_spatial_dims,
                init_features=config["init_features"],
                gradient_checkpointing=False,
            )
            params = count_parameters(processor)
            results.append((config["name"], params, config))
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue

    # Print table
    print(f"\n{'Configuration':<40} {'Total Params':<15}")
    print("-" * 55)
    for name, params, config in results:
        print(f"{name:<40} {format_number(params['total']):<15}")

    if detailed:
        print("\nDetailed breakdown:")
        for name, params, config in results:
            print(f"\n{name}:")
            print(f"  Init features:    {config['init_features']}")
            print(f"  Total:            {params['total']:,}")
            print(f"  Trainable:        {params['trainable']:,}")
            print(f"  Non-trainable:    {params['non_trainable']:,}")


def compare_azulaunet_configs(detailed: bool = False):
    """Compare parameter counts for different Azula UNet configurations."""
    print("\n" + "=" * 80)
    print("AZULA UNET PROCESSOR - Parameter Comparison")
    print("=" * 80)

    # Common parameters
    in_channels = 2
    out_channels = 2

    configs = [
        {
            "hid_channels": (32, 64, 128, 256),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": None,
            "name": "Tiny (32-256 ch, 2 blocks)",
        },
        {
            "hid_channels": (64, 128, 256, 512),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": None,
            "name": "Small (64-512 ch, 2 blocks)",
        },
        {
            "hid_channels": (64, 128, 256, 512),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": 256,
            "name": "Small-Noise (64-512 ch, noise=256)",
        },
        {
            "hid_channels": (64, 128, 256, 512),
            "hid_blocks": (3, 3, 3, 3),
            "n_noise_channels": None,
            "name": "Small-Deep (64-512 ch, 3 blocks)",
        },
        {
            "hid_channels": (64, 128, 256, 512),
            "hid_blocks": (3, 3, 3, 3),
            "n_noise_channels": 256,
            "name": "Small-Deep-Noise (64-512 ch, 3 blocks, noise=256)",
        },
        {
            "hid_channels": (96, 192, 384, 768),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": None,
            "name": "Medium (96-768 ch, 2 blocks)",
        },
        {
            "hid_channels": (96, 192, 384, 768),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": 256,
            "name": "Medium-Noise (96-768 ch, noise=256)",
        },
        {
            "hid_channels": (128, 256, 512, 1024),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": None,
            "name": "Large (128-1024 ch, 2 blocks)",
        },
        {
            "hid_channels": (128, 256, 512, 1024),
            "hid_blocks": (2, 2, 2, 2),
            "n_noise_channels": 256,
            "name": "Large-Noise (128-1024 ch, noise=256)",
        },
    ]

    results = []
    for config in configs:
        try:
            processor = AzulaUNetProcessor(
                in_channels=in_channels,
                out_channels=out_channels,
                hid_channels=config["hid_channels"],
                hid_blocks=config["hid_blocks"],
                norm="group",
                groups=8,
                ffn_factor=2,
                dropout=0.0,
                periodic=False,
                gradient_checkpointing=False,
                n_noise_channels=config["n_noise_channels"],
            )
            params = count_parameters(processor)
            results.append((config["name"], params, config))
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            continue

    # Print table
    print(f"\n{'Configuration':<50} {'Total Params':<15}")
    print("-" * 65)
    for name, params, config in results:
        print(f"{name:<50} {format_number(params['total']):<15}")

    if detailed:
        print("\nDetailed breakdown:")
        for name, params, config in results:
            print(f"\n{name}:")
            print(f"  Hid channels:     {config['hid_channels']}")
            print(f"  Hid blocks:       {config['hid_blocks']}")
            print(f"  N noise channels: {config['n_noise_channels']}")
            print(f"  Total:            {params['total']:,}")
            print(f"  Trainable:        {params['trainable']:,}")
            print(f"  Non-trainable:    {params['non_trainable']:,}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Count parameters for different processor configurations"
    )
    parser.add_argument(
        "--processor",
        type=str,
        choices=[
            "fno",
            "vit",
            "diffusion",
            "unet",
            "azulaunet",
            "all",
        ],
        default="all",
        help="Which processor to analyze (default: all)",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed parameter breakdown",
    )

    args = parser.parse_args()

    if args.processor in ["fno", "all"]:
        compare_fno_configs(detailed=args.detailed)

    if args.processor in ["vit", "all"]:
        compare_vit_configs(detailed=args.detailed)

    if args.processor in ["diffusion", "all"]:
        compare_diffusion_configs(detailed=args.detailed)

    if args.processor in ["unet", "all"]:
        compare_unet_configs(detailed=args.detailed)

    if args.processor in ["azulaunet", "all"]:
        compare_azulaunet_configs(detailed=args.detailed)

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
