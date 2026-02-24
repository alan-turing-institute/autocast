"""Hydra override parsing and manipulation utilities."""

from __future__ import annotations

import itertools


def normalized_override(override: str) -> str:
    """Strip leading ``+`` from a Hydra override."""
    return override[1:] if override.startswith("+") else override


def extract_override_value(overrides: list[str], key: str) -> str | None:
    """Return the last value for *key* in a Hydra-style overrides list."""
    for override in reversed(overrides):
        normalized = normalized_override(override)
        if normalized.startswith(f"{key}="):
            return normalized.split("=", 1)[1]
    return None


def contains_override(overrides: list[str], key_prefix: str) -> bool:
    """Return whether any override starts with *key_prefix*."""
    return any(normalized_override(o).startswith(key_prefix) for o in overrides)


def set_override(overrides: list[str], key: str, value: str) -> list[str]:
    """Return *overrides* with *key* set to *value*, replacing any prior value."""
    prefix = f"{key}="
    updated = [o for o in overrides if not normalized_override(o).startswith(prefix)]
    updated.append(f"{key}={value}")
    return updated


def strip_hydra_sweep_controls(overrides: list[str]) -> list[str]:
    """Remove ``hydra.mode`` and ``hydra.sweep.*`` overrides."""
    return [
        o
        for o in overrides
        if not normalized_override(o).startswith(("hydra.mode=", "hydra.sweep."))
    ]


def split_top_level_csv(value: str) -> list[str]:
    """Split a comma-separated *value* respecting quotes and brackets."""
    parts: list[str] = []
    current: list[str] = []
    in_quote: str | None = None
    depth = {"(": 0, "[": 0, "{": 0}
    close_map = {")": "(", "]": "[", "}": "{"}

    for char in value:
        if in_quote is not None:
            current.append(char)
            if char == in_quote:
                in_quote = None
            continue

        if char in {'"', "'"}:
            in_quote = char
            current.append(char)
            continue

        if char in depth:
            depth[char] += 1
            current.append(char)
            continue

        if char in close_map:
            depth[close_map[char]] = max(0, depth[close_map[char]] - 1)
            current.append(char)
            continue

        if char == "," and all(d == 0 for d in depth.values()):
            parts.append("".join(current).strip())
            current = []
            continue

        current.append(char)

    parts.append("".join(current).strip())
    return parts


def expand_sweep_overrides(overrides: list[str]) -> list[list[str]]:
    """Expand comma-separated values into a Cartesian product of job configs."""
    choice_groups: list[list[str]] = []

    for override in overrides:
        norm = normalized_override(override)
        if "=" not in norm:
            choice_groups.append([override])
            continue

        key, value = override.split("=", 1)
        values = split_top_level_csv(value)
        if len(values) <= 1:
            choice_groups.append([override])
            continue

        choice_groups.append([f"{key}={v}" for v in values])

    product_size = 1
    for group in choice_groups:
        product_size *= len(group)

    if product_size > 512:
        raise ValueError(
            f"Refusing to submit {product_size} sweep jobs (>512). "
            "Reduce sweep cardinality."
        )

    return [list(combo) for combo in itertools.product(*choice_groups)]


def hydra_string_list_literal(values: list[str]) -> str:
    """Format *values* as a Hydra list literal ``["a","b"]``."""
    escaped = [v.replace("\\", "\\\\").replace('"', '\\"') for v in values]
    quoted = [f'"{v}"' for v in escaped]
    return f"[{','.join(quoted)}]"
