"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Set

import yaml


def deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dicts with `extra` overriding `base`."""
    out = dict(base)
    for k, v in extra.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def _load_with_defaults(path: Path, seen: Set[Path]) -> Dict[str, Any]:
    """Recursively load YAML with defaults expansion."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Cyclic config defaults detected at: {path}")
    seen.add(path)

    cfg = _load_yaml(path)
    defaults = cfg.pop("defaults", [])

    merged: Dict[str, Any] = {}
    for entry in defaults:
        child = (path.parent / str(entry)).resolve()
        child_cfg = _load_with_defaults(child, seen=seen)
        merged = deep_merge(merged, child_cfg)

    merged = deep_merge(merged, cfg)
    return merged


def load_config(config_path: str) -> Dict[str, Any]:
    """Load config with recursive defaults expansion."""
    return _load_with_defaults(Path(config_path), seen=set())
