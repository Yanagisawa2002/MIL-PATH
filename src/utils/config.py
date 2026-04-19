"""YAML config loading with lightweight inheritance support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating the inputs."""
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Config at {path} must be a mapping."
        raise TypeError(msg)
    return data


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment config and recursively resolve `inherit_from`."""
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    parent = config.pop("inherit_from", None)
    if parent is None:
        return config

    parent_path = Path(parent)
    if not parent_path.is_absolute():
        candidates = [
            (config_path.parent / parent).resolve(),
            (config_path.parents[1] / Path(parent).name).resolve(),
            (Path.cwd() / parent).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                parent_path = candidate
                break
        else:
            parent_path = candidates[-1]
    base = load_experiment_config(parent_path)
    return deep_merge(base, config)


def prepare_experiment_config(config: dict[str, Any], repo_root: str | Path | None = None) -> dict[str, Any]:
    """Namespace mutable experiment paths by project name to avoid artifact overwrite."""
    prepared = deepcopy(config)
    project_name = prepared["project"]["name"]
    root = Path(repo_root).resolve() if repo_root is not None else Path.cwd().resolve()
    for key in ("processed_dir", "split_dir", "cache_dir"):
        raw = Path(prepared["paths"][key])
        if not raw.is_absolute():
            raw = (root / raw).resolve()
        if raw.name != project_name:
            raw = raw / project_name
        prepared["paths"][key] = str(raw)
    for key in ("raw_kg_csv", "raw_mech_csv", "raw_mech_json"):
        raw = Path(prepared["paths"][key])
        if not raw.is_absolute():
            raw = (root / raw).resolve()
        prepared["paths"][key] = str(raw)
    return prepared
