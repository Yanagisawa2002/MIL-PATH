"""Small file and serialization helpers used across scripts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    iterator = iter(rows)
    first = next(iterator, None)
    if first is None:
        with target.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(first.keys()))
        writer.writeheader()
        writer.writerow(first)
        for row in iterator:
            writer.writerow(row)
