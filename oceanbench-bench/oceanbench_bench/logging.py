"""Logging helpers for benchmark runs."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

try:
    import yaml
except Exception:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore[assignment]


def setup_logger(
    name: str = "oceanbench_bench",
    level: int = logging.INFO,
    stream: Optional[sys.__class__] = None,
) -> logging.Logger:
    """Configure and return a logger for benchmark output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        h = logging.StreamHandler(stream or sys.stdout)
        h.setLevel(level)
        logger.addHandler(h)
    return logger


def create_run_dir(base: Path, *, run_name: str) -> Path:
    """
    Create a run directory under ``base``.
    """
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    return run_dir


def save_config(run_dir: Path, config: Mapping[str, Any]) -> None:
    """
    Save the resolved configuration snapshot for a run.
    """
    path = run_dir / "config.yaml"
    if yaml is not None:
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(dict(config), f)
    else:  # pragma: no cover - fallback when PyYAML is unavailable
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


def save_json(run_dir: Path, name: str, payload: Mapping[str, Any]) -> None:
    """
    Save a small JSON artifact (graph, path, metrics).
    """
    path = run_dir / name
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def append_log(run_dir: Path, event: Mapping[str, Any]) -> None:
    """
    Append a single JSONL event to ``logs.jsonl``.
    """
    path = run_dir / "logs.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, default=str) + "\n")

