"""Save/load evaluation summaries for reproducibility and analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ModelRunSummary:
    """Summary of a single model run (name, metrics, timings)."""
    model_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0
    n_train: int = 0
    n_eval: int = 0
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Aggregate result of a field-model comparison run."""
    scenario_name: Optional[str] = None
    variable: str = ""
    seed: Optional[int] = None
    runs: List[ModelRunSummary] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EvaluationResult":
        runs = [
            ModelRunSummary(**r) if isinstance(r, dict) else r
            for r in d.get("runs", [])
        ]
        return cls(
            scenario_name=d.get("scenario_name"),
            variable=d.get("variable", ""),
            seed=d.get("seed"),
            runs=runs,
            metadata=d.get("metadata", {}),
        )


def save_results(result: EvaluationResult, path: str | Path) -> None:
    """Write result to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)


def load_results(path: str | Path) -> EvaluationResult:
    """Load result from JSON file."""
    with open(path) as f:
        d = json.load(f)
    return EvaluationResult.from_dict(d)
