"""
Receding-horizon Binney et al. (2013) waypoint-planning example.

Demonstrates replanning with a finite planning horizon and execution chunk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_config() -> dict:
    base = REPO_ROOT / "configs"
    with (base / "base.yaml").open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    with (base / "scenarios" / "binney_region.yaml").open("r", encoding="utf-8") as f:
        scenario_cfg = yaml.safe_load(f) or {}
    with (base / "tasks" / "binney_mapping.yaml").open("r", encoding="utf-8") as f:
        task_cfg = yaml.safe_load(f) or {}
    with (base / "policies" / "grg.yaml").open("r", encoding="utf-8") as f:
        policy_cfg = yaml.safe_load(f) or {}

    policy_cfg["policy"]["planning"]["mode"] = "receding_horizon"

    return {
        "seed": base_cfg.get("seed", 42),
        "scenario": scenario_cfg,
        "task": task_cfg,
        "policy": policy_cfg.get("policy", {}),
    }


def main() -> None:
    # For now we reuse the offline runner; a dedicated receding-horizon
    # runner can be added later if needed.
    from oceanbench_bench.runner import run_binney_offline
    from oceanbench_bench.logging import create_run_dir, save_config, save_json

    cfg = _load_config()
    run_root = REPO_ROOT / "results" / "binney2013_receding"
    run_dir = create_run_dir(run_root, run_name="run_seed_{:d}".format(int(cfg.get("seed", 42))))
    save_config(run_dir, cfg)

    result = run_binney_offline(cfg, run_dir=run_dir)
    save_json(run_dir, "metrics.json", result.get("metrics", {}))
    print("Receding-horizon Binney run (offline runner stub) completed.")
    print("Objective value:", result.get("objective_value"))
    print("Metrics:", result.get("metrics"))


if __name__ == "__main__":
    main()

