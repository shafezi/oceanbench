#!/usr/bin/env python
"""Compare 4 IPP methods across travel-time budgets.

This script runs Binney (GRG), DLP (persistent sampling), POMCP, and
AdaPP on the same Gulf-of-Mexico scenario and produces:

- RMSE vs travel-time budget plot (mean ± std across seeds)
- Trajectory overlay plots for one representative budget

Results are saved under results/compare_travel_budget/<scenario>/<timestamp>/

Usage
-----
    python examples/compare_4_methods_travel_budget.py

To customise, edit the scenario_cfg dict below or pass a YAML path.
"""

from __future__ import annotations

from pathlib import Path

from oceanbench_bench.compare_travel_budget import (
    ALL_METHODS,
    default_scenario_config,
    run_sweep,
)


def main() -> None:
    scenario_cfg = default_scenario_config()

    # ── Override defaults here if needed ──────────────────────────
    # 2x2 deg region: graph edge ≈ 11 km ≈ 11,000 s at 1 m/s
    # scenario_cfg["T_list"] = [110_000, 220_000, 330_000, 440_000]
    # scenario_cfg["seeds"] = [0, 1, 2]
    # scenario_cfg["T_plot"] = 220_000
    # scenario_cfg["graph_n_lat"] = 15
    # scenario_cfg["graph_n_lon"] = 15

    output_dir = Path("results") / "compare_travel_budget"

    result_dir = run_sweep(
        scenario_cfg=scenario_cfg,
        output_dir=output_dir,
        methods=ALL_METHODS,
    )

    print(f"\nDone. Results at: {result_dir}")


if __name__ == "__main__":
    main()
