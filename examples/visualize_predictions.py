"""
Visualize predictions and uncertainty from a saved result or from a quick run.

Use compare_field_models.py first to generate results; then optionally point
this script at the saved JSON and a figure path to regenerate plots.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    results_path = REPO_ROOT / "results" / "compare_field_models.json"
    if not results_path.exists():
        print("No results file found at", results_path)
        print("Run examples/compare_field_models.py first.")
        sys.exit(0)
    from oceanbench_bench import load_results
    result = load_results(results_path)
    print("Loaded result:", result.scenario_name, result.variable, len(result.runs), "runs")
    for run in result.runs:
        print(f"  {run.model_name}: {run.metrics}")


if __name__ == "__main__":
    main()
