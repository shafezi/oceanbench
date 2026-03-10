"""Pytest fixtures and path setup for OceanBench tests."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# So that "import oceanbench_core" etc. work when running tests from repo root
for pkg in ["oceanbench-core", "oceanbench-env", "oceanbench-models", "oceanbench-tasks", "oceanbench-bench", "oceanbench-policies", "oceanbench-data-provider"]:
    d = REPO_ROOT / pkg
    if d.exists():
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def small_observation_batch(seed):
    from oceanbench_core.types import ObservationBatch
    rng = np.random.default_rng(seed)
    n = 50
    return ObservationBatch(
        lats=rng.uniform(24, 36, n),
        lons=rng.uniform(-86, -74, n),
        values=20.0 + 0.1 * rng.standard_normal(n),
        variable="temp",
    )


@pytest.fixture
def small_query_points(seed):
    from oceanbench_core.types import QueryPoints
    rng = np.random.default_rng(seed + 1)
    n = 30
    return QueryPoints(
        lats=rng.uniform(24, 36, n),
        lons=rng.uniform(-86, -74, n),
    )
