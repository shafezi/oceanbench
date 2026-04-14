"""Tests for the POMCP policy adapter."""

import numpy as np
import pytest

from oceanbench_core import WaypointGraph
from oceanbench_core.types import ObservationBatch
from oceanbench_models.belief.field.gp import GPFieldModel
from oceanbench_policies.pomdp import (
    BeliefAdapter,
    POMCPAction,
    POMCPConfig,
    POMCPPolicy,
    POMDPObservation,
    POMDPState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REGION = {"lat_min": 24.0, "lat_max": 28.0, "lon_min": -86.0, "lon_max": -82.0}


@pytest.fixture
def small_graph():
    return WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")


@pytest.fixture
def fitted_belief(seed):
    rng = np.random.default_rng(seed)
    n = 30
    obs = ObservationBatch(
        lats=rng.uniform(24, 28, n),
        lons=rng.uniform(-86, -82, n),
        values=20.0 + 0.5 * rng.standard_normal(n),
        variable="temp",
    )
    model = GPFieldModel(
        {"lengthscale": 1.0, "variance": 1.0, "noise": 0.1,
         "include_time": False, "include_depth": False},
        seed=seed,
    )
    model.fit(obs)
    belief = BeliefAdapter(model, variable="temp", objective_c=1.0, measurement_noise_var=0.01)
    belief.seed_observations(obs)
    return belief


@pytest.fixture
def policy(small_graph, fitted_belief):
    config = POMCPConfig(
        max_depth=3,
        discount=0.9,
        uct_c=1.0,
        rollout_schedule="constant",
        rollout_kwargs={"n": 50},
        max_steps=10,
        seed=42,
    )
    return POMCPPolicy(graph=small_graph, belief=fitted_belief, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_act_returns_valid_action(policy, small_graph):
    """act() should return a POMCPAction whose target_node_id is in the graph."""
    nodes = list(small_graph.graph.nodes)
    state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)
    action = policy.act(state)

    assert isinstance(action, POMCPAction)
    assert action.target_node_id in nodes


def test_act_target_is_neighbor_or_self(policy, small_graph):
    """The returned action should be a graph neighbour of the current node (or self)."""
    nodes = list(small_graph.graph.nodes)
    start = nodes[12]
    state = POMDPState(node_id=start, time=np.datetime64("2014-01-01"), step=0)
    action = policy.act(state)

    neighbors = set(small_graph.graph.neighbors(start)) | {start}
    assert action.target_node_id in neighbors


def test_plan_k_steps_returns_list(policy, small_graph):
    """plan_k_steps() should return a non-empty list of POMCPAction."""
    nodes = list(small_graph.graph.nodes)
    state = POMDPState(node_id=nodes[0], time=np.datetime64("2014-01-01"), step=0)
    actions = policy.plan_k_steps(state)

    assert isinstance(actions, list)
    assert len(actions) >= 1
    assert all(isinstance(a, POMCPAction) for a in actions)


def test_observe_updates_belief(policy, small_graph, fitted_belief):
    """Calling observe should not raise and the belief should remain fitted."""
    obs = POMDPObservation(value=21.0, noise_var=0.01, lat=26.0, lon=-84.0)
    policy.observe(obs)
    assert fitted_belief.is_fitted


def test_trajectory_grows(policy, small_graph):
    """After acting, the trajectory log should have entries."""
    nodes = list(small_graph.graph.nodes)
    state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)
    policy.act(state)

    assert len(policy.trajectory) >= 1
    entry = policy.trajectory[0]
    assert "lat" in entry
    assert "lon" in entry
    assert "node_id" in entry


def test_reset_clears_trajectory(policy, small_graph):
    """reset() should clear the trajectory and step counter."""
    nodes = list(small_graph.graph.nodes)
    state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)
    policy.act(state)
    assert len(policy.trajectory) >= 1

    policy.reset()
    assert len(policy.trajectory) == 0
    assert policy.step_count == 0


def test_config_from_mapping():
    """POMCPConfig.from_mapping should parse a nested dict."""
    cfg = {
        "pomcp": {
            "max_depth": 5,
            "discount": 0.8,
            "uct_c": 2.0,
            "exploration": {"strategy": "successive_rejects"},
            "commitment": {"strategy": "fixed_k", "k": 3},
            "rollout_schedule": {"type": "increasing", "min_n": 50, "max_n": 200},
        },
        "seed": 99,
    }
    config = POMCPConfig.from_mapping(cfg)
    assert config.max_depth == 5
    assert config.discount == 0.8
    assert config.exploration_strategy == "successive_rejects"
    assert config.commitment_strategy == "fixed_k"
    assert config.commitment_kwargs["k"] == 3
    assert config.seed == 99
