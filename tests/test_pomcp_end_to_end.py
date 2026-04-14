"""End-to-end tests for the POMCP policy adapter."""

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
from oceanbench_policies.pomdp.rollout_schedule import (
    ConstantSchedule,
    IncreasingSchedule,
)


REGION = {"lat_min": 25.0, "lat_max": 27.0, "lon_min": -85.0, "lon_max": -83.0}


def _make_env(seed: int = 42):
    """Create a small 5x5 graph, fitted GP model, and synthetic truth."""
    rng = np.random.default_rng(seed)
    graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")

    # Fit GP on random observations.
    n = 20
    obs = ObservationBatch(
        lats=rng.uniform(25, 27, n),
        lons=rng.uniform(-85, -83, n),
        values=20.0 + rng.standard_normal(n),
        variable="temp",
    )
    model = GPFieldModel(
        {"lengthscale": 0.5, "variance": 1.0, "noise": 0.1,
         "include_time": False, "include_depth": False},
        seed=seed,
    )
    model.fit(obs)
    belief = BeliefAdapter(model, variable="temp", objective_c=1.0, measurement_noise_var=0.01)
    belief.seed_observations(obs)

    # Simple synthetic truth: lookup nearest.
    node_lats = np.array([graph.graph.nodes[n]["lat"] for n in graph.graph.nodes])
    node_lons = np.array([graph.graph.nodes[n]["lon"] for n in graph.graph.nodes])
    truth_values = 20.0 + np.sin(node_lats) + np.cos(node_lons)

    return graph, belief, truth_values, node_lats, node_lons


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_episode_runs(self):
        """A full episode should run without errors and produce a trajectory."""
        graph, belief, truth_values, _, _ = _make_env(seed=42)
        config = POMCPConfig(
            max_depth=3, discount=0.9, uct_c=1.0,
            rollout_schedule="constant", rollout_kwargs={"n": 30},
            max_steps=5, seed=42,
        )
        policy = POMCPPolicy(graph=graph, belief=belief, config=config)
        nodes = list(graph.graph.nodes)
        state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)

        for step in range(5):
            action = policy.act(state)
            # Simulate observation.
            obs = POMDPObservation(
                value=20.0 + np.random.default_rng(step).normal(),
                noise_var=0.01, lat=action.lat, lon=action.lon,
            )
            policy.observe(obs)
            state = POMDPState(
                node_id=action.target_node_id,
                time=state.time + np.timedelta64(1, "h"),
                step=step + 1,
            )

        assert len(policy.trajectory) >= 5

    def test_deterministic_under_fixed_seed(self):
        """Two runs with the same seed should produce the same action sequence."""
        actions_run1 = self._run_episode(seed=123)
        actions_run2 = self._run_episode(seed=123)

        assert len(actions_run1) == len(actions_run2)
        for a1, a2 in zip(actions_run1, actions_run2):
            assert a1.target_node_id == a2.target_node_id
            assert a1.lat == a2.lat
            assert a1.lon == a2.lon

    def test_trajectory_stays_within_graph(self):
        """All visited nodes should be valid graph nodes."""
        graph, belief, _, _, _ = _make_env(seed=42)
        config = POMCPConfig(
            max_depth=3, discount=0.9, uct_c=1.0,
            rollout_schedule="constant", rollout_kwargs={"n": 20},
            max_steps=8, seed=42,
        )
        policy = POMCPPolicy(graph=graph, belief=belief, config=config)
        nodes_set = set(graph.graph.nodes)
        state = POMDPState(node_id=list(graph.graph.nodes)[0], time=np.datetime64("2014-01-01"), step=0)

        for step in range(8):
            action = policy.act(state)
            assert action.target_node_id in nodes_set
            obs = POMDPObservation(value=20.0, noise_var=0.01, lat=action.lat, lon=action.lon)
            policy.observe(obs)
            state = POMDPState(
                node_id=action.target_node_id,
                time=state.time + np.timedelta64(1, "h"),
                step=step + 1,
            )

    def test_fixed_k_commitment_returns_multiple_actions(self):
        """With fixed_k=3 commitment, plan_k_steps should return up to 3 actions."""
        graph, belief, _, _, _ = _make_env(seed=42)
        config = POMCPConfig(
            max_depth=5, discount=0.9, uct_c=1.0,
            commitment_strategy="fixed_k", commitment_kwargs={"k": 3},
            rollout_schedule="constant", rollout_kwargs={"n": 50},
            max_steps=10, seed=42,
        )
        policy = POMCPPolicy(graph=graph, belief=belief, config=config)
        nodes = list(graph.graph.nodes)
        state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)

        actions = policy.plan_k_steps(state)
        # Should return between 1 and 3 actions (3 if tree is deep enough).
        assert 1 <= len(actions) <= 3

    def test_exploration_strategy_selection(self):
        """Each exploration strategy should be selectable and run without error."""
        for strategy in ["uct", "successive_rejects", "ugapeb"]:
            graph, belief, _, _, _ = _make_env(seed=42)
            config = POMCPConfig(
                max_depth=3, discount=0.9, uct_c=1.0,
                exploration_strategy=strategy,
                rollout_schedule="constant", rollout_kwargs={"n": 30},
                max_steps=3, seed=42,
            )
            policy = POMCPPolicy(graph=graph, belief=belief, config=config)
            nodes = list(graph.graph.nodes)
            state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)
            action = policy.act(state)
            assert isinstance(action, POMCPAction)

    # --- helpers ---

    @staticmethod
    def _run_episode(seed: int, n_steps: int = 5) -> list[POMCPAction]:
        graph, belief, _, _, _ = _make_env(seed=seed)
        config = POMCPConfig(
            max_depth=3, discount=0.9, uct_c=1.0,
            rollout_schedule="constant", rollout_kwargs={"n": 30},
            max_steps=n_steps, seed=seed,
        )
        policy = POMCPPolicy(graph=graph, belief=belief, config=config)
        rng = np.random.default_rng(seed)
        nodes = list(graph.graph.nodes)
        state = POMDPState(node_id=nodes[12], time=np.datetime64("2014-01-01"), step=0)

        actions = []
        for step in range(n_steps):
            action = policy.act(state, rng=rng)
            actions.append(action)
            obs = POMDPObservation(
                value=20.0 + rng.normal(), noise_var=0.01,
                lat=action.lat, lon=action.lon,
            )
            policy.observe(obs)
            state = POMDPState(
                node_id=action.target_node_id,
                time=state.time + np.timedelta64(1, "h"),
                step=step + 1,
            )
        return actions


class TestRolloutSchedule:
    def test_increasing_schedule_increases(self):
        sched = IncreasingSchedule(min_n=100, max_n=1000)
        r0 = sched.rollouts_for_step(0, 50)
        r49 = sched.rollouts_for_step(49, 50)
        assert r49 > r0

    def test_constant_schedule_is_constant(self):
        sched = ConstantSchedule(n=200)
        vals = [sched.rollouts_for_step(i, 50) for i in range(50)]
        assert all(v == 200 for v in vals)
