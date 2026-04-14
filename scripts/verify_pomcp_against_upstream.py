#!/usr/bin/env python
"""Verify OceanBench POMCP adapter against upstream AdaptiveSamplingPOMCP.

This script runs *both* the upstream POMCP engine and the OceanBench adapter
on a small toy scenario and compares:

  1. First K chosen actions (or distribution if stochastic).
  2. Reward curve trend (monotonicity is not guaranteed, but correlation
     should be high if both run the same algorithm).
  3. Runtime ballpark (should be same order of magnitude).

The goal is NOT byte-identical behaviour, but confirmation that the
OceanBench adapter executes the same algorithmic logic when configured
equivalently.

Usage
-----
    python scripts/verify_pomcp_against_upstream.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- sys.path for both OceanBench packages AND upstream ---
for pkg in [
    "oceanbench-core", "oceanbench-models", "oceanbench-env",
    "oceanbench-policies", "oceanbench-data-provider",
    "oceanbench-tasks", "oceanbench-bench",
]:
    d = REPO_ROOT / pkg
    if d.exists():
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)

# Also add the vendored upstream to sys.path (allowed only in this script).
VENDOR_ROOT = REPO_ROOT / "third_party" / "AdaptiveSamplingPOMCP"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))


# =========================================================================
# Shared toy scenario
# =========================================================================

SEED = 42
N_STATES = 25  # 5x5 grid
N_ACTIONS = 5  # STAY, +X, -X, +Y, -Y
MAX_DEPTH = 5
DISCOUNT = 0.8
UCT_C = 1.0
ROLLOUTS = 200
N_STEPS = 8

REGION = {"lat_min": 25.0, "lat_max": 27.0, "lon_min": -85.0, "lon_max": -83.0}


def _build_toy_gp(seed: int = SEED):
    """Build a fitted GP on a 5x5 grid for both upstream and adapter to use."""
    from oceanbench_core.types import ObservationBatch
    from oceanbench_models.belief.field.gp import GPFieldModel

    rng = np.random.default_rng(seed)
    n = 30
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
    return model


# =========================================================================
# OceanBench adapter run
# =========================================================================

def run_oceanbench_adapter() -> dict:
    """Run the OceanBench POMCP adapter and return actions + rewards."""
    from oceanbench_core import WaypointGraph
    from oceanbench_policies.pomdp import (
        BeliefAdapter, POMCPConfig, POMCPPolicy, POMDPObservation, POMDPState,
    )

    model = _build_toy_gp()
    graph = WaypointGraph.grid(REGION, 5, 5, speed_mps=1.0, connectivity="4")
    belief = BeliefAdapter(model, variable="temp", objective_c=1.0, measurement_noise_var=0.01)

    config = POMCPConfig(
        max_depth=MAX_DEPTH, discount=DISCOUNT, uct_c=UCT_C,
        rollout_schedule="constant", rollout_kwargs={"n": ROLLOUTS},
        max_steps=N_STEPS, seed=SEED,
    )
    policy = POMCPPolicy(graph=graph, belief=belief, config=config)
    rng = np.random.default_rng(SEED)

    nodes = list(graph.graph.nodes)
    start = nodes[12]
    state = POMDPState(node_id=start, time=np.datetime64("2014-01-01"), step=0)

    actions = []
    rewards = []
    t0 = time.perf_counter()

    for step in range(N_STEPS):
        action = policy.act(state, rng=rng)
        actions.append(action.target_node_id)
        reward = belief.reward_at(action.lat, action.lon)
        rewards.append(reward)

        obs = POMDPObservation(
            value=20.0 + rng.normal(0, 0.1),
            noise_var=0.01, lat=action.lat, lon=action.lon,
        )
        policy.observe(obs)
        state = POMDPState(
            node_id=action.target_node_id,
            time=state.time + np.timedelta64(1, "h"),
            step=step + 1,
        )

    elapsed = time.perf_counter() - t0
    return {"actions": actions, "rewards": rewards, "runtime": elapsed}


# =========================================================================
# Upstream run (best-effort, may fail if deps missing)
# =========================================================================

def run_upstream() -> dict | None:
    """Attempt to run the upstream POMCP on the same toy scenario.

    Returns None if upstream dependencies are not available.
    """
    try:
        from pomcp.pomcp import POMCP
        from pomcp.auxilliary import BuildTree
        from sample_sim.planning.pomcp_rollout_allocation.fixed import FixedRolloutAllocator
    except ImportError as e:
        print(f"[upstream] Cannot import upstream modules: {e}")
        print("[upstream] Skipping upstream run. This is expected if upstream deps")
        print("           (smallab, numba, gpytorch, etc.) are not installed.")
        return None

    # Build a simple transition matrix (5x5 grid, 5 actions).
    # Action 0: stay, 1: +row, 2: -row, 3: +col, 4: -col
    n = N_STATES
    ncols = 5
    transition = np.full((n, N_ACTIONS), -1, dtype=np.int32)
    for i in range(n):
        r, c = divmod(i, ncols)
        transition[i, 0] = i  # stay
        transition[i, 1] = i + ncols if r + 1 < 5 else i  # +row
        transition[i, 2] = i - ncols if r - 1 >= 0 else i  # -row
        transition[i, 3] = i + 1 if c + 1 < ncols else i   # +col
        transition[i, 4] = i - 1 if c - 1 >= 0 else i      # -col

    # Synthetic reward: just use a sine function.
    mean_vals = 20.0 + np.sin(np.arange(n).astype(float))
    std_vals = np.ones(n) * 0.5

    def toy_generator(s, a, s_history, extra_data):
        t_matrix, means, stds, obj_c = extra_data
        sprime = int(t_matrix[s, a])
        if sprime == -1:
            s_history.append((None, None))
            return s, s, 0
        o = sprime
        rw = float(means[sprime] + obj_c * stds[sprime])
        s_history.append((sprime, rw))
        return sprime, o, rw

    rs = np.random.RandomState(SEED)
    import logging
    logger_name = "verify_upstream"
    logging.getLogger(logger_name).setLevel(logging.WARNING)

    actions = []
    rewards = []
    start = 12

    import enum
    class ToyAction(enum.Enum):
        STAY = 0
        POS_R = 1
        NEG_R = 2
        POS_C = 3
        NEG_C = 4

    S = list(range(n))
    A = [a.value for a in ToyAction]
    O = list(range(n))
    extra = (transition, mean_vals, std_vals, 1.0)

    t0 = time.perf_counter()
    current = start

    for step in range(N_STEPS):
        allocator = FixedRolloutAllocator(ROLLOUTS)
        pomcp = POMCP(
            toy_generator, random_state=rs, rollout_allocator=allocator,
            logger=logger_name, gamma=DISCOUNT, c=UCT_C,
            start_states=[current], action_enum=ToyAction,
            extra_generator_data=extra,
        )
        pomcp.initialize(S, A, O)
        best_action = pomcp.Search()
        next_state = int(transition[current, best_action])
        reward = float(mean_vals[next_state] + std_vals[next_state])
        actions.append(next_state)
        rewards.append(reward)
        current = next_state

    elapsed = time.perf_counter() - t0
    return {"actions": actions, "rewards": rewards, "runtime": elapsed}


# =========================================================================
# Comparison
# =========================================================================

def compare(ob_result: dict, up_result: dict | None) -> None:
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)

    print(f"\nOceanBench adapter:")
    print(f"  Actions:  {ob_result['actions']}")
    print(f"  Rewards:  {[f'{r:.2f}' for r in ob_result['rewards']]}")
    print(f"  Runtime:  {ob_result['runtime']:.3f}s")

    if up_result is None:
        print("\nUpstream run skipped (dependencies not available).")
        print("To run the full comparison, install upstream dependencies:")
        print("  pip install smallab numba gpytorch tqdm func-timeout")
        return

    print(f"\nUpstream:")
    print(f"  Actions:  {up_result['actions']}")
    print(f"  Rewards:  {[f'{r:.2f}' for r in up_result['rewards']]}")
    print(f"  Runtime:  {up_result['runtime']:.3f}s")

    # Action overlap.
    match = sum(a == b for a, b in zip(ob_result["actions"], up_result["actions"]))
    print(f"\nAction match: {match}/{N_STEPS} ({100*match/N_STEPS:.0f}%)")

    # Reward correlation.
    ob_r = np.array(ob_result["rewards"])
    up_r = np.array(up_result["rewards"])
    if ob_r.std() > 0 and up_r.std() > 0:
        corr = np.corrcoef(ob_r, up_r)[0, 1]
        print(f"Reward correlation: {corr:.3f}")
    else:
        print("Reward correlation: N/A (constant rewards)")

    # Runtime ratio.
    ratio = ob_result["runtime"] / max(up_result["runtime"], 1e-9)
    print(f"Runtime ratio (OB/upstream): {ratio:.2f}x")

    print("\nNote: exact action match is NOT expected — the two implementations")
    print("use different RNG states and belief models.  The goal is to confirm")
    print("the same algorithmic structure (UCB selection, rollout, backprop).")


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("Running OceanBench adapter...")
    ob_result = run_oceanbench_adapter()

    print("Running upstream (if available)...")
    up_result = run_upstream()

    compare(ob_result, up_result)


if __name__ == "__main__":
    main()
