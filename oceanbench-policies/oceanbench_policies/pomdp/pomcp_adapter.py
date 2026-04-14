"""OceanBench POMCP policy adapter.

This module provides :class:`POMCPPolicy`, an online step-wise policy that
wraps the upstream AdaptiveSamplingPOMCP tree-search with OceanBench types.

The adapter:

1. Translates OceanBench :class:`WaypointGraph` nodes into a discrete
   state/action space consumable by the upstream POMCP engine.
2. Replaces the upstream ``Generator`` function with one backed by an
   OceanBench :class:`FieldBeliefModel` (via :class:`BeliefAdapter`).
3. Exposes ``act()`` and ``plan_k_steps()`` conforming to the OceanBench
   policy interface.
4. Delegates exploration, commitment, and rollout-schedule choices to the
   corresponding strategy objects.
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from oceanbench_core import WaypointGraph
from oceanbench_core.types import ObservationBatch, QueryPoints

from .state_models import BeliefAdapter, POMCPAction, POMDPObservation, POMDPState
from .exploration import (
    ExplorationStrategy,
    UCTExploration,
    build_exploration_strategy,
)
from .commitment import (
    CommitmentStrategy,
    NoneCommitment,
    TreeSnapshot,
    build_commitment_strategy,
)
from .rollout_schedule import (
    RolloutSchedule,
    ConstantSchedule,
    build_rollout_schedule,
)
from .utils import (
    build_transition_table,
    discretize_observation,
    node_coords,
    travel_time_seconds,
    advance_time,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class POMCPConfig:
    """Full configuration for the POMCP policy adapter."""

    # Tree search
    max_depth: int = 10
    discount: float = 0.95
    uct_c: float = 1.0

    # Action space
    action_space: str = "graph_neighbors"  # "graph_neighbors" | "knn"
    knn_k: int = 8
    include_stay: bool = True

    # Belief / reward
    objective_c: float = 1.0  # coefficient in mean + c*std

    # Observation discretisation
    obs_n_bins: int = 20
    obs_lo: float = -5.0
    obs_hi: float = 5.0

    # Exploration strategy
    exploration_strategy: str = "uct"
    exploration_kwargs: dict[str, Any] = field(default_factory=dict)

    # Commitment strategy
    commitment_strategy: str = "none"
    commitment_kwargs: dict[str, Any] = field(default_factory=dict)

    # Rollout schedule
    rollout_schedule: str = "constant"
    rollout_kwargs: dict[str, Any] = field(default_factory=dict)

    # Episode
    max_steps: int = 50

    # Reproducibility
    seed: int = 42

    @classmethod
    def from_mapping(cls, cfg: Mapping[str, Any]) -> "POMCPConfig":
        """Build from a flat or nested mapping (e.g. parsed YAML)."""
        pomcp = dict(cfg.get("pomcp", cfg))

        exploration_cfg = dict(pomcp.get("exploration", {}))
        commitment_cfg = dict(pomcp.get("commitment", {}))
        rollout_cfg = dict(pomcp.get("rollout_schedule", {}))
        belief_cfg = dict(pomcp.get("belief", {}))

        return cls(
            max_depth=int(pomcp.get("max_depth", 10)),
            discount=float(pomcp.get("discount", 0.95)),
            uct_c=float(pomcp.get("uct_c", 1.0)),
            action_space=str(pomcp.get("action_space", "graph_neighbors")),
            knn_k=int(pomcp.get("knn_k", 8)),
            include_stay=bool(pomcp.get("include_stay", True)),
            objective_c=float(belief_cfg.get("objective_c", pomcp.get("objective_c", 1.0))),
            obs_n_bins=int(pomcp.get("obs_n_bins", 20)),
            obs_lo=float(pomcp.get("obs_lo", -5.0)),
            obs_hi=float(pomcp.get("obs_hi", 5.0)),
            exploration_strategy=str(exploration_cfg.get("strategy", "uct")),
            exploration_kwargs={
                k: v
                for k, v in exploration_cfg.items()
                if k != "strategy"
            },
            commitment_strategy=str(commitment_cfg.get("strategy", "none")),
            commitment_kwargs={
                k: v
                for k, v in commitment_cfg.items()
                if k != "strategy"
            },
            rollout_schedule=str(rollout_cfg.get("type", "constant")),
            rollout_kwargs={k: v for k, v in rollout_cfg.items() if k != "type"},
            max_steps=int(pomcp.get("max_steps", cfg.get("max_steps", 50))),
            seed=int(cfg.get("seed", pomcp.get("seed", 42))),
        )


# ---------------------------------------------------------------------------
# Lightweight internal POMCP engine (pure-Python, no upstream deps)
# ---------------------------------------------------------------------------
# The upstream POMCP code is tightly coupled to its own GP, memory-mapped
# state spaces, and environment.  Rather than wrestling those imports, we
# reimplement the core UCB tree-search loop in ~120 lines.  The algorithmic
# logic (Simulate / Rollout / UCB) is identical to pomcp/pomcp.py; the
# difference is that the Generator is an OceanBench BeliefAdapter.
# ---------------------------------------------------------------------------


class _TreeNode:
    """A node in the POMCP search tree."""

    __slots__ = ("children", "visit_count", "reward_history", "is_action", "belief_particles")

    def __init__(self, *, is_action: bool = False) -> None:
        self.children: dict[int, _TreeNode] = {}
        self.visit_count: int = 0
        self.reward_history: list[float] = []
        self.is_action: bool = is_action
        self.belief_particles: list[int] = []  # state indices

    @property
    def value(self) -> float:
        if not self.reward_history:
            return 0.0
        return float(np.mean(self.reward_history))

    @property
    def value_std(self) -> float:
        if len(self.reward_history) < 2:
            return 0.0
        return float(np.std(self.reward_history))


class _POMCPEngine:
    """Minimal POMCP tree-search engine backed by BeliefAdapter.

    This reimplements the core algorithm from ``pomcp/pomcp.py`` without
    the upstream's dependency on ``smallab``, ``memory_mapped_dict``,
    ``numba``, or its own GP classes.
    """

    def __init__(
        self,
        actions: list[int],           # action indices (target node ids)
        transition: dict[int, list[int]],  # node_id -> [reachable node ids]
        belief: BeliefAdapter,
        graph: WaypointGraph,
        config: POMCPConfig,
        rng: np.random.Generator,
    ) -> None:
        self.actions = actions
        self.n_actions = len(actions)
        self.transition = transition
        self.belief = belief
        self.graph = graph
        self.cfg = config
        self.rng = rng

        # The tree root is keyed by -1 (convention from upstream).
        self.root = _TreeNode()

    # ----- public API -----------------------------------------------------

    def search(self, start_state: int, rollout_budget: int, exploration: ExplorationStrategy) -> None:
        """Run POMCP search from *start_state* for *rollout_budget* rollouts."""
        # Expand root if needed.
        if not self.root.children:
            for a_idx in range(self.n_actions):
                self.root.children[a_idx] = _TreeNode(is_action=True)

        rewards_per_arm: list[list[float]] = [
            self.root.children[i].reward_history for i in range(self.n_actions)
        ]

        rollouts_used = 0
        while exploration.should_continue(rewards_per_arm, rollouts_used):
            s = start_state
            self._simulate(s, self.root, 0)
            rollouts_used += 1

    def best_action_index(self, exploration: ExplorationStrategy) -> int:
        """Return the best root action index after search."""
        rewards_per_arm = [
            self.root.children[i].reward_history
            for i in range(self.n_actions)
        ]
        return exploration.best_arm(rewards_per_arm)

    def critical_path(self, max_depth: int) -> list[tuple[int, _TreeNode]]:
        """Trace the best path from root, returning ``[(action_idx, action_node), ...]``."""
        path: list[tuple[int, _TreeNode]] = []
        node = self.root
        for _ in range(max_depth):
            if not node.children:
                break
            # Pick child with highest mean value.
            best_a = max(
                node.children,
                key=lambda a: node.children[a].value,
            )
            action_node = node.children[best_a]
            path.append((best_a, action_node))
            # Descend through observation: pick most-visited child.
            if not action_node.children:
                break
            best_obs = max(
                action_node.children,
                key=lambda o: action_node.children[o].visit_count,
            )
            node = action_node.children[best_obs]
        return path

    # ----- internal -------------------------------------------------------

    def _simulate(self, s: int, node: _TreeNode, depth: int) -> float:
        gamma = self.cfg.discount
        if (gamma ** depth < 1e-4 and depth > 0) or depth >= self.cfg.max_depth:
            return 0.0

        # Leaf expansion.
        if node.visit_count == 0 and not node.is_action:
            if not node.children:
                for a_idx in range(self.n_actions):
                    node.children[a_idx] = _TreeNode(is_action=True)
            rollout_val = self._rollout(s, depth)
            node.visit_count += 1
            return rollout_val

        # Select action via UCB.
        a_idx = self._select_action_ucb(node)
        action_node = node.children[a_idx]

        # Simulate transition.
        s_prime = self._transition(s, a_idx)
        reward = self._reward(s_prime)
        obs = self._observe(s_prime)

        # Get or create observation child.
        if obs not in action_node.children:
            action_node.children[obs] = _TreeNode()
        obs_node = action_node.children[obs]

        # Recurse.
        future = gamma * self._simulate(s_prime, obs_node, depth + 1)
        total = reward + future

        # Backprop.
        node.visit_count += 1
        action_node.visit_count += 1
        action_node.reward_history.append(total)
        return total

    def _rollout(self, s: int, depth: int) -> float:
        gamma = self.cfg.discount
        if (gamma ** depth < 1e-4 and depth > 0) or depth >= self.cfg.max_depth:
            return 0.0
        a_idx = self.rng.integers(self.n_actions)
        s_prime = self._transition(s, a_idx)
        reward = self._reward(s_prime)
        return reward + gamma * self._rollout(s_prime, depth + 1)

    def _select_action_ucb(self, node: _TreeNode) -> int:
        c = self.cfg.uct_c
        N = max(node.visit_count, 1)
        best_val = -np.inf
        best_a = 0
        for a_idx, child in node.children.items():
            if child.visit_count == 0:
                return a_idx
            ucb = child.value + c * math.sqrt(math.log(N) / child.visit_count)
            if ucb > best_val:
                best_val = ucb
                best_a = a_idx
        return best_a

    def _transition(self, s: int, action_idx: int) -> int:
        """Return next state index given current state and action index."""
        reachable = self.transition.get(s)
        if reachable is None or action_idx >= len(reachable):
            return s  # stay
        return reachable[action_idx]

    def _reward(self, state_idx: int) -> float:
        """Compute reward at a state via the belief adapter."""
        lat, lon = node_coords(self.graph, state_idx)
        return self.belief.reward_at(lat, lon)

    def _observe(self, state_idx: int) -> int:
        """Generate a discretised observation at a state."""
        lat, lon = node_coords(self.graph, state_idx)
        pred = self.belief.predict_at(lat, lon)
        mean_val = float(pred.mean[0])
        std_val = float(pred.std[0]) if pred.std is not None else 0.0
        # Sample noisy observation.
        noise = self.rng.normal(0, math.sqrt(self.belief.measurement_noise_var))
        value = mean_val + noise
        return discretize_observation(
            value, n_bins=self.cfg.obs_n_bins, lo=self.cfg.obs_lo, hi=self.cfg.obs_hi,
        )


# ---------------------------------------------------------------------------
# Main policy class
# ---------------------------------------------------------------------------


class POMCPPolicy:
    """Online POMDP policy for adaptive sampling via POMCP.

    This is the main OceanBench adapter.  It conforms to the policy
    interface::

        action = policy.act(belief_state, task, constraints, rng)

    and optionally supports multi-step commitment::

        actions = policy.plan_k_steps(belief_state, task, constraints, rng)

    Parameters
    ----------
    graph:
        The WaypointGraph defining the discrete state/action space.
    belief:
        A :class:`BeliefAdapter` wrapping an OceanBench FieldBeliefModel.
    config:
        A :class:`POMCPConfig` (or raw mapping) with all parameters.
    truth_field:
        Optional :class:`OceanTruthField` for generating real observations
        during execution (as opposed to simulated ones from the belief).
    """

    def __init__(
        self,
        graph: WaypointGraph,
        belief: BeliefAdapter,
        config: POMCPConfig | Mapping[str, Any],
        *,
        truth_field: Any = None,
    ) -> None:
        if isinstance(config, Mapping):
            config = POMCPConfig.from_mapping(config)
        self.graph = graph
        self.belief = belief
        self.config = config
        self.truth_field = truth_field

        self._rng = np.random.default_rng(config.seed)

        # Build transition table.
        self._transition = build_transition_table(
            graph,
            action_space=config.action_space,
            knn_k=config.knn_k,
            include_stay=config.include_stay,
        )

        # Strategy objects.
        self._rollout_schedule = build_rollout_schedule(
            config.rollout_schedule, **config.rollout_kwargs,
        )
        self._commitment = build_commitment_strategy(
            config.commitment_strategy, **config.commitment_kwargs,
        )

        # Episode bookkeeping.
        self._step: int = 0
        self._trajectory: list[dict[str, Any]] = []

    # ----- public interface -----------------------------------------------

    def act(
        self,
        belief_state: POMDPState,
        task: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> POMCPAction:
        """Select a single action given the current POMDP state.

        This is the primary policy interface.
        """
        actions = self.plan_k_steps(belief_state, task, constraints, rng)
        return actions[0]

    def plan_k_steps(
        self,
        belief_state: POMDPState,
        task: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[POMCPAction]:
        """Return committed actions (k >= 1, determined by commitment strategy).

        Parameters
        ----------
        belief_state:
            Current POMDP state (node, time, step, budget).
        task:
            Optional task metadata (unused by core search but available for
            extensions).
        constraints:
            Optional constraints (e.g. budget limits).
        rng:
            Optional RNG override.

        Returns
        -------
        List of :class:`POMCPAction` to execute before re-planning.
        """
        if rng is None:
            rng = self._rng

        node_id = belief_state.node_id
        reachable = self._transition.get(node_id, [])
        if not reachable:
            # Nowhere to go — stay.
            lat, lon = node_coords(self.graph, node_id)
            return [POMCPAction(target_node_id=node_id, lat=lat, lon=lon, action_index=0)]

        n_actions = len(reachable)

        # Determine rollout budget for this step.
        rollout_budget = self._rollout_schedule.rollouts_for_step(
            self._step, self.config.max_steps,
        )

        # Build exploration strategy for this planning step.
        exploration = build_exploration_strategy(
            self.config.exploration_strategy,
            budget=rollout_budget,
            **{k: v for k, v in self.config.exploration_kwargs.items()
               if k not in ("budget",)},
        )
        # Override UCT constant if using UCT.
        if isinstance(exploration, UCTExploration):
            exploration.c = self.config.uct_c

        # Build the POMCP engine for this node.
        engine = _POMCPEngine(
            actions=reachable,
            transition=self._transition,
            belief=self.belief,
            graph=self.graph,
            config=self.config,
            rng=rng,
        )

        # Run tree search.
        engine.search(node_id, rollout_budget, exploration)

        # Extract critical path from tree.
        path = engine.critical_path(self.config.max_depth)
        if not path:
            # Fallback: pick best action from root.
            best_idx = engine.best_action_index(exploration)
            target = reachable[best_idx] if best_idx < len(reachable) else node_id
            lat, lon = node_coords(self.graph, target)
            return [POMCPAction(target_node_id=target, lat=lat, lon=lon, action_index=best_idx)]

        # Apply commitment strategy.
        committed: list[POMCPAction] = []
        current_node_id = node_id

        for depth, (a_idx, action_node) in enumerate(path):
            # The reachable set depends on where we are in the path.
            current_reachable = self._transition.get(current_node_id, reachable)
            if a_idx >= len(current_reachable):
                break
            target = current_reachable[a_idx]
            lat, lon = node_coords(self.graph, target)
            committed.append(POMCPAction(
                target_node_id=target, lat=lat, lon=lon, action_index=a_idx,
            ))

            # First action is always committed.
            if depth == 0:
                current_node_id = target
                continue

            # Ask commitment strategy whether to keep going.
            # Build a TreeSnapshot for this depth.
            arm_rewards = [
                action_node.children.get(i, _TreeNode(is_action=True)).reward_history
                for i in range(n_actions)
            ] if action_node.children else [[]]

            # Find second-best arm rewards.
            sorted_arms = sorted(
                range(len(arm_rewards)),
                key=lambda i: np.mean(arm_rewards[i]) if arm_rewards[i] else -np.inf,
                reverse=True,
            )
            best_r = arm_rewards[sorted_arms[0]] if sorted_arms else []
            second_r = arm_rewards[sorted_arms[1]] if len(sorted_arms) > 1 else []

            snapshot = TreeSnapshot(
                best_arm_rewards=best_r,
                second_best_arm_rewards=second_r,
                arm_rewards=arm_rewards,
                visit_count=action_node.visit_count,
            )

            if not self._commitment.should_commit_next(depth, snapshot):
                break
            current_node_id = target

        if not committed:
            # Safety fallback.
            best_idx = engine.best_action_index(exploration)
            target = reachable[best_idx] if best_idx < len(reachable) else node_id
            lat, lon = node_coords(self.graph, target)
            committed = [POMCPAction(
                target_node_id=target, lat=lat, lon=lon, action_index=best_idx,
            )]

        # Log trajectory.
        for action in committed:
            self._trajectory.append({
                "step": self._step,
                "node_id": action.target_node_id,
                "lat": action.lat,
                "lon": action.lon,
                "action_index": action.action_index,
                "rollout_budget": rollout_budget,
            })
            self._step += 1

        return committed

    def observe(self, observation: POMDPObservation) -> None:
        """Feed an observation back to update the belief model."""
        self.belief.update(observation)

    # ----- trajectory & state management ----------------------------------

    @property
    def trajectory(self) -> list[dict[str, Any]]:
        """Return the full trajectory log so far."""
        return list(self._trajectory)

    @property
    def step_count(self) -> int:
        return self._step

    def reset(self) -> None:
        """Reset policy state for a new episode."""
        self._step = 0
        self._trajectory.clear()
        self.belief.reset()
        self._rng = np.random.default_rng(self.config.seed)

    def seed(self, seed: int) -> None:
        """Set a new random seed."""
        self.config.seed = seed
        self._rng = np.random.default_rng(seed)
