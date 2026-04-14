"""Root action exploration strategies for POMCP.

Each strategy decides *which* root action to pull next and *when* to stop
allocating rollouts.  The three strategies from the paper are:

- **UCT** (default POMCP): actions are selected by UCB1 inside the tree.
- **Successive Rejects**: phase-based arm elimination.
- **UGapEb**: fixed-budget confidence-bound arm identification.

All strategies implement a common :class:`ExplorationStrategy` interface so
that :class:`~oceanbench_policies.pomdp.pomcp_adapter.POMCPPolicy` can swap
them via configuration.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ExplorationStrategy(ABC):
    """Base class for root-action exploration / rollout-allocation strategies."""

    @abstractmethod
    def select_arm(self, rewards_per_arm: list[list[float]]) -> int:
        """Given current reward history per arm, return the arm index to pull next."""

    @abstractmethod
    def should_continue(self, rewards_per_arm: list[list[float]], rollouts_used: int) -> bool:
        """Return True if more rollouts should be allocated."""

    @abstractmethod
    def best_arm(self, rewards_per_arm: list[list[float]]) -> int:
        """Return the recommended arm after exploration is finished."""

    def allocated_rollouts(self) -> Optional[int]:
        """If known in advance, return the total number of rollouts for this step.

        Return *None* for open-ended strategies (e.g. UGapEc).
        """
        return None


# ---------------------------------------------------------------------------
# UCT (standard POMCP)
# ---------------------------------------------------------------------------


class UCTExploration(ExplorationStrategy):
    """Standard UCB1 exploration — the default POMCP behaviour.

    Action selection in the tree is handled by the upstream POMCP ``SearchBest``
    via UCB.  At the root level this strategy simply runs a fixed number of
    rollouts (controlled by :class:`~rollout_schedule.RolloutSchedule`).

    Parameters
    ----------
    budget:
        Fixed rollout budget for this planning step.
    c:
        UCB exploration constant (higher = more exploration).
    """

    def __init__(self, budget: int = 500, c: float = 1.0) -> None:
        self.budget = budget
        self.c = c

    def select_arm(self, rewards_per_arm: list[list[float]]) -> int:
        n_arms = len(rewards_per_arm)
        total = sum(len(r) for r in rewards_per_arm)
        # Pull unvisited arms first.
        for i, r in enumerate(rewards_per_arm):
            if len(r) == 0:
                return i
        # UCB1.
        best_val = -np.inf
        best_arm = 0
        for i, r in enumerate(rewards_per_arm):
            mean = np.mean(r)
            ucb = mean + self.c * math.sqrt(math.log(total) / len(r))
            if ucb > best_val:
                best_val = ucb
                best_arm = i
        return best_arm

    def should_continue(self, rewards_per_arm: list[list[float]], rollouts_used: int) -> bool:
        return rollouts_used < self.budget

    def best_arm(self, rewards_per_arm: list[list[float]]) -> int:
        means = [np.mean(r) if r else -np.inf for r in rewards_per_arm]
        return int(np.argmax(means))

    def allocated_rollouts(self) -> Optional[int]:
        return self.budget


# ---------------------------------------------------------------------------
# Successive Rejects
# ---------------------------------------------------------------------------


class SuccessiveRejectsExploration(ExplorationStrategy):
    """Phase-based arm elimination (Audibert et al., 2010).

    Eliminates the worst-performing arm after each phase until a single
    arm remains.

    Parameters
    ----------
    budget:
        Total rollout budget for this planning step.
    """

    def __init__(self, budget: int = 500) -> None:
        self.budget = budget
        self._phase_schedule: list[tuple[int, int]] | None = None
        self._eliminated: set[int] = set()
        self._K: int = 0

    def _log_bar(self, K: int) -> float:
        return 0.5 + sum(1.0 / i for i in range(2, K + 1))

    def _n_k(self, k: int, K: int) -> int:
        if k == 0:
            return 0
        return math.ceil((1.0 / self._log_bar(K)) * ((self.budget - K) / (K + 1 - k)))

    def _init_schedule(self, K: int) -> None:
        self._K = K
        self._eliminated = set()
        self._phase_schedule = []
        for k in range(1, K):
            pulls = self._n_k(k, K) - self._n_k(k - 1, K)
            self._phase_schedule.append((k, max(pulls, 1)))

    def select_arm(self, rewards_per_arm: list[list[float]]) -> int:
        K = len(rewards_per_arm)
        if self._phase_schedule is None:
            self._init_schedule(K)

        # Pull unvisited non-eliminated arms first.
        for i in range(K):
            if i not in self._eliminated and len(rewards_per_arm[i]) == 0:
                return i

        # Find an active arm with fewest pulls.
        active = [i for i in range(K) if i not in self._eliminated]
        if not active:
            return 0
        return min(active, key=lambda i: len(rewards_per_arm[i]))

    def should_continue(self, rewards_per_arm: list[list[float]], rollouts_used: int) -> bool:
        if rollouts_used >= self.budget:
            return False

        K = len(rewards_per_arm)
        if self._phase_schedule is None:
            self._init_schedule(K)

        # Check if we should eliminate an arm based on phase boundaries.
        active = [i for i in range(K) if i not in self._eliminated]
        if len(active) <= 1:
            return False

        # Check phase completion.
        if self._phase_schedule:
            phase_k, pulls_needed = self._phase_schedule[0]
            target_per_arm = self._n_k(phase_k, K)
            all_done = all(
                len(rewards_per_arm[i]) >= target_per_arm
                for i in active
            )
            if all_done:
                # Eliminate worst arm.
                worst = min(active, key=lambda i: np.mean(rewards_per_arm[i]) if rewards_per_arm[i] else -np.inf)
                self._eliminated.add(worst)
                self._phase_schedule.pop(0)
                logger.debug("SR eliminated arm %d (phase %d)", worst, phase_k)
                # Continue if more than 1 arm remains.
                active = [i for i in range(K) if i not in self._eliminated]
                if len(active) <= 1:
                    return False

        return True

    def best_arm(self, rewards_per_arm: list[list[float]]) -> int:
        K = len(rewards_per_arm)
        active = [i for i in range(K) if i not in self._eliminated]
        if len(active) == 1:
            return active[0]
        means = [np.mean(rewards_per_arm[i]) if rewards_per_arm[i] else -np.inf for i in active]
        return active[int(np.argmax(means))]

    def allocated_rollouts(self) -> Optional[int]:
        return self.budget


# ---------------------------------------------------------------------------
# UGapEb  (fixed-budget)
# ---------------------------------------------------------------------------


class UGapEbExploration(ExplorationStrategy):
    """Fixed-budget confidence-bound arm identification (Gabillon et al., 2012).

    Parameters
    ----------
    budget:
        Total rollout budget for this planning step.
    a:
        Confidence scaling parameter. Estimated from budget and K.
    b:
        Confidence scaling factor (reward range).
    """

    def __init__(
        self,
        budget: int = 500,
        b: float = 1.0,
        epsilon: float = 0.5,
    ) -> None:
        self.budget = budget
        self.b = b
        self.epsilon = epsilon
        self._K: int = 0
        self._a: float = 0.0

    def _init_params(self, K: int) -> None:
        self._K = K
        # Estimate 'a' as in the paper (under-estimate of H_epsilon).
        denom = 4.0 * sum(
            (self.b ** 2) / (self.epsilon ** 2) for _ in range(K)
        )
        self._a = (self.budget - K) / max(denom, 1e-12)

    def _beta(self, num_pulls: int) -> float:
        if num_pulls <= 0:
            return float("inf")
        return self.b * math.sqrt(self._a / num_pulls)

    def _B(self, rewards: list[list[float]], arm_idx: int) -> float:
        """Gap metric for arm *arm_idx*."""
        total = sum(len(r) for r in rewards)
        lower = np.mean(rewards[arm_idx]) - self._beta(len(rewards[arm_idx]))
        max_gap = -np.inf
        for j, r in enumerate(rewards):
            if j != arm_idx and r:
                upper = np.mean(r) + self._beta(len(r))
                max_gap = max(max_gap, upper - lower)
        return max_gap if max_gap > -np.inf else 0.0

    def select_arm(self, rewards_per_arm: list[list[float]]) -> int:
        K = len(rewards_per_arm)
        if self._K == 0:
            self._init_params(K)

        # Pull unvisited arms first.
        for i, r in enumerate(rewards_per_arm):
            if len(r) == 0:
                return i

        # UGap arm selection: find J (min B_i), then find arm with largest
        # upper confidence that is not J, and pull whichever has wider beta.
        Bs = [self._B(rewards_per_arm, i) for i in range(K)]
        lt = int(np.argmin(Bs))

        ut_arm = None
        ut_val = -np.inf
        for i, r in enumerate(rewards_per_arm):
            if i != lt and r:
                upper = np.mean(r) + self._beta(len(r))
                if upper > ut_val:
                    ut_val = upper
                    ut_arm = i

        if ut_arm is None:
            return lt

        if self._beta(len(rewards_per_arm[lt])) > self._beta(len(rewards_per_arm[ut_arm])):
            return lt
        return ut_arm

    def should_continue(self, rewards_per_arm: list[list[float]], rollouts_used: int) -> bool:
        return rollouts_used < self.budget

    def best_arm(self, rewards_per_arm: list[list[float]]) -> int:
        K = len(rewards_per_arm)
        if self._K == 0:
            self._init_params(K)

        # J(rewards) — arm with smallest gap.
        Bs = [self._B(rewards_per_arm, i) for i in range(K)]
        return int(np.argmin(Bs))

    def allocated_rollouts(self) -> Optional[int]:
        return self.budget


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "uct": UCTExploration,
    "successive_rejects": SuccessiveRejectsExploration,
    "ugapeb": UGapEbExploration,
}


def build_exploration_strategy(
    name: str,
    budget: int = 500,
    **kwargs: Any,
) -> ExplorationStrategy:
    """Construct an exploration strategy by name.

    Parameters
    ----------
    name:
        One of ``"uct"``, ``"successive_rejects"``, ``"ugapeb"``.
    budget:
        Rollout budget for this planning step.
    **kwargs:
        Extra keyword arguments forwarded to the strategy constructor.
    """
    cls = _STRATEGIES.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown exploration strategy {name!r}. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return cls(budget=budget, **kwargs)
