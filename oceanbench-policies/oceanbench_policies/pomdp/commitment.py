"""Plan commitment strategies for POMCP.

After POMCP search completes, the commitment strategy decides how many
steps of the best critical path to *commit* before re-planning.  Strategies
from the paper:

- **NoneCommitment** — commit only the first action (standard POMCP).
- **FixedKCommitment** — commit exactly *k* steps.
- **TTestCommitment** — commit while best arm is statistically better than
  second-best (Welch's t-test).
- **UGapEcCommitment** — commit while UGapEc confidence criterion is met.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tree-info container passed by the adapter
# ---------------------------------------------------------------------------


class TreeSnapshot:
    """Lightweight snapshot of POMCP tree info at a given depth.

    The adapter populates this at each level of the critical path so
    commitment strategies can reason about per-arm statistics.

    Attributes
    ----------
    best_arm_rewards:
        Reward history for the best action at this depth.
    second_best_arm_rewards:
        Reward history for the second-best action.
    arm_rewards:
        Full ``[list_of_rewards_per_arm]`` at this depth.
    visit_count:
        Total visits to the belief node at this depth.
    """

    def __init__(
        self,
        best_arm_rewards: list[float],
        second_best_arm_rewards: list[float],
        arm_rewards: list[list[float]],
        visit_count: int,
    ) -> None:
        self.best_arm_rewards = best_arm_rewards
        self.second_best_arm_rewards = second_best_arm_rewards
        self.arm_rewards = arm_rewards
        self.visit_count = visit_count


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class CommitmentStrategy(ABC):
    """Decide how many steps of the POMCP plan to commit before re-planning."""

    @abstractmethod
    def should_commit_next(
        self,
        depth: int,
        snapshot: TreeSnapshot,
    ) -> bool:
        """Return True if the action at *depth* should be committed.

        The first action (depth=0) is always committed by the adapter.  This
        method is called starting from depth=1 onwards.
        """


# ---------------------------------------------------------------------------
# Concrete strategies
# ---------------------------------------------------------------------------


class NoneCommitment(CommitmentStrategy):
    """Commit only the first action (re-plan every step)."""

    def should_commit_next(self, depth: int, snapshot: TreeSnapshot) -> bool:
        return False


class FixedKCommitment(CommitmentStrategy):
    """Commit exactly *k* steps from the critical path.

    Parameters
    ----------
    k:
        Number of steps to commit.  ``k=1`` is equivalent to
        :class:`NoneCommitment`.
    """

    def __init__(self, k: int = 1) -> None:
        self.k = max(1, k)

    def should_commit_next(self, depth: int, snapshot: TreeSnapshot) -> bool:
        # depth is 0-indexed; the first action (depth=0) is always committed.
        return depth < self.k - 1


class TTestCommitment(CommitmentStrategy):
    """Commit while best arm is statistically distinguishable from second-best.

    Uses Welch's two-sample t-test.  If the p-value is below *p_threshold*
    (i.e. we can reject H0 that the means are equal) the action is committed.

    Parameters
    ----------
    p_threshold:
        Maximum p-value for committing.  Lower = more conservative.
    min_samples:
        Minimum number of reward samples per arm before t-test is attempted.
    """

    def __init__(self, p_threshold: float = 0.05, min_samples: int = 3) -> None:
        self.p_threshold = p_threshold
        self.min_samples = min_samples

    def should_commit_next(self, depth: int, snapshot: TreeSnapshot) -> bool:
        best = snapshot.best_arm_rewards
        second = snapshot.second_best_arm_rewards

        if len(best) < self.min_samples or len(second) < self.min_samples:
            return False

        _, p_value = stats.ttest_ind(best, second, equal_var=False)
        if np.isnan(p_value):
            return False

        commit = p_value < self.p_threshold
        if commit:
            logger.debug("t-test commit at depth %d (p=%.4f)", depth, p_value)
        return commit


class UGapEcCommitment(CommitmentStrategy):
    """Commit while UGapEc (fixed-confidence) indicates sufficient separation.

    If the confidence gap ``B_i`` for the best arm is below *epsilon*, we are
    confident enough in the best arm to commit that action.

    Parameters
    ----------
    delta:
        Confidence parameter for UGapEc.
    epsilon:
        Gap threshold — smaller means more confident.
    b:
        Reward range scaling factor.
    """

    def __init__(
        self,
        delta: float = 0.1,
        epsilon: float = 0.01,
        b: float = 1.0,
    ) -> None:
        self.delta = delta
        self.epsilon = epsilon
        self.b = b
        self.c = 0.5

    def _beta_fc(self, num_pulls: int, total_pulls: int, K: int) -> float:
        if num_pulls <= 0:
            return float("inf")
        top = self.c * math.log((4 * K * total_pulls ** 3) / max(self.delta, 1e-30))
        return self.b * math.sqrt(max(top, 0.0) / num_pulls)

    def _B(self, all_rewards: list[list[float]], arm_idx: int) -> float:
        K = len(all_rewards)
        total = sum(len(r) for r in all_rewards)
        n_i = len(all_rewards[arm_idx])
        if n_i == 0:
            return float("inf")
        lower = np.mean(all_rewards[arm_idx]) - self._beta_fc(n_i, total, K)
        max_gap = -np.inf
        for j, r in enumerate(all_rewards):
            if j != arm_idx and r:
                upper = np.mean(r) + self._beta_fc(len(r), total, K)
                max_gap = max(max_gap, upper - lower)
        return max_gap if max_gap > -np.inf else 0.0

    def should_commit_next(self, depth: int, snapshot: TreeSnapshot) -> bool:
        all_rewards = snapshot.arm_rewards
        K = len(all_rewards)

        if K < 2:
            return False
        # Need at least one sample per arm.
        if any(len(r) == 0 for r in all_rewards):
            return False

        # Find best arm (smallest gap).
        Bs = [self._B(all_rewards, i) for i in range(K)]
        best_arm = int(np.argmin(Bs))
        best_B = Bs[best_arm]

        commit = best_B < self.epsilon
        if commit:
            logger.debug("UGapEc commit at depth %d (B=%.4f < eps=%.4f)", depth, best_B, self.epsilon)
        return commit


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_STRATEGIES = {
    "none": NoneCommitment,
    "fixed_k": FixedKCommitment,
    "ttest": TTestCommitment,
    "ugapec": UGapEcCommitment,
}


def build_commitment_strategy(
    name: str,
    **kwargs: Any,
) -> CommitmentStrategy:
    """Construct a commitment strategy by name.

    Parameters
    ----------
    name:
        One of ``"none"``, ``"fixed_k"``, ``"ttest"``, ``"ugapec"``.
    **kwargs:
        Forwarded to the strategy constructor (e.g. ``k=3``, ``p_threshold=0.1``).
    """
    cls = _STRATEGIES.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown commitment strategy {name!r}. "
            f"Available: {sorted(_STRATEGIES)}"
        )
    return cls(**kwargs)
