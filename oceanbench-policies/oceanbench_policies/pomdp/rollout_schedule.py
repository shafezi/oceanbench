"""Rollout allocation schedules for POMCP.

A rollout schedule determines how many POMCP rollouts to allocate at each
planning step of the episode.  Options from the paper:

- **Constant** — same budget every step.
- **Increasing** — linearly increasing budget (more rollouts as belief improves).
- **BetaCDF** — budget follows a beta-CDF curve over the episode horizon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RolloutSchedule(ABC):
    """Determine rollout budget for each planning step."""

    @abstractmethod
    def rollouts_for_step(self, step: int, total_steps: int) -> int:
        """Return the number of rollouts to use at *step* (0-indexed).

        Parameters
        ----------
        step:
            Current planning step (0-indexed).
        total_steps:
            Total number of steps in the episode.
        """


# ---------------------------------------------------------------------------
# Concrete schedules
# ---------------------------------------------------------------------------


class ConstantSchedule(RolloutSchedule):
    """Fixed number of rollouts at every step.

    Parameters
    ----------
    n:
        Rollouts per step.
    """

    def __init__(self, n: int = 500) -> None:
        self.n = max(1, n)

    def rollouts_for_step(self, step: int, total_steps: int) -> int:
        return self.n


class IncreasingSchedule(RolloutSchedule):
    """Linearly increasing rollout budget from *min_n* to *max_n*.

    Parameters
    ----------
    min_n:
        Rollouts at step 0.
    max_n:
        Rollouts at the last step.
    """

    def __init__(self, min_n: int = 100, max_n: int = 1000) -> None:
        self.min_n = max(1, min_n)
        self.max_n = max(self.min_n, max_n)

    def rollouts_for_step(self, step: int, total_steps: int) -> int:
        if total_steps <= 1:
            return self.max_n
        frac = step / (total_steps - 1)
        return int(self.min_n + frac * (self.max_n - self.min_n))


class BetaCDFSchedule(RolloutSchedule):
    """Rollout budget follows a beta-CDF curve (paper's approach).

    The total rollout budget *max_rollouts* is distributed across the
    episode such that the cumulative allocation tracks the CDF of
    ``Beta(alpha, beta_param)`` evaluated at the fraction of episode
    completed.

    Parameters
    ----------
    alpha:
        Alpha parameter of the beta distribution.
    beta_param:
        Beta parameter of the beta distribution.
    max_rollouts:
        Total rollout budget for the entire episode.
    min_per_step:
        Minimum rollouts per step (to avoid degenerate cases).
    """

    def __init__(
        self,
        alpha: float = 2.0,
        beta_param: float = 5.0,
        max_rollouts: int = 5000,
        min_per_step: int = 10,
    ) -> None:
        self.alpha = alpha
        self.beta_param = beta_param
        self.max_rollouts = max_rollouts
        self.min_per_step = min_per_step
        self._used: int = 0

    def rollouts_for_step(self, step: int, total_steps: int) -> int:
        if total_steps <= 0:
            return self.min_per_step

        frac_now = (step + 1) / total_steps
        frac_prev = step / total_steps

        cdf_now = beta_dist.cdf(frac_now, self.alpha, self.beta_param)
        cdf_prev = beta_dist.cdf(frac_prev, self.alpha, self.beta_param)

        ideal = int((cdf_now - cdf_prev) * self.max_rollouts)
        n = max(ideal, self.min_per_step)
        self._used += n
        return n


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_SCHEDULES = {
    "constant": ConstantSchedule,
    "increasing": IncreasingSchedule,
    "beta": BetaCDFSchedule,
}


def build_rollout_schedule(
    name: str,
    **kwargs: Any,
) -> RolloutSchedule:
    """Construct a rollout schedule by name.

    Parameters
    ----------
    name:
        One of ``"constant"``, ``"increasing"``, ``"beta"``.
    **kwargs:
        Forwarded to the schedule constructor.
    """
    cls = _SCHEDULES.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Unknown rollout schedule {name!r}. Available: {sorted(_SCHEDULES)}"
        )
    return cls(**kwargs)
