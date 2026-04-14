"""Tests for POMCP plan commitment strategies."""

import numpy as np
import pytest

from oceanbench_policies.pomdp.commitment import (
    NoneCommitment,
    FixedKCommitment,
    TTestCommitment,
    UGapEcCommitment,
    TreeSnapshot,
    build_commitment_strategy,
)


def _make_snapshot(
    best: list[float],
    second: list[float],
    n_arms: int = 3,
    visits: int = 100,
) -> TreeSnapshot:
    """Helper to create a TreeSnapshot with given arm rewards."""
    arm_rewards = [best, second] + [
        [np.random.default_rng(i).normal(0, 1)] * 10 for i in range(n_arms - 2)
    ]
    return TreeSnapshot(
        best_arm_rewards=best,
        second_best_arm_rewards=second,
        arm_rewards=arm_rewards,
        visit_count=visits,
    )


# ---------------------------------------------------------------------------
# NoneCommitment
# ---------------------------------------------------------------------------


class TestNoneCommitment:
    def test_never_commits_beyond_first(self):
        c = NoneCommitment()
        snap = _make_snapshot([1.0] * 20, [0.5] * 20)
        assert not c.should_commit_next(1, snap)
        assert not c.should_commit_next(5, snap)


# ---------------------------------------------------------------------------
# FixedKCommitment
# ---------------------------------------------------------------------------


class TestFixedKCommitment:
    def test_commits_k_minus_1_additional(self):
        c = FixedKCommitment(k=3)
        snap = _make_snapshot([1.0] * 20, [0.5] * 20)
        # depth=0 (first action) is always committed by the adapter.
        # Commitment should say yes for depth=1 and depth=2 (for k=3 total: 0,1,2).
        assert c.should_commit_next(0, snap)  # 0 < 3-1=2 -> True
        assert c.should_commit_next(1, snap)  # 1 < 2 -> True
        assert not c.should_commit_next(2, snap)  # 2 < 2 -> False

    def test_k1_never_commits_extra(self):
        c = FixedKCommitment(k=1)
        snap = _make_snapshot([1.0] * 20, [0.5] * 20)
        assert not c.should_commit_next(0, snap)


# ---------------------------------------------------------------------------
# TTestCommitment
# ---------------------------------------------------------------------------


class TestTTestCommitment:
    def test_commits_when_clearly_different(self):
        """If best arm is much better, t-test should have low p-value -> commit."""
        c = TTestCommitment(p_threshold=0.05, min_samples=5)
        snap = _make_snapshot(
            best=[10.0] * 30,
            second=[0.0] * 30,
        )
        assert c.should_commit_next(1, snap)

    def test_does_not_commit_when_similar(self):
        """If arms are nearly identical, t-test p-value should be high -> no commit."""
        rng = np.random.default_rng(42)
        same_data = list(rng.normal(5.0, 0.01, 30))
        snap = _make_snapshot(best=same_data, second=same_data)
        c = TTestCommitment(p_threshold=0.05, min_samples=5)
        assert not c.should_commit_next(1, snap)

    def test_does_not_commit_with_few_samples(self):
        c = TTestCommitment(p_threshold=0.05, min_samples=10)
        snap = _make_snapshot(best=[10.0, 11.0], second=[0.0, 1.0])
        assert not c.should_commit_next(1, snap)


# ---------------------------------------------------------------------------
# UGapEcCommitment
# ---------------------------------------------------------------------------


class TestUGapEcCommitment:
    def test_commits_when_clear_winner(self):
        c = UGapEcCommitment(delta=0.1, epsilon=5.0, b=1.0)
        snap = _make_snapshot(
            best=[10.0] * 50,
            second=[0.0] * 50,
        )
        # With large epsilon and clear separation, should commit.
        assert c.should_commit_next(1, snap)

    def test_does_not_commit_with_empty_arms(self):
        c = UGapEcCommitment(delta=0.1, epsilon=0.01, b=1.0)
        snap = TreeSnapshot(
            best_arm_rewards=[],
            second_best_arm_rewards=[],
            arm_rewards=[[], [], []],
            visit_count=0,
        )
        assert not c.should_commit_next(1, snap)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.parametrize("name", ["none", "fixed_k", "ttest", "ugapec"])
    def test_build_known_strategies(self, name):
        strategy = build_commitment_strategy(name)
        assert strategy is not None

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_commitment_strategy("nonexistent")
