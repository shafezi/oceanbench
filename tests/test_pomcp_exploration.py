"""Tests for POMCP exploration strategies."""

import numpy as np
import pytest

from oceanbench_policies.pomdp.exploration import (
    UCTExploration,
    SuccessiveRejectsExploration,
    UGapEbExploration,
    build_exploration_strategy,
)


# ---------------------------------------------------------------------------
# UCT
# ---------------------------------------------------------------------------


class TestUCT:
    def test_selects_unvisited_first(self):
        uct = UCTExploration(budget=100, c=1.0)
        rewards = [[1.0, 2.0], [], [0.5]]
        arm = uct.select_arm(rewards)
        assert arm == 1, "Should select the unvisited arm"

    def test_should_continue_respects_budget(self):
        uct = UCTExploration(budget=10, c=1.0)
        rewards = [[1.0]] * 3
        assert uct.should_continue(rewards, 5)
        assert not uct.should_continue(rewards, 10)

    def test_best_arm_picks_highest_mean(self):
        uct = UCTExploration(budget=100)
        rewards = [[1.0, 2.0, 3.0], [0.1, 0.2], [10.0, 11.0]]
        assert uct.best_arm(rewards) == 2

    def test_c_affects_exploration(self):
        """Higher c should make UCT more likely to select less-visited arms."""
        rewards = [[1.0] * 100, [0.5] * 2]

        low_c = UCTExploration(budget=200, c=0.01)
        high_c = UCTExploration(budget=200, c=100.0)

        arm_low = low_c.select_arm(rewards)
        arm_high = high_c.select_arm(rewards)
        # With very high c, the under-visited arm 1 should be selected.
        assert arm_high == 1


# ---------------------------------------------------------------------------
# Successive Rejects
# ---------------------------------------------------------------------------


class TestSuccessiveRejects:
    def test_selects_valid_arm(self):
        sr = SuccessiveRejectsExploration(budget=50)
        rewards = [[], [], []]
        arm = sr.select_arm(rewards)
        assert 0 <= arm <= 2

    def test_should_continue_stops_at_budget(self):
        sr = SuccessiveRejectsExploration(budget=10)
        rewards = [[1.0]] * 3
        assert not sr.should_continue(rewards, 10)

    def test_best_arm_after_elimination(self):
        sr = SuccessiveRejectsExploration(budget=100)
        # Simulate many pulls.
        rewards = [[10.0] * 30, [1.0] * 30, [5.0] * 30]
        sr._init_schedule(3)
        sr._eliminated = {1}  # arm 1 eliminated
        best = sr.best_arm(rewards)
        assert best in (0, 2)


# ---------------------------------------------------------------------------
# UGapEb
# ---------------------------------------------------------------------------


class TestUGapEb:
    def test_selects_valid_arm(self):
        ugap = UGapEbExploration(budget=50, b=1.0, epsilon=0.5)
        rewards = [[], [], []]
        arm = ugap.select_arm(rewards)
        assert 0 <= arm <= 2

    def test_should_continue_stops_at_budget(self):
        ugap = UGapEbExploration(budget=10)
        rewards = [[1.0]] * 3
        assert not ugap.should_continue(rewards, 10)

    def test_best_arm_finds_smallest_gap(self):
        ugap = UGapEbExploration(budget=100, b=1.0, epsilon=0.5)
        # Arm 0 is clearly best.
        rewards = [[10.0] * 50, [1.0] * 50, [2.0] * 50]
        best = ugap.best_arm(rewards)
        assert best == 0

    def test_allocated_rollouts(self):
        ugap = UGapEbExploration(budget=42)
        assert ugap.allocated_rollouts() == 42


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    @pytest.mark.parametrize("name", ["uct", "successive_rejects", "ugapeb"])
    def test_build_known_strategies(self, name):
        strategy = build_exploration_strategy(name, budget=100)
        assert strategy is not None

    def test_build_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            build_exploration_strategy("nonexistent", budget=100)
