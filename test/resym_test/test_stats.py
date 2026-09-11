"""
The pre-registered statistics: Wilson intervals and the paired bootstrap.

Host-runnable, stdlib-only.
"""

from __future__ import annotations

import pytest

from resym.evaluation.stats import (
    paired_bootstrap,
    rate_summary,
    wilson_interval,
)


class TestWilsonInterval:
    def test_known_value(self):
        low, high = wilson_interval(8, 10)
        assert low == pytest.approx(0.4902, abs=1e-3)
        assert high == pytest.approx(0.9433, abs=1e-3)

    def test_empty_sample_is_maximally_uninformative(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_extremes_stay_inside_the_unit_interval(self):
        low_all, high_all = wilson_interval(10, 10)
        low_none, high_none = wilson_interval(0, 10)
        assert 0.0 <= low_none < high_none < 1.0
        assert 0.0 < low_all < high_all <= 1.0

    def test_interval_narrows_with_sample_size(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(500, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_rate_summary_renders(self):
        summary = rate_summary(3, 4)
        assert summary.rate == pytest.approx(0.75)
        assert "3/4" in summary.render()


class TestPairedBootstrap:
    def test_identical_sequences_give_a_zero_interval(self):
        result = paired_bootstrap([1, 0, 1, 0], [1, 0, 1, 0], iterations=200)
        assert result.mean_difference == 0.0
        assert result.low == 0.0 and result.high == 0.0
        assert not result.significantly_positive
        assert not result.significantly_negative

    def test_a_clear_improvement_is_significant(self):
        a = [1.0] * 20
        b = [0.0] * 10 + [1.0] * 10
        result = paired_bootstrap(a, b, iterations=2000, seed=1)
        assert result.mean_difference == pytest.approx(0.5)
        assert result.significantly_positive

    def test_symmetry(self):
        a = [1.0] * 20
        b = [0.0] * 10 + [1.0] * 10
        result = paired_bootstrap(b, a, iterations=2000, seed=1)
        assert result.significantly_negative

    def test_deterministic_in_the_seed(self):
        a = [0.11, 0.42, 0.87, 0.29, 0.64, 0.53, 0.91, 0.05]
        b = [0.08, 0.55, 0.61, 0.33, 0.60, 0.44, 0.72, 0.19]
        first = paired_bootstrap(a, b, iterations=500, seed=7)
        second = paired_bootstrap(a, b, iterations=500, seed=7)
        third = paired_bootstrap(a, b, iterations=500, seed=8)
        assert first == second
        assert (first.low, first.high) != (third.low, third.high)

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError, match="differ in length"):
            paired_bootstrap([1, 0], [1], iterations=10)

    def test_empty_pairing_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            paired_bootstrap([], [], iterations=10)
