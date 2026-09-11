"""
Reusable statistics for evaluation reports.

Proportion endpoints carry Wilson 95% confidence intervals; method comparisons on the
same (context, fault, seed) units use a paired bootstrap over per-unit differences.
Everything is deterministic in an explicit seed — reported numbers must be reproducible
from the run records alone.

Named ``stats`` (not ``statistics``) to stay clear of the standard library module.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from typing_extensions import Sequence

CONFIDENCE_Z = 1.959963984540054
"""
Two-sided 95% normal quantile.
"""


def wilson_interval(
    successes: int, total: int, z: float = CONFIDENCE_Z
) -> tuple[float, float]:
    """
    The Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because endpoint rates (false admission,
    correct repair) sit near 0 or 1 at these sample sizes. An empty sample is maximally
    uninformative: (0, 1).
    """
    if total == 0:
        return (0.0, 1.0)
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(frozen=True)
class RateSummary:
    """
    One proportion endpoint with its interval.
    """

    successes: int
    total: int
    low: float
    high: float

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    def render(self) -> str:
        return (
            f"{self.successes}/{self.total} = {self.rate:.3f} "
            f"[{self.low:.3f}, {self.high:.3f}]"
        )


def rate_summary(successes: int, total: int) -> RateSummary:
    low, high = wilson_interval(successes, total)
    return RateSummary(successes=successes, total=total, low=low, high=high)


@dataclass(frozen=True)
class PairedBootstrap:
    """A paired-difference estimate: mean of (a_i - b_i) with a percentile
    bootstrap 95% interval over resampled units."""

    mean_difference: float
    low: float
    high: float
    units: int
    iterations: int

    @property
    def significantly_positive(self) -> bool:
        """
        The whole interval lies above zero: a is reliably larger.
        """
        return self.low > 0.0

    @property
    def significantly_negative(self) -> bool:
        """
        The whole interval lies below zero: a is reliably smaller.
        """
        return self.high < 0.0

    def noninferior(self, margin: float) -> bool:
        """
        Whether the whole interval excludes degradation beyond margin.
        """
        if margin < 0:
            raise ValueError("A noninferiority margin must be non-negative")
        return self.low >= -margin

    def increase_bounded_by(self, margin: float) -> bool:
        """
        Whether the whole interval excludes an increase beyond margin.
        """
        if margin < 0:
            raise ValueError("A tolerance margin must be non-negative")
        return self.high <= margin

    def render(self) -> str:
        return (
            f"mean diff {self.mean_difference:+.4f} "
            f"[{self.low:+.4f}, {self.high:+.4f}] over {self.units} pairs"
        )


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = 10_000,
    seed: int = 0,
) -> PairedBootstrap:
    """Percentile bootstrap of the mean paired difference a_i - b_i.

    The pairing is positional: index i of both sequences must be the
    same experimental unit (same template, same scene seed, same
    episode index). Deterministic in ``seed``.
    """
    if len(a) != len(b):
        raise ValueError(f"Paired sequences differ in length: {len(a)} vs {len(b)}.")
    if not a:
        raise ValueError("Cannot bootstrap an empty pairing.")
    differences = [float(x) - float(y) for x, y in zip(a, b)]
    observed = sum(differences) / len(differences)
    generator = random.Random(seed)
    count = len(differences)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(count):
            total += differences[generator.randrange(count)]
        means.append(total / count)
    means.sort()
    low = means[max(0, math.ceil(0.025 * iterations) - 1)]
    high = means[min(iterations - 1, math.floor(0.975 * iterations))]
    return PairedBootstrap(
        mean_difference=observed,
        low=low,
        high=high,
        units=count,
        iterations=iterations,
    )
