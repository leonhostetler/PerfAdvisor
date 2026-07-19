"""Tests for kernel duration variance.

The previous form ``sum(x*x)/n - mean**2`` subtracts two large nearly equal
quantities; in nanosecond units on long kernels it loses precision and can go
negative, which was masked by a ``max(0.0, ...)`` clamp.
"""

from __future__ import annotations

import random
import statistics

from perf_advisor.analysis.metrics import _population_variance


def test_matches_stdlib_on_simple_input():
    vals = [10, 12, 23, 23, 16, 23, 21, 16]
    mean = sum(vals) / len(vals)
    assert abs(_population_variance(vals, mean) - statistics.pvariance(vals)) < 1e-9


def test_zero_variance_for_identical_values():
    vals = [50_000] * 1000
    assert _population_variance(vals, 50_000.0) == 0.0


def test_single_value_has_zero_variance():
    assert _population_variance([42], 42.0) == 0.0
    assert _population_variance([], 0.0) == 0.0


def test_never_negative_on_low_variance_long_kernels():
    """The regime where the old form lost the most precision.

    1-second kernels with microsecond jitter: summands reach ~1e18 while the
    variance itself is ~1e6.
    """
    random.seed(7)
    vals = [int(1_000_000_000 + random.gauss(0, 1_000)) for _ in range(50_000)]
    mean = sum(vals) / len(vals)
    var = _population_variance(vals, mean)
    assert var >= 0.0
    assert abs(var - statistics.pvariance(vals)) / statistics.pvariance(vals) < 1e-9


def test_more_accurate_than_naive_form():
    """Pins the improvement: the two-pass form tracks the exact value far more
    closely than the algebraically-equivalent naive form."""
    random.seed(11)
    vals = [int(1_000_000_000 + random.gauss(0, 1_000)) for _ in range(50_000)]
    n = len(vals)
    mean = sum(vals) / n
    exact = statistics.pvariance(vals)

    naive = sum(v * v for v in vals) / n - mean**2
    two_pass = _population_variance(vals, mean)

    assert abs(two_pass - exact) / exact < abs(naive - exact) / exact
