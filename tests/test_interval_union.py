"""Tests for concurrency-aware GPU busy time and idle-gap detection.

GPU kernels run concurrently across streams, so summing individual durations
double-counts wall-clock time and differencing consecutive events in start
order invents gaps that never existed. These tests pin down the merged-interval
behaviour that replaced both.
"""

from __future__ import annotations

from perf_advisor.analysis._utils import busy_time_ns, merge_intervals
from perf_advisor.analysis.metrics import _bucket_gaps, _kernel_gaps_ns, _window_busy_time
from perf_advisor.ingestion.base import KernelRow


def K(start: int, end: int, name: str = "k", stream: int = 0) -> KernelRow:
    return KernelRow(
        start_ns=start,
        end_ns=end,
        name=name,
        short_name=None,
        device_id=None,
        stream_id=stream,
        duration_ns=end - start,
        registers_per_thread=8,
        shared_mem_bytes=0,
        total_threads=1024.0,
    )


# --- merge_intervals -------------------------------------------------------


def test_merge_disjoint_intervals_are_unchanged():
    assert merge_intervals([(0, 10), (20, 30)]) == [(0, 10), (20, 30)]


def test_merge_overlapping_intervals():
    assert merge_intervals([(0, 10), (5, 20)]) == [(0, 20)]


def test_merge_touching_intervals_coalesce():
    """No idle time between them, so they must not produce a zero-length gap."""
    assert merge_intervals([(0, 10), (10, 20)]) == [(0, 20)]


def test_merge_fully_contained_interval():
    assert merge_intervals([(0, 100), (10, 20)]) == [(0, 100)]


def test_merge_handles_unsorted_input():
    assert merge_intervals([(20, 30), (0, 10)]) == [(0, 10), (20, 30)]


def test_merge_drops_zero_length_and_inverted():
    assert merge_intervals([(5, 5), (10, 4), (0, 3)]) == [(0, 3)]


def test_merge_empty():
    assert merge_intervals([]) == []


# --- busy_time_ns ----------------------------------------------------------


def test_busy_time_does_not_double_count_concurrency():
    """Two streams busy for the same wall-second is 1s of GPU busy time, not 2s."""
    assert busy_time_ns([(0, 1_000), (0, 1_000)]) == 1_000


def test_busy_time_sums_disjoint_spans():
    assert busy_time_ns([(0, 10), (20, 25)]) == 15


def test_busy_time_never_exceeds_span():
    evts = [(0, 100), (0, 100), (50, 100), (10, 90)]
    assert busy_time_ns(evts) <= 100


# --- gap detection ---------------------------------------------------------


def test_no_phantom_gap_from_overlapping_kernels():
    """Regression: a long kernel overlapping a short one used to invent a gap.

    long occupies [0, 100); short [10, 20) sits inside it; next is [200, 300).
    The only true idle window is [100, 200) = 100ns. Differencing in start
    order gave 200 - 20 = 180ns.
    """
    evts = [K(0, 100, "long", 0), K(10, 20, "short", 1), K(200, 300, "next", 0)]
    assert _kernel_gaps_ns(evts) == [100]


def test_gaps_between_disjoint_kernels():
    evts = [K(0, 10), K(30, 40), K(100, 110)]
    assert _kernel_gaps_ns(evts) == [20, 60]


def test_fully_overlapped_kernels_have_no_gaps():
    evts = [K(0, 100, "a", 0), K(0, 100, "b", 1), K(25, 75, "c", 2)]
    assert _kernel_gaps_ns(evts) == []


def test_back_to_back_kernels_produce_no_gap():
    assert _kernel_gaps_ns([K(0, 10), K(10, 20)]) == []


def test_single_kernel_has_no_gaps():
    assert _kernel_gaps_ns([K(0, 10)]) == []
    assert _kernel_gaps_ns([]) == []


# --- bucketing -------------------------------------------------------------


def test_bucket_gaps_assigns_correct_labels():
    total, buckets = _bucket_gaps([5_000, 50_000, 500_000_000])
    by_label = {b.label: b for b in buckets}
    assert by_label["<10us"].count == 1
    assert by_label["10-100us"].count == 1
    assert by_label[">100ms"].count == 1
    assert total > 0


def test_bucket_boundaries_are_exclusive_upper():
    _, buckets = _bucket_gaps([10_000])
    assert [b.label for b in buckets] == ["10-100us"]


def test_empty_buckets_are_omitted():
    _, buckets = _bucket_gaps([5_000])
    assert [b.label for b in buckets] == ["<10us"]


# --- windowed busy time ----------------------------------------------------


def test_window_busy_time_merges_concurrency():
    evts = [K(0, 1_000_000_000, "a", 0), K(0, 1_000_000_000, "b", 1)]
    assert _window_busy_time(evts, 0, 2_000_000_000) == 1.0


def test_window_busy_time_excludes_events_outside_window():
    evts = [K(0, 10), K(1_000_000_000, 2_000_000_000)]
    assert _window_busy_time(evts, 0, 100) == 10 / 1e9


def test_utilization_cannot_exceed_one_hundred_percent():
    """The headline symptom: 2s of kernel work in a 1s window is 100% busy."""
    evts = [K(0, 1_000_000_000, "a", 0), K(0, 1_000_000_000, "b", 1)]
    window_ns = 1_000_000_000
    busy_s = _window_busy_time(evts, 0, window_ns)
    util_pct = 100.0 * busy_s / (window_ns / 1e9)
    assert util_pct == 100.0

    work_s = sum(e.duration_ns for e in evts) / 1e9
    assert work_s == 2.0  # kernel work is still reported in full
