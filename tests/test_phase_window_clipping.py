"""Tests for phase-window event attribution.

Windowed metrics used to require full containment
(``start >= window_start AND end <= window_end``), so any event crossing a
phase boundary was dropped from every phase. These tests pin down the
overlap-and-clip behaviour that replaced it.
"""

from __future__ import annotations

import sqlite3

from perf_advisor.analysis.metrics import (
    _clipped_duration_ns,
    _overlaps,
    _window_busy_time,
    _window_idle_time,
    _window_kernel_time,
    _window_memcpy_by_kind,
    _window_memcpy_time,
)
from perf_advisor.ingestion.base import KernelRow, MemcpyRow


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


def M(start: int, end: int, direction: str = "HtoD", nbytes: int = 1024) -> MemcpyRow:
    return MemcpyRow(
        start_ns=start,
        end_ns=end,
        direction=direction,
        bytes=nbytes,
        duration_ns=end - start,
    )


# --- primitives ------------------------------------------------------------


def test_overlaps_detects_straddling_events():
    assert _overlaps(K(50, 150), 100, 200)  # straddles the start
    assert _overlaps(K(150, 250), 100, 200)  # straddles the end
    assert _overlaps(K(120, 130), 100, 200)  # fully inside
    assert _overlaps(K(50, 250), 100, 200)  # spans the whole window


def test_overlaps_excludes_disjoint_events():
    assert not _overlaps(K(0, 100), 100, 200)  # ends exactly at window start
    assert not _overlaps(K(200, 300), 100, 200)  # starts exactly at window end
    assert not _overlaps(K(0, 50), 100, 200)


def test_clipped_duration_trims_to_window():
    assert _clipped_duration_ns(K(50, 150), 100, 200) == 50
    assert _clipped_duration_ns(K(150, 250), 100, 200) == 50
    assert _clipped_duration_ns(K(50, 250), 100, 200) == 100  # spans whole window
    assert _clipped_duration_ns(K(120, 130), 100, 200) == 10  # fully contained
    assert _clipped_duration_ns(K(0, 50), 100, 200) == 0  # disjoint


# --- windowed totals -------------------------------------------------------


def test_straddling_kernel_is_no_longer_dropped():
    """The regression: this kernel used to contribute nothing to either phase."""
    evts = [K(50, 150)]
    assert _window_kernel_time(evts, 100, 200) == 50 / 1e9


def test_phase_times_sum_to_global_total():
    """Contiguous phases must partition kernel work with nothing lost."""
    evts = [K(0, 30), K(80, 120), K(150, 260), K(190, 210)]
    total = sum(e.duration_ns for e in evts) / 1e9
    boundaries = [(0, 100), (100, 200), (200, 300)]
    per_phase = sum(_window_kernel_time(evts, s, e) for s, e in boundaries)
    assert abs(per_phase - total) < 1e-12


def test_memcpy_time_clips_at_boundaries():
    assert _window_memcpy_time([M(50, 150)], 100, 200) == 50 / 1e9


def test_memcpy_by_kind_includes_straddling_transfer():
    out = _window_memcpy_by_kind([M(50, 150, "HtoD")], 100, 200)
    assert len(out) == 1
    assert out[0].kind == "HtoD"


def test_busy_time_clips_at_boundaries():
    assert _window_busy_time([K(50, 150)], 100, 200) == 50 / 1e9


def test_busy_time_never_exceeds_window_duration():
    evts = [K(0, 1000, "a", 0), K(0, 1000, "b", 1)]
    assert _window_busy_time(evts, 100, 200) == 100 / 1e9


def test_idle_gaps_use_clipped_intervals():
    """A kernel spanning the whole window leaves no idle time inside it."""
    total_idle, _ = _window_idle_time([K(0, 1000)], 100, 200)
    assert total_idle == 0.0


def test_idle_gap_between_straddling_kernels():
    """Window [100us, 200us); kernels clip to 100..120us and 180..200us.

    Uses microsecond-scale timings so the 60us gap survives the histogram's
    microsecond rounding — nanosecond-scale gaps are below reporting resolution.
    """
    us = 1_000
    evts = [K(50 * us, 120 * us), K(180 * us, 250 * us)]
    total_idle, buckets = _window_idle_time(evts, 100 * us, 200 * us)
    assert total_idle == 60e-6
    assert [(b.label, b.count) for b in buckets] == [("10-100us", 1)]


def test_ten_microsecond_gap_is_not_rounded_away():
    """Regression: bucket totals were rounded to milliseconds, so the two
    smallest buckets always reported total_s == 0.0."""
    us = 1_000
    evts = [K(0, 10 * us), K(30 * us, 40 * us)]
    total_idle, buckets = _window_idle_time(evts, 0, 40 * us)
    assert total_idle == 20e-6
    assert buckets[0].total_s > 0.0


# --- SQL-side windowing ----------------------------------------------------


def test_sql_min_max_clipping_semantics():
    """Guard the SQL clip expression used by the *_aggregates methods.

    Verifies SQLite resolves two-arg MIN/MAX as scalar functions inside SUM,
    which is what makes `SUM(MIN(end, ?) - MAX(start, ?))` a per-row clip.
    """
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE e (start INT, end INT)")
    # 50..150 straddles the start, 180..250 straddles the end, 120..130 inside.
    conn.executemany("INSERT INTO e VALUES (?, ?)", [(50, 150), (180, 250), (120, 130)])
    win_start, win_end = 100, 200
    row = conn.execute(
        "SELECT SUM(MIN(end, ?) - MAX(start, ?)) AS total, COUNT(*) AS n "
        "FROM e WHERE start < ? AND end > ?",
        (win_end, win_start, win_end, win_start),
    ).fetchone()
    conn.close()
    # 50 (100..150) + 20 (180..200) + 10 (120..130)
    assert row[0] == 80
    assert row[1] == 3
