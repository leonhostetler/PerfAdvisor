"""Tests for CPU sync-blocked time.

Durations used to be summed across host threads, so several threads blocked in
*Synchronize concurrently could report more blocked time than the profile's
wall clock. They are now merged.
"""

from __future__ import annotations

import sqlite3

import pytest

from perf_advisor.ingestion.nsys import NsysProfile


@pytest.fixture
def sync_profile(tmp_path):
    """Minimal nsys-shaped DB with two host threads blocked concurrently."""
    path = tmp_path / "sync.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO StringIds VALUES (7, 'cudaStreamSynchronize')")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
        "(start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER)"
    )
    # Two threads blocked over the SAME wall-clock second, plus a later
    # disjoint 0.5s block. Summed: 2.5s. Merged: 1.5s.
    conn.executemany(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, 7, ?)",
        [
            (0, 1_000_000_000, 1),
            (0, 1_000_000_000, 2),
            (2_000_000_000, 2_500_000_000, 3),
        ],
    )
    conn.commit()
    conn.close()
    return NsysProfile(path)


def test_concurrent_sync_calls_are_merged_not_summed(sync_profile):
    sync_s, _ = sync_profile.cpu_sync_blocked_s(span_s=10.0)
    assert sync_s == 1.5  # not 2.5


def test_pct_is_relative_to_profile_span(sync_profile):
    _, pct = sync_profile.cpu_sync_blocked_s(span_s=10.0)
    assert pct == 15.0


def test_pct_cannot_exceed_one_hundred(sync_profile):
    """Merged blocked time is bounded by the span, so the ratio is bounded too."""
    _, pct = sync_profile.cpu_sync_blocked_s(span_s=1.5)
    assert pct == 100.0


def test_zero_span_yields_none_pct(sync_profile):
    _, pct = sync_profile.cpu_sync_blocked_s(span_s=0.0)
    assert pct is None


def test_missing_runtime_table_returns_none(tmp_path):
    path = tmp_path / "empty.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()
    assert NsysProfile(path).cpu_sync_blocked_s(span_s=10.0) == (None, None)
