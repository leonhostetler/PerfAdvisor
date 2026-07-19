"""Tests for launch-overhead caching, name-collision merging, and phase reuse."""

from __future__ import annotations

import json
import sqlite3

import pytest

from perf_advisor.agent.tools import dispatch, tool_phase_summary
from perf_advisor.ingestion.nsys import NsysProfile


@pytest.fixture
def overhead_profile(tmp_path):
    """Two distinct demangled names that normalise to the same key.

    _normalize_demangled strips the 'std::enable_if<..., void>::type ' prefix,
    so these two collapse onto one key while SQL groups them separately.
    """
    path = tmp_path / "oh.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    conn.executemany(
        "INSERT INTO StringIds VALUES (?, ?)",
        [
            (1, "Kernel3D"),
            (2, "std::enable_if<(A), void>::type quda::Kernel3D<X>"),
            (3, "std::enable_if<(B), void>::type quda::Kernel3D<X>"),
        ],
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
        "(start INTEGER, end INTEGER, shortName INTEGER, demangledName INTEGER, "
        " correlationId INTEGER, streamId INTEGER)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME "
        "(start INTEGER, end INTEGER, nameId INTEGER, correlationId INTEGER)"
    )
    # Group A: 1 launch, 10us overhead.  Group B: 3 launches, 20us overhead each.
    kernels = [
        (10_000, 20_000, 1, 2, 1, 7),
        (20_000, 30_000, 1, 3, 2, 7),
        (40_000, 50_000, 1, 3, 3, 7),
        (60_000, 70_000, 1, 3, 4, 7),
    ]
    runtimes = [(0, 5_000, 1, 1), (0, 5_000, 1, 2), (20_000, 25_000, 1, 3), (40_000, 45_000, 1, 4)]
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,?,?,?,?)", kernels)
    conn.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?,?,?,?)", runtimes)
    conn.commit()
    conn.close()
    return NsysProfile(path)


def test_launch_overhead_is_cached(overhead_profile):
    first = overhead_profile.launch_overhead()
    assert overhead_profile._launch_overhead_cache is not None
    assert overhead_profile.launch_overhead() is first  # same object, not recomputed


def test_colliding_names_are_merged_not_overwritten(overhead_profile):
    """Both SQL groups normalise to one key; neither may be silently dropped."""
    oh = overhead_profile.launch_overhead()
    assert len(oh) == 1
    key = next(iter(oh))
    assert "enable_if" not in key  # prefix was stripped
    avg_us, max_us = oh[key]
    # Group A: 1 launch @ 10us. Group B: 3 launches @ 20us.
    # Count-weighted mean = (10*1 + 20*3) / 4 = 17.5us; max = 20us.
    assert avg_us == 17.5
    assert max_us == 20.0


# --- phase_summary reuse ---------------------------------------------------


class _FakeSummary:
    def __init__(self, n):
        self.phases = [_FakePhase(i) for i in range(n)]


class _FakePhase:
    def __init__(self, i):
        self.i = i

    def model_dump(self):
        return {"name": f"phase_{self.i}"}


def test_phase_summary_reuses_preseeded_phases(overhead_profile):
    """No max_phases requested -> return the pre-computed segmentation."""
    out = tool_phase_summary(overhead_profile, {}, summary=_FakeSummary(3))
    assert [p["name"] for p in out["phases"]] == ["phase_0", "phase_1", "phase_2"]


def test_phase_summary_reuses_when_count_matches(overhead_profile):
    out = tool_phase_summary(overhead_profile, {"max_phases": 3}, summary=_FakeSummary(3))
    assert len(out["phases"]) == 3


def test_phase_summary_recomputes_when_count_differs(overhead_profile):
    """A different phase count must not silently return the pre-seeded one."""
    out = tool_phase_summary(overhead_profile, {"max_phases": 1}, summary=_FakeSummary(3))
    assert [p.get("name") for p in out["phases"]] != ["phase_0", "phase_1", "phase_2"]


def test_dispatch_threads_summary_to_phase_summary(overhead_profile):
    raw = dispatch(overhead_profile, "phase_summary", {}, summary=_FakeSummary(2))
    assert len(json.loads(raw)["phases"]) == 2


def test_dispatch_without_summary_still_works(overhead_profile):
    raw = dispatch(overhead_profile, "profile_summary", {})
    assert "gpu_busy_s" in json.loads(raw)
