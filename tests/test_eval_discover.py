"""Profile discovery across the CUDA and HIP layouts.

Both benches now write the same flat shape — the HIP submit scripts flatten
rocprof-sys's per-run subdirectory into it — so only the extension differs. The
pre-unification subdirectory layout is still supported so older captures load.
"""

from __future__ import annotations

import json

import pytest

from perf_advisor.eval.discover import discover_runs

_META = {
    "tiny_kernels": {
        "run_id": "test_01",
        "expected_bottleneck": "kernel_launch_overhead",
        "suggestions": [],
    }
}


@pytest.fixture
def bench_dir(tmp_path):
    d = tmp_path / "bench"
    d.mkdir()
    (d / "ground_truth_meta.json").write_text(json.dumps(_META), encoding="utf-8")
    return d


def _make_run(profiles: object, subdir: str, run_id: str) -> object:
    p = profiles / subdir
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{run_id}.json").write_text(
        json.dumps({"scenario": "tiny_kernels", "expected_bottleneck": "kernel_launch_overhead"}),
        encoding="utf-8",
    )
    return p


def test_cuda_flat_single(tmp_path, bench_dir):
    prof = tmp_path / "profiles"
    d = _make_run(prof, "1gpu", "test_01")
    (d / "test_01.sqlite").touch()
    (runs,) = discover_runs(prof, bench_dir)
    assert [p.name for p in runs.sqlite_paths] == ["test_01.sqlite"]
    assert not runs.is_multi_rank


def test_hip_flat_single(tmp_path, bench_dir):
    prof = tmp_path / "profiles"
    d = _make_run(prof, "1gpu", "test_01")
    (d / "test_01.db").touch()
    (runs,) = discover_runs(prof, bench_dir)
    assert [p.name for p in runs.sqlite_paths] == ["test_01.db"]
    assert not runs.is_multi_rank


@pytest.mark.parametrize("ext", ["sqlite", "db"])
def test_flat_multi_rank_is_ordered_by_rank(tmp_path, bench_dir, ext):
    """Rank order is load-bearing: sqlite_paths[0] is the cross-rank reference."""
    prof = tmp_path / "profiles"
    d = _make_run(prof, "8gpu", "test_01")
    for i in (3, 0, 11, 2, 10, 1):          # created out of order, incl. >9
        (d / f"test_01.{i}.{ext}").touch()
    (runs,) = discover_runs(prof, bench_dir)
    assert [p.name for p in runs.sqlite_paths] == [
        f"test_01.{i}.{ext}" for i in (0, 1, 2, 3, 10, 11)
    ]
    assert runs.is_multi_rank


def test_legacy_rocpd_subdirectory_still_loads(tmp_path, bench_dir):
    """Captures taken before the layouts were unified must not stop working."""
    prof = tmp_path / "profiles"
    d = _make_run(prof, "4gpu", "test_01")
    sub = d / "test_01"
    sub.mkdir()
    for i in (2, 0, 1):
        (sub / f"rank_rocpd-{i}.db").touch()
    (runs,) = discover_runs(prof, bench_dir)
    assert [p.name for p in runs.sqlite_paths] == [f"rank_rocpd-{i}.db" for i in (0, 1, 2)]


def test_flat_layout_wins_over_a_leftover_subdirectory(tmp_path, bench_dir):
    """If a partial flatten left the directory behind, prefer the flat files."""
    prof = tmp_path / "profiles"
    d = _make_run(prof, "4gpu", "test_01")
    (d / "test_01.0.db").touch()
    (d / "test_01.1.db").touch()
    sub = d / "test_01"
    sub.mkdir()
    (sub / "weird.db").touch()
    (runs,) = discover_runs(prof, bench_dir)
    assert [p.name for p in runs.sqlite_paths] == ["test_01.0.db", "test_01.1.db"]


def test_run_with_no_profile_is_skipped(tmp_path, bench_dir):
    prof = tmp_path / "profiles"
    _make_run(prof, "1gpu", "test_01")       # json only, no profile
    assert discover_runs(prof, bench_dir) == []
