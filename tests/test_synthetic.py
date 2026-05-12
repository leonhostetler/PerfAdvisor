"""Fast tests using the synthetic SQLite fixture.

These run in CI and on any machine — no real Nsight profile required, no API key.
The fixture values are defined in conftest.py::_build_synthetic_db.

Expected values from the fixture (all derived from conftest constants):
  GPU kernel time   = 10 × 2 ms + 2 × 5 ms = 30 ms = 0.030 s
  Profile span      > 0.030 s (MPI/MEMCPY events push span wider — MPI_Barrier starts at 0.5 s)
  GPU utilization   < 100 % (other event sources push the span wider)
  Gaps              9 gaps of 5 µs (<10 µs bucket), 2 gaps of 1.1 ms (1-10 ms bucket)
  Streams           single stream (id=7)
  Top kernel        Kernel3D (short_name) with demangled name containing "MyFunctor"
  H2D memcpy        1 MB in 100 µs → ~10 GB/s
  NVTX range        "computePhase"
  MPI ops           MPI_Barrier (100 ms total), MPI_Allreduce (50 ms total)
  CPU sync          ~10 ms (cuStreamSynchronize)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perf_advisor.analysis.metrics import (
    compute_cpu_sync_time,
    compute_device_info,
    compute_gap_histogram,
    compute_gpu_kernel_time,
    compute_gpu_memcpy_time,
    compute_gpu_sync_time,
    compute_marker_ranges,
    compute_memcpy_by_kind,
    compute_mpi_ops,
    compute_profile_span,
    compute_profile_summary,
    compute_streams,
    compute_top_kernels,
)
from perf_advisor.ingestion.profile import NsysProfile

# ---------------------------------------------------------------------------
# Ingestion layer
# ---------------------------------------------------------------------------


def test_synthetic_profile_opens(synthetic_profile_path):
    assert Path(synthetic_profile_path).exists()


def test_synthetic_tables_present(synthetic_profile):
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in synthetic_profile.tables
    assert "CUPTI_ACTIVITY_KIND_MEMCPY" in synthetic_profile.tables
    assert "StringIds" in synthetic_profile.tables


def test_synthetic_has_mpi(synthetic_profile):
    assert synthetic_profile.has_mpi()


def test_synthetic_has_nvtx(synthetic_profile):
    assert synthetic_profile.has_nvtx()


def test_synthetic_string_resolution(synthetic_profile):
    row = synthetic_profile.query("SELECT shortName FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1")[0]
    name = synthetic_profile.resolve_string(row["shortName"])
    assert name and not name.startswith("<id:")


def test_synthetic_context_manager(synthetic_profile_path):
    with NsysProfile(synthetic_profile_path) as p:
        assert "CUPTI_ACTIVITY_KIND_KERNEL" in p.tables


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


def test_synthetic_gpu_kernel_time(synthetic_profile):
    t = compute_gpu_kernel_time(synthetic_profile)
    assert abs(t - 0.030) < 1e-4, f"Expected 0.030 s, got {t}"


def test_synthetic_profile_span_wider_than_kernel_time(synthetic_profile):
    span = compute_profile_span(synthetic_profile)
    kernel = compute_gpu_kernel_time(synthetic_profile)
    assert span > kernel


def test_synthetic_gpu_utilization_below_100(synthetic_profile):
    span = compute_profile_span(synthetic_profile)
    kernel = compute_gpu_kernel_time(synthetic_profile)
    util = 100.0 * kernel / span
    assert 0.0 < util < 100.0


def test_synthetic_gpu_memcpy_time(synthetic_profile):
    t = compute_gpu_memcpy_time(synthetic_profile)
    assert t > 0.0


def test_synthetic_gpu_sync_time_absent(synthetic_profile):
    # No CUPTI_ACTIVITY_KIND_SYNCHRONIZATION in the synthetic fixture
    t = compute_gpu_sync_time(synthetic_profile)
    assert t == 0.0


def test_synthetic_top_kernels_count(synthetic_profile):
    kernels = compute_top_kernels(synthetic_profile)
    assert len(kernels) == 2  # Kernel3D and Reduction2D


def test_synthetic_top_kernel_is_kernel3d(synthetic_profile):
    kernels = compute_top_kernels(synthetic_profile)
    assert kernels[0].short_name == "Kernel3D"
    assert "MyFunctor" in kernels[0].name


def test_synthetic_kernel_pct_sums_to_100(synthetic_profile):
    kernels = compute_top_kernels(synthetic_profile, limit=100)
    total = sum(k.pct_of_gpu_time for k in kernels)
    assert abs(total - 100.0) < 0.5


def test_synthetic_memcpy_h2d(synthetic_profile):
    kinds = compute_memcpy_by_kind(synthetic_profile)
    assert len(kinds) == 1
    assert kinds[0].kind == "Host-to-Device"
    assert kinds[0].total_bytes == 1_048_576
    assert kinds[0].effective_GBs > 5.0  # ~10 GB/s


def test_synthetic_gap_histogram(synthetic_profile):
    total_idle, buckets = compute_gap_histogram(synthetic_profile)
    assert total_idle > 0
    labels = {b.label for b in buckets}
    assert "<10us" in labels
    assert "1-10ms" in labels
    assert total_idle == pytest.approx(sum(b.total_s for b in buckets), rel=1e-3)


def test_synthetic_gap_counts(synthetic_profile):
    _, buckets = compute_gap_histogram(synthetic_profile)
    by_label = {b.label: b for b in buckets}
    # 9 inter-kernel gaps of 5 µs (between 10 Kernel3D instances)
    assert by_label["<10us"].count == 9
    # 2 gaps of 1.1 ms (Kernel3D→Reduction2D and Reduction2D→Reduction2D)
    assert by_label["1-10ms"].count == 2


def test_synthetic_single_stream(synthetic_profile):
    streams = compute_streams(synthetic_profile)
    assert len(streams) == 1
    assert streams[0].stream_id == 7
    assert streams[0].pct_of_gpu_time == pytest.approx(100.0, abs=0.1)


def test_synthetic_marker_range(synthetic_profile):
    ranges = compute_marker_ranges(synthetic_profile)
    assert len(ranges) == 1
    assert ranges[0].name == "computePhase"


def test_synthetic_mpi_ops(synthetic_profile):
    ops = compute_mpi_ops(synthetic_profile)
    assert len(ops) == 2
    op_names = {o.op for o in ops}
    assert "MPI_Barrier" in op_names
    assert "MPI_Allreduce" in op_names


def test_synthetic_mpi_barrier_time(synthetic_profile):
    ops = compute_mpi_ops(synthetic_profile)
    barrier = next(o for o in ops if o.op == "MPI_Barrier")
    assert abs(barrier.total_s - 0.1) < 1e-3  # 100 ms


def test_synthetic_device_info(synthetic_profile):
    info = compute_device_info(synthetic_profile)
    assert info.sm_count == 108
    assert info.max_threads_per_sm == 64 * 32  # 2048
    assert abs(info.peak_memory_bandwidth_GBs - 2000.0) < 1.0


def test_synthetic_occupancy_computed(synthetic_profile):
    device_info = compute_device_info(synthetic_profile)
    kernels = compute_top_kernels(synthetic_profile, device_info=device_info)
    # All kernels should have estimated_occupancy set
    assert all(k.estimated_occupancy is not None for k in kernels)
    assert all(0.0 < k.estimated_occupancy <= 1.0 for k in kernels)


def test_synthetic_bandwidth_pct_computed(synthetic_profile):
    device_info = compute_device_info(synthetic_profile)
    peak_bw = device_info.peak_memory_bandwidth_GBs
    kinds = compute_memcpy_by_kind(synthetic_profile, peak_bandwidth_GBs=peak_bw)
    assert kinds[0].pct_of_peak_bandwidth is not None
    assert kinds[0].pct_of_peak_bandwidth > 0.0


def test_synthetic_cpu_sync_time(synthetic_profile):
    kernel_s = compute_gpu_kernel_time(synthetic_profile)
    sync_s, pct = compute_cpu_sync_time(synthetic_profile, kernel_s)
    # cuStreamSynchronize is in RUNTIME but nameId doesn't resolve to a name containing
    # 'Synchronize' unless StringIds has the right entry — it does (id=7, "cuStreamSynchronize")
    assert sync_s is not None
    assert abs(sync_s - 0.01) < 1e-3  # 10 ms


def test_synthetic_full_summary(synthetic_profile):
    summary = compute_profile_summary(synthetic_profile)
    assert summary.profile_span_s > 0
    assert summary.gpu_utilization_pct > 0
    assert summary.mpi_present is True
    assert len(summary.top_kernels) == 2
    assert len(summary.mpi_ops) == 2


# ---------------------------------------------------------------------------
# Tool dispatch (no API key required)
# ---------------------------------------------------------------------------


def test_dispatch_profile_summary(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "profile_summary", {}))
    assert "gpu_utilization_pct" in result
    assert result["mpi_present"] is True
    assert result["markers_present"] is True
    assert result["format"] == "nsys"


def test_dispatch_top_kernels(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "top_kernels", {"limit": 5}))
    assert "kernels" in result
    assert result["kernels"][0]["short_name"] == "Kernel3D"


def test_dispatch_gap_histogram(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "gap_histogram", {}))
    assert result["total_idle_s"] > 0
    labels = {b["label"] for b in result["buckets"]}
    assert "<10us" in labels


def test_dispatch_memcpy_summary(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "memcpy_summary", {}))
    assert "transfers" in result
    assert len(result["transfers"]) == 1
    assert result["transfers"][0]["kind"] == "Host-to-Device"


def test_dispatch_marker_ranges(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "marker_ranges", {}))
    assert "ranges" in result
    assert result["ranges"][0]["name"] == "computePhase"


def test_dispatch_mpi_summary(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "mpi_summary", {}))
    assert "ops" in result
    op_names = {o["op"] for o in result["ops"]}
    assert "MPI_Barrier" in op_names


def test_dispatch_unknown_tool(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_profile, "nonexistent_tool", {}))
    assert "error" in result


def test_dispatch_sql_query(synthetic_profile):
    from perf_advisor.agent.tools import dispatch

    result = json.loads(
        dispatch(
            synthetic_profile,
            "sql_query",
            {"sql": "SELECT COUNT(*) AS n FROM CUPTI_ACTIVITY_KIND_KERNEL"},
        )
    )
    assert "rows" in result
    assert result["rows"][0]["n"] == 12  # 10 Kernel3D + 2 Reduction2D


# ---------------------------------------------------------------------------
# NsysProfile vendor-neutral helpers (Phase 3)
# ---------------------------------------------------------------------------


def test_nsys_format(synthetic_profile):
    from perf_advisor.ingestion.base import Format

    assert synthetic_profile.format == Format.NSYS


def test_nsys_capabilities_has_kernels(synthetic_profile):
    assert synthetic_profile.capabilities.has_kernels is True


def test_nsys_capabilities_has_memcpy(synthetic_profile):
    assert synthetic_profile.capabilities.has_memcpy is True


def test_nsys_capabilities_has_mpi(synthetic_profile):
    assert synthetic_profile.capabilities.has_mpi is True


def test_nsys_capabilities_has_markers(synthetic_profile):
    assert synthetic_profile.capabilities.has_markers is True


def test_nsys_capabilities_schema_version(synthetic_profile):
    assert synthetic_profile.capabilities.schema_version == "nsys"


def test_nsys_kernel_events_count(synthetic_profile):
    from perf_advisor.ingestion.base import KernelRow

    evts = synthetic_profile.kernel_events()
    assert len(evts) == 12  # 10 Kernel3D + 2 Reduction2D
    assert all(isinstance(e, KernelRow) for e in evts)


def test_nsys_kernel_events_limit(synthetic_profile):
    evts = synthetic_profile.kernel_events(limit=3)
    assert len(evts) == 3


def test_nsys_kernel_events_names(synthetic_profile):
    evts = synthetic_profile.kernel_events()
    names = {e.name for e in evts}
    assert any("MyFunctor" in n for n in names)
    assert any("Reduction2D" in n for n in names)


def test_nsys_kernel_events_short_name(synthetic_profile):
    evts = synthetic_profile.kernel_events()
    # demangledName != shortName → short_name should be populated
    k3d = [e for e in evts if e.short_name and "Kernel3D" in e.short_name]
    assert len(k3d) == 10


def test_nsys_memcpy_events(synthetic_profile):
    from perf_advisor.ingestion.base import MemcpyRow

    evts = synthetic_profile.memcpy_events()
    assert len(evts) == 1
    assert isinstance(evts[0], MemcpyRow)
    assert evts[0].direction == "Host-to-Device"
    assert evts[0].bytes == 1_048_576


def test_nsys_marker_ranges(synthetic_profile):
    from perf_advisor.ingestion.base import RangeRow

    evts = synthetic_profile.marker_ranges()
    assert len(evts) == 1
    assert isinstance(evts[0], RangeRow)
    assert evts[0].name == "computePhase"
    assert evts[0].category == "NVTX"


def test_nsys_mpi_ranges(synthetic_profile):
    from perf_advisor.ingestion.base import RangeRow

    evts = synthetic_profile.mpi_ranges()
    assert len(evts) == 2
    assert all(isinstance(e, RangeRow) for e in evts)
    names = {e.name for e in evts}
    assert "MPI_Barrier" in names
    assert "MPI_Allreduce" in names
    assert all(e.category == "MPI" for e in evts)
