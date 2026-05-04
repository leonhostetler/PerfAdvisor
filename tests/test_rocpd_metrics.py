"""Tests for compute_profile_summary() and metric helpers on a rocpd fixture.

Uses the synthetic_rocpd_path / synthetic_rocpd_profile fixtures from conftest.py
(CI-safe, in-memory build — no real profiles or API keys required).

Expected values derived from conftest constants:
  GPU kernel time   = 4×2 ms + 5 ms = 13 ms = 0.013 s
  Profile span      > 0.013 s  (region events push span to ~202 ms)
  GPU utilization   < 100 %    (13 ms GPU time / ~202 ms span ≈ 6.4 %)
  Gaps              3 × 5 µs (<10 µs bucket) + 1 × ~2 ms (1-10 ms bucket)
  Streams           1 stream (id=1)
  Top kernels       dslash_function (8 ms, 4 calls) > reduce_kernel (5 ms, 1 call)
  Memcpy kinds      3 — Device-to-Device (1 MB), Host-to-Device, Device-to-Host
  MPI               absent (rocprofv3 does not intercept MPI)
  Markers           absent (no rocTX in fixture; only HIP/HSA API regions)
  Device info       AMD Instinct MI250X, 110 CUs, 2048 max threads/CU, 1700 MHz clock
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def test_rocpd_gpu_kernel_time(synthetic_rocpd_profile):
    t = compute_gpu_kernel_time(synthetic_rocpd_profile)
    assert abs(t - 0.013) < 1e-5, f"Expected 0.013 s, got {t}"


def test_rocpd_profile_span_wider_than_kernel_time(synthetic_rocpd_profile):
    span = compute_profile_span(synthetic_rocpd_profile)
    kernel = compute_gpu_kernel_time(synthetic_rocpd_profile)
    assert span > kernel


def test_rocpd_profile_span_gt_100ms(synthetic_rocpd_profile):
    span = compute_profile_span(synthetic_rocpd_profile)
    # Regions start at ~1.0 s offset and end at ~1.2 s, so span > 100 ms
    assert span > 0.1


def test_rocpd_gpu_utilization_below_100(synthetic_rocpd_profile):
    span = compute_profile_span(synthetic_rocpd_profile)
    kernel = compute_gpu_kernel_time(synthetic_rocpd_profile)
    util = 100.0 * kernel / span
    assert 0.0 < util < 100.0


def test_rocpd_gpu_utilization_low(synthetic_rocpd_profile):
    # 13 ms of GPU time over ~202 ms span ≈ 6.4 %
    span = compute_profile_span(synthetic_rocpd_profile)
    kernel = compute_gpu_kernel_time(synthetic_rocpd_profile)
    util = 100.0 * kernel / span
    assert util < 20.0


def test_rocpd_gpu_memcpy_time(synthetic_rocpd_profile):
    t = compute_gpu_memcpy_time(synthetic_rocpd_profile)
    assert t > 0.0


def test_rocpd_gpu_sync_time_zero(synthetic_rocpd_profile):
    # RocpdProfile always returns 0.0 (no GPU sync table in rocpd)
    t = compute_gpu_sync_time(synthetic_rocpd_profile)
    assert t == 0.0


def test_rocpd_cpu_sync_time_none(synthetic_rocpd_profile):
    # RocpdProfile returns (None, None) — HIP API timing isn't mapped to sync overhead yet
    kernel_s = compute_gpu_kernel_time(synthetic_rocpd_profile)
    sync_s, pct = compute_cpu_sync_time(synthetic_rocpd_profile, kernel_s)
    assert sync_s is None
    assert pct is None


def test_rocpd_top_kernels_count(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    assert len(kernels) == 2  # dslash_function and reduce_kernel


def test_rocpd_top_kernel_is_dslash(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    assert "dslash_function" in kernels[0].name


def test_rocpd_top_kernel_calls(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    dslash = next(k for k in kernels if "dslash_function" in k.name)
    assert dslash.calls == 4


def test_rocpd_second_kernel_is_reduce(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    assert "reduce_kernel" in kernels[1].name
    assert kernels[1].calls == 1


def test_rocpd_kernel_pct_sums_to_100(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile, limit=100)
    total = sum(k.pct_of_gpu_time for k in kernels)
    assert abs(total - 100.0) < 0.5


def test_rocpd_top_kernel_time(synthetic_rocpd_profile):
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    dslash = next(k for k in kernels if "dslash_function" in k.name)
    # 4 × 2 ms = 8 ms = 0.008 s
    assert abs(dslash.total_s - 0.008) < 1e-4


def test_rocpd_kernel_short_name_none(synthetic_rocpd_profile):
    # RocpdProfile does not populate short_name (no separate display/mangled column)
    kernels = compute_top_kernels(synthetic_rocpd_profile)
    assert all(k.short_name is None for k in kernels)


def test_rocpd_kernel_occupancy_none(synthetic_rocpd_profile):
    # rocpd KernelRow lacks thread-count fields; occupancy cannot be estimated
    device_info = compute_device_info(synthetic_rocpd_profile)
    kernels = compute_top_kernels(synthetic_rocpd_profile, device_info=device_info)
    assert all(k.estimated_occupancy is None for k in kernels)


def test_rocpd_memcpy_kinds_count(synthetic_rocpd_profile):
    kinds = compute_memcpy_by_kind(synthetic_rocpd_profile)
    assert len(kinds) == 3


def test_rocpd_memcpy_kind_names(synthetic_rocpd_profile):
    kinds = compute_memcpy_by_kind(synthetic_rocpd_profile)
    names = {k.kind for k in kinds}
    assert names == {"Device-to-Device", "Host-to-Device", "Device-to-Host"}


def test_rocpd_memcpy_d2d_size(synthetic_rocpd_profile):
    kinds = compute_memcpy_by_kind(synthetic_rocpd_profile)
    d2d = next(k for k in kinds if k.kind == "Device-to-Device")
    assert d2d.total_bytes == 1_048_576  # 1 MB


def test_rocpd_memcpy_h2d_size(synthetic_rocpd_profile):
    kinds = compute_memcpy_by_kind(synthetic_rocpd_profile)
    h2d = next(k for k in kinds if k.kind == "Host-to-Device")
    assert h2d.total_bytes == 262_144  # 256 KB


def test_rocpd_memcpy_no_peak_bandwidth_pct(synthetic_rocpd_profile):
    # rocpd DeviceInfo.peak_memory_bandwidth_GBs is None → pct_of_peak_bandwidth is None
    kinds = compute_memcpy_by_kind(synthetic_rocpd_profile)
    assert all(k.pct_of_peak_bandwidth is None for k in kinds)


def test_rocpd_gap_histogram_nonempty(synthetic_rocpd_profile):
    total_idle, buckets = compute_gap_histogram(synthetic_rocpd_profile)
    assert total_idle > 0
    assert len(buckets) > 0


def test_rocpd_gap_sub10us_bucket(synthetic_rocpd_profile):
    _, buckets = compute_gap_histogram(synthetic_rocpd_profile)
    by_label = {b.label: b for b in buckets}
    # 3 × 5 µs inter-dslash gaps
    assert "<10us" in by_label
    assert by_label["<10us"].count == 3


def test_rocpd_gap_1to10ms_bucket(synthetic_rocpd_profile):
    _, buckets = compute_gap_histogram(synthetic_rocpd_profile)
    by_label = {b.label: b for b in buckets}
    # 1 × ~2 ms gap (last dslash → reduce_kernel)
    assert "1-10ms" in by_label
    assert by_label["1-10ms"].count == 1


def test_rocpd_gap_total_matches_sum(synthetic_rocpd_profile):
    total_idle, buckets = compute_gap_histogram(synthetic_rocpd_profile)
    assert total_idle == pytest.approx(sum(b.total_s for b in buckets), rel=1e-3)


def test_rocpd_single_stream(synthetic_rocpd_profile):
    streams = compute_streams(synthetic_rocpd_profile)
    assert len(streams) == 1
    assert streams[0].stream_id == 1
    assert streams[0].pct_of_gpu_time == pytest.approx(100.0, abs=0.1)


def test_rocpd_marker_ranges_empty(synthetic_rocpd_profile):
    # Fixture has only HIP/HSA API regions — no rocTX markers
    ranges = compute_marker_ranges(synthetic_rocpd_profile)
    assert ranges == []


def test_rocpd_mpi_ops_empty(synthetic_rocpd_profile):
    # rocprofv3 does not intercept MPI; fixture has no MPI category
    ops = compute_mpi_ops(synthetic_rocpd_profile)
    assert ops == []


def test_rocpd_device_info_vendor(synthetic_rocpd_profile):
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.vendor == "amd"


def test_rocpd_device_info_name(synthetic_rocpd_profile):
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.name == "AMD Instinct MI250X"


def test_rocpd_device_info_cu_count(synthetic_rocpd_profile):
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.sm_count == 110  # CUs mapped to sm_count


def test_rocpd_device_info_max_threads(synthetic_rocpd_profile):
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.max_threads_per_sm == 64 * 32  # wave_front_size × max_waves_per_cu


def test_rocpd_device_info_clock(synthetic_rocpd_profile):
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.clock_rate_MHz == pytest.approx(1700.0)


def test_rocpd_device_info_no_peak_bandwidth(synthetic_rocpd_profile):
    # rocpd extdata doesn't include peak memory bandwidth
    info = compute_device_info(synthetic_rocpd_profile)
    assert info.peak_memory_bandwidth_GBs is None


# ---------------------------------------------------------------------------
# compute_profile_summary() — integration / no-exception paths
# ---------------------------------------------------------------------------


def test_rocpd_summary_runs_without_exception(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary is not None


def test_rocpd_summary_mpi_present_false(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.mpi_present is False


def test_rocpd_summary_mpi_ops_empty(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.mpi_ops == []


def test_rocpd_summary_marker_ranges_empty(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.marker_ranges == []


def test_rocpd_summary_gpu_kernel_time(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert abs(summary.gpu_kernel_s - 0.013) < 1e-4


def test_rocpd_summary_span_gt_kernel(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.profile_span_s > summary.gpu_kernel_s


def test_rocpd_summary_utilization_positive(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert 0.0 < summary.gpu_utilization_pct < 100.0


def test_rocpd_summary_top_kernels_count(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert len(summary.top_kernels) == 2


def test_rocpd_summary_top_kernel_is_dslash(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert "dslash_function" in summary.top_kernels[0].name


def test_rocpd_summary_memcpy_three_kinds(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert len(summary.memcpy_by_kind) == 3


def test_rocpd_summary_has_phases(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert len(summary.phases) >= 1


def test_rocpd_summary_phases_within_limit(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile, max_phases=4)
    assert len(summary.phases) <= 4


def test_rocpd_summary_phases_non_overlapping(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    for i in range(1, len(summary.phases)):
        assert summary.phases[i].start_ns >= summary.phases[i - 1].end_ns


def test_rocpd_summary_phases_cover_span(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    first_start = summary.phases[0].start_ns
    last_end = summary.phases[-1].end_ns
    t0, t1 = synthetic_rocpd_profile.profile_bounds_ns()
    assert first_start == t0
    assert last_end == t1


def test_rocpd_summary_single_phase_mode(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile, max_phases=1)
    assert len(summary.phases) == 1


def test_rocpd_summary_cpu_sync_none(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.cpu_sync_blocked_s is None


def test_rocpd_summary_gpu_sync_zero(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.gpu_sync_s == 0.0


def test_rocpd_summary_device_info(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.device_info.vendor == "amd"
    assert summary.device_info.sm_count == 110


def test_rocpd_summary_streams(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert len(summary.streams) == 1
    assert summary.streams[0].pct_of_gpu_time == pytest.approx(100.0, abs=0.1)


def test_rocpd_summary_gap_histogram(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    assert summary.total_gpu_idle_s > 0
    labels = {b.label for b in summary.gap_histogram}
    assert "<10us" in labels
    assert "1-10ms" in labels


# ---------------------------------------------------------------------------
# Phase-level metrics
# ---------------------------------------------------------------------------


def test_rocpd_phase_gpu_utilization_nonnegative(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    for phase in summary.phases:
        assert phase.gpu_utilization_pct >= 0.0


def test_rocpd_phase_mpi_ops_empty(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    for phase in summary.phases:
        assert phase.mpi_ops == []


def test_rocpd_phase_start_end_ordered(synthetic_rocpd_profile):
    summary = compute_profile_summary(synthetic_rocpd_profile)
    for phase in summary.phases:
        assert phase.end_s > phase.start_s


# ---------------------------------------------------------------------------
# Tool dispatch — format field and no-exception path
# ---------------------------------------------------------------------------


def test_rocpd_dispatch_profile_summary(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "profile_summary", {}))
    assert "gpu_utilization_pct" in result
    assert result["mpi_present"] is False
    assert result["markers_present"] is False
    assert result["format"] == "rocpd"


def test_rocpd_dispatch_top_kernels(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "top_kernels", {"limit": 5}))
    assert "kernels" in result
    assert "dslash_function" in result["kernels"][0]["name"]


def test_rocpd_dispatch_gap_histogram(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "gap_histogram", {}))
    assert result["total_idle_s"] > 0
    labels = {b["label"] for b in result["buckets"]}
    assert "<10us" in labels


def test_rocpd_dispatch_memcpy_summary(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "memcpy_summary", {}))
    assert "transfers" in result
    assert len(result["transfers"]) == 3
    kinds = {t["kind"] for t in result["transfers"]}
    assert "Device-to-Device" in kinds


def test_rocpd_dispatch_marker_ranges_empty(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "marker_ranges", {}))
    assert result["ranges"] == []


def test_rocpd_dispatch_mpi_summary_empty(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(dispatch(synthetic_rocpd_profile, "mpi_summary", {}))
    assert result["ops"] == []


def test_rocpd_dispatch_sql_query(synthetic_rocpd_profile):
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(
        dispatch(
            synthetic_rocpd_profile,
            "sql_query",
            {"sql": "SELECT COUNT(*) AS n FROM rocpd_kernel_dispatch"},
        )
    )
    assert "rows" in result
    assert result["rows"][0]["n"] == 5  # 4 dslash + 1 reduce


def test_rocpd_dispatch_nsys_table_redirect(synthetic_rocpd_profile):
    """Querying an NSYS table against a rocpd profile should return a redirect hint."""
    import json

    from perf_advisor.agent.tools import dispatch

    result = json.loads(
        dispatch(
            synthetic_rocpd_profile,
            "sql_query",
            {"sql": "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"},
        )
    )
    # Should get an error with a redirect hint, not a crash
    assert "error" in result
    assert "rocpd" in result["error"].lower() or "redirect" in result.get("hint", "").lower()
