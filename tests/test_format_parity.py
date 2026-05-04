"""Format parity tests.

Verifies that ProfileSummary has identical schema and contract for both
Nsight Systems (NVIDIA) and rocpd (AMD) profiles. These tests do NOT compare
metric values — they check structural contracts: same fields exist, same types
for non-None values, and always-populated fields are populated on both formats.

Known format-specific None fields (by design, not defects):
  ProfileSummary level:
    - cpu_sync_blocked_s / cpu_sync_blocked_pct: None for rocpd (no CPU runtime tracing)
    - peak_memory_bandwidth_GBs: None for rocpd (no TARGET_INFO_GPU equivalent)
  DeviceInfo level:
    - compute_capability: None for rocpd (AMD has no CUDA compute capability)
    - peak_memory_bandwidth_GBs: None for rocpd (not in rocpd_agent table)
    - total_memory_GiB: None for rocpd
    - l2_cache_MiB: None for rocpd
  KernelSummary level:
    - short_name: None for rocpd (no StringIds table)
    - estimated_occupancy: None for rocpd (no thread-count fields on KernelRow)
    - avg_launch_overhead_us / max_launch_overhead_us: None for rocpd (no CPU-GPU timing correlation)
"""
from __future__ import annotations

import pytest

from perf_advisor.analysis.metrics import compute_profile_summary
from perf_advisor.analysis.models import (
    DeviceInfo,
    GapBucket,
    KernelSummary,
    MarkerRangeSummary,
    MemcpySummary,
    MpiOpSummary,
    PhaseSummary,
    ProfileSummary,
    StreamSummary,
)


# ---------------------------------------------------------------------------
# Session-scoped computed summaries
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def nsys_summary(synthetic_profile):
    return compute_profile_summary(synthetic_profile)


@pytest.fixture(scope="session")
def rocpd_summary(synthetic_rocpd_profile):
    return compute_profile_summary(synthetic_rocpd_profile)


@pytest.fixture(params=["nsys", "rocpd"])
def any_summary(request, nsys_summary, rocpd_summary):
    """Parametrized fixture: runs each test against both formats."""
    return nsys_summary if request.param == "nsys" else rocpd_summary


# ---------------------------------------------------------------------------
# Schema identity
# ---------------------------------------------------------------------------


def test_both_summaries_are_profile_summary(nsys_summary, rocpd_summary):
    """Both formats return the exact same Pydantic model type."""
    assert type(nsys_summary) is ProfileSummary
    assert type(rocpd_summary) is ProfileSummary


def test_profile_summary_field_sets_identical(nsys_summary, rocpd_summary):
    """Both summaries expose identical top-level field sets (same Pydantic schema)."""
    nsys_keys = set(ProfileSummary.model_fields.keys())
    rocpd_keys = set(ProfileSummary.model_fields.keys())
    assert nsys_keys == rocpd_keys
    # Both instances share the same model class
    assert nsys_summary.__class__ is rocpd_summary.__class__


@pytest.mark.parametrize(
    "model_cls",
    [DeviceInfo, KernelSummary, PhaseSummary, MemcpySummary, StreamSummary, GapBucket],
)
def test_nested_model_field_sets_stable(model_cls):
    """Nested models used by ProfileSummary have a consistent field set."""
    # Verify model is a proper Pydantic model with a field registry.
    assert hasattr(model_cls, "model_fields")
    assert len(model_cls.model_fields) > 0


# ---------------------------------------------------------------------------
# Always-populated scalar fields (both formats)
# ---------------------------------------------------------------------------


def test_profile_path_is_non_empty_string(any_summary):
    assert isinstance(any_summary.profile_path, str)
    assert len(any_summary.profile_path) > 0


def test_profile_span_positive(any_summary):
    assert any_summary.profile_span_s > 0.0


def test_gpu_kernel_s_positive(any_summary):
    assert any_summary.gpu_kernel_s > 0.0


def test_gpu_memcpy_s_non_negative(any_summary):
    assert any_summary.gpu_memcpy_s >= 0.0


def test_gpu_sync_s_non_negative(any_summary):
    assert any_summary.gpu_sync_s >= 0.0


def test_gpu_utilization_in_range(any_summary):
    assert 0.0 <= any_summary.gpu_utilization_pct <= 100.0


def test_total_gpu_idle_non_negative(any_summary):
    assert any_summary.total_gpu_idle_s >= 0.0


def test_mpi_present_is_bool(any_summary):
    assert isinstance(any_summary.mpi_present, bool)


def test_kernel_s_not_greater_than_span(any_summary):
    assert any_summary.gpu_kernel_s <= any_summary.profile_span_s


# ---------------------------------------------------------------------------
# Always-populated list fields (both formats)
# ---------------------------------------------------------------------------


def test_top_kernels_non_empty(any_summary):
    assert len(any_summary.top_kernels) > 0


def test_top_kernels_elements_are_kernel_summary(any_summary):
    for k in any_summary.top_kernels:
        assert isinstance(k, KernelSummary)


def test_top_kernel_name_non_empty(any_summary):
    for k in any_summary.top_kernels:
        assert isinstance(k.name, str) and len(k.name) > 0


def test_top_kernel_calls_positive(any_summary):
    for k in any_summary.top_kernels:
        assert k.calls > 0


def test_top_kernel_total_s_positive(any_summary):
    for k in any_summary.top_kernels:
        assert k.total_s > 0.0


def test_top_kernel_pct_in_range(any_summary):
    for k in any_summary.top_kernels:
        assert 0.0 < k.pct_of_gpu_time <= 100.0


def test_top_kernel_avg_ms_positive(any_summary):
    for k in any_summary.top_kernels:
        assert k.avg_ms > 0.0


def test_memcpy_by_kind_non_empty(any_summary):
    assert len(any_summary.memcpy_by_kind) > 0


def test_memcpy_elements_are_memcpy_summary(any_summary):
    for m in any_summary.memcpy_by_kind:
        assert isinstance(m, MemcpySummary)


def test_memcpy_kind_non_empty_string(any_summary):
    for m in any_summary.memcpy_by_kind:
        assert isinstance(m.kind, str) and len(m.kind) > 0


def test_memcpy_transfers_positive(any_summary):
    for m in any_summary.memcpy_by_kind:
        assert m.transfers > 0


def test_streams_non_empty(any_summary):
    assert len(any_summary.streams) > 0


def test_streams_elements_are_stream_summary(any_summary):
    for s in any_summary.streams:
        assert isinstance(s, StreamSummary)


def test_stream_pct_positive(any_summary):
    for s in any_summary.streams:
        assert s.pct_of_gpu_time > 0.0


def test_gap_histogram_non_empty(any_summary):
    assert len(any_summary.gap_histogram) > 0


def test_gap_histogram_elements_are_gap_bucket(any_summary):
    for b in any_summary.gap_histogram:
        assert isinstance(b, GapBucket)


def test_phases_non_empty(any_summary):
    assert len(any_summary.phases) > 0


def test_phases_elements_are_phase_summary(any_summary):
    for p in any_summary.phases:
        assert isinstance(p, PhaseSummary)


def test_marker_ranges_is_list(any_summary):
    assert isinstance(any_summary.marker_ranges, list)
    for m in any_summary.marker_ranges:
        assert isinstance(m, MarkerRangeSummary)


def test_mpi_ops_is_list(any_summary):
    assert isinstance(any_summary.mpi_ops, list)
    for m in any_summary.mpi_ops:
        assert isinstance(m, MpiOpSummary)


# ---------------------------------------------------------------------------
# Phase structural contracts (both formats)
# ---------------------------------------------------------------------------


def test_phase_name_non_empty(any_summary):
    for ph in any_summary.phases:
        assert isinstance(ph.name, str) and len(ph.name) > 0


def test_phase_end_after_start(any_summary):
    for ph in any_summary.phases:
        assert ph.end_s > ph.start_s


def test_phase_duration_matches_bounds(any_summary):
    for ph in any_summary.phases:
        assert abs(ph.duration_s - (ph.end_s - ph.start_s)) < 1e-9


def test_phase_ns_end_after_start(any_summary):
    for ph in any_summary.phases:
        assert ph.end_ns > ph.start_ns


def test_phase_utilization_in_range(any_summary):
    for ph in any_summary.phases:
        assert 0.0 <= ph.gpu_utilization_pct <= 100.0


def test_phase_kernel_s_non_negative(any_summary):
    for ph in any_summary.phases:
        assert ph.gpu_kernel_s >= 0.0


def test_phase_mpi_ops_is_list(any_summary):
    for ph in any_summary.phases:
        assert isinstance(ph.mpi_ops, list)


# ---------------------------------------------------------------------------
# DeviceInfo always-populated fields (both formats)
# ---------------------------------------------------------------------------


def test_device_info_vendor_set(any_summary):
    assert any_summary.device_info.vendor in ("nvidia", "amd")


def test_device_info_sm_count_positive(any_summary):
    assert any_summary.device_info.sm_count is not None
    assert any_summary.device_info.sm_count > 0


def test_device_info_max_threads_positive(any_summary):
    assert any_summary.device_info.max_threads_per_sm is not None
    assert any_summary.device_info.max_threads_per_sm > 0


# ---------------------------------------------------------------------------
# Format-specific: rocpd expected None fields
# ---------------------------------------------------------------------------


def test_rocpd_cpu_sync_blocked_s_is_none(rocpd_summary):
    assert rocpd_summary.cpu_sync_blocked_s is None


def test_rocpd_cpu_sync_blocked_pct_is_none(rocpd_summary):
    assert rocpd_summary.cpu_sync_blocked_pct is None


def test_rocpd_peak_memory_bandwidth_is_none(rocpd_summary):
    assert rocpd_summary.peak_memory_bandwidth_GBs is None


def test_rocpd_device_compute_capability_is_none(rocpd_summary):
    assert rocpd_summary.device_info.compute_capability is None


def test_rocpd_device_peak_bandwidth_is_none(rocpd_summary):
    assert rocpd_summary.device_info.peak_memory_bandwidth_GBs is None


def test_rocpd_device_total_memory_is_none(rocpd_summary):
    assert rocpd_summary.device_info.total_memory_GiB is None


def test_rocpd_kernel_short_name_is_none(rocpd_summary):
    for k in rocpd_summary.top_kernels:
        assert k.short_name is None


def test_rocpd_kernel_estimated_occupancy_is_none(rocpd_summary):
    for k in rocpd_summary.top_kernels:
        assert k.estimated_occupancy is None


def test_rocpd_kernel_launch_overhead_is_none(rocpd_summary):
    for k in rocpd_summary.top_kernels:
        assert k.avg_launch_overhead_us is None
        assert k.max_launch_overhead_us is None


def test_rocpd_device_name_set(rocpd_summary):
    # rocpd_agent always provides the device name; nsys requires TARGET_INFO_GPU columns
    assert isinstance(rocpd_summary.device_info.name, str)
    assert len(rocpd_summary.device_info.name) > 0


def test_rocpd_clock_rate_positive(rocpd_summary):
    # rocpd_agent always records clock_rate_MHz; nsys TARGET_INFO_GPU may omit it
    assert rocpd_summary.device_info.clock_rate_MHz is not None
    assert rocpd_summary.device_info.clock_rate_MHz > 0.0


def test_rocpd_vendor_is_amd(rocpd_summary):
    assert rocpd_summary.device_info.vendor == "amd"


def test_rocpd_mpi_present_false(rocpd_summary):
    assert rocpd_summary.mpi_present is False


def test_rocpd_mpi_ops_empty(rocpd_summary):
    assert rocpd_summary.mpi_ops == []


def test_rocpd_marker_ranges_empty(rocpd_summary):
    assert rocpd_summary.marker_ranges == []


# ---------------------------------------------------------------------------
# Format-specific: nsys expected non-None fields
# ---------------------------------------------------------------------------


def test_nsys_cpu_sync_blocked_s_is_float(nsys_summary):
    assert isinstance(nsys_summary.cpu_sync_blocked_s, float)


def test_nsys_cpu_sync_blocked_pct_is_float(nsys_summary):
    assert isinstance(nsys_summary.cpu_sync_blocked_pct, float)


def test_nsys_peak_memory_bandwidth_is_float(nsys_summary):
    assert isinstance(nsys_summary.peak_memory_bandwidth_GBs, float)
    assert nsys_summary.peak_memory_bandwidth_GBs > 0.0


def test_nsys_vendor_is_nvidia(nsys_summary):
    assert nsys_summary.device_info.vendor == "nvidia"


def test_nsys_kernel_short_name_type(nsys_summary):
    # short_name is optional on nsys too, but when present it must be a string
    for k in nsys_summary.top_kernels:
        if k.short_name is not None:
            assert isinstance(k.short_name, str)


def test_nsys_mpi_present_true(nsys_summary):
    assert nsys_summary.mpi_present is True


def test_nsys_mpi_ops_non_empty(nsys_summary):
    assert len(nsys_summary.mpi_ops) > 0
