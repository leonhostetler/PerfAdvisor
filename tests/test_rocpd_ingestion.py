"""Tests for RocpdProfile against the synthetic rocpd fixture.

All tests here are CI-safe: they use ``synthetic_rocpd_path`` which is
built in-memory from the fixture builder in conftest.py.

Real-profile smoke tests that require the local test6 capture are in
test_rocpd_schema.py (marked with the ``real_rocpd_path`` fixture, which
auto-skips when the file is absent).
"""

from __future__ import annotations

from perf_advisor.ingestion.base import Format, KernelRow, MemcpyRow
from perf_advisor.ingestion.detect import open_profile
from perf_advisor.ingestion.rocpd import RocpdProfile


def _open(path) -> RocpdProfile:
    return RocpdProfile(path)


# ---------------------------------------------------------------------------
# Format and basics
# ---------------------------------------------------------------------------


def test_format(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.format == Format.ROCPD


def test_repr(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert "RocpdProfile" in repr(p)
        assert "synthetic.rocpd" in repr(p)


def test_open_profile_returns_rocpd(synthetic_rocpd_path):
    with open_profile(synthetic_rocpd_path) as p:
        assert isinstance(p, RocpdProfile)


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_capabilities_has_kernels(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_kernels is True


def test_capabilities_has_memcpy(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_memcpy is True


def test_capabilities_has_runtime_api(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_runtime_api is True


def test_capabilities_no_mpi(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_mpi is False


def test_capabilities_no_markers(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_markers is False


def test_capabilities_no_cpu_samples(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.has_cpu_samples is False


def test_capabilities_schema_version(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        assert p.capabilities.schema_version == "3"


# ---------------------------------------------------------------------------
# String resolution
# ---------------------------------------------------------------------------


def test_resolve_string_valid(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        s = p.resolve_string(1)
    assert isinstance(s, str)
    assert len(s) > 0


def test_resolve_string_missing(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        s = p.resolve_string(99999)
    assert s == "<id:99999>"


# ---------------------------------------------------------------------------
# kernel_events
# ---------------------------------------------------------------------------


def test_kernel_events_count(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    assert len(evts) == 5  # 4× dslash + 1× reduce


def test_kernel_events_type(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    assert all(isinstance(e, KernelRow) for e in evts)


def test_kernel_events_names(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    names = {e.name for e in evts}
    assert any("dslash" in n for n in names)
    assert any("reduce" in n for n in names)


def test_kernel_events_no_kd_suffix(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    assert all(not e.name.endswith(".kd") for e in evts)


def test_kernel_events_duration_positive(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    assert all(e.duration_ns > 0 for e in evts)


def test_kernel_events_gpu_agent(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    assert all(e.device_id == 2 for e in evts)  # GPU agent id=2 in fixture


def test_kernel_events_total_time(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events()
    total_ms = sum(e.duration_ns for e in evts) / 1_000_000
    assert abs(total_ms - 13.0) < 0.01  # 4×2 ms + 5 ms


def test_kernel_events_limit(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.kernel_events(limit=2)
    assert len(evts) == 2


# ---------------------------------------------------------------------------
# memcpy_events
# ---------------------------------------------------------------------------


def test_memcpy_events_count(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    assert len(evts) == 3


def test_memcpy_events_type(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    assert all(isinstance(e, MemcpyRow) for e in evts)


def test_memcpy_directions(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    directions = {e.direction for e in evts}
    assert directions == {"Device-to-Device", "Host-to-Device", "Device-to-Host"}


def test_memcpy_bytes_positive(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    assert all(e.bytes > 0 for e in evts)


def test_memcpy_d2d_size(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    d2d = [e for e in evts if e.direction == "Device-to-Device"]
    assert len(d2d) == 1
    assert d2d[0].bytes == 1_048_576  # 1 MB


def test_memcpy_duration_positive(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        evts = p.memcpy_events()
    assert all(e.duration_ns > 0 for e in evts)


# ---------------------------------------------------------------------------
# marker_ranges and mpi_ranges
# ---------------------------------------------------------------------------


def test_marker_ranges_empty(synthetic_rocpd_path):
    """No rocTX markers in the sys-trace synthetic fixture."""
    with _open(synthetic_rocpd_path) as p:
        assert p.marker_ranges() == []


def test_mpi_ranges_empty(synthetic_rocpd_path):
    """No MPI in sys-trace rocprofv3 output (rocprof-sys needed for MPI)."""
    with _open(synthetic_rocpd_path) as p:
        assert p.mpi_ranges() == []


# ---------------------------------------------------------------------------
# device_info
# ---------------------------------------------------------------------------


def test_device_info_name(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        di = p.device_info()
    assert di.name == "AMD Instinct MI250X"


def test_device_info_cu_count(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        di = p.device_info()
    assert di.sm_count == 110  # cu_count from extdata


def test_device_info_max_threads_per_cu(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        di = p.device_info()
    # wave_front_size=64 × max_waves_per_cu=32 = 2048
    assert di.max_threads_per_sm == 2048


def test_device_info_clock_rate(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        di = p.device_info()
    assert di.clock_rate_MHz == 1700.0


def test_device_info_no_cuda_fields(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        di = p.device_info()
    assert di.compute_capability is None
    assert di.peak_memory_bandwidth_GBs is None


# ---------------------------------------------------------------------------
# Emptiness diagnostics
# ---------------------------------------------------------------------------


def test_emptiness_no_empty_tables(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        emp = p.emptiness
    assert len(emp.empty_tables) == 0


def test_emptiness_categories_populated(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        emp = p.emptiness
    assert "HSA_CORE_API" in emp.observed_categories
    assert "HIP_RUNTIME_API_EXT" in emp.observed_categories
    assert "HIP_COMPILER_API_EXT" in emp.observed_categories
    assert "HSA_AMD_EXT_API" in emp.observed_categories


def test_emptiness_no_truncation(synthetic_rocpd_path):
    with _open(synthetic_rocpd_path) as p:
        emp = p.emptiness
    assert emp.writer_truncation_suspected is False


# ---------------------------------------------------------------------------
# Real-profile smoke tests (auto-skip when local capture is absent)
# ---------------------------------------------------------------------------


def test_real_kernel_event_count(real_rocpd_path):
    with _open(real_rocpd_path) as p:
        evts = p.kernel_events()
    assert len(evts) == 401_777


def test_real_capabilities_schema_version(real_rocpd_path):
    with _open(real_rocpd_path) as p:
        assert p.capabilities.schema_version == "3"


def test_real_capabilities_has_kernels(real_rocpd_path):
    with _open(real_rocpd_path) as p:
        assert p.capabilities.has_kernels is True


def test_real_memcpy_directions_valid(real_rocpd_path):
    _known = {"Device-to-Device", "Host-to-Device", "Device-to-Host", "Peer-to-Peer"}
    with _open(real_rocpd_path) as p:
        evts = p.memcpy_events(limit=100)
    assert all(e.direction in _known for e in evts)
