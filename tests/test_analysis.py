"""Tests for the analysis/metrics layer against the real test profile."""

import pytest

from nsight_agent.analysis.metrics import (
    compute_gap_histogram,
    compute_gpu_kernel_time,
    compute_gpu_memcpy_time,
    compute_gpu_sync_time,
    compute_memcpy_by_kind,
    compute_mpi_ops,
    compute_nvtx_ranges,
    compute_profile_span,
    compute_profile_summary,
    compute_streams,
    compute_top_kernels,
)
from nsight_agent.ingestion.profile import NsysProfile

TEST_PROFILE = "/home/ads.leonhost/Downloads/nsight/nsys_4864_cgdef_2node_1rhs/cg_4864_1rhs.sqlite"


@pytest.fixture(scope="module")
def profile():
    p = NsysProfile(TEST_PROFILE)
    yield p
    p.close()


# --- Time budget ---

def test_profile_span(profile):
    span = compute_profile_span(profile)
    assert 95.0 < span < 115.0, f"Expected ~102s (true wall-clock span), got {span}"


def test_gpu_kernel_time(profile):
    t = compute_gpu_kernel_time(profile)
    assert 20.0 < t < 30.0, f"Expected ~24.5s, got {t}"


def test_gpu_utilization_reasonable(profile):
    span = compute_profile_span(profile)
    kernel = compute_gpu_kernel_time(profile)
    util = 100.0 * kernel / span
    assert 10.0 < util < 40.0, f"Expected ~24% (kernel time / true profile span), got {util}"


def test_gpu_memcpy_time(profile):
    t = compute_gpu_memcpy_time(profile)
    assert t > 0.0


def test_gpu_sync_time(profile):
    t = compute_gpu_sync_time(profile)
    assert t > 0.0


# --- Kernels ---

def test_top_kernels_returns_results(profile):
    kernels = compute_top_kernels(profile)
    assert len(kernels) > 0


def test_top_kernel_is_kernel3d(profile):
    kernels = compute_top_kernels(profile)
    # name is now the normalized demangled name; short_name holds the display label
    assert kernels[0].short_name == "Kernel3D"
    assert "dslash_functor" in kernels[0].name


def test_kernel_pct_sums_near_100(profile):
    kernels = compute_top_kernels(profile, limit=100)
    total_pct = sum(k.pct_of_gpu_time for k in kernels)
    assert 95.0 < total_pct <= 100.1, f"Expected ~100%, got {total_pct}"


# --- Memory ---

def test_memcpy_has_p2p(profile):
    kinds = compute_memcpy_by_kind(profile)
    kind_names = {m.kind for m in kinds}
    assert "Peer-to-Peer" in kind_names


def test_memcpy_total_bytes_positive(profile):
    kinds = compute_memcpy_by_kind(profile)
    assert all(m.total_bytes >= 0 for m in kinds)


# --- Gaps ---

def test_gap_histogram(profile):
    total_idle, buckets = compute_gap_histogram(profile)
    assert total_idle > 0
    labels = {b.label for b in buckets}
    assert "<10us" in labels
    assert total_idle == pytest.approx(sum(b.total_s for b in buckets), rel=1e-3)


# --- Streams ---

def test_single_dominant_stream(profile):
    streams = compute_streams(profile)
    assert streams[0].pct_of_gpu_time > 90.0


# --- NVTX ---

def test_nvtx_ranges(profile):
    ranges = compute_nvtx_ranges(profile)
    assert len(ranges) > 0
    names = {r.name for r in ranges}
    assert "invertMultiSrcQuda" in names


# --- MPI ---

def test_mpi_ops_present(profile):
    ops = compute_mpi_ops(profile)
    assert len(ops) > 0
    op_names = {o.op for o in ops}
    assert "MPI_Barrier" in op_names


def test_mpi_barrier_dominates(profile):
    ops = compute_mpi_ops(profile)
    barrier = next(o for o in ops if o.op == "MPI_Barrier")
    assert barrier.total_s > 30.0, f"Expected >30s of MPI_Barrier, got {barrier.total_s}"


# --- Full summary ---

def test_compute_profile_summary(profile):
    summary = compute_profile_summary(profile)
    assert summary.profile_span_s > 0
    assert summary.gpu_utilization_pct > 0
    assert summary.mpi_present is True
    assert len(summary.top_kernels) > 0
    assert len(summary.mpi_ops) > 0
