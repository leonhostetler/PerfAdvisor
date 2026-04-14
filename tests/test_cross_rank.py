"""Tests for perf_advisor.analysis.cross_rank.

All tests are purely in-memory — no SQLite files required.
ProfileSummary objects are constructed synthetically.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from perf_advisor.analysis.cross_rank import (
    OUTLIER_IDLE_THRESHOLD,
    align_phases,
    compute_cross_rank_summary,
    parse_rank_ids,
    select_primary_rank,
)
from perf_advisor.analysis.models import MpiOpSummary, PhaseSummary, ProfileSummary

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_phase(
    name: str,
    duration_s: float = 10.0,
    gpu_kernel_s: float = 8.0,
    gpu_idle_s: float = 2.0,
    mpi_ops: list[MpiOpSummary] | None = None,
) -> PhaseSummary:
    return PhaseSummary(
        name=name,
        start_s=0.0,
        end_s=duration_s,
        duration_s=duration_s,
        start_ns=0,
        end_ns=int(duration_s * 1e9),
        gpu_utilization_pct=100.0 * gpu_kernel_s / duration_s,
        gpu_kernel_s=gpu_kernel_s,
        gpu_memcpy_s=0.0,
        total_gpu_idle_s=gpu_idle_s,
        top_kernels=[],
        mpi_ops=mpi_ops or [],
    )


def _make_summary(
    gpu_kernel_s: float = 30.0,
    total_gpu_idle_s: float = 10.0,
    mpi_ops: list[MpiOpSummary] | None = None,
    phases: list[PhaseSummary] | None = None,
) -> ProfileSummary:
    span = gpu_kernel_s + total_gpu_idle_s
    return ProfileSummary(
        profile_path="/fake/rank.sqlite",
        profile_span_s=span,
        gpu_kernel_s=gpu_kernel_s,
        gpu_memcpy_s=0.0,
        gpu_sync_s=0.0,
        gpu_utilization_pct=100.0 * gpu_kernel_s / span,
        total_gpu_idle_s=total_gpu_idle_s,
        gap_histogram=[],
        top_kernels=[],
        memcpy_by_kind=[],
        streams=[],
        mpi_ops=mpi_ops or [],
        mpi_present=bool(mpi_ops),
        phases=phases or [],
    )


# ---------------------------------------------------------------------------
# parse_rank_ids
# ---------------------------------------------------------------------------


class TestParseRankIds:
    def test_single_varying_slot(self):
        paths = [
            Path("report.0.nid001429.50613743.0.sqlite"),
            Path("report.0.nid001429.50613743.1.sqlite"),
            Path("report.0.nid001429.50613743.2.sqlite"),
        ]
        ids, ok = parse_rank_ids(paths)
        assert ok is True
        assert ids == [0, 1, 2]

    def test_fallback_when_multiple_slots_vary(self):
        paths = [
            Path("rank0_job100.sqlite"),
            Path("rank1_job200.sqlite"),
        ]
        ids, ok = parse_rank_ids(paths)
        assert ok is False
        assert ids == [0, 1]

    def test_fallback_when_no_integers(self):
        paths = [Path("alpha.sqlite"), Path("beta.sqlite")]
        ids, ok = parse_rank_ids(paths)
        assert ok is False
        assert ids == [0, 1]

    def test_single_file(self):
        paths = [Path("rank3.sqlite")]
        ids, ok = parse_rank_ids(paths)
        # Only one file: no slot "varies", but index fallback gives [0]
        assert ids == [0]

    def test_non_contiguous_rank_ids(self):
        paths = [
            Path("profile.node0.rank4.sqlite"),
            Path("profile.node0.rank7.sqlite"),
        ]
        ids, ok = parse_rank_ids(paths)
        # Two integer slots vary (node and rank both stay same in node column;
        # but "rank4" vs "rank7" — the second int varies).
        # node slot: 0 vs 0 — constant. rank slot: 4 vs 7 — varies.
        assert ok is True
        assert ids == [4, 7]


# ---------------------------------------------------------------------------
# select_primary_rank
# ---------------------------------------------------------------------------


class TestSelectPrimaryRank:
    def test_clear_outlier(self):
        summaries = {
            0: _make_summary(total_gpu_idle_s=10.0),
            1: _make_summary(total_gpu_idle_s=11.0),
            2: _make_summary(total_gpu_idle_s=10.5),
            3: _make_summary(total_gpu_idle_s=25.0),  # outlier
        }
        rank, reason = select_primary_rank(summaries)
        assert rank == 3
        assert "25" in reason or "outlier" in reason.lower() or "median" in reason.lower()

    def test_no_outlier_returns_lowest_rank(self):
        summaries = {
            0: _make_summary(total_gpu_idle_s=10.0),
            1: _make_summary(total_gpu_idle_s=10.5),
            2: _make_summary(total_gpu_idle_s=10.2),
        }
        rank, reason = select_primary_rank(summaries)
        assert rank == 0
        assert "no clear outlier" in reason.lower() or "default" in reason.lower()

    def test_multiple_outliers_picks_worst(self):
        summaries = {
            0: _make_summary(total_gpu_idle_s=10.0),
            1: _make_summary(total_gpu_idle_s=20.0),
            2: _make_summary(total_gpu_idle_s=30.0),  # worst
        }
        rank, _ = select_primary_rank(summaries)
        assert rank == 2

    def test_threshold_boundary(self):
        # Exactly at threshold should NOT be considered an outlier (strictly greater)
        median = 10.0
        at_threshold = median * (1 + OUTLIER_IDLE_THRESHOLD)
        summaries = {
            0: _make_summary(total_gpu_idle_s=median),
            1: _make_summary(total_gpu_idle_s=at_threshold),
        }
        rank, _ = select_primary_rank(summaries)
        assert rank == 0  # no outlier, falls back to rank 0


# ---------------------------------------------------------------------------
# align_phases
# ---------------------------------------------------------------------------


class TestAlignPhases:
    def test_name_match(self):
        phases = [_make_phase("deflation"), _make_phase("cg_solve")]
        summaries = {
            0: _make_summary(phases=phases),
            1: _make_summary(phases=phases),
            2: _make_summary(phases=phases),
        }
        mode, msg = align_phases(summaries)
        assert mode == "name_match"
        assert msg is None

    def test_count_mismatch_fails(self):
        s0 = _make_summary(phases=[_make_phase("p1"), _make_phase("p2")])
        s1 = _make_summary(phases=[_make_phase("p1")])
        mode, msg = align_phases({0: s0, 1: s1})
        assert mode == "failed"
        assert "count" in msg.lower() or "differ" in msg.lower()

    def test_name_mismatch_with_duration_agreement_uses_index_order(self):
        phases_a = [
            _make_phase("deflation", duration_s=10.0),
            _make_phase("cg", duration_s=20.0),
        ]
        phases_b = [
            _make_phase("phase_1", duration_s=10.2),
            _make_phase("phase_2", duration_s=19.8),
        ]
        summaries = {0: _make_summary(phases=phases_a), 1: _make_summary(phases=phases_b)}
        mode, msg = align_phases(summaries)
        assert mode == "index_order"
        assert msg is not None

    def test_name_mismatch_with_duration_divergence_fails(self):
        phases_a = [_make_phase("deflation", duration_s=10.0)]
        phases_b = [_make_phase("other", duration_s=50.0)]  # wildly different
        summaries = {0: _make_summary(phases=phases_a), 1: _make_summary(phases=phases_b)}
        mode, msg = align_phases(summaries)
        assert mode == "failed"
        assert msg is not None


# ---------------------------------------------------------------------------
# compute_cross_rank_summary
# ---------------------------------------------------------------------------


class TestComputeCrossRankSummary:
    def _make_rank_summaries(self):
        """Three ranks with a clear GPU-kernel straggler on rank 2."""
        mpi_phase = [
            MpiOpSummary(op="MPI_Barrier", calls=10, total_s=2.0, avg_ms=200.0, max_ms=300.0),
            MpiOpSummary(op="MPI_Allreduce", calls=5, total_s=1.0, avg_ms=200.0, max_ms=250.0),
        ]
        phases_rank0 = [_make_phase("cg", gpu_kernel_s=8.0, gpu_idle_s=2.0, mpi_ops=mpi_phase)]
        phases_rank1 = [_make_phase("cg", gpu_kernel_s=8.5, gpu_idle_s=1.5, mpi_ops=mpi_phase)]
        phases_rank2 = [
            _make_phase(
                "cg",
                gpu_kernel_s=14.0,
                gpu_idle_s=2.0,
                mpi_ops=[
                    MpiOpSummary(
                        op="MPI_Barrier", calls=10, total_s=0.5, avg_ms=50.0, max_ms=100.0
                    ),
                    MpiOpSummary(
                        op="MPI_Allreduce", calls=5, total_s=0.3, avg_ms=60.0, max_ms=80.0
                    ),
                ],
            )
        ]
        return {
            0: _make_summary(gpu_kernel_s=8.0, total_gpu_idle_s=12.0, phases=phases_rank0),
            1: _make_summary(gpu_kernel_s=8.5, total_gpu_idle_s=11.5, phases=phases_rank1),
            2: _make_summary(gpu_kernel_s=14.0, total_gpu_idle_s=6.0, phases=phases_rank2),
        }

    def test_structure(self):
        summaries = self._make_rank_summaries()
        crs = compute_cross_rank_summary(summaries, primary_rank_id=0, phase_alignment="name_match")
        assert crs.num_ranks == 3
        assert crs.rank_ids == [0, 1, 2]
        assert crs.primary_rank_id == 0
        assert len(crs.phases) == 1
        assert len(crs.per_rank_overview) == 3

    def test_gpu_kernel_slowest_rank(self):
        summaries = self._make_rank_summaries()
        crs = compute_cross_rank_summary(summaries, primary_rank_id=0, phase_alignment="name_match")
        phase = crs.phases[0]
        assert phase.gpu_kernel_slowest_rank_id == 2
        assert phase.gpu_kernel_max_s == pytest.approx(14.0)

    def test_collective_imbalance_sorted_descending(self):
        summaries = self._make_rank_summaries()
        crs = compute_cross_rank_summary(summaries, primary_rank_id=0, phase_alignment="name_match")
        phase = crs.phases[0]
        scores = [c.imbalance_score for c in phase.collective_imbalance]
        assert scores == sorted(scores, reverse=True)

    def test_per_rank_data_present(self):
        summaries = self._make_rank_summaries()
        crs = compute_cross_rank_summary(summaries, primary_rank_id=0, phase_alignment="name_match")
        assert len(crs.phases[0].per_rank) == 3
        rank_ids_in_per_rank = [r.rank_id for r in crs.phases[0].per_rank]
        assert rank_ids_in_per_rank == [0, 1, 2]

    def test_mpi_wait_slowest_is_highest_mpi_rank(self):
        summaries = self._make_rank_summaries()
        crs = compute_cross_rank_summary(summaries, primary_rank_id=0, phase_alignment="name_match")
        # Ranks 0 and 1 each have MPI wait = 3.0s; rank 2 has 0.8s
        # So slowest MPI rank should be 0 or 1 (tied; max() picks 0 since it's first)
        phase = crs.phases[0]
        assert phase.mpi_wait_slowest_rank_id in (0, 1)
        assert phase.mpi_wait_max_s == pytest.approx(3.0)
