"""Cross-rank analysis for multi-rank MPI profiles.

Provides utilities to:
  - Parse rank IDs from a set of filenames
  - Select the primary (outlier) rank
  - Align execution phases across ranks
  - Compute cross-rank imbalance metrics
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

from .models import (
    CollectiveImbalance,
    CrossRankPhaseSummary,
    CrossRankSummary,
    ProfileSummary,
    RankOverview,
    RankPhaseStats,
)

# A rank is considered an outlier if its GPU idle time exceeds the median by
# more than this fraction.
OUTLIER_IDLE_THRESHOLD = 0.20

# Phases are considered duration-matched (for index-order fallback) if every
# rank's phase duration is within this fraction of the cross-rank mean.
PHASE_DURATION_TOLERANCE = 0.20


# ---------------------------------------------------------------------------
# Rank ID parsing
# ---------------------------------------------------------------------------


def parse_rank_ids(paths: list[Path]) -> tuple[list[int], bool]:
    """Derive rank IDs from a list of profile file paths.

    Strategy: extract all integers from each filename, then find which
    positional slot varies across the set.  If exactly one slot varies,
    use those values as rank IDs.  Otherwise fall back to [0, 1, 2, …].

    Returns (rank_ids, parsed_ok) where parsed_ok is False when the
    fallback was used.
    """
    names = [p.stem for p in paths]
    # Extract all non-negative integers from each stem, in order.
    int_seqs: list[list[int]] = [[int(m) for m in re.findall(r"\d+", name)] for name in names]

    # All stems must yield the same number of integer tokens to compare slots.
    lengths = {len(seq) for seq in int_seqs}
    if len(lengths) == 1 and lengths != {0}:
        n_slots = lengths.pop()
        varying = [i for i in range(n_slots) if len({seq[i] for seq in int_seqs}) > 1]
        if len(varying) == 1:
            slot = varying[0]
            return [seq[slot] for seq in int_seqs], True

    # Fallback: index order
    return list(range(len(paths))), False


# ---------------------------------------------------------------------------
# Primary rank selection
# ---------------------------------------------------------------------------


def select_primary_rank(
    summaries: dict[int, ProfileSummary],
) -> tuple[int, str]:
    """Choose the primary (outlier) rank based on GPU idle time.

    Returns (rank_id, reason) where reason is a human-readable string
    suitable for printing to the user.
    """
    idle_by_rank = {rid: s.total_gpu_idle_s for rid, s in summaries.items()}
    rank_ids = sorted(idle_by_rank)
    idle_values = [idle_by_rank[r] for r in rank_ids]
    med = statistics.median(idle_values)

    if med > 0:
        outliers = [r for r in rank_ids if idle_by_rank[r] > med * (1 + OUTLIER_IDLE_THRESHOLD)]
    else:
        outliers = []

    if outliers:
        primary = max(outliers, key=lambda r: idle_by_rank[r])
        idle = idle_by_rank[primary]
        pct_above = 100.0 * (idle - med) / med
        reason = (
            f"GPU idle {idle:.1f}s vs. median {med:.1f}s across ranks "
            f"({pct_above:.0f}% above median)"
        )
        return primary, reason

    # No clear outlier — use the lowest rank ID.
    primary = rank_ids[0]
    reason = (
        f"no clear outlier (all ranks within {int(OUTLIER_IDLE_THRESHOLD * 100)}%"
        f" of median GPU idle {med:.1f}s) — defaulting to rank {primary}"
    )
    return primary, reason


# ---------------------------------------------------------------------------
# Phase alignment
# ---------------------------------------------------------------------------


def align_phases(
    summaries: dict[int, ProfileSummary],
) -> tuple[str, str | None]:
    """Check that phases are consistent across ranks.

    Returns (alignment_mode, message) where:
      - alignment_mode is one of:
          "name_match"     — all ranks agree on phase names; proceed normally
          "index_order"    — names differ but durations match; use index order
          "failed"         — alignment cannot be established; abort cross-rank
      - message is None on success, or a human-readable description of the
        problem (and how the program will proceed) for display to the user.
    """
    rank_ids = sorted(summaries)
    phase_name_lists = {rid: [p.name for p in summaries[rid].phases] for rid in rank_ids}
    phase_counts = {rid: len(names) for rid, names in phase_name_lists.items()}

    # --- Check 1: all ranks must have the same number of phases ---
    if len(set(phase_counts.values())) > 1:
        detail_lines = [f"  rank {r}: {phase_counts[r]} phases" for r in rank_ids]
        detail = "\n".join(detail_lines)
        msg = "Phase count differs across ranks — cross-rank analysis cannot proceed.\n" + detail
        return "failed", msg

    # --- Check 2: phase names must match (or durations must agree) ---
    reference_names = phase_name_lists[rank_ids[0]]
    names_match = all(phase_name_lists[r] == reference_names for r in rank_ids[1:])

    if names_match:
        return "name_match", None

    # Names differ — check if durations agree within tolerance.
    n_phases = phase_counts[rank_ids[0]]
    for phase_idx in range(n_phases):
        durations = [summaries[r].phases[phase_idx].duration_s for r in rank_ids]
        mean_dur = statistics.mean(durations)
        if mean_dur <= 0:
            continue
        if any(abs(d - mean_dur) / mean_dur > PHASE_DURATION_TOLERANCE for d in durations):
            # Duration divergence — likely a phase detection artifact.
            worst_rank = max(
                rank_ids,
                key=lambda r: abs(summaries[r].phases[phase_idx].duration_s - mean_dur) / mean_dur,
            )
            worst_dur = summaries[worst_rank].phases[phase_idx].duration_s
            msg = (
                f"Phase names differ across ranks and phase {phase_idx} durations diverge "
                f"beyond {int(PHASE_DURATION_TOLERANCE * 100)}% tolerance "
                f"(rank {worst_rank}: {worst_dur:.2f}s vs. mean {mean_dur:.2f}s) — "
                f"this likely indicates a phase detection artifact. "
                f"Cross-rank analysis cannot proceed."
            )
            return "failed", msg

    # Names differ but durations agree — use index-order alignment.
    mismatched = []
    for r in rank_ids[1:]:
        if phase_name_lists[r] != reference_names:
            mismatched.append(r)
    msg = (
        f"Phase names differ across ranks (ranks {mismatched} disagree with rank {rank_ids[0]}) "
        f"but phase durations agree within {int(PHASE_DURATION_TOLERANCE * 100)}%. "
        f"Proceeding with index-order phase alignment."
    )
    return "index_order", msg


# ---------------------------------------------------------------------------
# Cross-rank aggregation
# ---------------------------------------------------------------------------


def _mpi_wait_s(summary: ProfileSummary, phase_idx: int) -> float:
    """Sum of all MPI op total_s for a given phase."""
    if phase_idx >= len(summary.phases):
        return 0.0
    return sum(op.total_s for op in summary.phases[phase_idx].mpi_ops)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def _imbalance(values: list[float]) -> float:
    """(max - min) / mean; 0 when mean is 0."""
    if not values:
        return 0.0
    mean = statistics.mean(values)
    if mean == 0:
        return 0.0
    return (max(values) - min(values)) / mean


def compute_cross_rank_summary(
    summaries: dict[int, ProfileSummary],
    primary_rank_id: int,
    phase_alignment: str,
) -> CrossRankSummary:
    """Compute cross-rank imbalance metrics from per-rank ProfileSummary objects."""
    rank_ids = sorted(summaries)

    # --- Per-rank overview (whole-profile) ---
    overviews = []
    for rid in rank_ids:
        s = summaries[rid]
        mpi_wait = sum(op.total_s for op in s.mpi_ops)
        overviews.append(
            RankOverview(
                rank_id=rid,
                gpu_kernel_s=s.gpu_kernel_s,
                gpu_idle_s=s.total_gpu_idle_s,
                mpi_wait_s=mpi_wait,
                gpu_utilization_pct=s.gpu_utilization_pct,
            )
        )

    # --- Per-phase cross-rank stats ---
    n_phases = len(summaries[rank_ids[0]].phases)
    phase_summaries: list[CrossRankPhaseSummary] = []

    for phase_idx in range(n_phases):
        phase_name = summaries[rank_ids[0]].phases[phase_idx].name

        gpu_kernel_vals = [summaries[r].phases[phase_idx].gpu_kernel_s for r in rank_ids]
        mpi_wait_vals = [_mpi_wait_s(summaries[r], phase_idx) for r in rank_ids]

        gpu_mean = statistics.mean(gpu_kernel_vals)
        mpi_mean = statistics.mean(mpi_wait_vals)

        gpu_slowest = rank_ids[gpu_kernel_vals.index(max(gpu_kernel_vals))]
        mpi_slowest = rank_ids[mpi_wait_vals.index(max(mpi_wait_vals))]

        # Per-collective imbalance: gather all op names present across ranks
        all_ops: set[str] = set()
        for r in rank_ids:
            for op in summaries[r].phases[phase_idx].mpi_ops:
                all_ops.add(op.op)

        collective_imbalances: list[CollectiveImbalance] = []
        for op_name in sorted(all_ops):
            op_vals = []
            for r in rank_ids:
                matching = [
                    o.total_s for o in summaries[r].phases[phase_idx].mpi_ops if o.op == op_name
                ]
                op_vals.append(matching[0] if matching else 0.0)
            op_mean = statistics.mean(op_vals)
            if op_mean == 0:
                continue
            slowest_r = rank_ids[op_vals.index(max(op_vals))]
            collective_imbalances.append(
                CollectiveImbalance(
                    op=op_name,
                    imbalance_score=round(_imbalance(op_vals), 3),
                    slowest_rank_id=slowest_r,
                    mean_s=round(op_mean, 4),
                    min_s=round(min(op_vals), 4),
                    max_s=round(max(op_vals), 4),
                )
            )
        # Sort by imbalance score descending so the worst offender is first
        collective_imbalances.sort(key=lambda x: x.imbalance_score, reverse=True)

        per_rank = [
            RankPhaseStats(
                rank_id=r,
                gpu_kernel_s=round(summaries[r].phases[phase_idx].gpu_kernel_s, 4),
                gpu_idle_s=round(summaries[r].phases[phase_idx].total_gpu_idle_s, 4),
                mpi_wait_s=round(_mpi_wait_s(summaries[r], phase_idx), 4),
            )
            for r in rank_ids
        ]

        phase_summaries.append(
            CrossRankPhaseSummary(
                phase_index=phase_idx,
                phase_name=phase_name,
                gpu_kernel_mean_s=round(gpu_mean, 4),
                gpu_kernel_std_s=round(_std(gpu_kernel_vals), 4),
                gpu_kernel_min_s=round(min(gpu_kernel_vals), 4),
                gpu_kernel_max_s=round(max(gpu_kernel_vals), 4),
                gpu_kernel_imbalance=round(_imbalance(gpu_kernel_vals), 3),
                gpu_kernel_slowest_rank_id=gpu_slowest,
                mpi_wait_mean_s=round(mpi_mean, 4),
                mpi_wait_std_s=round(_std(mpi_wait_vals), 4),
                mpi_wait_min_s=round(min(mpi_wait_vals), 4),
                mpi_wait_max_s=round(max(mpi_wait_vals), 4),
                mpi_wait_imbalance=round(_imbalance(mpi_wait_vals), 3),
                mpi_wait_slowest_rank_id=mpi_slowest,
                collective_imbalance=collective_imbalances,
                per_rank=per_rank,
            )
        )

    return CrossRankSummary(
        num_ranks=len(rank_ids),
        rank_ids=rank_ids,
        primary_rank_id=primary_rank_id,
        phase_alignment=phase_alignment,
        per_rank_overview=overviews,
        phases=phase_summaries,
    )
