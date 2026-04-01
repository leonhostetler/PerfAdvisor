"""Compute structured metrics from a NsysProfile.

Each function is a focused query that returns data for one section of ProfileSummary.
All SQL is written to be portable across profiles — tables are checked for existence
before querying, and optional tables (MPI, NVTX) degrade gracefully.
"""

from __future__ import annotations

from nsight_agent.ingestion.profile import NsysProfile

from .models import (
    GapBucket,
    KernelSummary,
    MemcpySummary,
    MpiOpSummary,
    NvtxRangeSummary,
    PhaseSummary,
    ProfileSummary,
    StreamSummary,
)
from .phases import PhaseWindow, detect_phases

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_profile_span(profile: NsysProfile) -> float:
    """True wall-clock duration from first to last captured event across all sources.

    Includes CPU-side CUDA API calls (RUNTIME) and NVTX annotations so that the
    reported span matches the timeline shown in the Nsight Systems GUI.
    """
    sources = [
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL",
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY",
    ]
    if profile.has_table("CUPTI_ACTIVITY_KIND_RUNTIME"):
        sources.append(
            "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME "
            "WHERE start IS NOT NULL AND end IS NOT NULL"
        )
    if profile.has_nvtx():
        sources.append(
            "SELECT start, end FROM NVTX_EVENTS "
            "WHERE start IS NOT NULL AND end IS NOT NULL AND end > start"
        )
    union_sql = " UNION ALL ".join(sources)
    row = profile.query(
        f"SELECT (MAX(end) - MIN(start)) / 1e9 AS span_s FROM ({union_sql})"
    )[0]
    return float(row["span_s"] or 0.0)


def compute_gpu_kernel_time(profile: NsysProfile) -> float:
    row = profile.query(
        "SELECT COALESCE(SUM(end - start), 0) / 1e9 AS t FROM CUPTI_ACTIVITY_KIND_KERNEL"
    )[0]
    return float(row["t"])


def compute_gpu_memcpy_time(profile: NsysProfile) -> float:
    row = profile.query(
        "SELECT COALESCE(SUM(end - start), 0) / 1e9 AS t FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    )[0]
    return float(row["t"])


def compute_gpu_sync_time(profile: NsysProfile) -> float:
    row = profile.query(
        "SELECT COALESCE(SUM(end - start), 0) / 1e9 AS t FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"
    )[0]
    return float(row["t"])


def compute_top_kernels(profile: NsysProfile, limit: int = 15) -> list[KernelSummary]:
    total_gpu_s = compute_gpu_kernel_time(profile)
    rows = profile.query(f"""
        SELECT
            s.value                             AS name,
            COUNT(*)                            AS calls,
            SUM(k.end - k.start) / 1e9         AS total_s,
            AVG(k.end - k.start) / 1e6         AS avg_ms,
            MIN(k.end - k.start) / 1e6         AS min_ms,
            MAX(k.end - k.start) / 1e6         AS max_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        GROUP BY k.shortName
        ORDER BY total_s DESC
        LIMIT {limit}
    """)
    return [
        KernelSummary(
            name=r["name"],
            calls=r["calls"],
            total_s=round(r["total_s"], 4),
            avg_ms=round(r["avg_ms"], 4),
            min_ms=round(r["min_ms"], 4),
            max_ms=round(r["max_ms"], 4),
            pct_of_gpu_time=round(100.0 * r["total_s"] / total_gpu_s, 1) if total_gpu_s else 0.0,
        )
        for r in rows
    ]


def compute_memcpy_by_kind(profile: NsysProfile) -> list[MemcpySummary]:
    rows = profile.query("""
        SELECT
            e.label                              AS kind,
            COUNT(*)                             AS transfers,
            SUM(m.bytes)                         AS total_bytes,
            SUM(m.end - m.start) / 1e9          AS total_s,
            CAST(SUM(m.bytes) AS REAL) / NULLIF(SUM(m.end - m.start), 0)
                                                 AS effective_GBs
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m
        JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
        GROUP BY m.copyKind
        ORDER BY total_s DESC
    """)
    return [
        MemcpySummary(
            kind=r["kind"],
            transfers=r["transfers"],
            total_bytes=r["total_bytes"] or 0,
            total_s=round(r["total_s"], 4),
            effective_GBs=round(r["effective_GBs"] or 0.0, 2),
        )
        for r in rows
    ]


def compute_gap_histogram(profile: NsysProfile) -> tuple[float, list[GapBucket]]:
    """Return (total_idle_s, gap_histogram) for inter-kernel idle time."""
    rows = profile.query("""
        WITH ordered AS (
            SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) AS rn
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        ),
        gaps AS (
            SELECT o2.start - o1.end AS gap_ns
            FROM ordered o1
            JOIN ordered o2 ON o2.rn = o1.rn + 1
            WHERE o2.start > o1.end
        )
        SELECT
            CASE
                WHEN gap_ns < 10000      THEN '<10us'
                WHEN gap_ns < 100000     THEN '10-100us'
                WHEN gap_ns < 1000000    THEN '100us-1ms'
                WHEN gap_ns < 10000000   THEN '1-10ms'
                WHEN gap_ns < 100000000  THEN '10-100ms'
                ELSE '>100ms'
            END                          AS label,
            COUNT(*)                     AS count,
            SUM(gap_ns) / 1e9           AS total_s
        FROM gaps
        GROUP BY label
        ORDER BY MIN(gap_ns)
    """)
    buckets = [
        GapBucket(label=r["label"], count=r["count"], total_s=round(r["total_s"], 3))
        for r in rows
    ]
    total_idle_s = sum(b.total_s for b in buckets)
    return total_idle_s, buckets


def compute_streams(profile: NsysProfile) -> list[StreamSummary]:
    total_gpu_s = compute_gpu_kernel_time(profile)
    rows = profile.query("""
        SELECT
            streamId                        AS stream_id,
            COUNT(*)                        AS kernel_calls,
            SUM(end - start) / 1e9         AS total_gpu_s
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY streamId
        ORDER BY total_gpu_s DESC
    """)
    return [
        StreamSummary(
            stream_id=r["stream_id"],
            kernel_calls=r["kernel_calls"],
            total_gpu_s=round(r["total_gpu_s"], 4),
            pct_of_gpu_time=round(100.0 * r["total_gpu_s"] / total_gpu_s, 1) if total_gpu_s else 0.0,
        )
        for r in rows
    ]


def compute_nvtx_ranges(profile: NsysProfile, limit: int = 20) -> list[NvtxRangeSummary]:
    if not profile.has_nvtx():
        return []
    rows = profile.query(f"""
        SELECT
            text                            AS name,
            COUNT(*)                        AS calls,
            SUM(end - start) / 1e9         AS total_s,
            AVG(end - start) / 1e6         AS avg_ms
        FROM NVTX_EVENTS
        WHERE eventType = 59
          AND end IS NOT NULL
          AND end > start
          AND text IS NOT NULL
        GROUP BY text
        ORDER BY total_s DESC
        LIMIT {limit}
    """)
    return [
        NvtxRangeSummary(
            name=r["name"],
            calls=r["calls"],
            total_s=round(r["total_s"], 3),
            avg_ms=round(r["avg_ms"], 3),
        )
        for r in rows
    ]


def compute_mpi_ops(profile: NsysProfile, limit: int = 10) -> list[MpiOpSummary]:
    if not profile.has_mpi():
        return []

    ops: list[MpiOpSummary] = []

    for table in ("MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS", "MPI_START_WAIT_EVENTS"):
        if not profile.has_table(table):
            continue
        rows = profile.query(f"""
            SELECT
                s.value                         AS op,
                COUNT(*)                        AS calls,
                SUM(end - start) / 1e9         AS total_s,
                AVG(end - start) / 1e6         AS avg_ms,
                MAX(end - start) / 1e6         AS max_ms
            FROM {table} e
            JOIN StringIds s ON e.textId = s.id
            WHERE end IS NOT NULL
            GROUP BY e.textId
            ORDER BY total_s DESC
            LIMIT {limit}
        """)
        ops.extend(
            MpiOpSummary(
                op=r["op"],
                calls=r["calls"],
                total_s=round(r["total_s"], 3),
                avg_ms=round(r["avg_ms"], 3),
                max_ms=round(r["max_ms"], 3),
            )
            for r in rows
        )

    # Deduplicate by op name, keeping highest total_s entry
    seen: dict[str, MpiOpSummary] = {}
    for op in sorted(ops, key=lambda x: x.total_s, reverse=True):
        seen.setdefault(op.op, op)
    return sorted(seen.values(), key=lambda x: x.total_s, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# Per-phase metric helpers
# ---------------------------------------------------------------------------


def _window_kernel_time(profile: NsysProfile, start_ns: int, end_ns: int) -> float:
    row = profile.query(f"""
        SELECT COALESCE(SUM(end - start), 0) / 1e9 AS t
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE start >= {start_ns} AND end <= {end_ns}
    """)[0]
    return float(row["t"])


def _window_memcpy_time(profile: NsysProfile, start_ns: int, end_ns: int) -> float:
    row = profile.query(f"""
        SELECT COALESCE(SUM(end - start), 0) / 1e9 AS t
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE start >= {start_ns} AND end <= {end_ns}
    """)[0]
    return float(row["t"])


def _window_idle_time(profile: NsysProfile, start_ns: int, end_ns: int) -> float:
    row = profile.query(f"""
        WITH ordered AS (
            SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) AS rn
            FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE start >= {start_ns} AND end <= {end_ns}
        ),
        gaps AS (
            SELECT o2.start - o1.end AS gap_ns
            FROM ordered o1
            JOIN ordered o2 ON o2.rn = o1.rn + 1
            WHERE o2.start > o1.end
        )
        SELECT COALESCE(SUM(gap_ns), 0) / 1e9 AS t
        FROM gaps
    """)[0]
    return float(row["t"])


def _window_top_kernels(
    profile: NsysProfile, start_ns: int, end_ns: int, total_kernel_s: float, limit: int = 5
) -> list[KernelSummary]:
    rows = profile.query(f"""
        SELECT
            s.value                             AS name,
            COUNT(*)                            AS calls,
            SUM(k.end - k.start) / 1e9         AS total_s,
            AVG(k.end - k.start) / 1e6         AS avg_ms,
            MIN(k.end - k.start) / 1e6         AS min_ms,
            MAX(k.end - k.start) / 1e6         AS max_ms
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= {start_ns} AND k.end <= {end_ns}
        GROUP BY k.shortName
        ORDER BY total_s DESC
        LIMIT {limit}
    """)
    return [
        KernelSummary(
            name=r["name"],
            calls=r["calls"],
            total_s=round(r["total_s"], 4),
            avg_ms=round(r["avg_ms"], 4),
            min_ms=round(r["min_ms"], 4),
            max_ms=round(r["max_ms"], 4),
            pct_of_gpu_time=round(100.0 * r["total_s"] / total_kernel_s, 1) if total_kernel_s else 0.0,
        )
        for r in rows
    ]


def _window_mpi_ops(
    profile: NsysProfile, start_ns: int, end_ns: int, limit: int = 5
) -> list[MpiOpSummary]:
    if not profile.has_mpi():
        return []
    ops: list[MpiOpSummary] = []
    for table in ("MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS", "MPI_START_WAIT_EVENTS"):
        if not profile.has_table(table):
            continue
        rows = profile.query(f"""
            SELECT
                s.value                         AS op,
                COUNT(*)                        AS calls,
                SUM(end - start) / 1e9         AS total_s,
                AVG(end - start) / 1e6         AS avg_ms,
                MAX(end - start) / 1e6         AS max_ms
            FROM {table} e
            JOIN StringIds s ON e.textId = s.id
            WHERE e.start >= {start_ns} AND e.end <= {end_ns}
              AND e.end IS NOT NULL
            GROUP BY e.textId
            ORDER BY total_s DESC
            LIMIT {limit}
        """)
        ops.extend(
            MpiOpSummary(
                op=r["op"],
                calls=r["calls"],
                total_s=round(r["total_s"], 3),
                avg_ms=round(r["avg_ms"], 3),
                max_ms=round(r["max_ms"], 3),
            )
            for r in rows
        )
    seen: dict[str, MpiOpSummary] = {}
    for op in sorted(ops, key=lambda x: x.total_s, reverse=True):
        seen.setdefault(op.op, op)
    return sorted(seen.values(), key=lambda x: x.total_s, reverse=True)[:limit]


def compute_phase_summary(
    profile: NsysProfile,
    phase: PhaseWindow,
    profile_start_ns: int,
) -> PhaseSummary:
    """Compute full metrics for a single PhaseWindow."""
    kernel_s = _window_kernel_time(profile, phase.start_ns, phase.end_ns)
    memcpy_s = _window_memcpy_time(profile, phase.start_ns, phase.end_ns)
    idle_s = _window_idle_time(profile, phase.start_ns, phase.end_ns)
    duration_s = (phase.end_ns - phase.start_ns) / 1e9
    start_s = (phase.start_ns - profile_start_ns) / 1e9

    return PhaseSummary(
        name=phase.name,
        start_s=round(start_s, 3),
        end_s=round(start_s + duration_s, 3),
        duration_s=round(duration_s, 3),
        gpu_utilization_pct=round(100.0 * kernel_s / duration_s, 1) if duration_s else 0.0,
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        total_gpu_idle_s=round(idle_s, 3),
        top_kernels=_window_top_kernels(profile, phase.start_ns, phase.end_ns, kernel_s),
        mpi_ops=_window_mpi_ops(profile, phase.start_ns, phase.end_ns),
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_profile_summary(profile: NsysProfile, max_phases: int = 6) -> ProfileSummary:
    """Compute all metrics for a profile and return a ProfileSummary.

    Set max_phases=1 to skip phase segmentation (returns a single phase).
    """
    span_s = compute_profile_span(profile)
    kernel_s = compute_gpu_kernel_time(profile)
    memcpy_s = compute_gpu_memcpy_time(profile)
    sync_s = compute_gpu_sync_time(profile)
    total_idle_s, gap_histogram = compute_gap_histogram(profile)

    phases_windows = detect_phases(profile, max_phases=max_phases)
    profile_start_ns = phases_windows[0].start_ns if phases_windows else 0
    phase_summaries = [
        compute_phase_summary(profile, pw, profile_start_ns)
        for pw in phases_windows
    ]

    return ProfileSummary(
        profile_path=str(profile.path),
        profile_span_s=round(span_s, 3),
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        gpu_sync_s=round(sync_s, 3),
        gpu_utilization_pct=round(100.0 * kernel_s / span_s, 1) if span_s else 0.0,
        total_gpu_idle_s=round(total_idle_s, 3),
        gap_histogram=gap_histogram,
        top_kernels=compute_top_kernels(profile),
        memcpy_by_kind=compute_memcpy_by_kind(profile),
        streams=compute_streams(profile),
        nvtx_ranges=compute_nvtx_ranges(profile),
        mpi_ops=compute_mpi_ops(profile),
        mpi_present=profile.has_mpi(),
        phases=phase_summaries,
    )
