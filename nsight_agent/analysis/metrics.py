"""Compute structured metrics from a NsysProfile.

Each function is a focused query that returns data for one section of ProfileSummary.
All SQL is written to be portable across profiles — tables are checked for existence
before querying, and optional tables (MPI, NVTX) degrade gracefully.
"""

from __future__ import annotations

import math
import time

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


def compute_top_kernels(
    profile: NsysProfile,
    limit: int = 15,
    device_info: dict | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[KernelSummary]:
    total_gpu_s = compute_gpu_kernel_time(profile)
    rows = profile.query(f"""
        SELECT
            s.value                                                         AS name,
            COUNT(*)                                                        AS calls,
            SUM(k.end - k.start) / 1e9                                     AS total_s,
            AVG(k.end - k.start) / 1e6                                     AS avg_ms,
            MIN(k.end - k.start) / 1e6                                     AS min_ms,
            MAX(k.end - k.start) / 1e6                                     AS max_ms,
            SUM(CAST(k.end - k.start AS REAL) * (k.end - k.start))        AS sum_sq_ns,
            SUM(k.end - k.start)                                            AS sum_ns,
            AVG(COALESCE(k.registersPerThread, 0))                         AS avg_registers,
            AVG(COALESCE(k.sharedMemoryExecuted,
                         k.staticSharedMemory + k.dynamicSharedMemory, 0)) AS avg_shared_mem,
            AVG(CAST(k.gridX * k.gridY * k.gridZ *
                     k.blockX * k.blockY * k.blockZ AS REAL))              AS avg_total_threads
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        GROUP BY k.shortName
        ORDER BY total_s DESC
        LIMIT {limit}
    """)
    sm_count = (device_info or {}).get("sm_count")
    max_threads_per_sm = (device_info or {}).get("max_threads_per_sm")
    result = []
    for r in rows:
        n = r["calls"]
        sum_ns = r["sum_ns"] or 0
        sum_sq_ns = r["sum_sq_ns"] or 0.0
        mean_ns = sum_ns / n if n else 0.0
        variance = (sum_sq_ns / n - mean_ns ** 2) if n > 1 else 0.0
        std_dev_ms = math.sqrt(max(0.0, variance)) / 1e6
        avg_ms = r["avg_ms"] or 0.0
        cv = round(std_dev_ms / avg_ms, 3) if avg_ms > 0 else 0.0

        occupancy = None
        if sm_count and max_threads_per_sm:
            avg_threads = r["avg_total_threads"] or 0.0
            if avg_threads > 0:
                occupancy = round(min(1.0, avg_threads / (sm_count * max_threads_per_sm)), 3)

        oh = (launch_overhead or {}).get(r["name"])
        result.append(KernelSummary(
            name=r["name"],
            calls=n,
            total_s=round(r["total_s"], 4),
            avg_ms=round(avg_ms, 4),
            min_ms=round(r["min_ms"], 4),
            max_ms=round(r["max_ms"], 4),
            pct_of_gpu_time=round(100.0 * r["total_s"] / total_gpu_s, 1) if total_gpu_s else 0.0,
            std_dev_ms=round(std_dev_ms, 4),
            cv=cv,
            avg_registers_per_thread=int(round(r["avg_registers"] or 0)),
            avg_shared_mem_bytes=int(round(r["avg_shared_mem"] or 0)),
            estimated_occupancy=occupancy,
            avg_launch_overhead_us=oh[0] if oh else None,
            max_launch_overhead_us=oh[1] if oh else None,
        ))
    return result


def compute_memcpy_by_kind(
    profile: NsysProfile,
    peak_bandwidth_GBs: float | None = None,
) -> list[MemcpySummary]:
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
            pct_of_peak_bandwidth=(
                round(100.0 * (r["effective_GBs"] or 0.0) / peak_bandwidth_GBs, 1)
                if peak_bandwidth_GBs and (r["effective_GBs"] or 0.0) > 0
                else None
            ),
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


def compute_device_info(profile: NsysProfile) -> dict:
    """Query TARGET_INFO_GPU for hardware properties used in occupancy and bandwidth metrics.

    Returns a dict with keys ``sm_count``, ``max_threads_per_sm``, ``peak_bandwidth_GBs``.
    Any value may be ``None`` if TARGET_INFO_GPU is absent or the column is missing.
    """
    empty: dict = {"sm_count": None, "max_threads_per_sm": None, "peak_bandwidth_GBs": None}
    if not profile.has_table("TARGET_INFO_GPU"):
        return empty

    cols = set(profile.columns("TARGET_INFO_GPU"))
    rows = profile.query("SELECT * FROM TARGET_INFO_GPU LIMIT 1")
    if not rows:
        return empty
    r = rows[0]

    sm_count = int(r["smCount"]) if "smCount" in cols and r["smCount"] else None

    max_threads_per_sm = None
    if "maxWarpsPerSm" in cols and "threadsPerWarp" in cols:
        warps = r["maxWarpsPerSm"]
        warp_size = r["threadsPerWarp"]
        if warps and warp_size:
            max_threads_per_sm = int(warps) * int(warp_size)

    # memoryBandwidth is stored in bytes/s
    peak_bandwidth_GBs = None
    if "memoryBandwidth" in cols and r["memoryBandwidth"]:
        peak_bandwidth_GBs = round(float(r["memoryBandwidth"]) / 1e9, 1)

    return {
        "sm_count": sm_count,
        "max_threads_per_sm": max_threads_per_sm,
        "peak_bandwidth_GBs": peak_bandwidth_GBs,
    }


def _compute_launch_overhead(profile: NsysProfile) -> dict[str, tuple[float, float]]:
    """Return {kernel_name: (avg_launch_us, max_launch_us)} via correlationId join.

    Measures CPU-to-GPU enqueue latency: the time from when the CPU issued
    cudaLaunchKernel to when the kernel actually started executing on the GPU.
    Returns an empty dict if RUNTIME is absent or lacks correlationId.
    """
    if not profile.has_table("CUPTI_ACTIVITY_KIND_RUNTIME"):
        return {}
    runtime_cols = set(profile.columns("CUPTI_ACTIVITY_KIND_RUNTIME"))
    kernel_cols = set(profile.columns("CUPTI_ACTIVITY_KIND_KERNEL"))
    if "correlationId" not in runtime_cols or "correlationId" not in kernel_cols:
        return {}

    rows = profile.query("""
        SELECT
            s.value                                                  AS name,
            AVG(CAST(k.start - rt.start AS REAL)) / 1000.0         AS avg_launch_us,
            MAX(k.start - rt.start) / 1000.0                        AS max_launch_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN CUPTI_ACTIVITY_KIND_RUNTIME rt ON k.correlationId = rt.correlationId
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= rt.start
        GROUP BY k.shortName
    """)
    return {
        r["name"]: (round(float(r["avg_launch_us"]), 2), round(float(r["max_launch_us"]), 2))
        for r in rows
        if r["avg_launch_us"] is not None and float(r["avg_launch_us"]) >= 0
    }


def compute_cpu_sync_time(profile: NsysProfile, gpu_kernel_s: float) -> tuple[float | None, float | None]:
    """Return (total_sync_s, pct_of_gpu_kernel_time) for CUDA synchronization API calls.

    Sums the CPU wall time spent in *Synchronize calls (cuEventSynchronize,
    cuStreamSynchronize, cuCtxSynchronize, etc.).  The percentage is relative to
    total GPU kernel time — a high value means GPU execution is being serialized
    by CPU-side sync barriers.

    Returns (None, None) if CUPTI_ACTIVITY_KIND_RUNTIME is not present or lacks nameId.
    """
    if not profile.has_table("CUPTI_ACTIVITY_KIND_RUNTIME"):
        return None, None
    if "nameId" not in set(profile.columns("CUPTI_ACTIVITY_KIND_RUNTIME")):
        return None, None

    rows = profile.query("""
        SELECT COALESCE(SUM(rt.end - rt.start), 0) / 1e9 AS sync_s
        FROM CUPTI_ACTIVITY_KIND_RUNTIME rt
        JOIN StringIds s ON rt.nameId = s.id
        WHERE s.value LIKE '%Synchronize%'
          AND rt.end IS NOT NULL
    """)
    if not rows or rows[0]["sync_s"] is None:
        return None, None

    sync_s = round(float(rows[0]["sync_s"]), 3)
    pct = round(100.0 * sync_s / gpu_kernel_s, 1) if gpu_kernel_s > 0 else None
    return sync_s, pct


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


def _compute_all_mpi_stats(
    profile: NsysProfile,
    phases: list[PhaseWindow],
    global_limit: int = 10,
    phase_limit: int = 5,
) -> tuple[list[MpiOpSummary], list[list[MpiOpSummary]]]:
    """Compute global and per-phase MPI stats in a single scan per table.

    Replaces separate ``compute_mpi_ops`` + ``_batch_window_mpi_ops`` calls with
    one query per MPI table using conditional aggregation (CASE WHEN per phase).
    Saves ~50% of MPI scan time when computing both global and phase-level stats.

    Returns ``(global_ops, per_phase_ops)`` where ``per_phase_ops[i]`` is the list
    for ``phases[i]``.
    """
    if not profile.has_mpi():
        return [], [[] for _ in phases]

    # Build conditional-aggregation columns for each phase.
    # We need COUNT, total_ns (for avg computation), and MAX per phase.
    phase_cols = []
    for i, p in enumerate(phases):
        pred = f"e.start >= {p.start_ns} AND e.end <= {p.end_ns}"
        phase_cols.append(f"""
            COUNT(CASE WHEN {pred} THEN 1 END)                    AS p{i}_calls,
            SUM(CASE WHEN {pred} THEN e.end - e.start ELSE 0 END) AS p{i}_sum_ns,
            MAX(CASE WHEN {pred} THEN e.end - e.start ELSE 0 END) AS p{i}_max_ns
        """)
    extra_cols = ",\n".join(phase_cols)

    global_seen: dict[str, MpiOpSummary] = {}
    phase_ops: list[dict[str, MpiOpSummary]] = [{} for _ in phases]

    for table in ("MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS", "MPI_START_WAIT_EVENTS"):
        if not profile.has_table(table):
            continue
        rows = profile.query(f"""
            SELECT
                s.value                         AS op,
                COUNT(*)                        AS total_calls,
                SUM(e.end - e.start) / 1e9     AS total_s,
                AVG(e.end - e.start) / 1e6     AS avg_ms,
                MAX(e.end - e.start) / 1e6     AS max_ms,
                {extra_cols}
            FROM {table} e
            JOIN StringIds s ON e.textId = s.id
            WHERE e.end IS NOT NULL
            GROUP BY e.textId
        """)
        for r in rows:
            op_name = r["op"]
            # Global (deduplicate by highest total_s across tables)
            if op_name not in global_seen or r["total_s"] > global_seen[op_name].total_s:
                global_seen[op_name] = MpiOpSummary(
                    op=op_name,
                    calls=r["total_calls"],
                    total_s=round(r["total_s"], 3),
                    avg_ms=round(r["avg_ms"], 3),
                    max_ms=round(r["max_ms"], 3),
                )
            # Per-phase
            for i in range(len(phases)):
                p_calls = r[f"p{i}_calls"] or 0
                p_sum_ns = r[f"p{i}_sum_ns"] or 0
                p_max_ns = r[f"p{i}_max_ns"] or 0
                if p_calls == 0:
                    continue
                if op_name not in phase_ops[i] or p_sum_ns > (phase_ops[i][op_name].total_s * 1e9):
                    phase_ops[i][op_name] = MpiOpSummary(
                        op=op_name,
                        calls=p_calls,
                        total_s=round(p_sum_ns / 1e9, 3),
                        avg_ms=round(p_sum_ns / p_calls / 1e6, 3),
                        max_ms=round(p_max_ns / 1e6, 3),
                    )

    global_ops = sorted(global_seen.values(), key=lambda x: x.total_s, reverse=True)[:global_limit]
    per_phase = [
        sorted(d.values(), key=lambda x: x.total_s, reverse=True)[:phase_limit]
        for d in phase_ops
    ]
    return global_ops, per_phase


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


def _window_idle_time(
    profile: NsysProfile, start_ns: int, end_ns: int
) -> tuple[float, list[GapBucket]]:
    """Return (total_idle_s, gap_histogram) for inter-kernel gaps within a time window."""
    rows = profile.query(f"""
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
    return sum(b.total_s for b in buckets), buckets


def _window_top_kernels(
    profile: NsysProfile,
    start_ns: int,
    end_ns: int,
    total_kernel_s: float,
    limit: int = 5,
    device_info: dict | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[KernelSummary]:
    rows = profile.query(f"""
        SELECT
            s.value                                                         AS name,
            COUNT(*)                                                        AS calls,
            SUM(k.end - k.start) / 1e9                                     AS total_s,
            AVG(k.end - k.start) / 1e6                                     AS avg_ms,
            MIN(k.end - k.start) / 1e6                                     AS min_ms,
            MAX(k.end - k.start) / 1e6                                     AS max_ms,
            SUM(CAST(k.end - k.start AS REAL) * (k.end - k.start))        AS sum_sq_ns,
            SUM(k.end - k.start)                                            AS sum_ns,
            AVG(COALESCE(k.registersPerThread, 0))                         AS avg_registers,
            AVG(COALESCE(k.sharedMemoryExecuted,
                         k.staticSharedMemory + k.dynamicSharedMemory, 0)) AS avg_shared_mem,
            AVG(CAST(k.gridX * k.gridY * k.gridZ *
                     k.blockX * k.blockY * k.blockZ AS REAL))              AS avg_total_threads
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= {start_ns} AND k.end <= {end_ns}
        GROUP BY k.shortName
        ORDER BY total_s DESC
        LIMIT {limit}
    """)
    sm_count = (device_info or {}).get("sm_count")
    max_threads_per_sm = (device_info or {}).get("max_threads_per_sm")
    result = []
    for r in rows:
        n = r["calls"]
        sum_ns = r["sum_ns"] or 0
        sum_sq_ns = r["sum_sq_ns"] or 0.0
        mean_ns = sum_ns / n if n else 0.0
        variance = (sum_sq_ns / n - mean_ns ** 2) if n > 1 else 0.0
        std_dev_ms = math.sqrt(max(0.0, variance)) / 1e6
        avg_ms = r["avg_ms"] or 0.0
        cv = round(std_dev_ms / avg_ms, 3) if avg_ms > 0 else 0.0

        occupancy = None
        if sm_count and max_threads_per_sm:
            avg_threads = r["avg_total_threads"] or 0.0
            if avg_threads > 0:
                occupancy = round(min(1.0, avg_threads / (sm_count * max_threads_per_sm)), 3)

        oh = (launch_overhead or {}).get(r["name"])
        result.append(KernelSummary(
            name=r["name"],
            calls=n,
            total_s=round(r["total_s"], 4),
            avg_ms=round(avg_ms, 4),
            min_ms=round(r["min_ms"], 4),
            max_ms=round(r["max_ms"], 4),
            pct_of_gpu_time=round(100.0 * r["total_s"] / total_kernel_s, 1) if total_kernel_s else 0.0,
            std_dev_ms=round(std_dev_ms, 4),
            cv=cv,
            avg_registers_per_thread=int(round(r["avg_registers"] or 0)),
            avg_shared_mem_bytes=int(round(r["avg_shared_mem"] or 0)),
            estimated_occupancy=occupancy,
            avg_launch_overhead_us=oh[0] if oh else None,
            max_launch_overhead_us=oh[1] if oh else None,
        ))
    return result


def _batch_window_mpi_ops(
    profile: NsysProfile,
    phases: list[PhaseWindow],
    limit: int = 5,
) -> list[list[MpiOpSummary]]:
    """Fetch per-phase MPI op summaries for all phases in one query per table.

    Replaces N_phases × N_tables separate queries with N_tables queries total,
    using a CASE expression to assign each event to its phase in one pass.
    Returns a list indexed by phase position.
    """
    if not profile.has_mpi() or not phases:
        return [[] for _ in phases]

    case_clauses = " ".join(
        f"WHEN e.start >= {p.start_ns} AND e.end <= {p.end_ns} THEN {i}"
        for i, p in enumerate(phases)
    )
    case_expr = f"CASE {case_clauses} ELSE -1 END"
    overall_min = min(p.start_ns for p in phases)
    overall_max = max(p.end_ns for p in phases)

    # phase_ops[i] maps op_name -> MpiOpSummary for phase i
    phase_ops: list[dict[str, MpiOpSummary]] = [{} for _ in phases]

    for table in ("MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS", "MPI_START_WAIT_EVENTS"):
        if not profile.has_table(table):
            continue
        rows = profile.query(f"""
            SELECT
                {case_expr}                     AS phase_idx,
                s.value                         AS op,
                COUNT(*)                        AS calls,
                SUM(e.end - e.start) / 1e9     AS total_s,
                AVG(e.end - e.start) / 1e6     AS avg_ms,
                MAX(e.end - e.start) / 1e6     AS max_ms
            FROM {table} e
            JOIN StringIds s ON e.textId = s.id
            WHERE e.start >= {overall_min} AND e.end <= {overall_max}
              AND e.end IS NOT NULL
            GROUP BY phase_idx, e.textId
            HAVING phase_idx >= 0
            ORDER BY phase_idx, total_s DESC
        """)
        for r in rows:
            pidx = int(r["phase_idx"])
            op_name = r["op"]
            if op_name not in phase_ops[pidx]:
                phase_ops[pidx][op_name] = MpiOpSummary(
                    op=op_name,
                    calls=r["calls"],
                    total_s=round(r["total_s"], 3),
                    avg_ms=round(r["avg_ms"], 3),
                    max_ms=round(r["max_ms"], 3),
                )

    return [
        sorted(d.values(), key=lambda x: x.total_s, reverse=True)[:limit]
        for d in phase_ops
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
    *,
    mpi_ops: list[MpiOpSummary] | None = None,
    device_info: dict | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> PhaseSummary:
    """Compute full metrics for a single PhaseWindow.

    Pass ``mpi_ops`` to supply pre-computed MPI data (avoids an extra query when
    all phases are processed together via _batch_window_mpi_ops).
    Pass ``device_info`` (from compute_device_info) to enable occupancy metrics.
    Pass ``launch_overhead`` (from _compute_launch_overhead) to annotate kernels with
    CPU-to-GPU enqueue latency.
    """
    kernel_s = _window_kernel_time(profile, phase.start_ns, phase.end_ns)
    memcpy_s = _window_memcpy_time(profile, phase.start_ns, phase.end_ns)
    idle_s, gap_histogram = _window_idle_time(profile, phase.start_ns, phase.end_ns)
    duration_s = (phase.end_ns - phase.start_ns) / 1e9
    start_s = (phase.start_ns - profile_start_ns) / 1e9

    if mpi_ops is None:
        mpi_ops = _window_mpi_ops(profile, phase.start_ns, phase.end_ns)

    return PhaseSummary(
        name=phase.name,
        start_s=round(start_s, 3),
        end_s=round(start_s + duration_s, 3),
        duration_s=round(duration_s, 3),
        gpu_utilization_pct=round(100.0 * kernel_s / duration_s, 1) if duration_s else 0.0,
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        total_gpu_idle_s=round(idle_s, 3),
        gap_histogram=gap_histogram,
        top_kernels=_window_top_kernels(
            profile, phase.start_ns, phase.end_ns, kernel_s,
            device_info=device_info, launch_overhead=launch_overhead,
        ),
        mpi_ops=mpi_ops,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_profile_summary(
    profile: NsysProfile,
    max_phases: int = 6,
    timings: dict[str, float] | None = None,
) -> ProfileSummary:
    """Compute all metrics for a profile and return a ProfileSummary.

    Set max_phases=1 to skip phase segmentation (returns a single phase).
    Pass a dict as ``timings`` to receive a breakdown:
    ``{"phase_detection_s": ..., "metrics_s": ...}``.
    """
    t_start = time.perf_counter()

    device_info = compute_device_info(profile)
    peak_bw = device_info.get("peak_bandwidth_GBs")

    span_s = compute_profile_span(profile)
    kernel_s = compute_gpu_kernel_time(profile)
    memcpy_s = compute_gpu_memcpy_time(profile)
    sync_s = compute_gpu_sync_time(profile)
    total_idle_s, gap_histogram = compute_gap_histogram(profile)
    launch_overhead = _compute_launch_overhead(profile)
    cpu_sync_s, cpu_sync_pct = compute_cpu_sync_time(profile, kernel_s)

    t_phase = time.perf_counter()
    phases_windows = detect_phases(profile, max_phases=max_phases)
    t_phase_done = time.perf_counter()

    profile_start_ns = phases_windows[0].start_ns if phases_windows else 0
    global_mpi_ops, all_phase_mpi = _compute_all_mpi_stats(profile, phases_windows)
    phase_summaries = [
        compute_phase_summary(
            profile, pw, profile_start_ns,
            mpi_ops=all_phase_mpi[i], device_info=device_info, launch_overhead=launch_overhead,
        )
        for i, pw in enumerate(phases_windows)
    ]

    if timings is not None:
        t_end = time.perf_counter()
        timings["phase_detection_s"] = t_phase_done - t_phase
        timings["metrics_s"] = (t_end - t_start) - (t_phase_done - t_phase)

    return ProfileSummary(
        profile_path=str(profile.path),
        profile_span_s=round(span_s, 3),
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        gpu_sync_s=round(sync_s, 3),
        gpu_utilization_pct=round(100.0 * kernel_s / span_s, 1) if span_s else 0.0,
        total_gpu_idle_s=round(total_idle_s, 3),
        gap_histogram=gap_histogram,
        top_kernels=compute_top_kernels(profile, device_info=device_info, launch_overhead=launch_overhead),
        memcpy_by_kind=compute_memcpy_by_kind(profile, peak_bandwidth_GBs=peak_bw),
        streams=compute_streams(profile),
        nvtx_ranges=compute_nvtx_ranges(profile),
        mpi_ops=global_mpi_ops,
        mpi_present=profile.has_mpi(),
        phases=phase_summaries,
        peak_memory_bandwidth_GBs=peak_bw,
        cpu_sync_blocked_s=cpu_sync_s,
        cpu_sync_blocked_pct=cpu_sync_pct,
    )
