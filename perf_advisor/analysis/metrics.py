"""Compute structured metrics from a Profile.

Each function is a focused computation that returns data for one section of ProfileSummary.
All vendor-specific SQL is contained in the Profile implementations (NsysProfile, RocpdProfile).
Analysis functions are vendor-neutral: they call only Profile Protocol methods.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

from perf_advisor.ingestion.base import KernelRow, MemcpyRow, Profile

from ._utils import _normalize_demangled
from .models import (
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
from .phases import PhaseWindow, detect_phases

# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------


def compute_profile_span(profile: Profile) -> float:
    """True wall-clock duration from first to last captured event."""
    t0, t1 = profile.profile_bounds_ns()
    return (t1 - t0) / 1e9 if t1 > t0 else 0.0


def compute_gpu_kernel_time(profile: Profile) -> float:
    if not profile.capabilities.has_kernels:
        return 0.0
    return sum(e.duration_ns for e in profile.kernel_events()) / 1e9


def compute_gpu_memcpy_time(profile: Profile) -> float:
    if not profile.capabilities.has_memcpy:
        return 0.0
    return sum(e.duration_ns for e in profile.memcpy_events()) / 1e9


def compute_gpu_sync_time(profile: Profile) -> float:
    return profile.gpu_sync_time_s()


def compute_cpu_sync_time(profile: Profile, kernel_s: float) -> tuple[float, float]:
    return profile.cpu_sync_blocked_s(kernel_s)


def _compute_launch_overhead(profile: Profile) -> dict[str, tuple[float, float]]:
    return profile.launch_overhead()


def compute_top_kernels(
    profile: Profile,
    limit: int = 15,
    device_info: DeviceInfo | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[KernelSummary]:
    if not profile.capabilities.has_kernels:
        return []
    total_gpu_s = compute_gpu_kernel_time(profile)
    evts = profile.kernel_events()
    return _aggregate_kernel_summaries(evts, total_gpu_s, limit, device_info, launch_overhead)


def _aggregate_kernel_summaries(
    evts: list[KernelRow],
    total_gpu_s: float,
    limit: int,
    device_info: DeviceInfo | None,
    launch_overhead: dict[str, tuple[float, float]] | None,
) -> list[KernelSummary]:
    groups: dict[str, list[KernelRow]] = defaultdict(list)
    for e in evts:
        groups[_normalize_demangled(e.name)].append(e)

    sm_count = device_info.sm_count if device_info else None
    max_threads_per_sm = device_info.max_threads_per_sm if device_info else None

    result: list[KernelSummary] = []
    for norm_name, grp in groups.items():
        n = len(grp)
        durations = [e.duration_ns for e in grp]
        total_ns = sum(durations)
        avg_ns = total_ns / n
        sum_sq = sum(d * d for d in durations)
        variance = (sum_sq / n - avg_ns**2) if n > 1 else 0.0
        std_dev_ms = math.sqrt(max(0.0, variance)) / 1e6
        avg_ms = avg_ns / 1e6
        cv = round(std_dev_ms / avg_ms, 3) if avg_ms > 0 else 0.0

        avg_registers = sum(e.registers_per_thread or 0 for e in grp) / n
        avg_shared = sum(e.shared_mem_bytes or 0 for e in grp) / n
        threads_vals = [e.total_threads for e in grp if e.total_threads is not None]
        avg_total_threads = sum(threads_vals) / len(threads_vals) if threads_vals else 0.0

        occupancy = None
        if sm_count and max_threads_per_sm and avg_total_threads > 0:
            occupancy = round(min(1.0, avg_total_threads / (sm_count * max_threads_per_sm)), 3)

        short_name = grp[0].short_name
        short = short_name if short_name and short_name != norm_name else None
        oh = (launch_overhead or {}).get(norm_name)

        result.append(
            KernelSummary(
                name=norm_name,
                short_name=short,
                calls=n,
                total_s=round(total_ns / 1e9, 4),
                avg_ms=round(avg_ms, 4),
                min_ms=round(min(durations) / 1e6, 4),
                max_ms=round(max(durations) / 1e6, 4),
                pct_of_gpu_time=round(100.0 * total_ns / 1e9 / total_gpu_s, 1)
                if total_gpu_s
                else 0.0,
                std_dev_ms=round(std_dev_ms, 4),
                cv=cv,
                avg_registers_per_thread=int(round(avg_registers)),
                avg_shared_mem_bytes=int(round(avg_shared)),
                estimated_occupancy=occupancy,
                avg_launch_overhead_us=oh[0] if oh else None,
                max_launch_overhead_us=oh[1] if oh else None,
            )
        )

    result.sort(key=lambda k: k.total_s, reverse=True)
    return result[:limit]


def compute_memcpy_by_kind(
    profile: Profile,
    peak_bandwidth_GBs: float | None = None,
) -> list[MemcpySummary]:
    if not profile.capabilities.has_memcpy:
        return []
    evts = profile.memcpy_events()
    by_dir: dict[str, dict] = {}
    for e in evts:
        d = e.direction
        if d not in by_dir:
            by_dir[d] = {"count": 0, "total_ns": 0, "total_bytes": 0}
        by_dir[d]["count"] += 1
        by_dir[d]["total_ns"] += e.duration_ns
        by_dir[d]["total_bytes"] += e.bytes
    result = []
    for direction, s in by_dir.items():
        total_s = s["total_ns"] / 1e9
        eff_GBs = s["total_bytes"] / s["total_ns"] if s["total_ns"] > 0 else 0.0
        result.append(
            MemcpySummary(
                kind=direction,
                transfers=s["count"],
                total_bytes=s["total_bytes"],
                total_s=round(total_s, 4),
                effective_GBs=round(eff_GBs, 2),
                pct_of_peak_bandwidth=(
                    round(100.0 * eff_GBs / peak_bandwidth_GBs, 1)
                    if peak_bandwidth_GBs and eff_GBs > 0
                    else None
                ),
            )
        )
    result.sort(key=lambda m: m.total_s, reverse=True)
    return result


def compute_gap_histogram(profile: Profile) -> tuple[float, list[GapBucket]]:
    """Return (total_idle_s, gap_histogram) for inter-kernel idle time."""
    if not profile.capabilities.has_kernels:
        return 0.0, []
    evts = sorted(profile.kernel_events(), key=lambda e: e.start_ns)
    gaps = [
        evts[i].start_ns - evts[i - 1].end_ns
        for i in range(1, len(evts))
        if evts[i].start_ns > evts[i - 1].end_ns
    ]
    label_order = ["<10us", "10-100us", "100us-1ms", "1-10ms", "10-100ms", ">100ms"]
    buckets_raw: dict[str, list[int]] = {lb: [] for lb in label_order}
    for g in gaps:
        if g < 10_000:
            buckets_raw["<10us"].append(g)
        elif g < 100_000:
            buckets_raw["10-100us"].append(g)
        elif g < 1_000_000:
            buckets_raw["100us-1ms"].append(g)
        elif g < 10_000_000:
            buckets_raw["1-10ms"].append(g)
        elif g < 100_000_000:
            buckets_raw["10-100ms"].append(g)
        else:
            buckets_raw[">100ms"].append(g)
    buckets = [
        GapBucket(label=lb, count=len(vs), total_s=round(sum(vs) / 1e9, 3))
        for lb in label_order
        for vs in [buckets_raw[lb]]
        if vs
    ]
    total_idle_s = sum(b.total_s for b in buckets)
    return total_idle_s, buckets


def compute_streams(profile: Profile) -> list[StreamSummary]:
    if not profile.capabilities.has_kernels:
        return []
    total_gpu_s = compute_gpu_kernel_time(profile)
    by_stream: dict[int, dict] = {}
    for e in profile.kernel_events():
        sid = e.stream_id if e.stream_id is not None else -1
        if sid not in by_stream:
            by_stream[sid] = {"count": 0, "total_ns": 0}
        by_stream[sid]["count"] += 1
        by_stream[sid]["total_ns"] += e.duration_ns
    result = [
        StreamSummary(
            stream_id=sid,
            kernel_calls=s["count"],
            total_gpu_s=round(s["total_ns"] / 1e9, 4),
            pct_of_gpu_time=round(100.0 * s["total_ns"] / 1e9 / total_gpu_s, 1)
            if total_gpu_s
            else 0.0,
        )
        for sid, s in by_stream.items()
    ]
    result.sort(key=lambda x: x.total_gpu_s, reverse=True)
    return result


def compute_marker_ranges(profile: Profile, limit: int = 20) -> list[MarkerRangeSummary]:
    """Return aggregated marker ranges (NVTX or rocTX), sorted by total time."""
    if not profile.capabilities.has_markers:
        return []
    evts = profile.marker_ranges()
    by_name: dict[str, dict] = {}
    for e in evts:
        n = e.name
        if n not in by_name:
            by_name[n] = {"count": 0, "total_ns": 0}
        by_name[n]["count"] += 1
        by_name[n]["total_ns"] += e.duration_ns
    result = [
        MarkerRangeSummary(
            name=n,
            calls=s["count"],
            total_s=round(s["total_ns"] / 1e9, 3),
            avg_ms=round(s["total_ns"] / s["count"] / 1e6, 3),
        )
        for n, s in by_name.items()
    ]
    result.sort(key=lambda x: x.total_s, reverse=True)
    return result[:limit]


def compute_device_info(profile: Profile) -> DeviceInfo:
    return profile.device_info()


def compute_mpi_ops(profile: Profile, limit: int = 10) -> list[MpiOpSummary]:
    if not profile.capabilities.has_mpi:
        return []
    return _aggregate_mpi_ops(profile.mpi_ranges(), limit)


def _aggregate_mpi_ops(evts: list, limit: int = 10) -> list[MpiOpSummary]:
    by_op: dict[str, dict] = {}
    for e in evts:
        op = e.name
        if op not in by_op:
            by_op[op] = {"calls": 0, "total_ns": 0, "max_ns": 0}
        by_op[op]["calls"] += 1
        by_op[op]["total_ns"] += e.duration_ns
        by_op[op]["max_ns"] = max(by_op[op]["max_ns"], e.duration_ns)
    result = [
        MpiOpSummary(
            op=op,
            calls=s["calls"],
            total_s=round(s["total_ns"] / 1e9, 3),
            avg_ms=round(s["total_ns"] / s["calls"] / 1e6, 3),
            max_ms=round(s["max_ns"] / 1e6, 3),
        )
        for op, s in by_op.items()
    ]
    result.sort(key=lambda x: x.total_s, reverse=True)
    return result[:limit]


def _compute_all_mpi_stats(
    profile: Profile,
    phases: list[PhaseWindow],
    global_limit: int = 10,
    phase_limit: int = 5,
) -> tuple[list[MpiOpSummary], list[list[MpiOpSummary]]]:
    """Compute global and per-phase MPI stats from cached mpi_ranges()."""
    if not profile.capabilities.has_mpi:
        return [], [[] for _ in phases]
    all_mpi = profile.mpi_ranges()
    global_ops = _aggregate_mpi_ops(all_mpi, global_limit)
    per_phase = []
    for phase in phases:
        window = [e for e in all_mpi if e.start_ns >= phase.start_ns and e.end_ns <= phase.end_ns]
        per_phase.append(_aggregate_mpi_ops(window, phase_limit))
    return global_ops, per_phase


# ---------------------------------------------------------------------------
# Per-phase metric helpers (operate on pre-cached event lists)
# ---------------------------------------------------------------------------


def _window_kernel_time(evts: list[KernelRow], start_ns: int, end_ns: int) -> float:
    return sum(e.duration_ns for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns) / 1e9


def _window_memcpy_time(evts: list[MemcpyRow], start_ns: int, end_ns: int) -> float:
    return sum(e.duration_ns for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns) / 1e9


def _window_idle_time(
    evts: list[KernelRow], start_ns: int, end_ns: int
) -> tuple[float, list[GapBucket]]:
    """Return (total_idle_s, gap_histogram) for inter-kernel gaps within a time window."""
    window = sorted(
        (e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns),
        key=lambda e: e.start_ns,
    )
    gaps = [
        window[i].start_ns - window[i - 1].end_ns
        for i in range(1, len(window))
        if window[i].start_ns > window[i - 1].end_ns
    ]
    label_order = ["<10us", "10-100us", "100us-1ms", "1-10ms", "10-100ms", ">100ms"]
    buckets_raw: dict[str, list[int]] = {lb: [] for lb in label_order}
    for g in gaps:
        if g < 10_000:
            buckets_raw["<10us"].append(g)
        elif g < 100_000:
            buckets_raw["10-100us"].append(g)
        elif g < 1_000_000:
            buckets_raw["100us-1ms"].append(g)
        elif g < 10_000_000:
            buckets_raw["1-10ms"].append(g)
        elif g < 100_000_000:
            buckets_raw["10-100ms"].append(g)
        else:
            buckets_raw[">100ms"].append(g)
    buckets = [
        GapBucket(label=lb, count=len(vs), total_s=round(sum(vs) / 1e9, 3))
        for lb in label_order
        for vs in [buckets_raw[lb]]
        if vs
    ]
    return sum(b.total_s for b in buckets), buckets


def _window_top_kernels(
    evts: list[KernelRow],
    start_ns: int,
    end_ns: int,
    total_kernel_s: float,
    limit: int = 5,
    device_info: DeviceInfo | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> list[KernelSummary]:
    window = [e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns]
    return _aggregate_kernel_summaries(window, total_kernel_s, limit, device_info, launch_overhead)


def _window_mpi_ops(evts: list, start_ns: int, end_ns: int, limit: int = 5) -> list[MpiOpSummary]:
    window = [e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns]
    return _aggregate_mpi_ops(window, limit)


def _window_memcpy_by_kind(
    evts: list[MemcpyRow],
    start_ns: int,
    end_ns: int,
    peak_bandwidth_GBs: float | None = None,
) -> list[MemcpySummary]:
    window = [e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns]
    by_dir: dict[str, dict] = {}
    for e in window:
        d = e.direction
        if d not in by_dir:
            by_dir[d] = {"count": 0, "total_ns": 0, "total_bytes": 0}
        by_dir[d]["count"] += 1
        by_dir[d]["total_ns"] += e.duration_ns
        by_dir[d]["total_bytes"] += e.bytes
    result = []
    for direction, s in by_dir.items():
        total_s = s["total_ns"] / 1e9
        eff_GBs = s["total_bytes"] / s["total_ns"] if s["total_ns"] > 0 else 0.0
        result.append(
            MemcpySummary(
                kind=direction,
                transfers=s["count"],
                total_bytes=s["total_bytes"],
                total_s=round(total_s, 4),
                effective_GBs=round(eff_GBs, 2),
                pct_of_peak_bandwidth=(
                    round(100.0 * eff_GBs / peak_bandwidth_GBs, 1)
                    if peak_bandwidth_GBs and eff_GBs > 0
                    else None
                ),
            )
        )
    result.sort(key=lambda m: m.total_s, reverse=True)
    return result


def _window_streams(evts: list[KernelRow], start_ns: int, end_ns: int) -> list[StreamSummary]:
    window = [e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns]
    total_ns = sum(e.duration_ns for e in window)
    by_stream: dict[int, dict] = {}
    for e in window:
        sid = e.stream_id if e.stream_id is not None else -1
        if sid not in by_stream:
            by_stream[sid] = {"count": 0, "total_ns": 0}
        by_stream[sid]["count"] += 1
        by_stream[sid]["total_ns"] += e.duration_ns
    result = [
        StreamSummary(
            stream_id=sid,
            kernel_calls=s["count"],
            total_gpu_s=round(s["total_ns"] / 1e9, 4),
            pct_of_gpu_time=round(100.0 * s["total_ns"] / total_ns, 1) if total_ns else 0.0,
        )
        for sid, s in by_stream.items()
    ]
    result.sort(key=lambda x: x.total_gpu_s, reverse=True)
    return result


def _window_marker_ranges(
    evts: list, start_ns: int, end_ns: int, limit: int = 20
) -> list[MarkerRangeSummary]:
    window = [e for e in evts if e.start_ns >= start_ns and e.end_ns <= end_ns]
    by_name: dict[str, dict] = {}
    for e in window:
        n = e.name
        if n not in by_name:
            by_name[n] = {"count": 0, "total_ns": 0}
        by_name[n]["count"] += 1
        by_name[n]["total_ns"] += e.duration_ns
    result = [
        MarkerRangeSummary(
            name=n,
            calls=s["count"],
            total_s=round(s["total_ns"] / 1e9, 3),
            avg_ms=round(s["total_ns"] / s["count"] / 1e6, 3),
        )
        for n, s in by_name.items()
    ]
    result.sort(key=lambda x: x.total_s, reverse=True)
    return result[:limit]


def compute_phase_summary(
    profile: Profile,
    phase: PhaseWindow,
    profile_start_ns: int,
    *,
    mpi_ops: list[MpiOpSummary] | None = None,
    device_info: DeviceInfo | None = None,
    launch_overhead: dict[str, tuple[float, float]] | None = None,
) -> PhaseSummary:
    """Compute full metrics for a single PhaseWindow.

    Pass ``mpi_ops`` to supply pre-computed MPI data (avoids an extra scan).
    Pass ``device_info`` (from compute_device_info) to enable occupancy metrics.
    Pass ``launch_overhead`` (from profile.launch_overhead()) to annotate kernels.
    """
    all_kernel = profile.kernel_events()
    all_memcpy = profile.memcpy_events()
    all_mpi = profile.mpi_ranges()

    kernel_s = _window_kernel_time(all_kernel, phase.start_ns, phase.end_ns)
    memcpy_s = _window_memcpy_time(all_memcpy, phase.start_ns, phase.end_ns)
    idle_s, gap_histogram = _window_idle_time(all_kernel, phase.start_ns, phase.end_ns)
    duration_s = (phase.end_ns - phase.start_ns) / 1e9
    start_s = (phase.start_ns - profile_start_ns) / 1e9

    if mpi_ops is None:
        mpi_ops = _window_mpi_ops(all_mpi, phase.start_ns, phase.end_ns)

    return PhaseSummary(
        name=phase.name,
        start_s=round(start_s, 3),
        end_s=round(start_s + duration_s, 3),
        duration_s=round(duration_s, 3),
        start_ns=phase.start_ns,
        end_ns=phase.end_ns,
        gpu_utilization_pct=round(100.0 * kernel_s / duration_s, 1) if duration_s else 0.0,
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        total_gpu_idle_s=round(idle_s, 3),
        gap_histogram=gap_histogram,
        top_kernels=_window_top_kernels(
            all_kernel,
            phase.start_ns,
            phase.end_ns,
            kernel_s,
            device_info=device_info,
            launch_overhead=launch_overhead,
        ),
        mpi_ops=mpi_ops,
    )


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def compute_profile_summary(
    profile: Profile,
    max_phases: int = 6,
    timings: dict[str, float] | None = None,
    verbose: bool = False,
    rank: int | None = None,
    forced_k: int | None = None,
) -> ProfileSummary:
    """Compute all metrics for a profile and return a ProfileSummary.

    Set max_phases=1 to skip phase segmentation (returns a single phase).
    Pass a dict as ``timings`` to receive a breakdown:
    ``{"phase_detection_s": ..., "metrics_s": ...}``.
    Set forced_k to override the automatic elbow-based k selection in the DP
    segmentation step (used in multi-rank mode after cross-rank consensus).
    """
    t_start = time.perf_counter()

    device_info = compute_device_info(profile)
    peak_bw = device_info.peak_memory_bandwidth_GBs

    span_s = compute_profile_span(profile)
    kernel_s = compute_gpu_kernel_time(profile)
    memcpy_s = compute_gpu_memcpy_time(profile)
    sync_s = compute_gpu_sync_time(profile)
    total_idle_s, gap_histogram = compute_gap_histogram(profile)
    loh = profile.launch_overhead()
    cpu_sync_s, cpu_sync_pct = profile.cpu_sync_blocked_s(kernel_s)

    t_phase = time.perf_counter()
    phases_windows = detect_phases(
        profile, max_phases=max_phases, verbose=verbose, rank=rank, forced_k=forced_k
    )
    t_phase_done = time.perf_counter()

    profile_start_ns = phases_windows[0].start_ns if phases_windows else 0
    global_mpi_ops, all_phase_mpi = _compute_all_mpi_stats(profile, phases_windows)
    phase_summaries = [
        compute_phase_summary(
            profile,
            pw,
            profile_start_ns,
            mpi_ops=all_phase_mpi[i],
            device_info=device_info,
            launch_overhead=loh,
        )
        for i, pw in enumerate(phases_windows)
    ]

    if timings is not None:
        t_end = time.perf_counter()
        timings["phase_detection_s"] = t_phase_done - t_phase
        timings["metrics_s"] = (t_end - t_start) - (t_phase_done - t_phase)

    return ProfileSummary(
        profile_path=str(profile.path),
        device_info=device_info,
        profile_span_s=round(span_s, 3),
        gpu_kernel_s=round(kernel_s, 3),
        gpu_memcpy_s=round(memcpy_s, 3),
        gpu_sync_s=round(sync_s, 3),
        gpu_utilization_pct=round(100.0 * kernel_s / span_s, 1) if span_s else 0.0,
        total_gpu_idle_s=round(total_idle_s, 3),
        gap_histogram=gap_histogram,
        top_kernels=compute_top_kernels(profile, device_info=device_info, launch_overhead=loh),
        memcpy_by_kind=compute_memcpy_by_kind(profile, peak_bandwidth_GBs=peak_bw),
        streams=compute_streams(profile),
        marker_ranges=compute_marker_ranges(profile),
        mpi_ops=global_mpi_ops,
        mpi_present=profile.capabilities.has_mpi,
        phases=phase_summaries,
        peak_memory_bandwidth_GBs=peak_bw,
        cpu_sync_blocked_s=cpu_sync_s,
        cpu_sync_blocked_pct=cpu_sync_pct,
    )
