"""Phase detection: partition a profile timeline into non-overlapping sequential phases.

Algorithm (hybrid):
1. Collect candidate phase boundaries from:
   - GPU activity start/end (first/last kernel timestamp)
   - GPU idle gap midpoints — only gaps > max(10 * median_gap, 1000 ns)
   - Start/end of NVTX top-level ranges that appear <= max_phases times
     (frequently-repeated ranges are iterations, not distinct phases)
   - End-of-cluster timestamps from MPI_Barrier bursts
2. Merge nearby boundary candidates (within 50ms or 0.5% of profile span):
   each candidate is tagged with a source priority; when two candidates
   are within tolerance the lower-priority one is discarded (no averaging).
3. Segment the profile at each surviving boundary
4. Fingerprint each segment: dominant kernel + GPU utilization
5a. Pre-pass: merge any adjacent pair with the same dominant kernel and
    GPU utilization difference < 15%, repeating until stable
5b. Merge the most similar adjacent pair, repeating until <= max_phases remain
6. Label each final segment using NVTX coverage, then GPU activity patterns
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from perf_advisor.ingestion.profile import NsysProfile


@dataclass
class PhaseWindow:
    """A named, non-overlapping time window representing one execution phase."""

    name: str
    start_ns: int
    end_ns: int


@dataclass
class _Seg:
    start_ns: int
    end_ns: int
    kernel_ns: int
    dominant_kernel: str | None

    @property
    def duration_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def gpu_util(self) -> float:
        return 100.0 * self.kernel_ns / self.duration_ns if self.duration_ns > 0 else 0.0


def _profile_bounds(profile: NsysProfile) -> tuple[int, int]:
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
    # Include OS runtime and MPI tables so that pre-CUDA activity (e.g.
    # MPI_Init, thread creation) is covered.  These tables may not exist in
    # all profiles, so each is guarded by has_table().
    if profile.has_table("OSRT_API"):
        sources.append(
            "SELECT start, end FROM OSRT_API WHERE start IS NOT NULL AND end IS NOT NULL"
        )
    for _mpi_tbl in ("MPI_OTHER_EVENTS", "MPI_COLLECTIVES_EVENTS", "MPI_P2P_EVENTS"):
        if profile.has_table(_mpi_tbl):
            sources.append(
                f"SELECT start, end FROM {_mpi_tbl} WHERE start IS NOT NULL AND end IS NOT NULL"
            )
    union_sql = " UNION ALL ".join(sources)
    row = profile.query(f"SELECT MIN(start) AS t0, MAX(end) AS t1 FROM ({union_sql})")[0]
    return int(row["t0"] or 0), int(row["t1"] or 0)


# A gap must exceed this multiple of the median inter-kernel gap to be treated
# as a phase boundary candidate.  Higher values = fewer, more dramatic gaps.
_GAP_OUTLIER_MULTIPLIER = 50


def _gap_boundaries(profile: NsysProfile) -> list[int]:
    """Return midpoints of GPU idle gaps that are statistical outliers.

    Only gaps larger than max(_GAP_OUTLIER_MULTIPLIER * median_gap, 1000 ns)
    are returned.  If no gap clears this threshold, returns an empty list.

    The median and outlier filter are computed entirely in SQL so that the
    query returns only the surviving rows regardless of kernel count.
    """
    rows = profile.query(f"""
        WITH ordered AS (
            SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) AS rn
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        ),
        gaps AS (
            SELECT o1.end AS gs, o2.start AS ge, o2.start - o1.end AS gap_ns
            FROM ordered o1
            JOIN ordered o2 ON o2.rn = o1.rn + 1
            WHERE o2.start > o1.end
        ),
        gap_count AS (
            SELECT COUNT(*) AS n FROM gaps
        ),
        median_gap AS (
            SELECT AVG(gap_ns) AS val
            FROM (
                SELECT gap_ns FROM gaps ORDER BY gap_ns
                LIMIT   2 - (SELECT n % 2 FROM gap_count)
                OFFSET  (SELECT (n - 1) / 2 FROM gap_count)
            )
        )
        SELECT gs, ge
        FROM gaps, median_gap
        WHERE gap_ns > MAX({_GAP_OUTLIER_MULTIPLIER} * val, 1000)
        ORDER BY gs
    """)
    return [(int(r["gs"]) + int(r["ge"])) // 2 for r in rows]


def _nvtx_phase_boundaries(
    profile: NsysProfile,
    span_ns: int,
    max_phases: int,
) -> tuple[list[int], list[tuple[int, int, str]]]:
    """Return (boundary_timestamps, all_long_nvtx_ranges_for_labeling).

    Only ranges that appear <= max_phases times contribute boundaries;
    frequently-repeated ranges are iterations, not phase transitions.
    """
    if not profile.has_nvtx():
        return [], []

    min_dur_ns = max(int(span_ns * 0.02), 100_000_000)  # >= 2% of span or >= 100ms
    rows = profile.query(f"""
        SELECT start, end, text AS name
        FROM NVTX_EVENTS
        WHERE eventType = 59
          AND end IS NOT NULL
          AND end > start
          AND text IS NOT NULL
          AND (end - start) >= {min_dur_ns}
        ORDER BY (end - start) DESC
        LIMIT 200
    """)
    if not rows:
        return [], []

    all_long: list[tuple[int, int, str]] = [
        (int(r["start"]), int(r["end"]), r["name"]) for r in rows
    ]

    # Filter to top-level: ranges not contained within any other range in this set
    top_level: list[tuple[int, int, str]] = []
    for i, (s, e, n) in enumerate(all_long):
        contained = any(
            j != i and all_long[j][0] <= s and all_long[j][1] >= e for j in range(len(all_long))
        )
        if not contained:
            top_level.append((s, e, n))

    # Only use boundaries from ranges that are rare (appear <= max_phases times)
    name_counts: Counter[str] = Counter(n for _, _, n in top_level)
    boundaries = [ts for s, e, n in top_level if name_counts[n] <= max_phases for ts in (s, e)]
    return boundaries, all_long


_MPI_BARRIER_ROW_LIMIT = 50_000


def _mpi_cluster_boundaries(profile: NsysProfile) -> list[int]:
    """Return end-of-cluster timestamps for MPI_Barrier bursts."""
    import sys

    if not profile.has_mpi() or not profile.has_table("MPI_COLLECTIVES_EVENTS"):
        return []
    rows = profile.query(f"""
        SELECT e.end
        FROM MPI_COLLECTIVES_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value = 'MPI_Barrier' AND e.end IS NOT NULL
        ORDER BY e.end
        LIMIT {_MPI_BARRIER_ROW_LIMIT + 1}
    """)
    if not rows:
        return []
    if len(rows) > _MPI_BARRIER_ROW_LIMIT:
        rows = rows[:_MPI_BARRIER_ROW_LIMIT]
        print(
            f"[perf_advisor] Warning: MPI_Barrier event count exceeds {_MPI_BARRIER_ROW_LIMIT:,};"
            " phase boundary detection truncated. Results may be approximate.",
            file=sys.stderr,
        )
    ends = [int(r["end"]) for r in rows]
    boundaries: list[int] = []
    cluster_end = ends[0]
    for t in ends[1:]:
        if t - cluster_end > 500_000_000:  # 500ms gap = new cluster
            boundaries.append(cluster_end)
        cluster_end = t
    boundaries.append(cluster_end)
    return boundaries


# Source priorities for candidate boundaries (higher = more authoritative).
# When two candidates are within the merge tolerance, the lower-priority one
# is discarded so the surviving boundary stays at an actual event timestamp.
_PRI_GAP = 1  # synthetic midpoint of an idle gap
_PRI_MPI = 2  # end of an MPI_Barrier cluster
_PRI_NVTX = 3  # NVTX range start/end
_PRI_GPU_EDGE = 4  # first/last kernel timestamp


def _merge_nearby(tagged: list[tuple[int, int]], tolerance_ns: int) -> list[int]:
    """Collapse tagged candidates within tolerance_ns of each other.

    Each element is (timestamp_ns, priority).  When two candidates are within
    tolerance, the lower-priority one is discarded; ties keep the existing
    entry.  No new synthetic timestamps are introduced.
    """
    if not tagged:
        return []
    result: list[tuple[int, int]] = [min(tagged, key=lambda x: x[0])]
    for ts, pri in sorted(tagged, key=lambda x: x[0])[1:]:
        last_ts, last_pri = result[-1]
        if ts - last_ts <= tolerance_ns:
            if pri > last_pri:
                result[-1] = (ts, pri)
            # else: keep existing (higher or equal priority)
        else:
            result.append((ts, pri))
    return [ts for ts, _ in result]


def _fingerprint(profile: NsysProfile, start_ns: int, end_ns: int) -> _Seg:
    """Lightweight fingerprint: dominant kernel + total kernel time in window.

    Total kernel time uses an overlap condition (k.start < end_ns AND k.end > start_ns)
    clamped to the window, so kernels that straddle a boundary contribute their partial
    time rather than being excluded entirely.  The dominant kernel is taken from kernels
    whose center falls inside the window, which avoids attributing a large straddling
    kernel to both adjacent segments.
    """
    # Total GPU time: all kernels overlapping the window, clamped to window edges.
    total_rows = profile.query(f"""
        SELECT SUM(MIN(k.end, {end_ns}) - MAX(k.start, {start_ns})) AS total_kernel_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        WHERE k.start < {end_ns} AND k.end > {start_ns}
    """)
    kernel_ns = int(total_rows[0]["total_kernel_ns"] or 0) if total_rows else 0

    # Dominant kernel: kernel type with most time whose center lies inside the window.
    dom_rows = profile.query(f"""
        SELECT s.value AS name, SUM(k.end - k.start) AS kns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE (k.start + k.end) / 2 >= {start_ns}
          AND (k.start + k.end) / 2 < {end_ns}
        GROUP BY k.shortName
        ORDER BY kns DESC
        LIMIT 1
    """)
    dominant_kernel = dom_rows[0]["name"] if dom_rows else None
    return _Seg(
        start_ns=start_ns, end_ns=end_ns, kernel_ns=kernel_ns, dominant_kernel=dominant_kernel
    )


def _similarity(a: _Seg, b: _Seg) -> float:
    """Score how similar two adjacent segments are; higher = merge first."""
    score = 0.0
    if a.dominant_kernel and a.dominant_kernel == b.dominant_kernel:
        score += 3.0
    util_diff = abs(a.gpu_util - b.gpu_util)
    if util_diff < 15:
        score += 2.0
    elif util_diff < 30:
        score += 1.0
    return score


def _merge_segs(a: _Seg, b: _Seg) -> _Seg:
    dk = a.dominant_kernel if a.kernel_ns >= b.kernel_ns else b.dominant_kernel
    return _Seg(
        start_ns=a.start_ns,
        end_ns=b.end_ns,
        kernel_ns=a.kernel_ns + b.kernel_ns,
        dominant_kernel=dk,
    )


def _label(seg: _Seg, nvtx_ranges: list[tuple[int, int, str]]) -> str:
    """Choose a human-readable label: NVTX coverage first, then activity pattern.

    Aggregates overlap across all ranges sharing the same name, so that a phase
    covered by many repeated short ranges (e.g., 24 solver iterations) is still
    correctly labeled by that range's name.
    """
    window = seg.end_ns - seg.start_ns
    coverage: dict[str, int] = {}
    for rs, re, rn in nvtx_ranges:
        overlap = max(0, min(re, seg.end_ns) - max(rs, seg.start_ns))
        coverage[rn] = coverage.get(rn, 0) + overlap
    if coverage:
        best_name = max(coverage, key=lambda n: coverage[n])
        if coverage[best_name] >= 0.3 * window:
            return best_name
    if seg.gpu_util < 5.0:
        return "idle"
    if seg.dominant_kernel:
        return seg.dominant_kernel
    return "unknown"


def _deduplicate_names(phases: list[PhaseWindow]) -> list[PhaseWindow]:
    counts: Counter[str] = Counter(p.name for p in phases)
    seen: Counter[str] = Counter()
    result = []
    for p in phases:
        if counts[p.name] > 1:
            seen[p.name] += 1
            result.append(PhaseWindow(f"{p.name} ({seen[p.name]})", p.start_ns, p.end_ns))
        else:
            result.append(p)
    return result


def detect_phases(
    profile: NsysProfile,
    max_phases: int = 6,
    verbose: bool = False,
    rank: int | None = None,
) -> list[PhaseWindow]:
    """Detect a flat, ordered, non-overlapping sequence of phases.

    Returns at most `max_phases` PhaseWindow objects covering the full profile span.
    Set max_phases=1 to skip segmentation and return a single phase.
    """
    tag = f"[phase rank={rank}]" if rank is not None else "[phase]"
    t0, t1 = _profile_bounds(profile)
    if verbose:
        print(f"{tag} Step 1 — profile bounds: t0={t0}ns, t1={t1}ns, span={t1 - t0}ns")
    if t0 == 0 and t1 == 0:
        return [PhaseWindow("unknown", 0, 0)]
    if max_phases <= 1:
        return [PhaseWindow("profile", t0, t1)]

    span_ns = t1 - t0
    tolerance_ns = max(5_000_000, int(span_ns * 0.005))  # 5ms or 0.5%, whichever larger

    # --- Collect candidate boundaries (tagged with source priority) ---
    tagged: list[tuple[int, int]] = []  # (timestamp_ns, priority)

    # GPU activity start/end are always phase boundaries on the true timeline
    gpu_row = profile.query("""
        SELECT MIN(start) AS gpu_start, MAX(end) AS gpu_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)[0]
    if gpu_row["gpu_start"]:
        tagged.append((int(gpu_row["gpu_start"]), _PRI_GPU_EDGE))
    if gpu_row["gpu_end"]:
        tagged.append((int(gpu_row["gpu_end"]), _PRI_GPU_EDGE))

    gap_bounds = _gap_boundaries(profile)
    tagged.extend((t, _PRI_GAP) for t in gap_bounds)
    nvtx_boundaries, all_nvtx = _nvtx_phase_boundaries(profile, span_ns, max_phases)
    tagged.extend((t, _PRI_NVTX) for t in nvtx_boundaries)
    mpi_bounds = _mpi_cluster_boundaries(profile)
    tagged.extend((t, _PRI_MPI) for t in mpi_bounds)

    if verbose:
        print(f"{tag} Step 2 — candidate boundaries ({len(tagged)} total):")
        gpu_se = [int(gpu_row["gpu_start"] or 0), int(gpu_row["gpu_end"] or 0)]
        print(f"         GPU start/end:   {gpu_se}")
        print(f"         GPU idle gaps:   {gap_bounds}")
        print(f"         NVTX boundaries: {nvtx_boundaries}")
        print(f"         MPI clusters:    {mpi_bounds}")
        print(f"         All candidates:  {sorted(tagged)}")

    # Clamp to (t0, t1) and drop candidates too close to the profile edges.
    tagged = [
        (c, p) for c, p in tagged if t0 < c < t1 and c - t0 > tolerance_ns and t1 - c > tolerance_ns
    ]
    boundaries = sorted(set([t0] + _merge_nearby(tagged, tolerance_ns) + [t1]))

    if verbose:
        print(
            f"{tag} Step 3 — after merging nearby (tolerance={tolerance_ns}ns), "
            f"{len(boundaries)} boundaries remain: {boundaries}"
        )

    # --- Build and fingerprint initial segments ---
    segs = [
        _fingerprint(profile, boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    if not segs:
        return [PhaseWindow("profile", t0, t1)]

    if verbose:
        print(f"{tag} Step 4 — {len(segs)} initial segments after fingerprinting:")
        for i, s in enumerate(segs):
            print(
                f"         [{i}] {s.start_ns}–{s.end_ns}ns  "
                f"gpu_util={s.gpu_util:.1f}%  dominant={s.dominant_kernel!r}"
            )

    # --- Pre-pass: merge adjacent segments that are clearly identical ---
    # Criteria: same dominant kernel (both non-None) AND gpu_util diff < 15%.
    # Repeat until a full pass produces no merges, since a merge can expose
    # a new qualifying pair at the same position.
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segs) - 1:
            a, b = segs[i], segs[i + 1]
            if (
                a.dominant_kernel is not None
                and a.dominant_kernel == b.dominant_kernel
                and abs(a.gpu_util - b.gpu_util) < 15.0
            ):
                if verbose:
                    print(
                        f"{tag} Step 5a — identical-phase merge: segs[{i}] + segs[{i + 1}] "
                        f"(kernel={a.dominant_kernel!r}, "
                        f"util_diff={abs(a.gpu_util - b.gpu_util):.1f}%)  "
                        f"{len(segs)} → {len(segs) - 1} segments"
                    )
                segs = segs[:i] + [_merge_segs(a, b)] + segs[i + 2 :]
                changed = True
            else:
                i += 1

    if verbose:
        print(f"{tag} Step 5a — {len(segs)} segments after identical-phase pre-pass")

    # --- Merge until <= max_phases ---
    while len(segs) > max_phases:
        scores = [_similarity(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        best_i = scores.index(max(scores))
        if verbose:
            print(
                f"{tag} Step 5b — merging segs[{best_i}] + segs[{best_i + 1}] "
                f"(similarity={scores[best_i]:.1f})  {len(segs)} → {len(segs) - 1} segments"
            )
        segs = segs[:best_i] + [_merge_segs(segs[best_i], segs[best_i + 1])] + segs[best_i + 2 :]

    if verbose:
        print(f"{tag} Step 5b — final {len(segs)} segments (at or below max_phases={max_phases})")

    # --- Label and return ---
    phases = [PhaseWindow(_label(s, all_nvtx), s.start_ns, s.end_ns) for s in segs]
    result = _deduplicate_names(phases)

    if verbose:
        print(f"{tag} Step 6 — labeled phases:")
        for p in result:
            print(f"         {p.name!r}  {p.start_ns}–{p.end_ns}ns")

    return result
