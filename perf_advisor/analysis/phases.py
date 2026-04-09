"""Phase detection: partition a profile timeline into non-overlapping sequential phases.

Algorithm (hybrid):
1. Collect candidate phase boundaries from:
   - Top-N largest GPU idle gaps (gap midpoints)
   - Start/end of NVTX top-level ranges that appear <= max_phases times
     (repeated ranges are iterations within a phase, not distinct phases)
   - End-of-cluster timestamps from MPI_Barrier bursts
2. Merge nearby boundary candidates (within 50ms or 0.5% of profile span)
3. Segment the profile at each surviving boundary
4. Fingerprint each segment: dominant kernel + GPU utilization
5. Merge the most similar adjacent pair, repeating until <= max_phases remain
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
    union_sql = " UNION ALL ".join(sources)
    row = profile.query(f"SELECT MIN(start) AS t0, MAX(end) AS t1 FROM ({union_sql})")[0]
    return int(row["t0"] or 0), int(row["t1"] or 0)


def _gap_boundaries(profile: NsysProfile, n: int) -> list[int]:
    """Return midpoints of the N largest GPU idle gaps."""
    rows = profile.query(f"""
        WITH ordered AS (
            SELECT start, end, ROW_NUMBER() OVER (ORDER BY start) AS rn
            FROM CUPTI_ACTIVITY_KIND_KERNEL
        ),
        gaps AS (
            SELECT o1.end AS gs, o2.start AS ge
            FROM ordered o1
            JOIN ordered o2 ON o2.rn = o1.rn + 1
            WHERE o2.start > o1.end
        )
        SELECT gs, ge, ge - gs AS gap_ns
        FROM gaps
        ORDER BY gap_ns DESC
        LIMIT {n}
    """)
    return [(r["gs"] + r["ge"]) // 2 for r in rows]


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


def _merge_nearby(timestamps: list[int], tolerance_ns: int) -> list[int]:
    """Collapse timestamps that are within tolerance_ns of each other."""
    if not timestamps:
        return []
    merged = [sorted(timestamps)[0]]
    for t in sorted(timestamps)[1:]:
        if t - merged[-1] <= tolerance_ns:
            merged[-1] = (merged[-1] + t) // 2
        else:
            merged.append(t)
    return merged


def _fingerprint(profile: NsysProfile, start_ns: int, end_ns: int) -> _Seg:
    """Lightweight fingerprint: dominant kernel + total kernel time in window."""
    rows = profile.query(f"""
        SELECT s.value AS name, SUM(k.end - k.start) AS kernel_ns
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE k.start >= {start_ns} AND k.end <= {end_ns}
        GROUP BY k.shortName
        ORDER BY kernel_ns DESC
        LIMIT 1
    """)
    kernel_ns = int(rows[0]["kernel_ns"]) if rows else 0
    dominant_kernel = rows[0]["name"] if rows else None
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


def detect_phases(profile: NsysProfile, max_phases: int = 6) -> list[PhaseWindow]:
    """Detect a flat, ordered, non-overlapping sequence of phases.

    Returns at most `max_phases` PhaseWindow objects covering the full profile span.
    Set max_phases=1 to skip segmentation and return a single phase.
    """
    t0, t1 = _profile_bounds(profile)
    if t0 == 0 and t1 == 0:
        return [PhaseWindow("unknown", 0, 0)]
    if max_phases <= 1:
        return [PhaseWindow("profile", t0, t1)]

    span_ns = t1 - t0
    tolerance_ns = max(50_000_000, int(span_ns * 0.005))  # 50ms or 0.5%, whichever larger

    # --- Collect candidate boundaries ---
    candidates: list[int] = []

    # GPU activity start/end are always phase boundaries on the true timeline
    gpu_row = profile.query("""
        SELECT MIN(start) AS gpu_start, MAX(end) AS gpu_end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)[0]
    if gpu_row["gpu_start"]:
        candidates.append(int(gpu_row["gpu_start"]))
    if gpu_row["gpu_end"]:
        candidates.append(int(gpu_row["gpu_end"]))

    candidates.extend(_gap_boundaries(profile, n=min(20, max_phases * 4)))
    nvtx_boundaries, all_nvtx = _nvtx_phase_boundaries(profile, span_ns, max_phases)
    candidates.extend(nvtx_boundaries)
    candidates.extend(_mpi_cluster_boundaries(profile))

    # Clamp to (t0, t1) and merge nearby
    candidates = [c for c in candidates if t0 < c < t1]
    boundaries = sorted(set([t0] + _merge_nearby(candidates, tolerance_ns) + [t1]))

    # --- Build and fingerprint initial segments ---
    segs = [
        _fingerprint(profile, boundaries[i], boundaries[i + 1])
        for i in range(len(boundaries) - 1)
        if boundaries[i + 1] > boundaries[i]
    ]
    if not segs:
        return [PhaseWindow("profile", t0, t1)]

    # --- Merge until <= max_phases ---
    while len(segs) > max_phases:
        scores = [_similarity(segs[i], segs[i + 1]) for i in range(len(segs) - 1)]
        best_i = scores.index(max(scores))
        segs = segs[:best_i] + [_merge_segs(segs[best_i], segs[best_i + 1])] + segs[best_i + 2 :]

    # --- Label and return ---
    phases = [PhaseWindow(_label(s, all_nvtx), s.start_ns, s.end_ns) for s in segs]
    return _deduplicate_names(phases)
