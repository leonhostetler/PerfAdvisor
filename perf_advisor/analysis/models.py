"""Pydantic data models for profile analysis output.

These models are the structured representation that the Claude agent reasons over.
All times are in seconds, all sizes in bytes unless noted.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator


class KernelSummary(BaseModel):
    name: str
    short_name: str | None = Field(
        default=None,
        description=(
            "Short display name from StringIds (e.g. 'Kernel3D');"
            " name holds the full normalized demangled name"
        ),
    )
    calls: int
    total_s: float
    avg_ms: float
    min_ms: float
    max_ms: float
    pct_of_gpu_time: float = Field(description="Fraction of total GPU kernel time (0–100)")
    std_dev_ms: float = 0.0
    cv: float = Field(
        default=0.0,
        description=(
            "Coefficient of variation (std_dev / avg);"
            " high value signals load imbalance or wavefront irregularity"
        ),
    )
    avg_registers_per_thread: int = 0
    avg_shared_mem_bytes: int = 0
    estimated_occupancy: float | None = Field(
        default=None,
        description=(
            "Estimated wave occupancy (0–1):"
            " avg launch threads / (device units × max threads per unit)"
        ),
    )
    avg_launch_overhead_us: float | None = Field(
        default=None,
        description=(
            "Avg CPU-to-GPU enqueue latency in µs:"
            " time from launch API call on CPU to kernel start on GPU"
        ),
    )
    max_launch_overhead_us: float | None = Field(
        default=None,
        description="Max CPU-to-GPU enqueue latency in µs across all launches of this kernel",
    )


class MemcpySummary(BaseModel):
    kind: str  # e.g. "Host-to-Device", "Peer-to-Peer"
    transfers: int
    total_bytes: int
    total_s: float
    effective_GBs: float
    pct_of_peak_bandwidth: float | None = Field(
        default=None,
        description="Effective bandwidth as % of device peak (requires TARGET_INFO_GPU)",
    )


class MpiOpSummary(BaseModel):
    op: str  # e.g. "MPI_Barrier", "MPI_Allreduce"
    calls: int
    total_s: float
    avg_ms: float
    max_ms: float


class MarkerRangeSummary(BaseModel):
    name: str
    calls: int
    total_s: float
    avg_ms: float


class GapBucket(BaseModel):
    label: str  # e.g. "<10us", "1-10ms"
    count: int
    total_s: float


class StreamSummary(BaseModel):
    stream_id: int
    kernel_calls: int
    total_gpu_s: float
    pct_of_gpu_time: float


class DeviceInfo(BaseModel):
    """Hardware properties extracted from the profile's device metadata table.

    All fields are optional: they will be None if the metadata table is absent
    or the profile format does not expose a given property.
    """

    vendor: str | None = Field(default=None, description="GPU vendor: 'nvidia' or 'amd'")
    name: str | None = Field(
        default=None,
        description="GPU device name (e.g., 'NVIDIA A100-SXM4-40GB' or 'AMD Instinct MI250X')",
    )
    compute_capability: str | None = Field(
        default=None, description="CUDA compute capability (NVIDIA only), e.g. '8.0'"
    )
    sm_count: int | None = Field(
        default=None,
        description="Number of SMs (NVIDIA) or compute units / CUs (AMD)",
    )
    max_threads_per_sm: int | None = Field(
        default=None,
        description="Max concurrent threads per SM or CU (maxWarpsPerSm × threadsPerWarp)",
    )
    peak_memory_bandwidth_GBs: float | None = Field(
        default=None, description="Peak HBM/DRAM bandwidth in GB/s"
    )
    total_memory_GiB: float | None = Field(default=None, description="Total GPU memory in GiB")
    l2_cache_MiB: float | None = Field(default=None, description="L2 cache size in MiB")
    max_threads_per_block: int | None = Field(default=None, description="Maximum threads per block")
    max_registers_per_block: int | None = Field(
        default=None, description="Maximum registers per block"
    )
    max_shared_mem_per_block_KiB: float | None = Field(
        default=None, description="Standard shared memory limit per block in KiB"
    )
    max_shared_mem_per_block_optin_KiB: float | None = Field(
        default=None, description="Opt-in (carveout) shared memory limit per block in KiB"
    )
    clock_rate_MHz: float | None = Field(default=None, description="GPU clock rate in MHz")


class PhaseSummary(BaseModel):
    """Metrics for a single execution phase within a profile.

    Note: ``total_gpu_idle_s`` counts only inter-kernel gaps within the phase
    window (kernel-end to next-kernel-start). Idle time before the first kernel
    or after the last kernel in the phase is not included — it contributes to
    the denominator of ``gpu_utilization_pct`` but not to ``total_gpu_idle_s``.
    """

    name: str
    start_s: float  # seconds from profile start
    end_s: float
    duration_s: float
    start_ns: int  # absolute timestamp (nanoseconds); pass to windowed tools
    end_ns: int
    gpu_utilization_pct: float
    gpu_kernel_s: float
    gpu_memcpy_s: float
    total_gpu_idle_s: float
    gap_histogram: list[GapBucket] = Field(default_factory=list)
    top_kernels: list[KernelSummary]
    mpi_ops: list[MpiOpSummary] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    """Top-level summary of a single GPU profile.

    This is the primary input to the hypothesis-generation agent.
    """

    # Source
    profile_path: str

    # Overall timing
    profile_span_s: float  # wall-clock duration captured in profile
    gpu_kernel_s: float  # total time all kernels were running on GPU
    gpu_memcpy_s: float  # total time spent in memory transfers
    gpu_sync_s: float  # total time in GPU sync operations
    gpu_utilization_pct: float  # gpu_kernel_s / profile_span_s * 100

    # GPU idle
    total_gpu_idle_s: float  # sum of all inter-kernel gaps
    gap_histogram: list[GapBucket]

    # Breakdown tables
    top_kernels: list[KernelSummary]
    memcpy_by_kind: list[MemcpySummary]
    streams: list[StreamSummary]
    marker_ranges: list[MarkerRangeSummary] = Field(default_factory=list)

    # GPU hardware info (absent in older or format-limited profiles)
    device_info: DeviceInfo = Field(
        default_factory=DeviceInfo,
        description=(
            "Hardware properties from device metadata; injected into the agent system prompt"
        ),
    )
    peak_memory_bandwidth_GBs: float | None = Field(
        default=None,
        description="Device peak memory bandwidth in GB/s",
    )

    # CPU–GPU overlap (absent if runtime API tracing was not captured)
    cpu_sync_blocked_s: float | None = Field(
        default=None,
        description="Total CPU time spent in GPU sync calls (*Synchronize) during the profile",
    )
    cpu_sync_blocked_pct: float | None = Field(
        default=None,
        description=(
            "cpu_sync_blocked_s as a fraction of total GPU kernel time (0–100);"
            " high value means GPU is being serialized by CPU sync barriers"
        ),
    )

    # MPI (absent if profile has no MPI tables)
    mpi_ops: list[MpiOpSummary] = Field(default_factory=list)
    mpi_present: bool = False
    phases: list[PhaseSummary] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Multi-rank / cross-rank models
# ---------------------------------------------------------------------------


class RankOverview(BaseModel):
    """Whole-profile stats for a single MPI rank."""

    rank_id: int
    gpu_kernel_s: float
    gpu_idle_s: float
    mpi_wait_s: float
    gpu_utilization_pct: float


class RankPhaseStats(BaseModel):
    """Per-phase stats for a single MPI rank."""

    rank_id: int
    gpu_kernel_s: float
    gpu_idle_s: float
    mpi_wait_s: float


class CollectiveImbalance(BaseModel):
    """Cross-rank imbalance metrics for a single MPI collective operation."""

    op: str
    imbalance_score: float = Field(
        description="(max - min) / mean across ranks; 0 = perfectly balanced"
    )
    slowest_rank_id: int
    mean_s: float
    min_s: float
    max_s: float


class CrossRankPhaseSummary(BaseModel):
    """Cross-rank metrics for a single execution phase."""

    phase_index: int
    phase_name: str
    # GPU kernel time stats across ranks
    gpu_kernel_mean_s: float
    gpu_kernel_std_s: float
    gpu_kernel_min_s: float
    gpu_kernel_max_s: float
    gpu_kernel_imbalance: float = Field(description="(max - min) / mean; 0 = perfectly balanced")
    gpu_kernel_slowest_rank_id: int
    # MPI wait time stats across ranks
    mpi_wait_mean_s: float
    mpi_wait_std_s: float
    mpi_wait_min_s: float
    mpi_wait_max_s: float
    mpi_wait_imbalance: float
    mpi_wait_slowest_rank_id: int
    # Per-collective breakdown
    collective_imbalance: list[CollectiveImbalance] = Field(default_factory=list)
    # Raw per-rank data (for agent reasoning about specific ranks)
    per_rank: list[RankPhaseStats] = Field(default_factory=list)


class CrossRankSummary(BaseModel):
    """Cross-rank analysis summary for a multi-rank MPI job."""

    num_ranks: int
    rank_ids: list[int]
    primary_rank_id: int
    phase_alignment: str = Field(
        description="'name_match' | 'index_order' — how phases were aligned across ranks"
    )
    per_rank_overview: list[RankOverview]
    phases: list[CrossRankPhaseSummary]


# ---------------------------------------------------------------------------
# Profile comparison models
# ---------------------------------------------------------------------------


class ScalarDiff(BaseModel):
    a: float
    b: float
    delta_pct: float | None = Field(default=None, description="(b-a)/a*100; None when a==0")


class KernelDiff(BaseModel):
    name: str
    short_name: str | None = None
    only_in_a: bool = False
    only_in_b: bool = False
    calls_a: int | None = None
    calls_b: int | None = None
    total_s_a: float | None = None
    total_s_b: float | None = None
    avg_ms_a: float | None = None
    avg_ms_b: float | None = None
    pct_gpu_time_a: float | None = None
    pct_gpu_time_b: float | None = None
    total_s_delta_pct: float | None = None


class MemcpyDiff(BaseModel):
    kind: str
    only_in_a: bool = False
    only_in_b: bool = False
    total_s_a: float | None = None
    total_s_b: float | None = None
    effective_GBs_a: float | None = None
    effective_GBs_b: float | None = None
    total_s_delta_pct: float | None = None


class MpiDiff(BaseModel):
    op: str
    only_in_a: bool = False
    only_in_b: bool = False
    calls_a: int | None = None
    calls_b: int | None = None
    total_s_a: float | None = None
    total_s_b: float | None = None
    avg_ms_a: float | None = None
    avg_ms_b: float | None = None
    total_s_delta_pct: float | None = None


class PhaseDiff(BaseModel):
    """Per-phase scalar diffs between two matched phases (phase_aware mode only)."""

    phase_name: str
    phase_index: int
    duration_s: ScalarDiff
    gpu_utilization_pct: ScalarDiff
    gpu_kernel_s: ScalarDiff
    gpu_memcpy_s: ScalarDiff
    total_gpu_idle_s: ScalarDiff


class ProfileDiff(BaseModel):
    """Structured comparison between two ProfileSummary objects."""

    profile_a_name: str
    profile_b_name: str
    comparison_mode: str = Field(description="'phase_aware' | 'summary' | 'summary_no_kernel'")
    phases_match: bool
    kernel_overlap_pct: float = Field(
        description="Jaccard similarity of kernel names (|intersection|/|union| * 100)"
    )
    # Top-level scalar diffs
    profile_span_s: ScalarDiff
    gpu_utilization_pct: ScalarDiff
    gpu_kernel_s: ScalarDiff
    gpu_memcpy_s: ScalarDiff
    gpu_sync_s: ScalarDiff
    total_gpu_idle_s: ScalarDiff
    # CPU–GPU overlap (None when cpu_sync_blocked_s is absent from both profiles)
    cpu_sync_blocked_s: ScalarDiff | None = None
    cpu_sync_blocked_pct: ScalarDiff | None = None
    # Stream topology
    stream_count_a: int = 0
    stream_count_b: int = 0
    dominant_stream_pct_a: float | None = None
    dominant_stream_pct_b: float | None = None
    # Per-entity diffs (kernel_diffs empty for summary_no_kernel mode)
    kernel_diffs: list[KernelDiff] = Field(default_factory=list)
    memcpy_diffs: list[MemcpyDiff] = Field(default_factory=list)
    mpi_diffs: list[MpiDiff] = Field(default_factory=list)
    # Per-phase diffs (populated only in phase_aware mode)
    phase_diffs: list[PhaseDiff] = Field(default_factory=list)


class ComparisonDiff(BaseModel):
    metric: str
    phase: str = Field(
        default="whole_profile",
        description=(
            "Phase this difference belongs to. Use the exact phase name from phase_diffs "
            "when the difference is scoped to a specific phase; use 'whole_profile' for "
            "top-level or cross-phase metrics."
        ),
    )
    profile_a: str
    profile_b: str
    magnitude_pct: float | None = None
    note: str


class ComparisonReport(BaseModel):
    """LLM output schema for profile comparison."""

    narrative: str
    key_differences: list[ComparisonDiff] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Hypothesis report (persisted alongside the LLM interaction log)
# ---------------------------------------------------------------------------


BottleneckType = Literal[
    "compute_bound",
    "memory_bound",
    "mpi_latency",
    "mpi_imbalance",
    "cpu_launch_overhead",
    "synchronization",
    "io",
    "other",
]

ExpectedImpact = Literal["high", "medium", "low"]

ActionCategory = Literal[
    "runtime_config",
    "launch_config",
    "code_optimization",
    "algorithm",
]

Confidence = Literal["high", "medium", "low"]

# Common near-misses the models emit instead of the canonical enum values.
# Applied after lowercasing and normalising separators to underscores.
_BOTTLENECK_ALIASES: dict[str, str] = {
    "compute": "compute_bound",
    "computebound": "compute_bound",
    "memory": "memory_bound",
    "memorybound": "memory_bound",
    "bandwidth_bound": "memory_bound",
    "mpi": "mpi_latency",
    "mpi_communication": "mpi_latency",
    "communication": "mpi_latency",
    "mpi_load_imbalance": "mpi_imbalance",
    "load_imbalance": "mpi_imbalance",
    "imbalance": "mpi_imbalance",
    "launch_overhead": "cpu_launch_overhead",
    "kernel_launch_overhead": "cpu_launch_overhead",
    "cpu_overhead": "cpu_launch_overhead",
    "sync": "synchronization",
    "synchronisation": "synchronization",
    "i_o": "io",
    "disk": "io",
}

_ACTION_ALIASES: dict[str, str] = {
    "config": "runtime_config",
    "runtime": "runtime_config",
    "env": "runtime_config",
    "environment": "runtime_config",
    "launch": "launch_config",
    "occupancy": "launch_config",
    "code": "code_optimization",
    "kernel_optimization": "code_optimization",
    "optimization": "code_optimization",
    "algorithmic": "algorithm",
}

_IMPACT_ALIASES: dict[str, str] = {
    "very_high": "high",
    "critical": "high",
    "moderate": "medium",
    "med": "medium",
    "minor": "low",
    "negligible": "low",
}


def _canonicalize(value: object, aliases: dict[str, str], valid: frozenset[str]) -> str | None:
    """Normalise a free-form LLM string to a canonical enum value.

    Returns None when the value cannot be mapped, so the caller can decide
    whether to substitute a default or leave the field unset.
    """
    if not isinstance(value, str):
        return None
    key = re.sub(r"[\s\-/]+", "_", value.strip().lower()).strip("_")
    if key in valid:
        return key
    if key in aliases:
        return aliases[key]
    return None


class Hypothesis(BaseModel):
    """A single ranked performance hypothesis produced by the agent.

    Field values arriving from the LLM are free-form text, so the validators
    below canonicalise near-miss spellings (``memory-bound`` → ``memory_bound``)
    rather than rejecting them.  Values that cannot be mapped fall back to a
    safe default and are recorded in ``coercion_notes`` so a run is never
    silently scored against an unrecognised label.
    """

    model_config = ConfigDict(extra="allow")

    bottleneck_type: BottleneckType = "other"
    phase: str = "whole_profile"
    description: str = ""
    evidence: str = ""
    suggestion: str = ""
    expected_impact: ExpectedImpact = "medium"
    action_category: ActionCategory | None = None
    confidence: Confidence = "low"
    runtime_fraction_pct: float | None = None
    estimated_speedup_pct_lower: float | None = None
    estimated_speedup_pct_upper: float | None = None
    coercion_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Fields whose raw LLM value could not be mapped to a canonical enum "
            "value and were replaced by a default. Empty on a clean parse."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_enums(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        notes: list[str] = []

        for field, aliases, valid, default in (
            (
                "bottleneck_type",
                _BOTTLENECK_ALIASES,
                frozenset(get_args(BottleneckType)),
                "other",
            ),
            ("expected_impact", _IMPACT_ALIASES, frozenset(get_args(ExpectedImpact)), "medium"),
            ("action_category", _ACTION_ALIASES, frozenset(get_args(ActionCategory)), None),
            ("confidence", _IMPACT_ALIASES, frozenset(get_args(Confidence)), "low"),
        ):
            raw = out.get(field)
            if raw is None:
                continue
            mapped = _canonicalize(raw, aliases, valid)
            if mapped is None:
                notes.append(f"{field}={raw!r} unrecognised, defaulted to {default!r}")
                out[field] = default
            else:
                out[field] = mapped

        # runtime_fraction_pct is a percentage; clamp rather than reject so a
        # slightly out-of-range model estimate does not discard the hypothesis.
        rf = out.get("runtime_fraction_pct")
        if isinstance(rf, (int, float)) and not isinstance(rf, bool):
            if not 0.0 <= float(rf) <= 100.0:
                notes.append(f"runtime_fraction_pct={rf!r} out of range, clamped to [0, 100]")
                out["runtime_fraction_pct"] = max(0.0, min(100.0, float(rf)))
        elif rf is not None:
            notes.append(f"runtime_fraction_pct={rf!r} not numeric, dropped")
            out["runtime_fraction_pct"] = None

        if notes:
            out["coercion_notes"] = list(out.get("coercion_notes") or []) + notes
        return out


class HypothesisReport(BaseModel):
    """Structured output of a single PerfAdvisor analyze run.

    Written as ``{profile_stem}_{timestamp}_hypotheses.json`` next to the
    LLM interaction log when ``--log`` is requested.  Downstream agents can
    consume this file to act on hypotheses without re-running PerfAdvisor.
    """

    profile_path: str
    generated_at: str = Field(description="ISO 8601 timestamp of when the run started")
    provider: str = Field(description="LLM provider used (anthropic, openai, gemini, claude_code)")
    model: str = Field(description="Model identifier used for hypothesis generation")
    hypotheses: list[dict] = Field(
        default_factory=list,
        description=(
            "Ranked list of validated hypothesis dicts (see the Hypothesis model). "
            "Each dict has: bottleneck_type, phase, description, evidence, "
            "suggestion, expected_impact, action_category, confidence, "
            "runtime_fraction_pct, estimated_speedup_pct_lower/upper."
        ),
    )
