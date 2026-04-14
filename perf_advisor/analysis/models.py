"""Pydantic data models for profile analysis output.

These models are the structured representation that the Claude agent reasons over.
All times are in seconds, all sizes in bytes unless noted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
            "Estimated wave occupancy (0–1): avg launch threads / (SM count × max threads per SM)"
        ),
    )
    avg_launch_overhead_us: float | None = Field(
        default=None,
        description=(
            "Avg CPU-to-GPU enqueue latency in µs:"
            " time from cudaLaunchKernel on CPU to kernel start on GPU"
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


class NvtxRangeSummary(BaseModel):
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
    """Hardware properties extracted from TARGET_INFO_GPU.

    All fields are optional: they will be None if TARGET_INFO_GPU is absent
    from the profile (e.g., older Nsight Systems exports).
    """

    name: str | None = Field(
        default=None, description="GPU device name (e.g., 'NVIDIA A100-SXM4-40GB')"
    )
    compute_capability: str | None = Field(
        default=None, description="CUDA compute capability, e.g. '8.0'"
    )
    sm_count: int | None = Field(
        default=None, description="Number of streaming multiprocessors"
    )
    max_threads_per_sm: int | None = Field(
        default=None,
        description="Max concurrent threads per SM (maxWarpsPerSm × threadsPerWarp)",
    )
    peak_memory_bandwidth_GBs: float | None = Field(
        default=None, description="Peak HBM/DRAM bandwidth in GB/s"
    )
    total_memory_GiB: float | None = Field(
        default=None, description="Total GPU memory in GiB"
    )
    l2_cache_MiB: float | None = Field(
        default=None, description="L2 cache size in MiB"
    )
    max_threads_per_block: int | None = Field(
        default=None, description="Maximum threads per block"
    )
    max_registers_per_block: int | None = Field(
        default=None, description="Maximum registers per block"
    )
    max_shared_mem_per_block_KiB: float | None = Field(
        default=None, description="Standard shared memory limit per block in KiB"
    )
    max_shared_mem_per_block_optin_KiB: float | None = Field(
        default=None, description="Opt-in (carveout) shared memory limit per block in KiB"
    )
    clock_rate_MHz: float | None = Field(
        default=None, description="GPU clock rate in MHz"
    )


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
    start_ns: int  # absolute CUPTI timestamp (nanoseconds); pass to windowed tools
    end_ns: int
    gpu_utilization_pct: float
    gpu_kernel_s: float
    gpu_memcpy_s: float
    total_gpu_idle_s: float
    gap_histogram: list[GapBucket] = Field(default_factory=list)
    top_kernels: list[KernelSummary]
    mpi_ops: list[MpiOpSummary] = Field(default_factory=list)


class ProfileSummary(BaseModel):
    """Top-level summary of a single Nsight Systems profile.

    This is the primary input to the hypothesis-generation agent.
    """

    # Source
    profile_path: str

    # Overall timing
    profile_span_s: float  # wall-clock duration captured in profile
    gpu_kernel_s: float  # total time all kernels were running on GPU
    gpu_memcpy_s: float  # total time spent in memory transfers
    gpu_sync_s: float  # total time in CUDA sync operations
    gpu_utilization_pct: float  # gpu_kernel_s / profile_span_s * 100

    # GPU idle
    total_gpu_idle_s: float  # sum of all inter-kernel gaps
    gap_histogram: list[GapBucket]

    # Breakdown tables
    top_kernels: list[KernelSummary]
    memcpy_by_kind: list[MemcpySummary]
    streams: list[StreamSummary]
    nvtx_ranges: list[NvtxRangeSummary] = Field(default_factory=list)

    # GPU hardware info (from TARGET_INFO_GPU, absent in older profiles)
    device_info: DeviceInfo = Field(
        default_factory=DeviceInfo,
        description=(
            "Hardware properties from TARGET_INFO_GPU; injected into the agent system prompt"
        ),
    )
    peak_memory_bandwidth_GBs: float | None = Field(
        default=None,
        description="Device peak memory bandwidth in GB/s (from TARGET_INFO_GPU.memoryBandwidth)",
    )

    # CPU–GPU overlap (absent if CUPTI_ACTIVITY_KIND_RUNTIME is not captured)
    cpu_sync_blocked_s: float | None = Field(
        default=None,
        description="Total CPU time spent in CUDA sync calls (*Synchronize) during the profile",
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
    gpu_kernel_imbalance: float = Field(
        description="(max - min) / mean; 0 = perfectly balanced"
    )
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
    # Per-entity diffs (kernel_diffs empty for summary_no_kernel mode)
    kernel_diffs: list[KernelDiff] = Field(default_factory=list)
    memcpy_diffs: list[MemcpyDiff] = Field(default_factory=list)
    mpi_diffs: list[MpiDiff] = Field(default_factory=list)


class ComparisonDiff(BaseModel):
    metric: str
    profile_a: str
    profile_b: str
    magnitude_pct: float | None = None
    note: str


class ComparisonReport(BaseModel):
    """LLM output schema for profile comparison."""

    narrative: str
    key_differences: list[ComparisonDiff] = Field(default_factory=list)
