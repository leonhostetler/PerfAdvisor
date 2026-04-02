"""Pydantic data models for profile analysis output.

These models are the structured representation that the Claude agent reasons over.
All times are in seconds, all sizes in bytes unless noted.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class KernelSummary(BaseModel):
    name: str
    calls: int
    total_s: float
    avg_ms: float
    min_ms: float
    max_ms: float
    pct_of_gpu_time: float = Field(description="Fraction of total GPU kernel time (0–100)")
    std_dev_ms: float = 0.0
    cv: float = Field(default=0.0, description="Coefficient of variation (std_dev / avg); high value signals load imbalance or wavefront irregularity")
    avg_registers_per_thread: int = 0
    avg_shared_mem_bytes: int = 0
    estimated_occupancy: float | None = Field(default=None, description="Estimated wave occupancy (0–1): avg launch threads / (SM count × max threads per SM)")
    avg_launch_overhead_us: float | None = Field(default=None, description="Avg CPU-to-GPU enqueue latency in µs: time from cudaLaunchKernel on CPU to kernel start on GPU")
    max_launch_overhead_us: float | None = Field(default=None, description="Max CPU-to-GPU enqueue latency in µs across all launches of this kernel")


class MemcpySummary(BaseModel):
    kind: str                    # e.g. "Host-to-Device", "Peer-to-Peer"
    transfers: int
    total_bytes: int
    total_s: float
    effective_GBs: float
    pct_of_peak_bandwidth: float | None = Field(default=None, description="Effective bandwidth as % of device peak (requires TARGET_INFO_GPU)")


class MpiOpSummary(BaseModel):
    op: str                      # e.g. "MPI_Barrier", "MPI_Allreduce"
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
    label: str                   # e.g. "<10us", "1-10ms"
    count: int
    total_s: float


class StreamSummary(BaseModel):
    stream_id: int
    kernel_calls: int
    total_gpu_s: float
    pct_of_gpu_time: float


class PhaseSummary(BaseModel):
    """Metrics for a single execution phase within a profile."""

    name: str
    start_s: float              # seconds from profile start
    end_s: float
    duration_s: float
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
    profile_span_s: float        # wall-clock duration captured in profile
    gpu_kernel_s: float          # total time all kernels were running on GPU
    gpu_memcpy_s: float          # total time spent in memory transfers
    gpu_sync_s: float            # total time in CUDA sync operations
    gpu_utilization_pct: float   # gpu_kernel_s / profile_span_s * 100

    # GPU idle
    total_gpu_idle_s: float      # sum of all inter-kernel gaps
    gap_histogram: list[GapBucket]

    # Breakdown tables
    top_kernels: list[KernelSummary]
    memcpy_by_kind: list[MemcpySummary]
    streams: list[StreamSummary]
    nvtx_ranges: list[NvtxRangeSummary] = Field(default_factory=list)

    # GPU hardware info (from TARGET_INFO_GPU, absent in older profiles)
    peak_memory_bandwidth_GBs: float | None = Field(default=None, description="Device peak memory bandwidth in GB/s (from TARGET_INFO_GPU.memoryBandwidth)")

    # CPU–GPU overlap (absent if CUPTI_ACTIVITY_KIND_RUNTIME is not captured)
    cpu_sync_blocked_s: float | None = Field(default=None, description="Total CPU time spent in CUDA sync calls (*Synchronize) during the profile")
    cpu_sync_blocked_pct: float | None = Field(default=None, description="cpu_sync_blocked_s as a fraction of total GPU kernel time (0–100); high value means GPU is being serialized by CPU sync barriers")

    # MPI (absent if profile has no MPI tables)
    mpi_ops: list[MpiOpSummary] = Field(default_factory=list)
    mpi_present: bool = False
    phases: list[PhaseSummary] = Field(default_factory=list)
