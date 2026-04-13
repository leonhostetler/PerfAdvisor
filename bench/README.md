# Synthetic CUDA Benchmark

A controllable benchmark suite that generates Nsight Systems profiles with known,
reproducible performance bottlenecks. Designed to evaluate PerfAdvisor's bottleneck
detection accuracy without leaking hints into the profile metadata — kernel names,
NVTX annotations, scenario IDs, and run filenames are all deliberately neutral.

Ground-truth JSON files (one per run, written by rank 0) record the actual scenario
name and expected bottleneck label for offline comparison against advisor output.

## How to Use

**1. Build** (from the `bench/` directory, on a Perlmutter login node):

```bash
bash build.sh
```

This produces `synthetic_cuda_benchmark_mpi`. The script loads `cudatoolkit`,
detects the Cray PE GTL library automatically, and compiles for `sm_80` (A100).
Use `--arch sm_90` for H100, `--debug` for Nsight source-level correlation.

**2. Submit** the three job scripts in order or independently:

```bash
sbatch submit_1gpu.sbatch   # runs #1–#8:       single-GPU scenarios
sbatch submit_4gpu.sbatch   # runs #11–#13:     intra-node multi-GPU scenarios
sbatch submit_8gpu.sbatch   # runs #14, #16:    inter-node MPI scenarios
```

All scripts must be submitted from the `bench/` directory — they use `pwd` to
locate the binary and write profiles to `bench/profiles/{1gpu,4gpu,8gpu}/`.

**3. Outputs** per run:

- `<id>.<rank>.nsys-rep` — Nsight Systems profile (one per rank for MPI runs)
- `<id>.<rank>.sqlite`   — SQLite export for direct SQL querying
- `<id>.json`            — ground-truth: scenario name, expected bottleneck, params

---

## Run Reference

### Single-GPU runs — `submit_1gpu.sbatch` → `profiles/1gpu/`

1 node · 1 rank · GPU 0

---

**run_01 — kernel launch overhead**

10,000 back-to-back launches of a tiny kernel (16 K elements, 2 inner ops each).
The GPU is almost never busy; the timeline is dominated by launch latency gaps.

*Resolution:* Fuse work into fewer, larger kernels. Use
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
to record and replay launch sequences with near-zero CPU overhead. For
producer–consumer pipelines, persistent kernels eliminate re-launch entirely.

---

**run_02 — CPU sync stall**

2,000 kernel launches each followed immediately by `cudaStreamSynchronize`. The
CPU blocks after every launch waiting for the GPU to drain, preventing any overlap.

*Resolution:* Move synchronization to stage boundaries rather than after each
launch. Use CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`) to chain
dependent work without stalling the CPU. Profile with Nsight to identify which
syncs are on the critical path.

---

**run_03 — PCIe transfer-bound (with pipelining)**

Large H2D → compute → D2H pipeline split across 16 chunks using multiple streams.
Arithmetic intensity is low (`--work-iters 2`) so transfers dominate. Streams
are configured to allow overlap.

*Resolution:* Use pinned (page-locked) host memory (`cudaMallocHost`) to maximize
PCIe throughput. Verify that H2D, compute, and D2H phases genuinely overlap in the
timeline — if they do not, check stream dependencies. Consider keeping data
GPU-resident across iterations to eliminate round-trip transfers.

---

**run_04 — transfer-compute overlap missing**

Identical workload to run_03 but with `--force-serialize`: all streams are
flushed after each chunk, collapsing the pipeline into a sequential H2D → compute
→ D2H sequence with no overlap.

*Resolution:* Issue `cudaMemcpyAsync` on a transfer stream while a separate
compute stream processes the previous chunk. Ensure no implicit synchronization
(e.g., `cudaDeviceSynchronize`, host-pinned memory faults) breaks the pipeline.
Compare the Nsight timeline against run_03 to see the overlap that is missing.

---

**run_05 — HBM bandwidth-bound (coalesced)**

Strided memory kernel with `--stride 1` (fully coalesced). 128 M floats, 1
work iteration — essentially a memory copy with trivial compute. SM utilization
is high but throughput is capped by HBM bandwidth (~2 TB/s on A100).

*Resolution:* Increase arithmetic intensity: perform more computation per cache
line fetched. Shared-memory tiling, register blocking, or algorithm restructuring
can reuse each loaded value multiple times before it is evicted.

---

**run_06 — uncoalesced memory access**

Same kernel as run_05 but with `--stride 32`. Each warp's 32 threads touch
addresses 32 floats apart — 32 separate cache lines per warp instead of one,
causing a ~32× increase in memory transactions.

*Resolution:* Restructure the data layout so consecutive threads access
consecutive addresses (Array-of-Structs → Struct-of-Arrays). If the access
pattern is algorithm-driven, stage data through shared memory with a coalesced
global load before the strided processing step.

---

**run_08 — low SM occupancy (moderate, ~37% on A100)**

Shared memory per block set to 48 KB via `--smem-kb 48`. On A100 (164 KB
opt-in smem per SM), this limits concurrent warps to roughly 3 blocks per SM.

*Resolution:* Reduce per-block shared memory usage: smaller tiles, fewer
buffered elements, or split the kernel into two passes. Use `__launch_bounds__`
or the occupancy calculator to understand the register/smem trade-off. Verify
with Nsight Compute's occupancy limiter analysis.

---

### Intra-node multi-GPU runs — `submit_4gpu.sbatch` → `profiles/4gpu/`

1 node · 4 ranks · 1 GPU per rank (GPUs 0–3)

---

**run_11 — unnecessary host staging, intra-node P2P**

4-rank ring `MPI_Sendrecv` with 512 MB halo buffers, explicit D2H → MPI →
H2D staging. Every transfer copies data to the host before sending and back
to the device after receiving, even though all GPUs are on the same node.

*Resolution:* Use CUDA-aware MPI (`--cuda-aware-mpi 1`) so the MPI library
transfers data directly between device buffers — GTL handles the intra-node
path. Eliminate the explicit `cudaMemcpy` D2H/H2D calls flanking each
`MPI_Sendrecv`. Compare against run_14/run_16 to quantify the inter-node
version of the same trade-off.

---

**run_12 — MPI load imbalance (barrier stall)**

4 ranks perform SFU-heavy compute where rank *k* does `work_iters × (k+1)`
iterations, plus a `--stagger-us 1000` CPU sleep proportional to rank. All
ranks then call `MPI_Barrier`. Fast ranks arrive early and idle, visible in
the Nsight timeline as long CPU waits inside MPI.

*Resolution:* Balance work across ranks — profile per-rank compute time and
equalize iteration counts. For inherently uneven workloads, use asynchronous
collectives (`MPI_Ibarrier`) so fast ranks can queue follow-on work instead of
spinning. Alternatively, restructure algorithms to avoid global barriers on the
critical path.

---

**run_13 — host-staged collective (MPI_Allreduce)**

4 ranks each do 4 iterations of compute then D2H → `MPI_Allreduce` → H2D.
Every reduction round-trips through host memory, adding two PCIe transfers per
collective call.

*Resolution:* Replace with a CUDA-aware collective that operates on device
buffers directly: CUDA-aware `MPI_Allreduce` (if GTL supports it), or
[NCCL](https://developer.nvidia.com/nccl) `ncclAllReduce` which uses NVLink
for intra-node and the network for inter-node without involving the CPU.

---

### Inter-node MPI runs — `submit_8gpu.sbatch` → `profiles/8gpu/`

2 nodes · 8 ranks · 1 GPU per rank · `--distribution=cyclic` (every ring
neighbor pair crosses the Slingshot network)

---

**run_14 — inter-node halo exchange, host-staged (baseline)**

8-rank ring `MPI_Sendrecv` with 64 MB halos, host-staged (D2H → MPI → H2D).
This is the reference baseline showing what a standard non-GPU-aware halo
exchange looks like in a profile: prominent D2H and H2D `cudaMemcpy` events
bracketing each MPI call.

*Resolution:* See run_16 for the GPU-Direct version. Adopt CUDA-aware MPI to
remove the staging copies; or, if GPU-Direct is unavailable, use double-buffering
so the next halo is staged concurrently with the current compute phase.

---

**run_16 — inter-node GPU-Direct RDMA (optimized baseline)**

8-rank ring `MPI_Sendrecv` with device pointers and CUDA-aware MPI
(`--cuda-aware-mpi 1`). GTL uses GDRCopy to read/write GPU memory directly
from the Slingshot NIC — no host copies, data stays in device memory throughout.

*Resolution:* This is the optimized target for the inter-node scenarios.
Remaining latency is network-bound. Further gains require reducing message count
(aggregation), topology-aware routing (NCCL hierarchical algorithms), or
overlapping communication with computation using asynchronous MPI operations
and CUDA streams.
