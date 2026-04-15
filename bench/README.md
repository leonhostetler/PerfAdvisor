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
sbatch submit_1gpu.sbatch   # tests #1–#4:      single-GPU scenarios
sbatch submit_4gpu.sbatch   # tests #5–#7:      intra-node multi-GPU scenarios
sbatch submit_8gpu.sbatch   # test  #8:         inter-node MPI scenario
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

**test_01 — kernel launch overhead**

10,000 back-to-back launches of a tiny kernel (16 K elements, 2 inner ops each).
The GPU is almost never busy; the timeline is dominated by launch latency gaps.

*Resolution:* Fuse work into fewer, larger kernels. Use
[CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
to record and replay launch sequences with near-zero CPU overhead. For
producer–consumer pipelines, persistent kernels eliminate re-launch entirely.

---

**test_02 — CPU sync stall**

2,000 kernel launches each followed immediately by `cudaStreamSynchronize`. The
CPU blocks after every launch waiting for the GPU to drain, preventing any overlap.

*Resolution:* Move synchronization to stage boundaries rather than after each
launch. Use CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`) to chain
dependent work without stalling the CPU. Profile with Nsight to identify which
syncs are on the critical path.

---

**test_03 — PCIe transfer-bound**

Large H2D → compute → D2H pipeline split across 16 chunks using multiple streams
(`--work-iters 128`). Each chunk's compute stage (~25 µs) is visible in the
timeline but dwarfed by PCIe transfers (~250 µs each way) — the GPU is spending
almost all its time moving data across PCIe rather than computing.

*Resolution:* Increase arithmetic intensity so more computation is done per byte
transferred. Consider keeping data GPU-resident across iterations to eliminate the
round-trip H2D/D2H entirely. Compare the Nsight timeline against test_04 to see
the additional cost of losing pipeline overlap.

---

**test_04 — transfer-compute overlap missing**

Identical workload to test_03 but all streams are flushed after each chunk,
collapsing the pipeline into a strictly sequential H2D → compute → D2H sequence
with no overlap.

*Resolution:* Issue `cudaMemcpyAsync` on a transfer stream while a separate
compute stream processes the previous chunk. Ensure no implicit synchronization
(e.g., `cudaDeviceSynchronize`, host-pinned memory faults) breaks the pipeline.
Compare the Nsight timeline against test_03 to see the overlap that is missing.

---

### Intra-node multi-GPU runs — `submit_4gpu.sbatch` → `profiles/4gpu/`

1 node · 4 ranks · 1 GPU per rank (GPUs 0–3)

---

**test_05 — unnecessary host staging, intra-node P2P**

4-rank ring `MPI_Sendrecv` with 64 MB halo buffers, explicit D2H → MPI →
H2D staging. Every transfer copies data to the host before sending and back
to the device after receiving, even though all GPUs are on the same node.

*Resolution:* Use CUDA-aware MPI (`--cuda-aware-mpi 1`) so the MPI library
transfers data directly between device buffers — GTL handles the intra-node
path. Eliminate the explicit `cudaMemcpy` D2H/H2D calls flanking each
`MPI_Sendrecv`. Compare against test_08 to quantify the inter-node
version of the same trade-off.

---

**test_06 — MPI load imbalance (barrier stall)**

4 ranks perform SFU-heavy compute where rank *k* does `work_iters × (k+1)`
iterations (rank 0: 16 iters, rank 3: 64 iters). All ranks then call
`MPI_Barrier`. Fast ranks finish their GPU work early and idle at the barrier,
visible in the Nsight timeline as long CPU waits inside MPI.

*Resolution:* Balance work across ranks — profile per-rank compute time and
equalize iteration counts. For inherently uneven workloads, use asynchronous
collectives (`MPI_Ibarrier`) so fast ranks can queue follow-on work instead of
spinning. Alternatively, restructure algorithms to avoid global barriers on the
critical path.

---

**test_07 — host-staged collective (MPI_Allreduce)**

4 ranks each run one kernel (128 M-element FMA with `--work-iters 4` inner
iterations per element) then D2H → `MPI_Allreduce` → H2D.
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

**test_08 — inter-node halo exchange, host-staged (baseline)**

8-rank ring `MPI_Sendrecv` with 64 MB halos, host-staged (D2H → MPI → H2D).
This is the reference baseline showing what a standard non-GPU-aware halo
exchange looks like in a profile: prominent D2H and H2D `cudaMemcpy` events
bracketing each MPI call.

*Resolution:* See test_09 for the GPU-Direct version. Adopt CUDA-aware MPI to
remove the staging copies; or, if GPU-Direct is unavailable, use double-buffering
so the next halo is staged concurrently with the current compute phase.

---

## Evaluating PerfAdvisor

`ground_truth_meta.json` (this directory) holds the static evaluation rubric — one
entry per scenario, keyed by the `scenario` field written into the runtime
ground-truth JSONs. Each entry records:

- `nsys_signature` — observable facts about the Nsight Systems timeline
- `suggestions` — 3 expected recommendations, each with `action`, `mechanism`,
  and `rationale`

Use the `perf-advisor evaluate` subcommand to run PerfAdvisor against all benchmark
profiles and score its output automatically.

**Full evaluation** (from the repo root, after generating profiles):

```bash
python -m perf_advisor evaluate bench/profiles/ \
    --ground-truth bench/ \
    --model claude-opus-4-6 \
    --output eval_results.json
```

This runs PerfAdvisor on every profile found under `bench/profiles/{1gpu,4gpu,8gpu}/`,
scores each run's hypotheses on two axes, and writes results to `eval_results.json`.

**Rescore without re-running PerfAdvisor** (after editing the rubric or judge prompt):

```bash
python -m perf_advisor evaluate \
    --cached eval_results.json \
    --ground-truth bench/
```

Loads saved hypotheses from the previous run and re-runs only the judge calls.
Useful for iterating on `ground_truth_meta.json` without paying for hypothesis
generation again.

**Bottleneck-detection-only check** (no judge API calls, fast):

```bash
python -m perf_advisor evaluate bench/profiles/ \
    --ground-truth bench/ \
    --skip-judge --yes
```

**Per-suggestion detail** (shows judge score and explanation for every suggestion):

```bash
python -m perf_advisor evaluate bench/profiles/ \
    --ground-truth bench/ \
    --output eval_results.json \
    --verbose
```

**Use a different model** (e.g. for cross-model comparison):

```bash
python -m perf_advisor evaluate bench/profiles/ \
    --ground-truth bench/ \
    --model openai:gpt-4o \
    --judge-model openai:gpt-4o-mini \
    --output eval_gpt4o.json
```

### Scoring

| Axis                 | Method                                                               | Output        |
| -------------------- | -------------------------------------------------------------------- | ------------- |
| Bottleneck detection | Enum map + keyword fallback; deterministic                           | ✓ / ✗ per run |
| Suggestion coverage  | LLM judge (default: Haiku); scores 0 / 1 / 2 per expected suggestion | raw score + % |
| False positives      | Count hypotheses with no match to expected bottleneck                | count per run |

The judge model defaults to `claude-haiku-4-5-20251001` regardless of the
hypothesis model. Override with `--judge-model`.

## Eval Results — 2026-04-15

8 test scenarios: tiny_kernels, sync_stall, transfer_bound, overlap_missing, p2p_staged, mpi_barrier_stall,
 mpi_allreduce, mpi_halo_exchange

```
  ┌────────────────────────┬──────────┬──────────────┬───────────────┬────────┐                                        
  │         Model          │ Detected │ Avg Coverage │ Avg False Pos │ Errors │                            
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤
  │ claude-opus-4-6        │ 8/8      │ 62.5%        │ 2.75          │ 0      │                                      
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤
  │ gpt-5.4                │ 8/8      │ 54.2%        │ 2.50          │ 0      │                                        
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤                                        
  │ claude-haiku-4-5       │ 8/8      │ 50.0%        │ 3.00          │ 0      │                                        
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤                                        
  │ gpt-4o                 │ 6/8      │ 31.1%        │ 2.00          │ 0      │                            
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤                                        
  │ gemini-2.5-flash       │ 5/8      │ 18.8%        │ 1.25          │ 2      │                                      
  ├────────────────────────┼──────────┼──────────────┼───────────────┼────────┤                                        
  │ gemini-3.1-pro-preview │ 0/8      │ 0.0%         │ 0             │ 8/8    │                            
  └────────────────────────┴──────────┴──────────────┴───────────────┴────────┘                                                         
```

Key observations:

- Opus-4-6 leads on detection rate (8/8) and suggestion coverage (62.5%), though false positives are slightly higher
  than gpt-5.4.

- GPT-5.4 is close runner-up — 8/8 detection, 54.2% coverage, and the lowest false positives among the 8/8 group
  (2.50 avg).

- Haiku-4-5 matches on detection but has the most false positives (3.0 avg), and drops to 50% coverage.

- GPT-4o misses two scenarios entirely (sync_stall, overlap_missing) and coverage drops sharply on the multi-GPU MPI 
  tests (test_05, test_08 both 0%).

- Gemini-2.5-flash was partially functional — 2 API errors (tests 03, 04), and while it detected 5/8 bottlenecks,  
  coverage was weak (0% on both 4-GPU MPI tests). Also missed test_07 entirely.

- Gemini-3.1-pro-preview was fully down — all 8 runs hit 503 UNAVAILABLE errors; zero usable results.
  
  Common weakness across all models: MPI-specific suggestions (CUDA-aware MPI, GPU-Direct RDMA, NCCL, peer access) are 
  consistently missed or only partially covered. The single-GPU kernel scenarios score better than the multi-GPU MPI  
  tests across the board.
