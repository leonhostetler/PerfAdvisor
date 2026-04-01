# Todo List — nsight_agent Improvements

## 1. Richer GPU metrics

Add analytical per-kernel metrics to `ProfileSummary` and `PhaseSummary`:

- **Arithmetic intensity proxy**: extract `registersPerThread` and `sharedMemoryExecuted` from `CUPTI_ACTIVITY_KIND_KERNEL` and combine with grid/block dims to flag kernels that are register-bound or occupancy-limited.
- **Estimated occupancy**: compute `(gridX * gridY * gridZ * blockX * blockY * blockZ) / (SM count * max threads per SM)` using kernel table + `TARGET_INFO_GPU`. Expose as a field in `KernelSummary`.
- **Memory bandwidth utilization %**: compare effective GB/s from `CUPTI_ACTIVITY_KIND_MEMCPY` against device peak bandwidth (from `TARGET_INFO_GPU`) to give a % of peak rather than raw numbers.
- **Kernel duration CV (coefficient of variation)**: add `std_dev` to `KernelSummary` and compute `cv = std_dev / avg`. A high CV on a frequently-called kernel signals load imbalance or wavefront irregularity. Currently only min/max/avg are exposed.

## 2. Grounding the model's output

Prevent the model from leaking training knowledge into hypotheses:

- **Grounding instruction in prompts**: add to both `_SYSTEM_PROMPT_API` and `_format_summary_prompt` in `nsight_agent/agent/loop.py`:
  
  > "Ground all hypotheses strictly in the provided numbers. Do not infer algorithm names, library internals, or solver types from prior knowledge. Describe only what the data shows."
- **Evidence validation in post-processing**: after `_extract_hypotheses`, scan each `evidence` string and flag hypotheses that cite no specific numbers from the profile data as low-confidence. This does not require an additional LLM call.

## 3. Per-phase gap histogram

The current `gap_histogram` is global. A `>100ms` gap in teardown looks the same as one mid-compute.

- Add a `gap_histogram: list[GapBucket]` field to `PhaseSummary`.
- Extend `_window_idle_time` (in `nsight_agent/analysis/metrics.py`) to return a bucketed histogram rather than just total idle seconds.
- This lets the model distinguish one large synchronization barrier from thousands of small launch gaps within the same phase.

## 4. CPU–GPU overlap metrics

Add structured metrics for CPU-side behavior during GPU execution, using `CUPTI_ACTIVITY_KIND_RUNTIME`:

- **CPU launch overhead**: time from `cudaLaunchKernel` API call on CPU to kernel `start` on GPU (enqueue latency). Expose as avg and max per kernel.
- **CPU utilization during GPU execution**: fraction of GPU-active time where the CPU thread is busy vs. blocked (e.g., in `cudaDeviceSynchronize` or `MPI_Barrier`). A low fraction means GPU execution is being serialized by CPU synchronization points.

## 5. Multi-rank comparison

For MPI profiles that include per-rank sub-profiles, add a structured diff capability:

- Add a `compare` subcommand to `nsight_agent/__main__.py` accepting two `.sqlite` paths.
- Implement a `compare_profiles` tool (in `nsight_agent/agent/tools.py`) that diffs two `ProfileSummary` objects and highlights rank imbalance — the dominant cause when `MPI_Barrier` accounts for a large fraction of runtime.
- Expose the diff result to the agent so it can reason about which ranks are slow and why.

## 6. Iterative hypothesis refinement

Extend the multi-turn API backend to perform a second reasoning pass:

- After the agent produces its initial hypothesis JSON, inject a follow-up user turn: "For each hypothesis, identify what additional profile data would confirm or refute it, then issue those queries."
- Increase `MAX_TURNS` accordingly and add a "confirmation" phase to the agent loop in `nsight_agent/agent/loop.py`.
- This turns the current single-pass analysis into a two-stage reasoning loop: hypothesis generation → evidence gathering → final ranked output.

## 7. Hypothesis persistence and diffing (verification)

Implement the "Verification" step described in CLAUDE.md:

- After a run, save hypotheses as a JSON file next to the profile (e.g., `{stem}_{timestamp}_hypotheses.json`).
- Add a `diff` subcommand to `__main__.py` that loads two hypothesis JSON files and shows which bottlenecks were resolved, which worsened, and which are new.
- This enables tracking improvement across profiling iterations without re-running the full agent.

## 8. Runtime performance (analysis speed)

Currently the analyzer takes several minutes on a 2 GB profile. Likely causes and fixes:

- **Phase detection issues many fine-grained queries**: `_fingerprint()` and the per-phase helpers each issue separate SQL queries per phase window. Batch these into a single query with conditional aggregation or a `CASE`-based partition, reducing round-trips.
- **SQLite WAL / page cache**: open the profile with `PRAGMA cache_size = -65536` (64 MB) and `PRAGMA mmap_size = 2147483648` (2 GB) in `NsysProfile.__init__`. For a 2 GB file this can eliminate repeated disk reads.
- **Index creation**: add `CREATE INDEX IF NOT EXISTS idx_kernel_start ON CUPTI_ACTIVITY_KIND_KERNEL(start)` on first open. The window queries (`WHERE start >= X AND end <= Y`) do full table scans on an unindexed 2 GB table.
- **Parallel phase computation**: `compute_phase_summary` is called sequentially per phase. Since each call is an independent set of SQL queries, run them in a `ThreadPoolExecutor` — SQLite read-only connections are safe to share across threads in WAL mode.
- **Profile the profiler**: run `python -m cProfile -s cumulative -m nsight_agent summary <profile>` to identify which queries dominate before optimizing further.
