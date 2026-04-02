# Todo List — nsight_agent Improvements

## 1. Richer GPU metrics ✓ DONE (2026-04-02)

Added analytical per-kernel metrics to `KernelSummary` (propagates to both `ProfileSummary` and `PhaseSummary`).

Changes made:

- **`nsight_agent/analysis/models.py`**: Added to `KernelSummary`: `std_dev_ms`, `cv`, `avg_registers_per_thread`, `avg_shared_mem_bytes`, `estimated_occupancy`. Added `pct_of_peak_bandwidth` to `MemcpySummary`. Added `peak_memory_bandwidth_GBs` to `ProfileSummary`.

- **`nsight_agent/analysis/metrics.py`**: Added `compute_device_info()` — queries `TARGET_INFO_GPU` for `smCount`, `maxWarpsPerSm × threadsPerWarp` (max threads per SM), and `memoryBandwidth` (bytes/s → GB/s). Updated `compute_top_kernels()` and `_window_top_kernels()` to compute `std_dev_ms`/`cv` via sum-of-squares (SQLite has no STDDEV), `avg_registers_per_thread`, `avg_shared_mem_bytes` (with `sharedMemoryExecuted` → static+dynamic fallback), and `estimated_occupancy` = avg launch threads / (SM count × max threads per SM). Updated `compute_memcpy_by_kind()` to compute `pct_of_peak_bandwidth`. Updated `compute_profile_summary()` to call `compute_device_info()` and thread results through.

Sample values on test profile (A100 SXM4-40GB, 108 SMs):

- `Kernel3D`: cv=1.93 (high — load imbalance), regs=64, shmem=71 KB, occupancy=0.796
- `Kernel2D`: occupancy=1.0 (saturates SMs), cv=0.73
- Device-to-Device memcpy: 495 GB/s = 31.9% of 1555 GB/s peak

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

## 8. Runtime performance (analysis speed) ✓ DONE (2026-04-02)

Profiled with `python -m cProfile -s cumulative -m nsight_agent summary <profile>` on the 565 MB
test profile. Baseline: **18.9s** total (80 SQL queries). After fixes: **14.2s** (70 queries, −25%).

Changes made:

- **Index creation** (`NsysProfile._ensure_indexes()`): on first open, creates
  `idx_kernel_start ON CUPTI_ACTIVITY_KIND_KERNEL(start)` and equivalent indexes on all three MPI
  tables (`MPI_COLLECTIVES_EVENTS`, `MPI_P2P_EVENTS`, `MPI_START_WAIT_EVENTS`). Done via a brief
  writable connection before the read-only connection opens; silently skipped on read-only
  filesystems. Eliminated full-table sorts for all range queries and `_fingerprint()` calls.

- **SQLite PRAGMA tuning** (`NsysProfile.__init__`): `PRAGMA cache_size = -65536` (64 MB page
  cache) and `PRAGMA mmap_size = 2147483648` (2 GB memory-mapped I/O) set on every connection.

- **Single-pass MPI stats** (`_compute_all_mpi_stats()`): replaced the separate `compute_mpi_ops`
  (global) + `_batch_window_mpi_ops` (per-phase) calls — which each did a full scan of 6.4M MPI
  rows — with one query per MPI table using CASE-based conditional aggregation. Computes global
  stats and all per-phase breakdowns simultaneously. MPI scan time: 10.75s → 7.96s.

- **Self-profiling timers**: `compute_profile_summary` accepts an optional `timings` dict and
  reports `phase_detection_s` and `metrics_s`. `cmd_analyze` times `run_agent` separately and
  prints a timing breakdown table after hypotheses (suppressed by `--quiet` and `--json`).

Remaining (not implemented): parallel `compute_phase_summary` calls via `ThreadPoolExecutor`
(~1s potential saving on 6 phases; modest benefit on the current test profile).

## 9. Multi-provider LLM support (OpenAI, Gemini)

Extend the agent loop to support inference backends beyond the Anthropic API:

- Add a `--provider` flag to `nsight_agent/__main__.py` accepting `anthropic` (default), `openai`, and `gemini`. Wire through to `run_agent()` and `_run_api()`.
- Abstract the inference call in `nsight_agent/agent/loop.py` behind a thin provider interface so each backend can translate the shared tool-use conversation format into provider-specific API calls. Key differences to handle:
  - OpenAI uses `openai.OpenAI()` with `client.chat.completions.create(tools=..., tool_choice=...)` — tool schemas are compatible with OpenAI's function-calling format but need `"type": "function"` wrappers.
  - Gemini uses `google-generativeai` with `genai.GenerativeModel(...).start_chat()` — function declarations use a different schema format; multi-turn state is managed via a chat session rather than a messages list.
- Pre-seeding (`_preseed_messages`) is Anthropic-specific; each provider backend should handle its own conversation initialization.
- Add optional dependencies to `pyproject.toml`: `openai` and `google-generativeai` (both optional extras).
- The `--model` flag should accept provider-prefixed model IDs (e.g., `openai:gpt-4o`, `gemini:gemini-2.0-flash`) to make the provider unambiguous when `--provider` is omitted.
