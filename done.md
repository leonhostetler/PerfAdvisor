# Done — nsight_agent Improvements

## 1. Richer GPU metrics ✓ DONE (2026-04-02)

Added analytical per-kernel metrics to `KernelSummary` (propagates to both `ProfileSummary` and `PhaseSummary`).

Changes made:

- **`nsight_agent/analysis/models.py`**: Added to `KernelSummary`: `std_dev_ms`, `cv`, `avg_registers_per_thread`, `avg_shared_mem_bytes`, `estimated_occupancy`. Added `pct_of_peak_bandwidth` to `MemcpySummary`. Added `peak_memory_bandwidth_GBs` to `ProfileSummary`.

- **`nsight_agent/analysis/metrics.py`**: Added `compute_device_info()` — queries `TARGET_INFO_GPU` for `smCount`, `maxWarpsPerSm × threadsPerWarp` (max threads per SM), and `memoryBandwidth` (bytes/s → GB/s). Updated `compute_top_kernels()` and `_window_top_kernels()` to compute `std_dev_ms`/`cv` via sum-of-squares (SQLite has no STDDEV), `avg_registers_per_thread`, `avg_shared_mem_bytes` (with `sharedMemoryExecuted` → static+dynamic fallback), and `estimated_occupancy` = avg launch threads / (SM count × max threads per SM). Updated `compute_memcpy_by_kind()` to compute `pct_of_peak_bandwidth`. Updated `compute_profile_summary()` to call `compute_device_info()` and thread results through.

Sample values on test profile (A100 SXM4-40GB, 108 SMs):

- `Kernel3D`: cv=1.93 (high — load imbalance), regs=64, shmem=71 KB, occupancy=0.796
- `Kernel2D`: occupancy=1.0 (saturates SMs), cv=0.73
- Device-to-Device memcpy: 495 GB/s = 31.9% of 1555 GB/s peak

## 2. Per-phase gap histogram ✓ DONE (2026-04-02)

The current `gap_histogram` is global. A `>100ms` gap in teardown looks the same as one mid-compute.

Changes made:

- **`nsight_agent/analysis/models.py`**: Added `gap_histogram: list[GapBucket] = Field(default_factory=list)` to `PhaseSummary`.
- **`nsight_agent/analysis/metrics.py`**: Changed `_window_idle_time` signature from `→ float` to `→ tuple[float, list[GapBucket]]`. Now uses the same CASE-based bucketing as `compute_gap_histogram` (6 buckets: `<10us` through `>100ms`), scoped to the phase window. Updated `compute_phase_summary` to unpack both and pass `gap_histogram` to `PhaseSummary`.

## 3. CPU–GPU overlap metrics ✓ DONE (2026-04-02)

Add structured metrics for CPU-side behavior during GPU execution, using `CUPTI_ACTIVITY_KIND_RUNTIME`:

Changes made:

- **`nsight_agent/analysis/models.py`**: Added `avg_launch_overhead_us` and `max_launch_overhead_us` (both `float | None`) to `KernelSummary`. Added `cpu_sync_blocked_s` and `cpu_sync_blocked_pct` (`float | None`) to `ProfileSummary`.

- **`nsight_agent/analysis/metrics.py`**: Added `_compute_launch_overhead(profile)` — joins `CUPTI_ACTIVITY_KIND_KERNEL` with `CUPTI_ACTIVITY_KIND_RUNTIME` on `correlationId`, computes avg/max `(k.start − rt.start)` per kernel name. Added `compute_cpu_sync_time(profile, gpu_kernel_s)` — sums time in `*Synchronize` API calls (nameId → StringIds join) and expresses it as a % of GPU kernel time. Both degrade gracefully to `{}` / `(None, None)` if RUNTIME table is absent. Updated `compute_top_kernels`, `_window_top_kernels`, `compute_phase_summary`, and `compute_profile_summary` to thread `launch_overhead` through and populate all new fields.

Sample values on test profile (A100, CG run):

- `Kernel3D`: avg launch overhead = 73.7µs, max = 39.2ms (high max suggests occasional GPU stalls)
- `Reduction2D`: avg = 6.2µs (well-pipelined, low overhead)
- CPU sync blocked: 5.0s = **20.5% of GPU kernel time** (cuEventSynchronize dominates at 86K calls)

## 4. Runtime performance (analysis speed) ✓ DONE (2026-04-02)

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

## 5. Multi-provider LLM support (OpenAI, Gemini) ✓ DONE (2026-04-02)

Extend the agent loop to support inference backends beyond the Anthropic API.

Changes made:

- **`nsight_agent/agent/loop.py`**: Added `_parse_provider_and_model()` — resolves (provider, model_id) from optional prefix in model string, explicit `--provider`, then auto-detects from env vars (ANTHROPIC → OPENAI → GOOGLE). Added `_schemas_to_openai()` — wraps Anthropic tool schemas in `{"type":"function","function":{...}}`. Added `_preseed_messages_openai()` — injects pre-computed summaries as OpenAI-format tool call/result pairs. Added `_run_openai()` — OpenAI function-calling loop using `client.chat.completions.create`; serializes messages to dict for history. Added `_run_gemini()` — Gemini `start_chat()` loop using `google-generativeai`; pre-seeding not supported so summary is injected into the initial user message; function responses sent via `genai.protos.FunctionResponse`. Updated `run_agent()` to accept `provider: str | None`, route through `_parse_provider_and_model`, and dispatch to the correct backend.

- **`nsight_agent/__main__.py`**: Added `--provider {anthropic,openai,gemini}` flag to `p_analyze`. Updated `--model` help text to document provider-prefix syntax. Threaded `provider=args.provider` through to `run_agent()`.

- **`pyproject.toml`**: Added optional extras `openai` (`openai>=1.0`) and `gemini` (`google-generativeai>=0.5`). Install with `pip install 'nsight-agent[openai]'` or `pip install 'nsight-agent[gemini]'`.
