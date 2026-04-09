# Todo List — perf_advisor Improvements

## 1. Iterative hypothesis refinement

Extend the multi-turn API backend to perform a second reasoning pass:

- After the agent produces its initial hypothesis JSON, inject a follow-up user turn: "For each hypothesis, identify what additional profile data would confirm or refute it, then issue those queries."
- Increase `MAX_TURNS` accordingly and add a "confirmation" phase to the agent loop in `perf_advisor/agent/loop.py`.
- This turns the current single-pass analysis into a two-stage reasoning loop: hypothesis generation → evidence gathering → final ranked output.

## 2. Hypothesis evaluation agent

A second agent pass that runs after the primary agent and scores/filters hypotheses for grounding, applicability, and whether the optimization is already implemented. Opt-in: only runs when at least one of `--build-log`, `--run-log`, or `--source-dir` is provided (unless lightweight mode is added — see open questions).

### New files

**`perf_advisor/agent/eval_tools.py`**

Filesystem tools for the evaluator (separate from the profile query tools in `tools.py`):

- `search_codebase(pattern, directory, file_glob=None)` — shells out to `rg` (ripgrep, fallback `grep -r`); returns list of `(file_path, line_number, line_content)`, capped at ~50 results. Used first to locate relevant files by grepping for functor names extracted from demangled kernel names, then to check for specific patterns within those files (e.g., `__ldg`, `cudaMemcpyAsync`, `cudaStreamCreate`).
- `read_file_section(path, start_line, end_line)` — reads a slice of a source file. Used after `search_codebase` to verify a narrow claim, not to understand the full code.
- `read_log(path, max_lines=500)` — reads a build or run log, truncated. Strategy: first 200 lines + last 100 lines (build log has compiler invocation at the top; run log has runtime output at the end). Called once at the start of the evaluation pass.

All three must validate that paths stay within the user-provided root (`--source-dir` or the log path's parent) to prevent path traversal.

**`perf_advisor/agent/evaluator.py`**

```python
def run_evaluator(
    hypotheses: list[dict],
    profile_summary: ProfileSummary,
    build_log: str | None,
    run_log: str | None,
    source_dir: str | None,
    model: str | None = None,
    provider: str | None = None,
    token_usage: dict | None = None,
) -> list[dict]
```

- **System prompt**: adversarial posture — *"You are a skeptical performance engineer. For each hypothesis: (1) verify the cited profile evidence is accurate and sufficient; (2) check whether this optimization is already applied (using build/run logs and codebase search); (3) determine whether it is architecturally applicable given the hardware and software context."*
- **Pre-seeded input**: hypotheses JSON + profile summary injected as pre-seeded tool results before the loop starts (same pattern as the primary agent's profile summary injection).
- **Output tool**: `submit_evaluations` — takes the full hypothesis list with added fields per hypothesis:
  - `evaluation_status`: `"confirmed"` | `"already_implemented"` | `"not_applicable"` | `"insufficient_evidence"` | `"speculative"`
  - `evaluation_rationale`: one sentence explaining the verdict
  - `confidence_score`: float 0.0–1.0
  - `filtered`: bool — whether to suppress from default CLI output
- **Loop depth**: shallow, max ~5 turns. The evaluator makes a small number of targeted tool calls (read log, grep for functor name, check for pattern), then submits verdicts.
- **Default model**: `claude-haiku-4-5-20251001` — classification/verification, not open-ended reasoning.
- **Demangled names as navigation keys**: the evaluator extracts the functor name from the demangled kernel name (e.g., `dslash_functor` from the full `quda::Kernel3D<dslash_functor<...>>` string) and uses it as the first `search_codebase` query to locate the 1–3 relevant source files before doing any pattern checks. This depends on todo item 2 (demangled name resolution) being implemented first.

### Modified files

**`perf_advisor/__main__.py`**

New arguments on `p_analyze`:

- `--build-log PATH` — path to compiler build log
- `--run-log PATH` — path to application stdout/stderr from the profiled run
- `--source-dir PATH` — root of source tree for codebase search
- `--no-evaluate` — skip evaluation pass even if context is provided
- `--evaluate-model MODEL` — model for evaluation pass (default: haiku)
- `--show-filtered` — include filtered hypotheses in output (shown dimmed)

`cmd_analyze` pipeline becomes:

1. `compute_profile_summary()`
2. `run_agent()` → raw hypotheses
3. `run_evaluator()` (if context provided and not `--no-evaluate`) → evaluated hypotheses
4. Display: `evaluation_status` column added to table; filtered hypotheses hidden by default

**`perf_advisor/analysis/models.py`** (optional)

Add evaluation fields to the hypothesis TypedDict so `--json` output is self-describing. Can defer this and keep hypotheses as plain dicts if implementing incrementally.

### Open design questions (resolve before implementing)

1. **Lightweight mode without filesystem context**: even with no build/run logs or source dir, the evaluator could do profile-grounded filtering — checking that the primary agent's cited numbers match the actual profile summary, flagging suggestions that contradict what the profile shows (e.g., suggesting shared memory when `avg_shared_mem_bytes` is already near the hardware max). Should the evaluator run in this lightweight mode by default even when no external context is provided? Pro: improves output quality at low cost. Con: extra latency and cost on every run.

2. **Should the evaluator ever add hypotheses?** Scope-limited answer is no. But it could in principle notice something while reading the build log that the primary agent missed (e.g., fast math is disabled, a relevant compiler flag is absent). If yes, needs a `submit_hypotheses` tool and increases scope significantly. Recommendation: keep it purely evaluative for now.

3. **Dependency on todo item 2**: demangled kernel names are the natural codebase navigation keys. Without them, the evaluator can only search by short names ("Kernel3D"), which is far less precise. Strongly recommend implementing item 2 before item 6. 

## 3. Nsight Systems version compatibility

Ensure perf-advisor works with profiles generated by older and newer Nsight Systems versions. The SQLite schema has changed across releases — tables, columns, and enum values have been added, renamed, or restructured.

### Known variation points

- **`CUPTI_ACTIVITY_KIND_KERNEL`**: column set varies — older versions may lack `sharedMemoryExecuted`, `registersPerThread`, or `launchType`. The `sharedMemoryExecuted` fallback to `staticSharedMemory + dynamicSharedMemory` in `metrics.py` already handles one case; audit for others.
- **`TARGET_INFO_GPU`**: column names and units have changed across versions (e.g., `memoryBandwidth` vs. `memBandwidth`; bandwidth in bytes/s vs. GB/s). `compute_device_info()` in `metrics.py` is the primary risk area.
- **`MPI_COLLECTIVES_EVENTS` / `MPI_P2P_EVENTS` / `MPI_START_WAIT_EVENTS`**: table names and schema differ between MPI capture modes and Nsight versions. The current code already checks for table existence (`_compute_all_mpi_stats`); verify that all column references are guarded similarly.
- **`StringIds`**: the `demangledName` / `shortName` / `mangledName` columns in `StringIds` (used for kernel name resolution) may be absent or differently named in very old profiles. After todo item 2 (demangled name resolution) is implemented, this becomes a more critical fallback path.
- **`NVTX_EVENTS`**: schema differences between NVTX v2 and v3 payloads; phase detection in `metrics.py` relies on this table.

### Implementation approach

1. **Audit all column references** in `metrics.py` and `profile.py` against a matrix of known Nsight Systems versions (at minimum: 2022.x, 2023.x, 2024.x, 2025.x).
2. **Wrap column access in helpers** that gracefully degrade: if a column is absent, return `None` for optional fields rather than raising `OperationalError`. Pydantic models already use `float | None` for most derived metrics, so the data model is compatible.
3. **Emit a version warning** when the Nsight Systems version (available in `TARGET_INFO_SESSION` or similar metadata table, if present) is outside the tested range.
4. **Add version-tagged test fixtures**: small synthetic SQLite files or real profile slices representing different schema versions, used in `tests/` to guard against regressions.

### Open design questions

1. **How to detect Nsight version**: some profiles have a metadata table (e.g., `COLLECTOR_INFO` or `TARGET_INFO_SESSION`) with a version string; others do not. Fall back to schema introspection (`PRAGMA table_info(...)`) to detect which columns are present.
2. **Minimum supported version**: pick a floor (e.g., Nsight Systems 2022.1) below which we explicitly warn but do not guarantee results. This bounds the compatibility matrix.

## 4. Multi-profiler support (long-term)

Extend perf-advisor to ingest profiles from HPC profilers other than Nsight Systems. The analysis and agent layers already work through `ProfileSummary` and would be largely unaffected; the work is in ingestion adapters and making the agent prompt architecture-aware.

### Priority targets

1. **AMD rocprof** — highest value first target. AMD GPUs dominate the current TOP500 (Frontier, El Capitan, LUMI). QUDA supports HIP. rocprof output formats (CSV, JSON, and SQLite-like format in ROCm 6+) are more tractable than most alternatives. Conceptual mapping to Nsight's data model is close: kernels, dispatches, memory transfers, hardware counters.

2. **Score-P / OTF2** — covers European HPC sites and wraps CUDA, HIP, OpenMP, and MPI in a single trace. OTF2 binary format has a Python reader (`otf2` package).

3. **Intel VTune** — for Intel GPU systems (Aurora/Ponte Vecchio). Proprietary directory structure; lower priority than the above two.

### Main difficulties

- **Format diversity**: Nsight's SQLite is unusually convenient to query directly. rocprof outputs flat CSV or JSON; OTF2 is a compact binary trace format. Each needs its own ingestion layer.

- **Semantic gaps**: not every profiler exposes the same metrics. SM occupancy, shared memory usage, and launch overhead are Nsight-specific. rocprof has equivalent hardware counters but under different names and requiring explicit counter selection at collection time. Some `KernelSummary` fields (e.g., `estimated_occupancy`) would be absent or require a different derivation per profiler.

- **Agent prompt coupling**: the system prompt and bottleneck taxonomy use NVIDIA terminology (SMs, warps, HBM bandwidth). For AMD this means wavefronts, CUs, and different memory hierarchy language. The prompt would need to be adapted per hardware architecture or the agent will produce lower-quality hypotheses.

- **`sql_query` tool**: exposes raw Nsight SQL to the agent and is the tightest coupling point. For other profilers it must either be abstracted into a profiler-agnostic query interface or replaced with a per-profiler tool. This is the hardest part of the extension.

### Architectural approach

Introduce a `ProfileAdapter` abstraction: each profiler has an adapter that reads its native format and populates a normalized `ProfileSummary`. The analysis and agent layers are unchanged. The `sql_query` tool becomes optional or adapter-specific — adapters that can provide a queryable backend expose it; others do not.

A `--profiler {nsight,rocprof,scorep}` flag on the CLI selects the adapter. Auto-detection from file extension is a nice-to-have.

## 5. CI

Add a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs linting and tests on every push and pull request.

### What to add

- **`.github/workflows/ci.yml`** — runs on `push` and `pull_request` to `master`:
  1. `ruff check . && ruff format --check .`
  2. `pytest`
- Python version matrix: at minimum 3.11 and 3.12.
- No real profile or API key required — all tests use the synthetic fixture in `conftest.py`.

### Notes

- The `[dev]` extras in `pyproject.toml` (`pytest`, `pytest-cov`, `ruff`) are already the right install target for CI: `pip install -e ".[dev]"`.
- Optional providers (`openai`, `gemini`) do not need to be installed in CI unless provider-specific tests are added.

## 6. Distribution / PyPI release

Work required to publish perf-advisor to PyPI.

### What to add

- **`pyproject.toml`** — add `build` and `twine` to a new `[project.optional-dependencies]` group (e.g., `publish`) or to `dev`.
- **Release process documentation** — document the steps somewhere (a `CONTRIBUTING.md`, a comment block in `pyproject.toml`, or a `## Release` section in the README):
  1. Bump `version` in `pyproject.toml`
  2. Tag the commit: `git tag v0.x.y && git push --tags`
  3. Build: `python -m build`
  4. Upload: `twine upload dist/*`

### Notes

- `pyproject.toml` already has `description`, `authors`, `classifiers`, and `[project.urls]` — the package is PyPI-ready metadata-wise.
- Depends on the repo being public on GitHub first.
- Add `--version` flag to the CLI at this point: `perf-advisor --version` via `argparse action="version"` wired to `importlib.metadata.version("perf-advisor")`. Deferred from todo 11 because versioning hadn't been decided yet.

## 7. Threading for concurrent operations

Add threading where operations are naturally parallel to reduce wall-clock time.

### Priority targets

- **Multi-rank analysis** (`perf_advisor/agent/compare.py`) — when analyzing N rank profiles, ingestion and `compute_profile_summary()` for each rank are independent. Use `concurrent.futures.ThreadPoolExecutor` to load and summarize all ranks in parallel before the comparison agent runs.
- **Per-profile agent runs** — if a future mode runs the primary agent independently on each rank before aggregating, those agent calls can also be parallelized (subject to API rate limits).

### Implementation notes

- Use `concurrent.futures.ThreadPoolExecutor` (stdlib, no extra deps). SQLite connections are not thread-safe across threads; each worker must open its own connection rather than sharing a `NsysProfile` instance.
- Expose a `--workers N` CLI flag (default: number of rank profiles, capped at a reasonable max like 8) to let users control parallelism.
- Ensure exceptions from worker threads are propagated and surfaced cleanly (i.e., don't silently swallow errors from a failed rank load).
- Token/cost accounting (`token_usage` dict) will need a lock or per-thread accumulation with a final merge if agent calls are parallelized.
