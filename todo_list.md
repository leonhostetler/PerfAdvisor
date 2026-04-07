# Todo List — nsight_agent Improvements

## 1. Multi-rank MPI analysis

Add a `mpianalyze` subcommand that accepts N rank files, runs `compute_profile_summary` on one "primary" rank (default: rank 0), pre-aggregates cross-rank MPI metrics into a compact structure, then sends both to the LLM together. The agent generates hypotheses as usual but now has cross-rank context to ground MPI-related ones. No new agent prompt needed — the cross-rank MPI summary is injected as an additional pre-seeded tool result.

Open design questions:

1. **Which rank is "primary"?** Rank 0 is the natural default, but the slow outlier rank might be more informative. Options: let pre-aggregation auto-identify the outlier, or add `--primary-rank`.
2. **What goes in the pre-aggregated MPI structure?** At minimum: per-rank GPU kernel time, MPI wait time, top-kernel breakdown, plus cross-rank stats (mean/std/min/max with rank IDs). Key derived metric: imbalance score per collective = `(max − min) / mean` across ranks.
3. **File selection UX:** A glob pattern (e.g., `report.0.*.sqlite`) with automatic rank-ID parsing from the filename would be more practical than listing 64+ files manually.
4. **Does `mpianalyze` need its own system prompt?** Proposed answer: no — just inject the cross-rank summary as an additional pre-seeded tool result. The data speaks for itself.

## 2. Profile comparison

Add a `compare` subcommand that takes two `.sqlite` profiles and returns a structured comparison narrative: what changed, what improved, what worsened, and what is unexplained. Not improvement hypotheses — just a clear before/after diff. Natural use cases: before/after an optimization, or `ev0` vs `ev1024` on the same rank.

## 3. Iterative hypothesis refinement

Extend the multi-turn API backend to perform a second reasoning pass:

- After the agent produces its initial hypothesis JSON, inject a follow-up user turn: "For each hypothesis, identify what additional profile data would confirm or refute it, then issue those queries."
- Increase `MAX_TURNS` accordingly and add a "confirmation" phase to the agent loop in `nsight_agent/agent/loop.py`.
- This turns the current single-pass analysis into a two-stage reasoning loop: hypothesis generation → evidence gathering → final ranked output.

## 6. Hypothesis evaluation agent

A second agent pass that runs after the primary agent and scores/filters hypotheses for grounding, applicability, and whether the optimization is already implemented. Opt-in: only runs when at least one of `--build-log`, `--run-log`, or `--source-dir` is provided (unless lightweight mode is added — see open questions).

### New files

**`nsight_agent/agent/eval_tools.py`**

Filesystem tools for the evaluator (separate from the profile query tools in `tools.py`):

- `search_codebase(pattern, directory, file_glob=None)` — shells out to `rg` (ripgrep, fallback `grep -r`); returns list of `(file_path, line_number, line_content)`, capped at ~50 results. Used first to locate relevant files by grepping for functor names extracted from demangled kernel names, then to check for specific patterns within those files (e.g., `__ldg`, `cudaMemcpyAsync`, `cudaStreamCreate`).
- `read_file_section(path, start_line, end_line)` — reads a slice of a source file. Used after `search_codebase` to verify a narrow claim, not to understand the full code.
- `read_log(path, max_lines=500)` — reads a build or run log, truncated. Strategy: first 200 lines + last 100 lines (build log has compiler invocation at the top; run log has runtime output at the end). Called once at the start of the evaluation pass.

All three must validate that paths stay within the user-provided root (`--source-dir` or the log path's parent) to prevent path traversal.

**`nsight_agent/agent/evaluator.py`**

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

**`nsight_agent/__main__.py`**

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

**`nsight_agent/analysis/models.py`** (optional)

Add evaluation fields to the hypothesis TypedDict so `--json` output is self-describing. Can defer this and keep hypotheses as plain dicts if implementing incrementally.

### Open design questions (resolve before implementing)

1. **Lightweight mode without filesystem context**: even with no build/run logs or source dir, the evaluator could do profile-grounded filtering — checking that the primary agent's cited numbers match the actual profile summary, flagging suggestions that contradict what the profile shows (e.g., suggesting shared memory when `avg_shared_mem_bytes` is already near the hardware max). Should the evaluator run in this lightweight mode by default even when no external context is provided? Pro: improves output quality at low cost. Con: extra latency and cost on every run.

2. **Should the evaluator ever add hypotheses?** Scope-limited answer is no. But it could in principle notice something while reading the build log that the primary agent missed (e.g., fast math is disabled, a relevant compiler flag is absent). If yes, needs a `submit_hypotheses` tool and increases scope significantly. Recommendation: keep it purely evaluative for now.

3. **Dependency on todo item 2**: demangled kernel names are the natural codebase navigation keys. Without them, the evaluator can only search by short names ("Kernel3D"), which is far less precise. Strongly recommend implementing item 2 before item 6.

## 7. Multi-model ensemble hypothesis generation

Run `run_agent()` against N models in parallel and merge all hypothesis lists before display.

### Motivation

Different models have different training cutoffs and reasoning patterns for GPU performance tuning. Ensemble coverage reduces blind spots any single model has. The marginal implementation cost is low if the evaluator (item 6) is already running: the merged list is passed to the evaluator just as a single-model list would be, and the evaluator handles deduplication/filtering.

### Design

**New CLI flag on `p_analyze`:**

- `--multi-model model1,model2,...` — run the primary agent with each specified model (or provider-prefixed model, e.g. `anthropic:claude-opus-4-6,openai:gpt-4o`). Models run concurrently via `ThreadPoolExecutor`. Results are merged before the evaluator step.

**Deduplication / merge step** (`nsight_agent/agent/merge.py` or inline in `__main__.py`):

Two options, in order of implementation complexity:

1. **Structural deduplication** — group hypotheses that share the same `bottleneck_type` and the same top-mentioned kernel name; keep the best-worded version (longest `description` or highest `confidence`). Simple, no extra LLM call.
2. **Synthesis pass** — a short Haiku call that receives all N lists and outputs one deduplicated, best-of-N list with a `source_models` field per hypothesis. Cleaner output but adds latency and cost.

Recommendation: start with structural deduplication (option 1). Add `source_models: list[str]` field to each hypothesis so the output table can show which model(s) agreed.

**`token_usage` aggregation:** sum input/output tokens across all model runs and display the total cost.

### Ordering and dependencies

- Implement after item 6 (evaluator) is working — the evaluator naturally absorbs the merged list without changes.
- Item 5 (`action_category`) should be implemented first so categories are consistent across models before merging.
- No dependency on item 2, but demangled names improve deduplication accuracy (structural grouping by kernel name is more precise with full demangled names).

### Open design questions

1. **Default behavior**: should `--multi-model` be the only entry point, or should there be a `--ensemble` flag that auto-selects all available providers? Auto-selection is convenient but makes cost unpredictable.
2. **Disagreement signal**: when two models disagree (one flags a kernel as memory-bound, another as compute-bound), should that disagreement be surfaced to the user rather than silently deduplicating? Could add a `consensus: "agree" | "disagree" | "unique"` field per hypothesis.
3. **Synthesis model choice**: if using option 2 (synthesis pass), Haiku is the right default (classification task, not open-ended reasoning). But the synthesis prompt needs to be careful not to hallucinate evidence that wasn't in any model's output.

## 8. Pre-flight cost estimate and confirmation

After `compute_profile_summary()` and before `run_agent()`, estimate the LLM cost and prompt the user for confirmation.

### Why input tokens are predictable

The primary agent input is almost entirely determined before the first API call:

- The system prompt is a fixed string (known at import time)
- The pre-seeded profile summary is the serialized JSON from `compute_profile_summary()`, already in hand
- The tool schemas are fixed

This means input token count can be estimated locally with high accuracy (character / 4 heuristic, ~5% error for JSON/English) or exactly via Anthropic's `client.messages.count_tokens()` API.

Output tokens are variable (depends on number of agent turns and verbosity). Use a fixed heuristic: `MAX_TURNS × ~600 tokens/turn + ~800 tokens for final hypothesis output`. Display as a range rather than a single number to communicate uncertainty.

### Implementation

In `cmd_analyze`, between `compute_profile_summary()` and `run_agent()`:

1. Serialize system prompt + profile summary JSON; compute estimated input token count.
2. Apply output-token heuristic to get an estimated range.
3. Look up per-token prices from a small hardcoded table (with a note that prices may be outdated).
4. Print: `"Estimated cost: $X.XX–$Y.YY (Xk input tokens, ~Yk output tokens est.). Proceed? [Y/n]"`
5. Skip confirmation if `--yes` or `--quiet` is passed.

**New CLI flags on `p_analyze`:**

- `--yes` — skip confirmation prompt and proceed automatically
- `--exact-token-count` — use Anthropic's `count_tokens` API for a precise input count instead of the local heuristic (adds one small API call before the main run)

**Hardcoded price table** (`nsight_agent/agent/pricing.py` or inline dict):

- Keyed by `(provider, model_id)` → `(input_price_per_1k, output_price_per_1k)`
- Include a `PRICES_DATE` constant and print it alongside the estimate so users know when it was last updated
- Fallback: if model not in table, print token counts only without a dollar estimate

**Multi-model interaction (item 7):** when `--multi-model` is used, sum estimates across all N models and show per-model and total cost before confirming. This is the most valuable use of the confirmation step — ensemble runs can be significantly more expensive.

### Open design questions

1. **Token counting method default**: local heuristic (fast, free, ~5% error) vs. `count_tokens` API call (exact, tiny cost, adds ~1s latency). Recommendation: default to local heuristic; `--exact-token-count` opts into the API call.
2. **Gemini**: no cheap local token counter available. Fall back to char/4 heuristic with a note that Gemini token counts are approximate.
3. **Price table maintenance**: prices change. Options: (a) hardcode with a date and accept drift, (b) fetch from a known URL at runtime (adds network dependency), (c) allow override via env var `NSIGHT_AGENT_PRICE_INPUT_PER_1K` / `NSIGHT_AGENT_PRICE_OUTPUT_PER_1K`. Recommendation: (a) for now, with clear `PRICES_DATE` labeling.

## 9. Nsight Systems version compatibility

Ensure nsight-agent works with profiles generated by older and newer Nsight Systems versions. The SQLite schema has changed across releases — tables, columns, and enum values have been added, renamed, or restructured.

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

## 10. Update README.md

Update `README.md` after all other todo items are complete to reflect the final feature set, CLI flags, and architecture.

Sections to update or add:

- **Feature list** — add action categories, evaluator pass, multi-model ensemble, pre-flight cost estimate, version compatibility note
- **CLI reference** — document all new flags added by items 5–9 (`--build-log`, `--run-log`, `--source-dir`, `--no-evaluate`, `--evaluate-model`, `--show-filtered`, `--multi-model`, `--yes`, `--exact-token-count`)
- **Architecture diagram / description** — reflect the two-stage pipeline (primary agent → evaluator) and optional ensemble path
- **Supported Nsight Systems versions** — document the tested version range (from item 9)
- **Cost and token usage** — brief note on the pre-flight estimate feature and how to suppress it

This item has no implementation work; it is purely documentation. Do it last.

## 11. Pre-release / public readiness

Checklist of work required before nsight-agent can be made public. None of these items depend on items 1–10 being complete; they can be done in parallel.

### Legal and licensing

- **Add `LICENSE` file** — required before any public release. Pick a license (MIT is the simplest for a tool like this).
- **Add `[project.urls]`, `description`, `authors`, and `classifiers` to `pyproject.toml`** — needed for a proper PyPI listing and expected even for a GitHub-only release.

### Repository hygiene

- **`.gitignore`** — verify it covers: `.venv/`, `__pycache__/`, `*.sqlite` (exported profile files), `*_prompt.txt` / `*_response.txt` (agent output files saved next to profiles), `.env`.
- **`LEON.md`** — currently modified and tracked by git. Decide whether this is personal notes that should be removed before the repo goes public.
- **`CHANGELOG`** — add a changelog file, even if it just has a single `0.1.0` entry.

### CI

- Add a GitHub Actions workflow (`.github/workflows/ci.yml`) that runs `ruff check . && ruff format --check .` and `pytest` on push and pull request.
- This requires the portable test fixtures below — CI cannot use the real test profile at `/home/ads.leonhost/Downloads/...`.

### Portable test fixtures

- Current tests are integration tests tied to a hardcoded absolute path. They cannot run in CI or on any other machine.
- Create a small synthetic SQLite fixture (a few hundred rows of fake `CUPTI_ACTIVITY_KIND_KERNEL`, `TARGET_INFO_GPU`, etc.) that covers the core metric functions. Store it in `tests/fixtures/`.
- Keep the real-profile tests as an optional slow suite (e.g., `pytest -m slow`) that skips automatically if the file is absent.
- Add mock-based tests for the agent loop: tool dispatch, multi-turn logic, provider routing. These should not require a real API key.

### User-facing error handling

- **Missing API key**: currently raises a raw SDK exception. Catch and emit a clean error message pointing to the relevant env var.
- **Profile not found or not a valid SQLite**: emit a clear error before attempting any queries.
- **`--version` flag**: add to the CLI (`nsight-agent --version`). Wire to the version string in `pyproject.toml` via `importlib.metadata.version("nsight-agent")`.

### Privacy and data disclosure

The README Risks section covers hallucination and SQL injection but does not explicitly state that profile data (kernel names, NVTX annotations, timing data) is sent to a third-party LLM API. For HPC users at institutions with data governance policies, this must be stated clearly and prominently — not buried in a risks subsection. Add a dedicated **Data Privacy** notice near the top of the README and in the `analyze` command's help text.

### Distribution

- Add `build` and `twine` to the `[dev]` extras in `pyproject.toml`.
- Document the release process (tag → `python -m build` → `twine upload`) somewhere, even if just in a `CONTRIBUTING.md` or a comment in `pyproject.toml`.

### Priority order

License → `.gitignore` / `LEON.md` → portable test fixtures → CI → clean error handling → `--version` → privacy disclosure → PyPI metadata → distribution docs.

## 12. Multi-profiler support (long-term)

Extend nsight-agent to ingest profiles from HPC profilers other than Nsight Systems. The analysis and agent layers already work through `ProfileSummary` and would be largely unaffected; the work is in ingestion adapters and making the agent prompt architecture-aware.

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
