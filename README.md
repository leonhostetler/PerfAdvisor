# nsight-agent

An agentic performance analyzer for NVIDIA Nsight Systems profiles. It extracts structured metrics from a `.sqlite` profile, then uses an LLM to reason over the data and produce a ranked list of actionable performance hypotheses.

---

## Provenance

This tool was built with heavy AI coding assistance ("vibe coded"). The SQL queries, metric calculations, and analysis logic have not been exhaustively validated — they look plausible but may contain subtle errors in unit conversions, aggregations, or edge-case handling. Treat the numbers as starting points for investigation rather than ground truth, and verify anything surprising directly against the SQLite database or the Nsight Systems GUI.

---

## How it works

The pipeline has two distinct stages:

### Stage 1 — Pure analysis (no LLM)

`compute_profile_summary()` queries the SQLite profile directly and builds a structured `ProfileSummary`:

- **Phase detection** — segments the timeline into non-overlapping execution phases (initialization, main compute, teardown, etc.) using kernel density clustering on the GPU utilization timeline
- **Per-kernel metrics** — grouped by full demangled template name (e.g. `quda::Kernel3D<quda::dslash_functor, ...>`) rather than the short display name (`Kernel3D`), so each distinct template instantiation is tracked separately; total/avg/min/max time, coefficient of variation, register usage, shared memory, estimated SM occupancy, CPU launch overhead
- **Memory transfer summary** — by direction (H2D/D2H/D2D), with effective bandwidth vs. peak
- **MPI breakdown** — per-operation total and call count (collectives, P2P, wait)
- **GPU idle histogram** — bucketed gap distribution (`<10µs` through `>100ms`)
- **CPU–GPU overlap** — time the CPU spent blocked in `*Synchronize` calls
- **Stream utilization** — per-CUDA-stream GPU time and percentage
- **NVTX ranges** — if the application uses NVTX annotations

This stage runs entirely locally. No LLM is contacted. You can inspect the output with the `summary` subcommand.

### Stage 2 — Hypothesis generation (LLM)

The agent is given the `ProfileSummary` and a set of tools it can call to query the profile further. It works through the data systematically and produces a JSON array of ranked hypotheses, each with:

- `bottleneck_type` — one of: `compute_bound`, `memory_bound`, `mpi_latency`, `mpi_imbalance`, `cpu_launch_overhead`, `synchronization`, `io`, `other`
- `description` — what the bottleneck is
- `evidence` — specific numbers from the profile supporting the hypothesis
- `suggestion` — a concrete, actionable recommendation
- `expected_impact` — `high` / `medium` / `low`
- `action_category` — effort required to act on the suggestion:
  - `runtime_config` — env vars, MPI params, driver flags, library options (no rebuild needed)
  - `launch_config` — block/grid dimensions, shared memory, occupancy tuning (recompile only)
  - `code_optimization` — kernel rewrites, memory layout, stream pipelining, async transfers
  - `algorithm` — solver change, preconditioner, deflation, mathematical reformulation

**Tools available to the agent** (all read-only SQL queries against the local profile):

| Tool              | What it returns                                                         |
| ----------------- | ----------------------------------------------------------------------- |
| `profile_summary` | Wall-clock span, GPU kernel time, utilization, which tables are present |
| `phase_summary`   | Per-phase metrics (GPU util, top kernels, MPI, idle gaps)               |
| `top_kernels`     | Top kernels by total GPU time                                           |
| `gap_histogram`   | Idle-gap distribution                                                   |
| `memcpy_summary`  | Memory transfer breakdown by kind                                       |
| `mpi_summary`     | MPI operation breakdown                                                 |
| `nvtx_ranges`     | NVTX annotation ranges                                                  |
| `stream_summary`  | Per-stream GPU utilization                                              |
| `sql_query`       | Arbitrary read-only SQL for targeted follow-up                          |

The `profile_summary` and `phase_summary` results are pre-seeded from Stage 1 — the agent does not need to call those tools and will not spend tokens on them.

After the run, the agent saves its full prompt and response as text files next to the profile:

```
{profile_stem}_{timestamp}_prompt.txt
{profile_stem}_{timestamp}_response.txt
```

---

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

For optional LLM providers:

```bash
pip install -e ".[openai]"    # OpenAI GPT models
pip install -e ".[gemini]"    # Google Gemini models
pip install -e ".[dev]"       # Tests and linting
```

### Preparing a profile

Nsight Systems profiles must be exported to SQLite before use:

```bash
nsys export --type sqlite --output profile.sqlite profile.nsys-rep
```

---

## Usage

### Print metrics without running the LLM

```bash
nsight-agent summary profile.sqlite
nsight-agent summary profile.sqlite --json        # machine-readable JSON
nsight-agent summary profile.sqlite --max-phases 3
```

This runs Stage 1 only — fast, free, no API key required.

### Run the full agent

```bash
nsight-agent analyze profile.sqlite
```

The provider is auto-detected from your environment (see Provider Selection below).

**Useful flags:**

```bash
# Print the full ProfileSummary JSON sent to the model (for debugging)
nsight-agent analyze profile.sqlite --verbose

# Suppress per-turn agent logging and timing table
nsight-agent analyze profile.sqlite --quiet

# Output raw hypothesis JSON (suitable for piping or scripting)
nsight-agent analyze profile.sqlite --json

# Allow the model to draw on application-specific knowledge from training data.
# By default, suggestions are grounded strictly in the profile data, which reduces
# hallucinations (e.g. suggested environment variables that do not exist). With
# --allow-app-knowledge the model may produce more specific, targeted suggestions
# by drawing on its training knowledge of the application — but it may also
# confidently suggest configuration options, environment variables, or tuning
# parameters that are incorrect or do not exist.
nsight-agent analyze profile.sqlite --allow-app-knowledge

# Limit phase detection (fewer phases = less context = fewer tokens)
nsight-agent analyze profile.sqlite --max-phases 3
nsight-agent analyze profile.sqlite --max-phases 1   # disable phase segmentation entirely

# Override the agent turn limit (default: 20).
# Lower values reduce cost and risk of runaway tool calls; higher values give
# the model more room on complex profiles. A wrap-up warning is injected 3 turns
# before the limit; if the limit is still hit, one extra no-tool turn is made to
# extract whatever the model has gathered rather than discarding it.
nsight-agent analyze profile.sqlite --max-turns 10   # tighter limit for cheaper models
nsight-agent analyze profile.sqlite --max-turns 30   # more room for complex profiles

# Skip the pre-flight confirmation prompt (useful in scripts or batch jobs)
nsight-agent analyze profile.sqlite --yes

# Use Anthropic's count_tokens API for an exact input token count instead of the
# char/4 heuristic (adds one small API call; falls back to heuristic for other providers)
nsight-agent analyze profile.sqlite --exact-token-count
```

### Compare two profiles

```bash
nsight-agent compare profile_a.sqlite profile_b.sqlite
```

Produces a structured narrative and a key-differences table ordered by magnitude of change.
Both profiles are summarized independently, then a pre-computed structural diff is injected
into a single LLM prompt (no tool-use loop). Three comparison modes are selected automatically:

- **`phase_aware`** — same phase count and names; full per-phase analysis
- **`summary`** — phases differ but kernel overlap ≥ 20%; per-kernel diff included
- **`summary_no_kernel`** — phases differ and overlap < 20%; top-level metrics only

```bash
# Suppress verbose output
nsight-agent compare profile_a.sqlite profile_b.sqlite --quiet

# Output raw JSON
nsight-agent compare profile_a.sqlite profile_b.sqlite --json

# Skip the pre-flight confirmation prompt
nsight-agent compare profile_a.sqlite profile_b.sqlite --yes

# Exact token count via Anthropic API
nsight-agent compare profile_a.sqlite profile_b.sqlite --exact-token-count
```

---

## Provider selection

Provider resolution order (first match wins):

1. Provider prefix in `--model`: `openai:gpt-4o`, `gemini:gemini-2.0-flash`, `anthropic:claude-opus-4-6`
2. Bare provider name in `--model`: `openai`, `gemini`, `anthropic` (uses that provider's default model)
3. Auto-detect from environment: `ANTHROPIC_API_KEY` → `OPENAI_API_KEY` → `GOOGLE_API_KEY`
4. Fallback to `claude -p` subprocess (Claude Code CLI, no API key required)

### Anthropic (default)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
nsight-agent analyze profile.sqlite
nsight-agent analyze profile.sqlite --model claude-haiku-4-5-20251001   # faster, cheaper
```

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
nsight-agent analyze profile.sqlite --model openai:gpt-4o
nsight-agent analyze profile.sqlite --model openai:gpt-4o-mini
nsight-agent analyze profile.sqlite --model openai   # uses gpt-4o (default)
```

### Google Gemini

```bash
export GOOGLE_API_KEY=...
nsight-agent analyze profile.sqlite --model gemini:gemini-2.0-flash
nsight-agent analyze profile.sqlite --model gemini:gemini-1.5-pro
nsight-agent analyze profile.sqlite --model gemini   # uses gemini-2.0-flash (default)
```

### Claude Code fallback (no API key)

If no API key is set and `claude` is on your PATH, the agent falls back to a single `claude -p` call with the full `ProfileSummary` as a text prompt. This uses the Claude Code CLI's own authentication. No extra setup required, but this mode does not support multi-turn tool calls.

---

## Token expenditure

### Prompt caching

All three provider backends (Anthropic, OpenAI, Gemini) are stateless: every API call replays the full conversation history from turn 1. For a 20-turn analysis, the system prompt and pre-seeded profile summary are re-sent on every single turn — without caching, these account for the majority of total token cost.

**Anthropic — sliding cache (implemented):** nsight-agent uses Anthropic's prompt caching API with a sliding cache cursor. Turn 1 writes the system prompt and pre-seeded profile summary to cache (billed at 1.25× normal). Each subsequent turn reads the previous turn's cache hit (billed at 0.10×) and writes the new tool-result exchange to cache. The cached prefix grows by one exchange per turn, so later turns read an increasingly large prefix at 0.10× while paying full price only for the new incremental content. On a typical 18-turn run this produces a ~75–80% reduction in billable input tokens.

The pre-flight estimate for Anthropic runs shows the expected cache_write, cache_read, and cost-equivalent token counts instead of a raw input total.

**OpenAI — automatic caching:** OpenAI automatically caches repeated input prefixes longer than 1,024 tokens at a ~50% discount. No developer action is required; nsight-agent does not do anything special for OpenAI and the savings happen transparently.

**Google Gemini — not implemented:** Gemini supports explicit context caching via an "upload once, reference by ID" API that is architecturally different from the per-request marker approach. It is not currently implemented in nsight-agent.

---

### Pre-flight estimate

Before every `analyze` and `compare` run, nsight-agent prints an input/output token estimate and prompts for confirmation.

For Anthropic runs (with sliding prompt cache):

```
Token estimate (Anthropic, sliding prompt cache):
  Cache write: ~46,500 tokens  (billed at 1.25×)
  Cache read:  ~351,000 tokens (billed at 0.10×)
  Non-cached:  ~0 tokens
  Output:      ~3,800 – 12,800 (5 – 20 turns)
  Cost-equiv:  ~81,600 tokens  (heuristic)
  Model:       claude-opus-4-6 (anthropic)
Proceed? [Y/n]
```

For OpenAI and Gemini runs (full session total):

```
Token estimate (total across up to 20 turns):
  Input:  ~370,000 (heuristic)
  Output: ~3,800 – 12,800
  Model:  gpt-4o (openai)
Proceed? [Y/n]
```

- The estimate is skipped (and the prompt suppressed) in `--quiet` and `--json` modes.
- Pass `--yes` to skip the confirmation automatically (useful in scripts).
- Pass `--exact-token-count` to use Anthropic's `count_tokens` API for a precise input count instead of the character-count heuristic. No-ops with a note for other providers.
- The confirmation prompt is also skipped when stdin is not a TTY (piped or batch environments).
- The output range reflects the `--max-turns` value (default 20): `~3,800 – 12,800` tokens at 20 turns. Pass `--max-turns` to adjust both the actual limit and the displayed range.

### What drives token usage

The agent is multi-turn: each tool call costs one LLM round-trip. A typical analysis on a moderately complex profile (3–6 phases, a few dominant kernels, MPI present) uses roughly:

| Component                                 | Approximate tokens (input)                    |
| ----------------------------------------- | --------------------------------------------- |
| System prompt                             | ~400                                          |
| Pre-seeded profile + phase summary        | 3,000–12,000 (scales with phases and kernels) |
| Per-tool-call result (5–12 calls typical) | 500–2,000 each                                |
| Output hypothesis JSON                    | 1,000–2,500                                   |

**Total typical range: 15,000–60,000 input tokens + 1,000–3,000 output tokens per run.**

Profiles with many MPI operations, many phases, or dense kernel tables will be at the high end.

### Minimizing token expenditure

**Reduce phase context** (largest single lever):

```bash
nsight-agent analyze profile.sqlite --max-phases 2
nsight-agent analyze profile.sqlite --max-phases 1   # global metrics only, no phases
```

Each phase adds its own per-phase kernel table, MPI breakdown, and gap histogram to the pre-seeded context. Reducing from 6 to 1 can cut pre-seed size by 60–80%.

**Cap the turn count** (reduces worst-case cost and avoids runaway tool loops):

```bash
nsight-agent analyze profile.sqlite --max-turns 10   # good default for Haiku
nsight-agent analyze profile.sqlite --max-turns 5    # cheapest, summarizes after 5 tool calls
```

Smaller models like Haiku tend to use more turns for the same analysis. A lower `--max-turns` bounds the cost while the built-in wrap-up warning and forced final turn ensure you still get output rather than an error.

**Prompt caching** (already active for Anthropic — no action required):

Anthropic runs benefit from sliding prompt caching automatically. The pre-flight estimate will show the expected cache breakdown. For long runs (many turns, large profiles) the cache savings dominate: a 20-turn run on a large profile that would cost ~500k billed tokens without caching costs ~100k with it.

**Use a smaller model:**

```bash
# Anthropic — ~20× cheaper than Opus, good for straightforward profiles
nsight-agent analyze profile.sqlite --model claude-haiku-4-5-20251001

# OpenAI
nsight-agent analyze profile.sqlite --model openai:gpt-4o-mini

# Gemini — generous free tier
nsight-agent analyze profile.sqlite --model gemini:gemini-2.0-flash
```

**Inspect before analyzing:**

Run `summary` first to understand whether the full agent analysis is warranted:

```bash
nsight-agent summary profile.sqlite
```

If the bottleneck is already obvious from the summary table (e.g., one kernel dominates at 95% GPU time), you may not need the LLM at all.

**Use the JSON output for batch workflows:**

```bash
nsight-agent analyze profile.sqlite --quiet --json > hypotheses.json
```

`--quiet` suppresses verbose turn-by-turn output but doesn't affect token usage. `--json` skips the Rich table rendering.

**Claude Code fallback** sends a single large prompt with no follow-up tool calls. It uses more input tokens upfront (full summary) but zero tokens for tool interactions. This can be cheaper for simple profiles but less thorough.

---

## Risks

**Read-only, local operation.** The agent can only issue `SELECT` queries against the local SQLite file. It cannot write to the profile, modify your code, or access the network (beyond the LLM API call itself).

**SQL injection via `sql_query` tool.** The agent has access to a `sql_query` tool that executes arbitrary SQL. Because this runs against a local read-only connection, the blast radius is limited to the profile database. However, if you are using a profile that was provided by an untrusted third party, a specially crafted profile could attempt to influence the agent's SQL tool calls through embedded data (e.g., misleading kernel names or NVTX strings). Treat externally-sourced profiles with the same caution as any untrusted file.

**LLM hallucination.** The model may produce hypotheses that sound plausible but are not grounded in the profile data. Always cross-check the `evidence` field against the actual numbers — the `summary` subcommand and the saved `_prompt.txt` / `_response.txt` files provide the ground truth the model was given.

**Cost runaway.** If a profile is extremely large (many phases, dense MPI tables), the pre-seeded context can be very large, and if the agent makes many `sql_query` calls, costs can accumulate. Use `--max-phases 1` or a cheaper model for initial exploration.

**Saved files.** The agent writes `{stem}_{timestamp}_prompt.txt` and `_response.txt` next to the profile after each run. These files contain the full profile metrics. If the profile directory is shared or version-controlled, ensure these files are in `.gitignore`.

---

## Development

```bash
pytest                                          # run all tests
pytest tests/test_analysis.py::test_name -v    # single test
ruff check . && ruff format .                   # lint and format
```
