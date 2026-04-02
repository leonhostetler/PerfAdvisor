# nsight-agent

An agentic performance analyzer for NVIDIA Nsight Systems profiles. It extracts structured metrics from a `.sqlite` profile, then uses an LLM to reason over the data and produce a ranked list of actionable performance hypotheses.

---

## How it works

The pipeline has two distinct stages:

### Stage 1 — Pure analysis (no LLM)

`compute_profile_summary()` queries the SQLite profile directly and builds a structured `ProfileSummary`:

- **Phase detection** — segments the timeline into non-overlapping execution phases (initialization, main compute, teardown, etc.) using kernel density clustering on the GPU utilization timeline
- **Per-kernel metrics** — total/avg/min/max time, coefficient of variation, register usage, shared memory, estimated SM occupancy, CPU launch overhead
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

# Limit phase detection (fewer phases = less context = fewer tokens)
nsight-agent analyze profile.sqlite --max-phases 3
nsight-agent analyze profile.sqlite --max-phases 1   # disable phase segmentation entirely
```

---

## Provider selection

Provider resolution order (first match wins):

1. Provider prefix in the model string: `openai:gpt-4o`, `gemini:gemini-2.0-flash`, `anthropic:claude-opus-4-6`
2. `--provider {anthropic,openai,gemini}` flag
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
nsight-agent analyze profile.sqlite --provider openai --model gpt-4o-mini
```

### Google Gemini

```bash
export GOOGLE_API_KEY=...
nsight-agent analyze profile.sqlite --model gemini:gemini-2.0-flash
nsight-agent analyze profile.sqlite --provider gemini --model gemini-1.5-pro
```

### Claude Code fallback (no API key)

If no API key is set and `claude` is on your PATH, the agent falls back to a single `claude -p` call with the full `ProfileSummary` as a text prompt. This uses the Claude Code CLI's own authentication. No extra setup required, but this mode does not support multi-turn tool calls.

---

## Token expenditure

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
