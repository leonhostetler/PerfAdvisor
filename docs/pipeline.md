# Pipeline Internals

## Stage 1 — Ingestion and Analysis

Stage 1 runs entirely locally. No LLM is contacted. The output is a `ProfileSummary` Pydantic model that serves as the input to Stage 2.

### Profile ingestion (`ingestion/`)

`open_profile(path)` auto-detects the format (Nsight Systems or rocpd) by sniffing the SQLite schema and returns a `Profile`-protocol object (`NsysProfile` or `RocpdProfile`). Both implementations share the same interface:

- Open the file read-only via the SQLite URI (`mode=ro`)
- Configure page cache and mmap for fast repeated queries on large profiles
- Cache string-table lookups (`StringIds` for NSYS, `rocpd_string` for rocpd) so `resolve_string(id)` returns human-readable text on both formats
- Expose `ProfileCapabilities` flags (`has_kernels`, `has_memcpy`, `has_mpi`, `has_markers`, …) so metric functions degrade gracefully when a capability is absent
- Provide vendor-neutral query helpers (`kernel_events()`, `memcpy_events()`, `marker_ranges()`, `mpi_ranges()`, `device_info()`) that hide per-format SQL differences from the analysis layer

Two raw query methods are exposed for internal and LLM use:

| Method | Used by | Behaviour |
| ------ | ------- | --------- |
| `query(sql)` | Internal metric functions | No row cap; trusted callers include their own `LIMIT` clauses |
| `query_safe(sql)` | `sql_query` tool (LLM-issued) | Row cap (default 200); installs a SQLite progress handler that checks a `threading.Event` every 1000 VM instructions so long-running queries can be interrupted |

### compute_profile_summary() (`analysis/metrics.py`)

`compute_profile_summary()` orchestrates Stage 1. It calls individual metric functions in sequence and assembles the results into a `ProfileSummary`:

1. **Profile span** — `MAX(end) - MIN(start)` across all event tables (kernels, memcpy, runtime, markers, MPI), so the span matches the profiler's timeline view
2. **Device info** — SM/CU count, peak memory bandwidth, total HBM, L2 cache, clock rate, and thread/register/shared-memory limits from device metadata tables
3. **Phase detection** — described below; produces a list of `PhaseWindow` objects
4. **Per-kernel metrics** — grouped by full demangled name (not the short display name), so each distinct template instantiation is tracked separately; computes total/avg/min/max time, coefficient of variation, register usage, shared memory, estimated SM occupancy, and CPU launch overhead
5. **Memory transfer summary** — bytes and bandwidth by direction (H2D, D2H, D2D)
6. **MPI breakdown** — per-operation total time and call count (collectives, P2P, barrier, wait)
7. **GPU idle histogram** — gap distribution bucketed into `<10µs`, `10–100µs`, `100µs–1ms`, `1–10ms`, `10–100ms`, `>100ms`
8. **CPU–GPU overlap** — total time the CPU spent blocked in `*Synchronize` calls
9. **Stream utilization** — GPU time and percentage per GPU stream
10. **Marker ranges** — NVTX annotations (NVIDIA) or rocTX ranges (AMD), if present

Per-phase variants of steps 4–9 are computed for each `PhaseWindow` using windowed SQL queries (`start < phase_end AND end > phase_start`).

### Phase detection (`analysis/phases.py`)

See [Phase Detection](phase_detection.md) for a full description of the algorithm, the motivation for profile segmentation, and the pathologies a robust implementation must handle.

---

## Stage 2 — Agent Loop

### System prompt and pre-seeding

Before the first API call, two things are prepared:

1. **System prompt** — instructs the model to act as a GPU performance engineer, defines the hypothesis JSON schema, includes a format-specific SQLite schema reference (NSYS or rocpd column names, FK conventions, which aggregate functions are safe vs. unavailable in SQLite), a capability section listing what this profile does and does not contain, and optionally includes a hardware context block (GPU name, SM/CU count, peak bandwidth, etc.) derived from `device_info`.

2. **Pre-seeding** — the `profile_summary` and `phase_summary` tool results (already computed in Stage 1) are injected into the conversation history as a fake assistant-turn / user-turn exchange before the first real API call. This means the model starts with the full profile overview already in context and never needs to call those two tools, saving 2–3 turns of latency and cost.

For multi-rank runs, `cross_rank_summary` is pre-seeded alongside the primary rank's data as a third fake tool result.

### Tool dispatch (`agent/tools.py`)

`dispatch(profile, tool_name, args)` maps a tool name to its implementation function and returns the result as a JSON string. All tool implementations take a `Profile` (either `NsysProfile` or `RocpdProfile`) and an `args` dict; they call the same metric functions as Stage 1 but can accept `start_ns` / `end_ns` window parameters for per-phase queries.

The nine tools the agent can call:

| Tool | Returns |
| ---- | ------- |
| `profile_summary` | Wall-clock span, GPU utilization, which tables are present |
| `phase_summary` | Per-phase GPU util, top kernels, MPI ops, idle gaps |
| `top_kernels` | Top kernels by GPU time; accepts optional time window |
| `gap_histogram` | GPU idle gap distribution; accepts optional time window |
| `memcpy_summary` | Memory transfer breakdown by direction |
| `mpi_summary` | MPI operation breakdown; accepts optional time window |
| `marker_ranges` | Marker annotation ranges (NVTX or rocTX) |
| `stream_summary` | Per-CUDA-stream GPU time and percentage |
| `sql_query` | Arbitrary read-only SQL via `query_safe()` |
| `get_table_schema` | Column names for a named table |

`profile_summary` and `phase_summary` are pre-seeded and the model is instructed not to call them again.

### Multi-turn loop

The loop runs for up to `max_turns` turns (default 20):

1. Send the accumulated message history to the LLM
2. If `stop_reason == "end_turn"` (Anthropic) or `finish_reason == "stop"` (OpenAI/Gemini), extract the hypothesis JSON array from the last text block and return
3. If the model emitted tool calls, dispatch each one against the local profile and append the results as a new user message
4. Advance the sliding cache window (Anthropic only) and loop

**Turn limit management.** Three turns before the limit, a wrap-up warning is injected alongside the tool results: `"You have N turns remaining. Output your final hypothesis JSON array now if you have sufficient evidence."` If the limit is still reached, one additional forced turn is made without exposing any tools, prompting the model to output whatever it has gathered. This ensures the run always produces output rather than an error.

### Sliding prompt cache (Anthropic)

Every API call replays the full conversation history. Without caching, the system prompt and pre-seeded profile summary are billed at full input rate on every turn. The Anthropic backend uses a sliding cache window to avoid this:

- The pre-seed block (system + tools + pre-seeded results) carries a permanent `cache_control: ephemeral` marker, billed at 1.25× on the first call and 0.10× on all subsequent calls
- Each new user message (tool results) receives a floating `cache_control` marker; the oldest floating marker is stripped when a third one would be added, keeping the total at Anthropic's maximum of 4 markers
- On turn N, the entire context through turn N−1 is read from cache at 0.10×; only the new tool-result exchange is billed at 1.25×

For a typical 18-turn run this produces a ~75–80% reduction in billable input tokens.

See [prompt_caching.md](prompt_caching.md) for full per-provider details.

### Provider backends

All three backends share the same tool schemas and dispatch logic; only the message format and SDK differ.

**Anthropic** (`_run_api`) — native tool-use format; sliding prompt cache as described above; pre-seeds using the assistant/user fake-exchange pattern supported by the Anthropic message format.

**OpenAI** (`_run_openai`) — translates Anthropic tool schemas to OpenAI's `{"type":"function","function":{...}}` wrapper; pre-seeds using OpenAI's `tool_calls` / `role:"tool"` message format; automatic ~50% prefix caching happens transparently without developer action. Auto-detects whether the model requires `max_tokens` or `max_completion_tokens` and retries once on `BadRequestError` if the wrong parameter is used.

**Gemini** (`_run_gemini`) — translates schemas to `FunctionDeclaration` objects; pre-creates a named `CachedContent` object containing the system instruction, tool declarations, and pre-computed profile summary before the first turn (10-minute TTL, billed at 0.25×); subsequent turns reference the cache by name. If cache creation fails (token minimum not met, unsupported model), falls back to injecting the summary into the first user message.

**Claude Code fallback** (`_run_claude_code`) — invokes `claude -p <prompt> --output-format json` as a subprocess; no API key required; no tool-use loop (single call with full summary in prompt). Used automatically when no API key is found in the environment.

---

## Multi-rank path

When more than one profile is passed on the command line, Stage 1 runs independently on every rank profile. The results are used to:

1. **Select a primary rank.** The rank with the highest GPU idle time is selected as the primary rank for the full Stage 2 analysis (this is the rank being held up the most by MPI waits, memcpy, or sync). If no rank exceeds the median by more than 20%, rank 0 (or the value of `--primary-rank`) is used.

2. **Align phases across ranks.** Phase names are matched by name first. If names differ but durations agree within 20%, index-order alignment is used with a warning. If counts differ or durations diverge beyond tolerance, cross-rank analysis is aborted and the run falls back to single-rank analysis on the primary rank.

3. **Compute CrossRankSummary.** For each phase, `(max − min) / mean` imbalance scores are computed across ranks for GPU kernel time and MPI wait time, along with the worst collective operation per phase. This is pre-seeded into Stage 2 alongside the primary rank's data.

The agent can reference specific ranks and per-phase imbalance scores when generating hypotheses without additional tool calls.

See [cross_rank_analysis.md](cross_rank_analysis.md) for a full step-by-step description of each stage of the multi-rank path, including how rank IDs are parsed, how consensus k is selected, and how each column of the imbalance tables is calculated.

---

## Compare path

`compare` runs Stage 1 on both profiles, then makes a single LLM call (no tool-use loop) with a pre-computed structural diff injected into the prompt.

`compute_profile_diff()` computes the diff and selects one of three comparison modes:

| Mode | Condition | What the LLM receives |
| ---- | --------- | --------------------- |
| `phase_aware` | Same phase count and names | Full per-phase kernel and MPI diff |
| `summary` | Phases differ; kernel name overlap ≥ 20% | Top-level + per-kernel diff |
| `summary_no_kernel` | Phases differ; overlap < 20% | Top-level metrics only |

The LLM produces a structured response with a narrative and a `key_differences` array ordered by magnitude of change.
