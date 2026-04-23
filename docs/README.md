# PerfAdvisor — Architecture Overview

PerfAdvisor ingests an Nsight Systems SQLite profile, extracts structured metrics locally (Stage 1), then drives an LLM agent that calls read-only tools against the profile to produce a ranked list of performance hypotheses (Stage 2).

---

## Pipeline

```
.sqlite file
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Stage 1 — Ingestion & Analysis  (local, no LLM) │
│                                                   │
│  NsysProfile (SQLite wrapper)                     │
│       └─► compute_profile_summary()               │
│               ├── phase detection                 │
│               ├── per-kernel metrics              │
│               ├── MPI breakdown                   │
│               ├── memory transfer summary         │
│               ├── GPU idle histogram              │
│               └── device info                     │
│                                                   │
│  Output: ProfileSummary (Pydantic)                │
└──────────────────────┬──────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2 — Hypothesis Generation  (LLM)          │
│                                                   │
│  Pre-seed: inject ProfileSummary as fake          │
│  tool results (saves 2–3 API round-trips)         │
│                                                   │
│  Multi-turn agent loop (up to 20 turns):          │
│    LLM ──tool_call──► dispatch() ──► NsysProfile  │
│         ◄──result──────────────────               │
│                                                   │
│  Output: list[Hypothesis] (JSON)                  │
└─────────────────────────────────────────────────┘
```

For multi-rank analysis, Stage 1 runs on every rank profile before Stage 2. For `compare`, Stage 1 runs on both profiles and a single LLM call replaces the tool-use loop.

---

## Module map

```
perf_advisor/
├── ingestion/
│   └── profile.py          NsysProfile: read-only SQLite wrapper, string ID
│                           resolution, query helpers
├── analysis/
│   ├── models.py           Pydantic models: ProfileSummary, PhaseSummary,
│   │                       KernelSummary, Hypothesis, CrossRankSummary, …
│   ├── metrics.py          compute_profile_summary() and individual metric
│   │                       functions (kernels, MPI, memcpy, streams, …)
│   ├── phases.py           detect_phases(): hybrid boundary detection +
│   │                       similarity-based merge algorithm
│   ├── cross_rank.py       Multi-rank summary, primary rank selection,
│   │                       phase alignment across ranks
│   └── diff.py             compute_profile_diff() for the compare subcommand
├── agent/
│   ├── loop.py             run_agent(): provider selection, pre-seeding,
│   │                       multi-turn tool-use loop (Anthropic/OpenAI/Gemini),
│   │                       sliding prompt cache, turn limit management
│   ├── tools.py            Tool implementations + schemas; dispatch()
│   ├── compare.py          run_compare(): single-call compare agent
│   ├── preflight.py        Token estimation and Anthropic count_tokens API
│   └── logger.py           LLMLogger: real-time API request/response logging
├── eval/
│   ├── discover.py         Discover benchmark run directories + ground truth
│   ├── scorer.py           Two-tier scoring: bottleneck detection + LLM judge
│   └── report.py           Print and save evaluation results
└── __main__.py             CLI: analyze / compare / summary / evaluate
```

---

## Key design decisions

**SQLite as the intermediate format.** The profile is never loaded into memory wholesale. All metric queries run as SQL against the read-only connection; pandas is used only for small aggregations. The agent can also issue arbitrary `SELECT` queries via the `sql_query` tool.

**Pre-seeding saves API round-trips.** `profile_summary` and `phase_summary` are computed in Stage 1 and injected into the conversation as fake tool results before the first real API call. The LLM never needs to call those tools, saving 2–3 turns of latency and cost at the start of every run.

**Hypothesis objects are plain dicts.** The agent outputs a raw JSON array; no further parsing is imposed until the data is written to disk as a `HypothesisReport`. This keeps the extraction logic (`_extract_hypotheses`) minimal and provider-agnostic.

**Profiles are immutable.** Stage 1 and Stage 2 are strictly read-only. The agent can call `sql_query` for follow-up questions, but the connection is opened with `mode=ro` and `query_safe()` enforces a row cap and supports interrupt on a `threading.Event`.

---

## Further reading

| Document | Contents |
| -------- | -------- |
| [pipeline.md](pipeline.md) | Stage 1 internals (phase detection, metrics), Stage 2 agent loop, multi-rank path, compare path |
| [data-models.md](data-models.md) | Pydantic schema reference: all fields and types |
| [agent-tools.md](agent-tools.md) | The nine tools exposed to Claude, `dispatch()`, how to add a new tool |
| [providers.md](providers.md) | Provider resolution, prompt caching per backend, adding a new provider |
| [prompt_caching.md](prompt_caching.md) | Motivation for prompt caching; per-provider implementation details (Anthropic sliding window, OpenAI auto-caching, Gemini explicit context cache) |
| [evaluation.md](evaluation.md) | Eval subcommand, benchmark layout, scoring rubric, adding scenarios |
