# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An agentic performance analyzer and tuner that ingests NVIDIA Nsight Systems profiles (`.nsys-rep` / `.sqlite`), extracts structured performance data, generates hypotheses for bottlenecks, and proposes or applies optimizations — driven by a Claude-powered agent loop.

## Architecture

The system is organized as a pipeline of agents:

1. **Ingestion** — Parse Nsight Systems exports (SQLite or JSON) into structured data models (CUDA kernels, NVTX ranges, CPU/GPU overlap, memory transfers, streams, sync events).
2. **Analysis** — Compute derived metrics (kernel occupancy, PCIe bandwidth utilization, SM efficiency, memory-bound vs. compute-bound classification, CPU-GPU overlap gaps).
3. **Hypothesis Generation** — Claude agent reads the structured analysis and proposes ranked hypotheses (e.g., "kernel X is memory-bound — consider tiling", "H2D transfers overlap poorly with compute").
4. **Tuning Actions** — Optionally apply or scaffold suggested changes (CUDA code modifications, launch config changes, stream reordering).
5. **Verification** — Re-profile and diff before/after metrics to validate improvements.

## Nsight Systems Data Model

Nsight Systems `.nsys-rep` files are SQLite databases. Key tables:

- `CUPTI_ACTIVITY_KIND_KERNEL` — GPU kernel execution (duration, grid/block dims, SM count, registers, shared mem)
- `CUPTI_ACTIVITY_KIND_MEMCPY` — H2D/D2H/D2D transfers (bytes, duration, async flag)
- `CUPTI_ACTIVITY_KIND_RUNTIME` — CUDA API calls on CPU side
- `NVTX_EVENTS` — User-annotated ranges (used for labeling regions)
- `TARGET_INFO_*` — Device properties (SM count, clock rates, memory bandwidth)

Export to SQLite for direct querying:

```bash
nsys export --type sqlite --output profile.sqlite profile.nsys-rep
```

Export to JSON for LLM-readable summaries:

```bash
nsys stats --report gputrace --format json profile.nsys-rep
```

## Claude Agent Integration

This project uses the Anthropic SDK (`anthropic` Python package). The agent loop pattern:

- Use `claude-opus-4-6` for hypothesis generation and reasoning (complex multi-step)
- Use `claude-haiku-4-5-20251001` for fast classification or structured extraction tasks
- Tools exposed to the agent: `query_profile`, `compute_metrics`, `get_kernel_details`, `compare_profiles`
- Agent outputs structured `Hypothesis` objects with: bottleneck type, affected kernels/ranges, confidence, suggested fix, expected speedup range

## Test Profile

**Location:** `/home/ads.leonhost/Downloads/nsight/nsys_4864_cgdef_2node_1rhs/`

**Application:** MILC/QUDA lattice QCD (`ks_spectrum_hisq`) — staggered quark spectrum with HISQ action, conjugate gradient solver with optional eigenvector deflation.

**Run:** 2 nodes × 4 GPUs (NERSC Perlmutter, NVIDIA A100 HBM40g), 8 MPI ranks total. Job ran two configurations: `ev1024` (CG with 1024-eigenvector deflation) and `ev0` (plain CG, no deflation).

**Key files:**

- `cg_4864_1rhs.sqlite` — main analysis target (CG run, 1 RHS)
- `cgdef_4864_1rhs.nsys-rep` — CG with deflation variant (`.nsys-rep` only, export to SQLite before querying)
- `report.0.nid001429.50613743.{1,2}.sqlite` — per-rank profiles for rank 0, steps 1 (ev1024) and 2 (ev0)

**Profile characteristics:**

- Captured with: `-t cuda,nvtx,osrt,mpi --mpi-impl=mpich`
- Top kernels by call count: `Kernel3D` (478K calls, ~0.05ms avg), `Kernel2D` (87K calls), `Reduction2D` (1.5K calls, ~0.24ms avg), `MultiReduction` (318 calls)
- Heavy MPI: ~1.3M P2P events, ~309K collective events — communication overhead is a key analysis target
- `StringIds` table resolves integer IDs for kernel names (`demangledName`, `shortName`, `mangledName` columns are foreign keys into `StringIds`)

**Export command for cgdef profile:**

```bash
nsys export --type sqlite --output cgdef_4864_1rhs.sqlite /home/ads.leonhost/Downloads/nsight/nsys_4864_cgdef_2node_1rhs/cgdef_4864_1rhs.nsys-rep
```

## Project Structure

```
nsight_agent/
├── ingestion/profile.py   # NsysProfile: SQLite wrapper, string/enum resolution, query helpers
├── analysis/
│   ├── models.py          # Pydantic models: ProfileSummary, KernelSummary, MpiOpSummary, etc.
│   └── metrics.py         # compute_profile_summary() and individual metric functions
├── agent/
│   ├── tools.py           # Tool implementations + schemas exposed to Claude; dispatch()
│   └── loop.py            # run_agent(): Anthropic client loop with tool-use
└── __main__.py            # CLI: `python -m nsight_agent [analyze|summary] <profile.sqlite>`
tests/
├── test_ingestion.py      # NsysProfile unit tests
└── test_analysis.py       # Metrics unit tests (require real test profile)
```

## Development Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the agent on a profile
python -m nsight_agent analyze /path/to/profile.sqlite

# Print structured metrics without running the agent
python -m nsight_agent summary /path/to/profile.sqlite

# Run tests
pytest

# Run a single test
pytest tests/test_analysis.py::test_mpi_barrier_dominates -v

# Lint / format
ruff check . && ruff format .
```

## Key Design Decisions

- **SQLite as the intermediate format** — query profiles directly with SQL rather than loading everything into memory; use pandas only for metric aggregation.
- **Hypothesis objects are serializable** — all agent outputs are Pydantic models so they can be stored, diffed, and fed back into subsequent agent turns.
- **Profiles are immutable inputs** — never modify `.nsys-rep` files; work from exports or cached SQLite copies.
- **Agentic loop is restartable** — persist agent state (hypotheses, tool call history) to disk so long-running analyses can resume.
