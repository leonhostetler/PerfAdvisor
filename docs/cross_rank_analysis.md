# Cross-Rank Analysis

## Purpose and Motivation

When more than one profile is passed to `analyze`, PerfAdvisor runs a cross-rank analysis pass
before Stage 2 (hypothesis generation). The goal is to distinguish two fundamentally different
performance problems that look similar on a single rank's profile:

- **Genuine bottlenecks**: the application does too much GPU work, has memory-bandwidth issues, or
  launches kernels inefficiently — *every* rank is slow for the same reason.
- **Load imbalance**: one or more ranks finish their compute work earlier than others and spend
  most of their time waiting at MPI barriers or collective operations — the fast ranks are
  effectively throttled by the slow one.

Without cross-rank context, a hypothesis agent examining rank 0 in isolation cannot tell whether
rank 0's 40% GPU idle time is caused by a kernel-level bottleneck on rank 0 or by rank 0 waiting
for a slower rank 1. The `CrossRankSummary` object resolves this by quantifying imbalance across
all ranks before Stage 2 starts, so the agent has the full picture pre-seeded into context.

---

## How It Fits in the Pipeline

Cross-rank analysis is part of Stage 1 (local, no LLM). It runs between per-rank
`ProfileSummary` computation and Stage 2 pre-seeding. See [pipeline.md](pipeline.md) for the
full pipeline diagram.

The path is activated whenever `analyze` receives more than one `.sqlite` file:

```
python -m perf_advisor analyze rank*.sqlite
# or equivalently
python -m perf_advisor analyze rank0.sqlite rank1.sqlite rank2.sqlite ...
```

---

## Step-by-Step Walkthrough

### Step 1 — Parse Rank IDs from Filenames

`parse_rank_ids(paths)` (`cross_rank.py`) attempts to derive MPI rank IDs from the profile
filenames rather than from the file order on the command line.

**Algorithm:**
1. Strip the file extension and extract all non-negative integer tokens from each stem in order.
   Example: `report.0.nid001429.50613743.1.sqlite` → `[0, 1429, 50613743, 1]`.
2. Require that all stems produce the same number of integer tokens. If not, fall back.
3. Find every positional slot where the integers vary across files. If exactly one slot varies,
   use those values as rank IDs.
4. If zero or more than one slot varies, fall back to index order `[0, 1, 2, ...]` and set
   `parsed_ok = False`.

**Why this matters:** The rank ID is what the agent and the user see in tables and hypotheses.
Parsing from the filename rather than from the command-line position avoids confusion when files
are given out of order or when the naming convention is deterministic (e.g.,
`report.0.nid.50613743.{0,1,2,3}.sqlite` where position 4 is the rank).

If parsing fails, a warning is printed and index order is used. This does not affect
the correctness of the analysis — rank IDs are just labels.

---

### Step 2 — Pass 1: Cost-Curve Extraction for Consensus k

Before running the full per-rank pipeline, a lightweight first pass extracts each rank's phase
detection cost curve without running the labeling and fingerprinting steps.

`compute_phase_cost_curve(profile, max_phases, rank)` (`phases.py`) runs Steps 1–9 of the
[phase detection algorithm](phase_detection.md) through k-selection, returning:
- `selected_k` — the elbow-selected number of phases for this rank
- `cost_curve` — a dict mapping `k → total DP cost` for `k = 1..K`

This is done for every rank before any full `ProfileSummary` is computed, so all per-rank phase
structures can be compared before committing to a particular k.

---

### Step 3 — Select a Consensus k

`select_consensus_k(cost_curves, selected_ks, max_phases)` (`cross_rank.py`) picks a single k
to force on all ranks so that phase boundaries are comparable across ranks. Two sequential
checks guard the selection.

**Check 1 — Pre-consensus spread:**
Compute `spread = max(selected_ks) - min(selected_ks)` over all ranks. If `spread >
_K_SPREAD_THRESHOLD` (default: 2), the ranks have structurally different workloads — they do not
agree on how many phases the profile contains. Cross-rank analysis cannot proceed. An abort
message is returned and the run falls back to single-rank analysis on the primary rank.

**Fast path:** if all ranks selected the same k, return that k immediately.

**k averaging:** if the spread is within threshold but ranks disagree, compute the
duration-weighted average cost curve over the common k range and apply the same elbow criterion
(`_DP_ELBOW_THRESHOLD = 5%`) to the averaged curve to pick `consensus_k`.

**Check 2 — Post-consensus cost excess:**
For each rank where `consensus_k < rank's selected_k` (the consensus forces fewer phases than
the rank would prefer), compute:

```
excess = (cost_at_consensus_k - cost_at_optimal_k) / cost_at_optimal_k
```

A positive excess means the forced k produces a worse segmentation. If `excess >
_COST_EXCESS_THRESHOLD` (default: 15%) for any rank, the consensus cannot be imposed without
significant distortion — abort cross-rank analysis.

If both checks pass, `consensus_k` is returned and used as `forced_k` in Step 4.

---

### Step 4 — Pass 2: Full Per-Rank ProfileSummary

With `consensus_k` established, `compute_profile_summary(profile, forced_k=consensus_k)` is
called for every rank. This runs the full phase detection pipeline (see
[phase_detection.md](phase_detection.md)) with Step 9's k-selection overridden to use
`consensus_k` instead of the per-rank elbow value.

The result is a `dict[int, ProfileSummary]` — one `ProfileSummary` per rank, with all phases
aligned to the same k.

---

### Step 5 — Select the Primary Rank

`select_primary_rank(summaries)` (`cross_rank.py`) identifies which rank is the most interesting
for the Stage 2 agent to analyze in detail.

**Heuristic:** the rank with the highest GPU idle time is the one being held up the most by MPI
waits, slow communication, or kernel-level bottlenecks. That is the rank whose profile will
reveal the most about the overall job's critical path.

**Computation:**
1. Collect `total_gpu_idle_s` from each rank's `ProfileSummary`.
2. Compute the median across all ranks.
3. If any rank's idle time exceeds `median × (1 + OUTLIER_IDLE_THRESHOLD)` (default: 20%),
   select the rank with the highest idle time as the primary.
4. If no rank qualifies as an outlier (all ranks are within 20% of the median), default to the
   lowest rank ID.

The primary rank is the one whose `ProfileSummary` is pre-seeded into Stage 2 and used for
all single-rank tool calls during hypothesis generation. The `CrossRankSummary` is also
pre-seeded so the agent sees both the primary rank's detailed data and the cross-rank
imbalance picture simultaneously.

The `--primary-rank N` CLI flag bypasses automatic selection and forces rank N.

---

### Step 6 — Phase Alignment Check

`align_phases(summaries)` (`cross_rank.py`) verifies that the phases detected across ranks are
comparable before computing cross-rank statistics.

**Check 1 — Phase count:** all ranks must have the same number of phases. If counts differ,
alignment fails immediately and cross-rank analysis is aborted. This check should almost never
fire if `consensus_k` was selected correctly in Step 3, but can still occur if one rank's profile
has no GPU kernel activity at all (e.g., it is a pure CPU rank).

**Check 2 — Phase names:** compare each rank's ordered list of phase names against rank 0's.
Two outcomes:

- **`name_match`:** all ranks agree on phase names. Proceed with name-based alignment. A
  duration divergence warning may still be emitted (see below).

- **`index_order`:** names differ across ranks (e.g., different NVTX label coverage), but
  phase durations agree within `PHASE_DURATION_TOLERANCE` (default: 20%). Proceed using
  position-based matching (phase 0 matches phase 0, etc.) and emit a warning. This is a
  graceful fallback for cases where NVTX annotations are inconsistently applied across ranks.

**Check 3 — Duration divergence (always run):** for each phase index, compute the mean duration
across all ranks and check whether any rank's duration deviates by more than 20% from the mean.
If names matched but durations diverge, emit a warning alongside the `name_match` result — the
alignment may be unreliable. If names did *not* match and durations also diverge, return
`"failed"` — the ranks have structurally different phase structures and cross-rank statistics
would be meaningless.

The `alignment` string (`"name_match"` or `"index_order"`) is stored in `CrossRankSummary` and
shown in the output so users can judge whether the phase matching is reliable.

---

### Step 7 — Compute CrossRankSummary

`compute_cross_rank_summary(summaries, primary_rank_id, phase_alignment)` (`cross_rank.py`)
aggregates the per-rank `ProfileSummary` objects into a `CrossRankSummary`.

Two parts are computed:

**Part A — Per-rank overview** (`per_rank_overview: list[RankOverview]`): one entry per rank,
covering the full profile (all phases combined). Used to produce the Multi-rank Overview table.

**Part B — Per-phase cross-rank stats** (`phases: list[CrossRankPhaseSummary]`): one entry per
phase, covering statistics across all ranks for that phase. Used to produce the Cross-rank Phase
Imbalance table.

---

## Output: Two Tables

### Table 1 — Multi-rank Overview

Printed first. One row per rank; columns:

| Column | Source | How calculated |
|---|---|---|
| **Rank** | `RankOverview.rank_id` | Integer rank ID (parsed in Step 1) |
| **GPU Kernel (s)** | `RankOverview.gpu_kernel_s` | `ProfileSummary.gpu_kernel_s` — total time all GPU kernels ran across the full profile |
| **GPU Idle (s)** | `RankOverview.gpu_idle_s` | `ProfileSummary.total_gpu_idle_s` — sum of all inter-kernel gaps (kernel-end to next-kernel-start) across the full profile |
| **MPI Wait (s)** | `RankOverview.mpi_wait_s` | `sum(op.total_s for op in ProfileSummary.mpi_ops)` — total time in all MPI operations across the full profile |
| **GPU Util %** | `RankOverview.gpu_utilization_pct` | `ProfileSummary.gpu_utilization_pct` — `gpu_kernel_s / profile_span_s × 100` |
| **Primary** | — | Star (★) marks `CrossRankSummary.primary_rank_id` — the rank selected for detailed Stage 2 analysis |

**Note on GPU Idle vs. MPI Wait:** GPU idle (`total_gpu_idle_s`) counts inter-kernel gaps on the
GPU timeline — time when no kernel was executing. MPI wait (`mpi_wait_s`) counts time the CPU
spent inside MPI calls. These overlap but are not identical: an `MPI_Allreduce` call may keep the
GPU idle (the GPU is waiting for the collective to finish) but the GPU idle counter only sees the
gap between the last kernel before the collective and the first kernel after. A rank that spends
3s in MPI collectives may show 2.5s of GPU idle (if the collective is fully serializing) or 0.1s
(if the collective is communication-computation overlapped via async streams). Comparing both
columns together reveals whether MPI is actually serializing the GPU or is being effectively
pipelined.

---

### Table 2 — Cross-rank Phase Imbalance

One row per phase. Each row summarizes the imbalance across all ranks for that phase. Columns:

#### Phase

The name of the execution phase. In `name_match` mode, this is the phase name as reported by
all ranks (they agree). In `index_order` mode, the name is taken from rank 0's `ProfileSummary`
and may not match the names other ranks would use for this position.

Phase names come from the phase detection algorithm (see [phase_detection.md](phase_detection.md)):
NVTX range labels, dominant kernel short names, or `"idle"` / `"unknown"` fallbacks.

#### GPU Kernel Imbalance

**Formula:**

```
gpu_kernel_imbalance = (max_gpu_kernel_s - min_gpu_kernel_s) / mean_gpu_kernel_s
```

where `gpu_kernel_s` for each rank is `PhaseSummary.gpu_kernel_s` — the total time GPU kernels
were executing during this phase on that rank.

**Interpretation:** 0% means all ranks spent exactly the same time running GPU kernels in this
phase. 100% means the slowest rank ran kernels for twice as long as the mean (or the fastest ran
for zero time). Values above 50% are highlighted red; 20–50% yellow; below 20% green.

A high GPU kernel imbalance indicates that one rank is doing significantly more GPU compute work
than the others in this phase — either from algorithmic load imbalance (different problem sizes
per rank) or from hardware differences (different GPU clock behavior, thermal throttling).

#### Slowest Rank (GPU column)

The rank ID with the highest `gpu_kernel_s` in this phase. This is the rank that the job is
waiting for on the GPU compute side.

**Calculation:** `rank_ids[gpu_kernel_vals.index(max(gpu_kernel_vals))]`

#### MPI Wait Imbalance

**Formula:**

```
mpi_wait_imbalance = (max_mpi_wait_s - min_mpi_wait_s) / mean_mpi_wait_s
```

where `mpi_wait_s` for each rank is:

```python
sum(op.total_s for op in PhaseSummary.mpi_ops)
```

i.e., the total time the CPU spent inside any MPI operation during this phase, summed across all
operation types (`MPI_Allreduce`, `MPI_Barrier`, `MPI_Waitall`, etc.).

**Interpretation:** A high MPI wait imbalance means one rank is spending far more time inside MPI
calls than the others. This typically happens when a slow rank holds up a collective: all other
ranks arrive at the collective quickly and block waiting for the slow rank, so the fast ranks show
high MPI wait time while the slow rank shows low MPI wait time (it enters the collective last and
exits first). The **slowest rank** column (MPI) identifies which rank spent the most time inside
MPI calls — that is usually the rank that is *waiting the longest*, not the rank causing the
imbalance.

To distinguish who is *causing* the imbalance (doing more compute work) from who is *suffering*
the imbalance (waiting the most at collectives), compare the GPU Kernel Slowest Rank with the MPI
Wait Slowest Rank: if they are different ranks, the GPU-slowest rank is likely the root cause and
the MPI-slowest rank is the one being held up.

#### Slowest Rank (MPI column)

The rank ID with the highest total MPI wait time in this phase.

**Calculation:** `rank_ids[mpi_wait_vals.index(max(mpi_wait_vals))]`

#### Top Collective Imbalance

The single MPI collective operation with the highest imbalance score in this phase. Shows the
operation name and its imbalance score as a percentage.

**How it is selected:** for each MPI operation name that appears in at least one rank's phase:

1. Collect `total_s` for that operation from every rank's `PhaseSummary.mpi_ops`. Ranks that did
   not execute this operation contribute 0.0.
2. Compute:
   ```
   imbalance_score = (max_s - min_s) / mean_s
   ```
   where `mean_s = statistics.mean(op_vals)` over all ranks (including zeros).
   Skip the operation if `mean_s == 0`.
3. Record `slowest_rank_id` as the rank with the highest `total_s` for this operation.

All operations are sorted by imbalance score descending; the top entry is shown in the table.

**Interpretation:** A high collective imbalance score on a specific operation (e.g.,
`MPI_Allreduce: 87%`) tells you *which* communication pattern is most imbalanced in this phase.
The full per-collective breakdown is available in `CrossRankPhaseSummary.collective_imbalance`
(accessed via `--json` or agent reasoning) — the table only shows the worst offender.

Imbalance color thresholds match the GPU and MPI columns: ≥50% red, 20–49% yellow, <20% green.

---

## Data Models

The cross-rank analysis produces these Pydantic models (defined in `analysis/models.py`):

```
CrossRankSummary
├── num_ranks: int
├── rank_ids: list[int]
├── primary_rank_id: int
├── phase_alignment: str              # "name_match" | "index_order"
├── per_rank_overview: list[RankOverview]
│   └── RankOverview
│       ├── rank_id
│       ├── gpu_kernel_s              # whole-profile GPU kernel time
│       ├── gpu_idle_s                # whole-profile inter-kernel gap time
│       ├── mpi_wait_s                # whole-profile MPI operation total time
│       └── gpu_utilization_pct
└── phases: list[CrossRankPhaseSummary]
    └── CrossRankPhaseSummary
        ├── phase_index, phase_name
        ├── gpu_kernel_{mean,std,min,max}_s
        ├── gpu_kernel_imbalance      # (max-min)/mean
        ├── gpu_kernel_slowest_rank_id
        ├── mpi_wait_{mean,std,min,max}_s
        ├── mpi_wait_imbalance        # (max-min)/mean
        ├── mpi_wait_slowest_rank_id
        ├── collective_imbalance: list[CollectiveImbalance]
        │   └── CollectiveImbalance
        │       ├── op                # e.g. "MPI_Allreduce"
        │       ├── imbalance_score   # (max-min)/mean
        │       ├── slowest_rank_id
        │       ├── mean_s, min_s, max_s
        └── per_rank: list[RankPhaseStats]
            └── RankPhaseStats
                ├── rank_id
                ├── gpu_kernel_s      # phase-scoped GPU kernel time
                ├── gpu_idle_s        # phase-scoped inter-kernel gap time
                └── mpi_wait_s        # phase-scoped MPI total time
```

The `CrossRankSummary` is pre-seeded into Stage 2 as a third fake tool result alongside the
primary rank's `ProfileSummary`. The agent can reference per-rank data and per-collective
imbalance scores from `collective_imbalance` and `per_rank` without additional tool calls.

---

## Failure Modes and Fallbacks

The cross-rank path has three distinct failure points. In all cases the run falls back to
single-rank analysis on the primary rank rather than failing entirely.

| Failure | Detected in | Condition | What happens |
|---|---|---|---|
| Phase count diverges | Step 3 (`select_consensus_k`) | Per-rank elbow-selected k values span more than `_K_SPREAD_THRESHOLD` (2) | Abort message shown in red panel; `consensus_k = None`; each rank uses its own k in Pass 2; no `CrossRankSummary` |
| Consensus forces poor segmentation | Step 3 (`select_consensus_k`) | Any rank's cost at `consensus_k` exceeds its optimal cost by more than 15% | Same outcome as above |
| Phase alignment fails | Step 6 (`align_phases`) | Phase counts differ after Pass 2, or names differ and durations also diverge >20% | Abort message shown in red panel; `CrossRankSummary` is not computed |
| Phase names differ but durations agree | Step 6 (`align_phases`) | Names differ; durations agree within 20% | Warning shown in yellow panel; proceed with index-order alignment; `CrossRankSummary` is computed and includes a note in `phase_alignment = "index_order"` |

---

## Imbalance Score Color Scale

Used consistently in the phase imbalance table for all three imbalance columns:

| Score | Color | Interpretation |
|---|---|---|
| ≥ 50% | Red | Severe imbalance — slowest rank takes ≥1.5× the mean time |
| 20–49% | Yellow | Moderate imbalance — worth investigating |
| < 20% | Green | Within normal variation |

The 50% threshold corresponds to `(max - min) / mean ≥ 0.5`, which means the slowest rank
takes at least 50% more than the mean (e.g., 3 ranks at 1s and 1 rank at 2s → imbalance =
(2−1)/1.25 = 0.8, or 80%).

---

## What Is Passed to the LLM

The `CrossRankSummary` is pre-seeded into Stage 2 as a third fake tool result (alongside
`profile_summary` and `phase_summary`) before the first real API call. The agent never needs to
call a tool to retrieve it.

The content is `CrossRankSummary.model_dump()` serialized as a single JSON string — the
complete object with no fields omitted. This means the agent receives:

- `num_ranks`, `rank_ids`, `primary_rank_id`, `phase_alignment`
- The full `per_rank_overview` list — GPU kernel time, GPU idle time, MPI wait time, and GPU
  utilization for every rank across the whole profile
- The full `phases` list — for each phase:
  - `gpu_kernel_{mean,std,min,max}_s` and `gpu_kernel_imbalance` across all ranks
  - `gpu_kernel_slowest_rank_id`
  - `mpi_wait_{mean,std,min,max}_s` and `mpi_wait_imbalance` across all ranks
  - `mpi_wait_slowest_rank_id`
  - The complete `collective_imbalance` list sorted by imbalance score — every MPI operation
    with its `imbalance_score`, `slowest_rank_id`, `mean_s`, `min_s`, `max_s`
  - The complete `per_rank` list — raw `gpu_kernel_s`, `gpu_idle_s`, and `mpi_wait_s` for
    every individual rank within that phase

The agent is instructed not to call `cross_rank_summary` as a tool (since it is already in
context), and can reason directly over the pre-seeded data when generating hypotheses — for
example, citing the specific collective with the highest imbalance score or comparing a
particular rank's phase-level MPI wait against the mean without issuing any additional tool
calls.

On the Anthropic backend, the pre-seed block (system prompt + all three pre-seeded tool
results) is cached as a single unit with a permanent `cache_control: ephemeral` marker, so the
cross-rank data is billed at the reduced cache-read rate on every turn after the first.

---

## Further Reading

| Document | Relevant sections |
|---|---|
| [phase_detection.md](phase_detection.md) | The full algorithm that produces the per-rank phases aligned in Step 6; the `compute_phase_cost_curve` function used in Step 2 |
| [pipeline.md](pipeline.md) | Where cross-rank analysis sits in the Stage 1 / Stage 2 pipeline; multi-rank path description; pre-seeding of `CrossRankSummary` into Stage 2 |
| `analysis/cross_rank.py` | Implementation of all functions described in Steps 1–7 |
| `analysis/models.py` | Pydantic schema for `CrossRankSummary`, `CrossRankPhaseSummary`, `CollectiveImbalance`, `RankOverview`, `RankPhaseStats` |
| `__main__.py: cmd_analyze()` | CLI orchestration of the two-pass multi-rank path and `_print_cross_rank_tables()` |
