"""Scoring logic: bottleneck detection and LLM-judge suggestion coverage.

Two-tier scoring:
  1. Bottleneck detection — a hypothesis counts only if BOTH its
     ``bottleneck_type`` enum is in the expected set AND its text names a
     scenario-discriminating mechanism; deterministic. Reported rank-aware
     (detection@1 / detection@3 / MRR), not "any hypothesis anywhere".
  2. Suggestion coverage  — single-turn LLM judge; scores each expected
     suggestion 0 (absent) / 1 (partial) / 2 (full). The verdict is constrained
     by a JSON schema at decode time (``_JUDGE_SCHEMA``) on providers that
     support it, with a prompt-only fallback and a skip-not-zero safety net.

Why the conjunction: the advisor's ``bottleneck_type`` vocabulary (8 coarse
values) is strictly coarser than the benchmark's 8 scenarios, so several
scenarios share an enum — all three host-staging MPI scenarios are legitimately
``mpi_latency``. Scoring on the enum alone therefore cannot distinguish them,
and a profile-blind model that emits one hypothesis of every enum value scores
a perfect run. The enum is kept as a cheap sanity gate; the keyword carries the
discrimination. See ``tests/test_eval_scoring.py`` for the regression test that
holds this property.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Bottleneck detection
# ---------------------------------------------------------------------------

# Runs whose expected_bottleneck is in this set represent an *optimal* execution
# path — no deficiency was injected and no bottleneck detection is expected.
# They are still profiled and shown in the table but excluded from the detection
# accuracy numerator/denominator so they don't penalise the score.
# Currently empty: the benchmark has no optimal-path scenario. "p2p_direct_transfer"
# (scen_h) and "gpu_direct_transfer" (scen_n) were removed on 2026-07-20 along with
# the scenarios that produced them, leaving the suite measuring recall only — no
# captured profile is free of an injected deficiency, so nothing observes whether the
# advisor over-reports. The machinery below is deliberately retained rather than
# deleted: re-adding a clean-profile control means adding one label here and a run to
# an sbatch script. See todo_list.md and bench/README.md § Run Reference.
OPTIMAL_PATH_BOTTLENECKS: frozenset[str] = frozenset()

# Maps expected_bottleneck (ground-truth JSON) → advisor bottleneck_type enum values.
# Intentionally loose: some advisor types map to multiple expected labels because
# the advisor's vocabulary is coarser than the benchmark's.
_BOTTLENECK_ENUM_MAP: dict[str, set[str]] = {
    "kernel_launch_overhead": {"cpu_launch_overhead"},
    "cpu_sync_stall": {"synchronization"},
    "pcie_transfer_bound": {"memory_bound", "io"},
    "cpu_gpu_overlap_missing": {"synchronization", "io"},
    "unnecessary_host_staging_intranode": {"mpi_latency"},
    "mpi_load_imbalance": {"mpi_imbalance"},
    "host_staged_collective": {"mpi_latency"},
    "host_staged_halo_exchange": {"mpi_latency"},
}

# Scenario DISCRIMINATORS, checked against lowercased
# (description + evidence + suggestion) text. Required *in addition to* the enum
# match above — see the module docstring.
#
# These are deliberately NOT domain vocabulary. Terms like "memcpy", "h2d",
# "kernel launch", or "p2p" describe the benchmark's subject matter and appear in
# almost any hypothesis about these profiles, so they award credit without
# evidence that the advisor read the profile. Only terms that name the specific
# pathology or its specific fix belong here.
#
# The three host-staging MPI scenarios (p2p_staged / mpi_allreduce /
# mpi_halo_exchange) all map to the ``mpi_latency`` enum, so their keyword sets
# are kept mutually disjoint — that disjointness is the only thing separating
# them. Shared fix vocabulary ("cuda-aware", "host staging", "device pointer") is
# deliberately absent from all three: an advisor that recommends CUDA-aware MPI
# for every MPI profile has not demonstrated it distinguished them.
_BOTTLENECK_KEYWORDS: dict[str, list[str]] = {
    "kernel_launch_overhead": [
        "launch overhead",
        "launch latency",
        "launch-bound",
        "launch gap",
        "frequent launch",
        "tiny kernel",
        "small kernel",
        "cuda graph",
        "cudagraph",
        "hip graph",          # AMD HIP equivalent of CUDA Graphs
        "hipgraph",
        "kernel fusion",
        "fuse kernel",
        "fusing kernel",
        "persistent kernel",
    ],
    "cpu_sync_stall": [
        "sync stall",
        "cudastreamsynchronize",
        "hipstreamsynchronize",   # AMD HIP equivalent
        "cudadevicesynchronize",
        "hipdevicesynchronize",
        "stream synchronize",
        "device synchronize",
        "blocking sync",
        "synchronize after every",
        "synchronizes after every",
        "sync after each",
        "synchronize after each",
    ],
    "pcie_transfer_bound": [
        "pcie",
        "transfer-bound",
        "transfer bound",
        "transfer dominated",
        "transfer-dominated",
        "bandwidth-limited",
        "bandwidth limited",
        "arithmetic intensity",
        "gpu-resident",
        "gpu resident",
        "device-resident",
        "device resident",
        "round-trip",
        "round trip",
    ],
    "cpu_gpu_overlap_missing": [
        "overlap missing",
        "missing overlap",
        "no overlap",
        "lack of overlap",
        "not overlapped",
        "transfer-compute overlap",
        "compute overlap",
        "copy/compute",
        "copy-compute",
        "async memcpy",
        "asynchronous memcpy",
        "cudamemcpyasync",
        "hipmemcpyasync",
        "serialized transfer",
        "double buffer",
        "double-buffer",
        "transfer stream",
        "separate stream",
    ],
    # --- the three mpi_latency scenarios; keyword sets must stay disjoint ---
    #
    # p2p_staged (test_05) and mpi_halo_exchange (test_08) are the *same workload* —
    # a ring MPI_Sendrecv of 64 MB halo buffers, host-staged — differing only in
    # whether the ranks share a node. So "halo" and "sendrecv" describe both and
    # discriminate neither; the only real signal is node locality, which is what
    # these two lists encode. "mpich_gpu_support_enabled" is likewise the fix for
    # all three and so belongs to none of them.
    "unnecessary_host_staging_intranode": [
        "nvlink",
        "xgmi",               # AMD Infinity Fabric inter-GCD link
        "infinity fabric",
        "peer access",
        "peer-to-peer",
        "peer to peer",
        "cudaipc",
        "cuda ipc",
        "intra-node",
        "intranode",
        "same node",
    ],
    "host_staged_collective": [
        "allreduce",
        "mpi_allreduce",
        "nccl",
        "rccl",               # AMD equivalent of NCCL
        "collective",
    ],
    "host_staged_halo_exchange": [
        "rdma",
        "gpu-direct",
        "gpudirect",
        "gdrcopy",            # GPU-Direct RDMA copy path on both NVIDIA and AMD
        "inter-node",
        "internode",
        "across nodes",
        "cross-node",
    ],
    # mpi_imbalance is the only scenario using that enum, so these may be looser.
    "mpi_load_imbalance": [
        "load imbalance",
        "load balance",
        "imbalance",
        "imbalanced",
        "rank imbalance",
        "uneven work",
        "straggler",
        "barrier",
        "mpi_barrier",
        "barrier wait",
        "barrier stall",
    ],
}


# A plausible-sounding advisor that never opens a profile. Kept here rather than
# only in the tests so the report can print the floor it scores on the very runs
# being reported: if a real model isn't clearly above this line, the numbers are
# not measuring analysis. See tests/test_eval_scoring.py.
PROFILE_BLIND_BASELINE: list[dict[str, str]] = [
    {"bottleneck_type": t, "description": d, "evidence": "", "suggestion": ""}
    for t, d in (
        ("cpu_launch_overhead", "many small kernel launches dominate the timeline"),
        ("synchronization", "the host blocks on stream synchronization"),
        ("memory_bound", "kernels appear limited by memory bandwidth"),
        ("mpi_latency", "MPI communication overhead is significant"),
        ("io", "host-device data transfers take substantial time"),
        ("compute_bound", "kernels are compute bound"),
        ("other", "consider general tuning"),
    )
]


def baseline_detection_at_k(expected: list[str], k: int) -> tuple[int, int]:
    """(hits, total) the profile-blind baseline scores on these expected labels."""
    hits = 0
    for e in expected:
        detected, _, idx = score_bottleneck(PROFILE_BLIND_BASELINE, e)
        if detected and idx is not None and idx < k:
            hits += 1
    return hits, len(expected)


def _hypothesis_text(h: dict[str, Any]) -> str:
    """Combine the most informative fields into a single lowercase text blob."""
    return " ".join(str(h.get(k, "")) for k in ("description", "evidence", "suggestion")).lower()


def _hypothesis_matches(h: dict[str, Any], expected_bottleneck: str) -> bool:
    """True when a hypothesis identifies ``expected_bottleneck``.

    Requires BOTH the coarse enum gate and a scenario-discriminating keyword.
    Either alone is insufficient: the enum cannot separate scenarios that share
    a ``bottleneck_type``, and a bare keyword hit can come from a passing
    mention inside an otherwise unrelated diagnosis.
    """
    expected_enum_values = _BOTTLENECK_ENUM_MAP.get(expected_bottleneck, set())
    if h.get("bottleneck_type", "") not in expected_enum_values:
        return False
    keywords = _BOTTLENECK_KEYWORDS.get(expected_bottleneck, [])
    text = _hypothesis_text(h)
    return any(kw in text for kw in keywords)


def score_bottleneck(
    hypotheses: list[dict[str, Any]],
    expected_bottleneck: str,
) -> tuple[bool, str | None, int | None]:
    """Return (detected, match_type, hypothesis_index).

    ``hypothesis_index`` is the 0-based rank of the highest-ranked matching
    hypothesis, which is what the rank-aware aggregates below consume — a
    correct diagnosis buried at position 5 is not the same result as one at
    position 0. ``match_type`` is "enum+keyword" or None.
    """
    for i, h in enumerate(hypotheses):
        if _hypothesis_matches(h, expected_bottleneck):
            return True, "enum+keyword", i
    return False, None, None


def matches_also_true(h: dict[str, Any], also_true: list[dict[str, Any]] | None) -> bool:
    """True when a hypothesis reports a known-true secondary observation.

    ``also_true`` entries come from ``ground_truth_meta.json`` and describe things
    that are factually correct about the captured profile but are not the injected
    deficiency — e.g. the harness ``barrier_sync()`` genuinely dominating the MPI
    profiles. Matching uses the same enum-AND-keyword conjunction as detection, so
    an entry cannot degrade into a blanket excuse for anything vaguely related.
    """
    if not also_true:
        return False
    btype = h.get("bottleneck_type", "")
    text = _hypothesis_text(h)
    for entry in also_true:
        types = entry.get("bottleneck_type") or []
        keywords = entry.get("keywords") or []
        if btype in types and any(kw in text for kw in keywords):
            return True
    return False


def false_positive_count(
    hypotheses: list[dict[str, Any]],
    expected_bottleneck: str,
    also_true: list[dict[str, Any]] | None = None,
) -> int:
    """Count hypotheses that identify neither the expected bottleneck nor a
    known-true secondary observation.

    Uses the same predicate as detection for the primary, so the two metrics stay
    consistent. Without ``also_true`` this metric conflates two opposite
    behaviours — inventing an irrelevant problem, and correctly spotting a second
    real one — which are exactly what a benchmark should be able to tell apart.
    Detection is unaffected: an ``also_true`` match never counts as finding the
    injected bottleneck.
    """
    return sum(
        1
        for h in hypotheses
        if not _hypothesis_matches(h, expected_bottleneck)
        and not matches_also_true(h, also_true)
    )


def secondary_true_count(
    hypotheses: list[dict[str, Any]],
    expected_bottleneck: str,
    also_true: list[dict[str, Any]] | None = None,
) -> int:
    """Count hypotheses excused as known-true secondary observations.

    Reported alongside the false-positive count so the exclusion is visible rather
    than silently shrinking the number.
    """
    return sum(
        1
        for h in hypotheses
        if not _hypothesis_matches(h, expected_bottleneck)
        and matches_also_true(h, also_true)
    )


# ---------------------------------------------------------------------------
# LLM judge for suggestion coverage
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are scoring whether a GPU performance advisor covered a specific expected recommendation.

Expected recommendation:
  Action: {action}
  Mechanism (specific API or technique): {mechanism}
  Rationale: {rationale}

Performance advisor output — all hypotheses:
{hypotheses_text}

Does the advisor's output cover this expected recommendation?

  0 = Not covered — the recommendation is absent, too vague, or contradicted
  1 = Partially covered — correct direction but the specific mechanism is missing or wrong
  2 = Fully covered — both the action and the specific mechanism are clearly present

Respond with ONLY a single-line JSON object, nothing else:
{{"score": 0|1|2, "explanation": "one sentence explaining the score"}}\
"""

# Constrains the judge's reply to a valid verdict at the decoding level, so a
# malformed reply becomes impossible rather than merely handled. The prompt above
# still describes the shape: it steers the wording, and it is the only thing
# holding the format together on the fallback path below.
#
# ``additionalProperties: false`` plus every property listed in ``required`` is
# mandatory for both providers' strict modes.
_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "enum": [0, 1, 2],
            "description": "0 = not covered, 1 = partially covered, 2 = fully covered",
        },
        "explanation": {
            "type": "string",
            "description": "One sentence explaining the score.",
        },
    },
    "required": ["score", "explanation"],
    "additionalProperties": False,
}

# Emitted once if a judge model turns out not to support structured outputs, so a
# silent capability downgrade doesn't go unnoticed.
_STRUCTURED_OUTPUT_WARNED: set[str] = set()


def _warn_unstructured(provider: str, model: str, exc: Exception) -> None:
    key = f"{provider}:{model}"
    if key in _STRUCTURED_OUTPUT_WARNED:
        return
    _STRUCTURED_OUTPUT_WARNED.add(key)
    warnings.warn(
        f"Judge model {key} rejected structured outputs ({exc}); falling back to "
        f"prompt-only JSON for this run. Verdicts are no longer format-guaranteed, "
        f"so malformed replies will be skipped rather than scored.",
        RuntimeWarning,
        stacklevel=3,
    )


def _format_hypotheses_for_judge(hypotheses: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, h in enumerate(hypotheses, 1):
        lines.append(f"Hypothesis {i}:")
        lines.append(f"  bottleneck_type: {h.get('bottleneck_type', '—')}")
        lines.append(f"  description:     {h.get('description', '—')}")
        lines.append(f"  evidence:        {h.get('evidence', '—')}")
        lines.append(f"  suggestion:      {h.get('suggestion', '—')}")
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _judge_anthropic(prompt: str, model: str) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    # A verdict is ~60 tokens. The headroom is for the unstructured fallback path,
    # where nothing constrains the reply: Opus-tier models run without thinking
    # unless asked and can then write reasoning into the visible response, which a
    # 256-token cap would truncate mid-JSON. Unused headroom costs nothing —
    # max_tokens is a ceiling, not a reservation.
    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # `output_config` is passed through `extra_body` rather than as a typed
    # keyword: it is only a named parameter from anthropic ~0.7x onward, and this
    # project is run against older SDKs too (0.50.0 in the `ai` conda env). The
    # server accepts the field either way, so extra_body is version-stable.
    try:
        msg = client.messages.create(
            **kwargs,
            extra_body={"output_config": {"format": {
                "type": "json_schema",
                "schema": _JUDGE_SCHEMA,
            }}},
        )
    except anthropic.BadRequestError as exc:
        # Structured outputs are model-gated (Haiku 4.5, Opus 4.8, Sonnet 5,
        # Fable 5, and a couple of legacy Opus builds). --judge-model accepts any
        # model, so fall back rather than failing the run.
        _warn_unstructured("anthropic", model, exc)
        msg = client.messages.create(**kwargs)

    return json.loads(_strip_fences(msg.content[0].text))


def _judge_openai(prompt: str, model: str) -> dict[str, Any]:
    import openai

    client = openai.OpenAI()
    # Responses API: gpt-5.x / o-series reasoning models need room for reasoning
    # tokens before the JSON answer, so give them a much larger output budget
    # than the small non-reasoning cap.
    reasoning = model.lower().startswith(("gpt-5", "o1", "o3", "o4"))
    kwargs: dict[str, Any] = dict(
        model=model,
        instructions="Respond only with the requested JSON. No prose, no markdown fences.",
        input=prompt,
        max_output_tokens=4096 if reasoning else 256,
    )
    if reasoning:
        kwargs["reasoning"] = {"effort": "medium"}

    try:
        resp = client.responses.create(
            **kwargs,
            text={"format": {
                "type": "json_schema",
                "name": "judge_verdict",
                "schema": _JUDGE_SCHEMA,
                "strict": True,
            }},
        )
    except openai.BadRequestError as exc:
        _warn_unstructured("openai", model, exc)
        resp = client.responses.create(**kwargs)

    return json.loads(_strip_fences(resp.output_text or ""))


def _coerce_judge_score(result: dict[str, Any]) -> tuple[int, str | None]:
    """Extract the judge's 0/1/2 verdict from its parsed JSON reply.

    Returns ``(score, error)``. A reply that parsed as JSON but does not carry a
    valid verdict — missing key, non-numeric, or outside the rubric's 0–2 range —
    yields ``(-1, reason)`` so it is *skipped*, not silently recorded as 0.
    ``json.loads`` succeeding on the wrong shape raises no exception, so without
    this the ``.get("score", 0)`` default would score a malformed judge reply as
    "not covered" and deflate the model under test.

    Structured outputs (``_JUDGE_SCHEMA``) should make this unreachable on the
    happy path. It stays as the safety net for the unstructured fallback taken
    when a judge model doesn't support them.
    """
    if "score" not in result:
        return -1, f"judge reply carried no 'score' key: {result!r}"
    raw = result["score"]
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        return -1, f"judge score was not numeric: {raw!r}"
    try:
        value = int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return -1, f"judge score was not numeric: {raw!r}"
    if value not in (0, 1, 2):
        return -1, f"judge score {value} outside the 0-2 rubric"
    return value, None


@dataclass
class SuggestionScore:
    action: str
    mechanism: str
    score: int  # 0 = not covered, 1 = partial, 2 = full, -1 = skipped/error
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "mechanism": self.mechanism,
            "score": self.score,
            "explanation": self.explanation,
        }


def judge_suggestion(
    hypotheses: list[dict[str, Any]],
    suggestion: dict[str, Any],
    judge_model: str,
    judge_provider: str,
) -> SuggestionScore:
    """Score one expected suggestion against the full advisor hypothesis list."""
    prompt = _JUDGE_PROMPT.format(
        action=suggestion.get("action", ""),
        mechanism=suggestion.get("mechanism", ""),
        rationale=suggestion.get("rationale", ""),
        hypotheses_text=_format_hypotheses_for_judge(hypotheses),
    )

    try:
        if judge_provider == "anthropic":
            result = _judge_anthropic(prompt, judge_model)
        elif judge_provider == "openai":
            result = _judge_openai(prompt, judge_model)
        else:
            return SuggestionScore(
                action=suggestion.get("action", ""),
                mechanism=suggestion.get("mechanism", ""),
                score=-1,
                explanation=f"Judge not implemented for provider '{judge_provider}'; skipped.",
            )
        score, coercion_error = _coerce_judge_score(result)
        explanation = coercion_error or str(result.get("explanation", ""))
    except Exception as exc:
        score = -1
        explanation = f"Judge call failed: {exc}"

    return SuggestionScore(
        action=suggestion.get("action", ""),
        mechanism=suggestion.get("mechanism", ""),
        score=score,
        explanation=explanation,
    )


def suggestion_coverage_pct(scores: list[SuggestionScore]) -> float | None:
    """Return coverage as a percentage: sum(scores) / (2 × N) × 100.

    Skipped/errored scores (score == -1) are excluded from both numerator and
    denominator so a partial judge failure doesn't deflate the result.

    Returns ``None`` — not ``0.0`` — when nothing was judged (judge skipped, or
    every call failed). "No measurement" and "covered nothing" are different
    results, and collapsing them makes a model whose judge calls all errored
    look like one that recommended nothing useful.
    """
    valid = [s for s in scores if s.score >= 0]
    if not valid:
        return None
    return 100.0 * sum(s.score for s in valid) / (2 * len(valid))


# ---------------------------------------------------------------------------
# Run-level result
# ---------------------------------------------------------------------------


@dataclass
class RunScore:
    run_id: str
    scenario: str
    expected_bottleneck: str
    sqlite_paths: list[str]
    hypotheses: list[dict[str, Any]]
    # Detection
    bottleneck_detected: bool
    match_type: str | None  # "enum" | "keyword" | None
    matched_hypothesis_idx: int | None
    # Suggestions
    suggestion_scores: list[SuggestionScore] = field(default_factory=list)
    coverage_pct: float | None = None  # 0–100, or None when nothing was judged
    # False positives
    false_positive_count: int = 0
    # Hypotheses excused as known-true secondary observations (see also_true)
    secondary_true_count: int = 0
    # Timing
    elapsed_s: float = 0.0
    # Token accounting for the hypothesis-generation run (None for the claude_code
    # fallback, which does not report usage). Recorded per run because it cannot be
    # recovered later without paying for another eval pass.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_creation_tokens: int | None = None
    cache_read_tokens: int | None = None
    # Which repetition this is, when --repeats > 1. The agent loop is stochastic,
    # so a single sample per (model, scenario) cannot separate a real difference
    # between models from run-to-run noise.
    repeat: int = 0
    # Optimal-path flag: True when no bottleneck was injected (e.g. GPU-Direct path).
    # These runs are shown in the table but excluded from detection accuracy tallies.
    is_optimal_path: bool = False
    # Error (set when PerfAdvisor itself failed)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario": self.scenario,
            "expected_bottleneck": self.expected_bottleneck,
            "sqlite_paths": self.sqlite_paths,
            "hypotheses": self.hypotheses,
            "bottleneck_detected": self.bottleneck_detected,
            "match_type": self.match_type,
            "matched_hypothesis_idx": self.matched_hypothesis_idx,
            "suggestion_scores": [s.to_dict() for s in self.suggestion_scores],
            "coverage_pct": self.coverage_pct,
            "false_positive_count": self.false_positive_count,
            "secondary_true_count": self.secondary_true_count,
            "elapsed_s": self.elapsed_s,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "repeat": self.repeat,
            "is_optimal_path": self.is_optimal_path,
            "error": self.error,
        }

    @property
    def false_positive_rate(self) -> float | None:
        """FP as a fraction of hypotheses emitted, or None if none were.

        The raw count is not comparable across models: one emitting 4 hypotheses
        and one emitting 10 are not on the same scale, and a model that errors out
        early scores a flatteringly low count. The rate is what belongs in a
        cross-model table.
        """
        if not self.hypotheses:
            return None
        return self.false_positive_count / len(self.hypotheses)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunScore:
        """Reconstruct from a saved output JSON (for --cached / --rescore)."""
        scores = [
            SuggestionScore(
                action=s["action"],
                mechanism=s["mechanism"],
                score=s["score"],
                explanation=s["explanation"],
            )
            for s in d.get("suggestion_scores", [])
        ]
        return cls(
            run_id=d["run_id"],
            scenario=d["scenario"],
            expected_bottleneck=d["expected_bottleneck"],
            sqlite_paths=d.get("sqlite_paths", []),
            hypotheses=d.get("hypotheses", []),
            bottleneck_detected=d.get("bottleneck_detected", False),
            match_type=d.get("match_type"),
            matched_hypothesis_idx=d.get("matched_hypothesis_idx"),
            suggestion_scores=scores,
            coverage_pct=d.get("coverage_pct"),
            false_positive_count=d.get("false_positive_count", 0),
            secondary_true_count=d.get("secondary_true_count", 0),
            elapsed_s=d.get("elapsed_s", 0.0),
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            cache_creation_tokens=d.get("cache_creation_tokens"),
            cache_read_tokens=d.get("cache_read_tokens"),
            repeat=d.get("repeat", 0),
            is_optimal_path=d.get("is_optimal_path", False),
            error=d.get("error"),
        )


def scorable_runs(results: list[RunScore]) -> list[RunScore]:
    """Runs that count toward detection accuracy: completed, non-optimal-path."""
    return [r for r in results if not r.error and not r.is_optimal_path]


def detection_at_k(results: list[RunScore], k: int) -> tuple[int, int]:
    """Return (hits, total) where a hit is a match at rank < k.

    detection@1 asks "was the top hypothesis right"; the old any-hypothesis-
    anywhere tally is detection@∞, which a model can inflate by emitting one
    hypothesis of every kind.
    """
    scorable = scorable_runs(results)
    hits = sum(
        1
        for r in scorable
        if r.bottleneck_detected
        and r.matched_hypothesis_idx is not None
        and r.matched_hypothesis_idx < k
    )
    return hits, len(scorable)


def mean_reciprocal_rank(results: list[RunScore]) -> float | None:
    """Mean of 1/(rank+1) over scorable runs; misses contribute 0.

    Single number combining hit rate and ranking quality: 1.0 = always first,
    0.5 = always second, 0.0 = never found. None when there is nothing to score.
    """
    scorable = scorable_runs(results)
    if not scorable:
        return None
    total = 0.0
    for r in scorable:
        if r.bottleneck_detected and r.matched_hypothesis_idx is not None:
            total += 1.0 / (r.matched_hypothesis_idx + 1)
    return total / len(scorable)


def repeat_indices(results: list[RunScore]) -> list[int]:
    """Distinct repetition indices present in a result set (sorted)."""
    return sorted({r.repeat for r in results})


def detection_by_repeat(results: list[RunScore], k: int) -> list[tuple[int, int]]:
    """(hits, total) for detection@k, computed separately per repetition.

    Kept per-repeat rather than pooled: pooling 8 scenarios x N repeats into one
    proportion treats every observation as independent, but repeats of the same
    scenario are correlated and scenarios differ in difficulty. One observation
    per repetition is the honest unit for a spread.
    """
    return [
        detection_at_k([r for r in results if r.repeat == i], k)
        for i in repeat_indices(results)
    ]


def mrr_by_repeat(results: list[RunScore]) -> list[float]:
    """Mean reciprocal rank computed separately per repetition."""
    out = []
    for i in repeat_indices(results):
        v = mean_reciprocal_rank([r for r in results if r.repeat == i])
        if v is not None:
            out.append(v)
    return out


def detection_stability(results: list[RunScore]) -> list[tuple[str, str, int, int]]:
    """Per-scenario detection counts across repetitions.

    Returns ``(run_id, scenario, hits, observations)``. ``observations`` counts
    only non-errored repeats for that run, so a scenario whose 3rd repeat crashed
    reports k/4 rather than silently scoring the crash as a miss.

    This is the most informative view when repeats > 1: an aggregate hides the
    difference between "found it every time", "found it sometimes" and "never
    found it", which are three different capability claims.
    """
    order: list[tuple[str, str]] = []
    hits: dict[tuple[str, str], int] = {}
    obs: dict[tuple[str, str], int] = {}
    for r in results:
        if r.error or r.is_optimal_path:
            continue
        key = (r.run_id, r.scenario)
        if key not in obs:
            order.append(key)
            hits[key] = 0
            obs[key] = 0
        obs[key] += 1
        if r.bottleneck_detected:
            hits[key] += 1
    return [(rid, scen, hits[(rid, scen)], obs[(rid, scen)]) for rid, scen in order]


def score_run(
    run_id: str,
    gt_runtime: dict[str, Any],
    gt_meta: dict[str, Any] | None,
    hypotheses: list[dict[str, Any]],
    sqlite_paths: list[str],
    judge_model: str,
    judge_provider: str,
    skip_judge: bool = False,
    elapsed_s: float = 0.0,
    token_usage: dict[str, int | None] | None = None,
    repeat: int = 0,
) -> RunScore:
    """Score one run's hypotheses against ground truth."""
    scenario = gt_runtime.get("scenario", "unknown")
    expected_bottleneck = gt_runtime.get("expected_bottleneck", "unknown")
    is_optimal = expected_bottleneck in OPTIMAL_PATH_BOTTLENECKS

    detected, match_type, matched_idx = score_bottleneck(hypotheses, expected_bottleneck)
    also_true = (gt_meta or {}).get("also_true")
    fp_count = false_positive_count(hypotheses, expected_bottleneck, also_true)
    secondary = secondary_true_count(hypotheses, expected_bottleneck, also_true)

    suggestion_scores: list[SuggestionScore] = []
    if gt_meta and not skip_judge:
        for sugg in gt_meta.get("suggestions", []):
            ss = judge_suggestion(hypotheses, sugg, judge_model, judge_provider)
            suggestion_scores.append(ss)

    cov = suggestion_coverage_pct(suggestion_scores)

    return RunScore(
        run_id=run_id,
        scenario=scenario,
        expected_bottleneck=expected_bottleneck,
        sqlite_paths=sqlite_paths,
        hypotheses=hypotheses,
        bottleneck_detected=detected,
        match_type=match_type,
        matched_hypothesis_idx=matched_idx,
        suggestion_scores=suggestion_scores,
        coverage_pct=cov,
        false_positive_count=fp_count,
        secondary_true_count=secondary,
        elapsed_s=elapsed_s,
        input_tokens=(token_usage or {}).get("input_tokens"),
        output_tokens=(token_usage or {}).get("output_tokens"),
        cache_creation_tokens=(token_usage or {}).get("cache_creation_tokens"),
        cache_read_tokens=(token_usage or {}).get("cache_read_tokens"),
        repeat=repeat,
        is_optimal_path=is_optimal,
    )
