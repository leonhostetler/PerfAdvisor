"""Scoring logic: bottleneck detection and LLM-judge suggestion coverage.

Two-tier scoring:
  1. Bottleneck detection — enum match then keyword fallback; deterministic.
  2. Suggestion coverage  — single-turn LLM judge; scores each expected
     suggestion 0 (absent) / 1 (partial) / 2 (full).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Bottleneck detection
# ---------------------------------------------------------------------------

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

# Keywords checked against lowercased (description + evidence + suggestion) text.
# Used as a fallback when the bottleneck_type enum doesn't match.
_BOTTLENECK_KEYWORDS: dict[str, list[str]] = {
    "kernel_launch_overhead": [
        "launch overhead",
        "launch latency",
        "tiny kernel",
        "cuda graph",
        "kernel launch",
        "launch gap",
        "frequent launch",
    ],
    "cpu_sync_stall": [
        "sync stall",
        "cudastreamsynchronize",
        "stream synchronize",
        "blocking sync",
        "cpu block",
        "cpu wait",
        "synchronize after every",
        "sync after each",
    ],
    "pcie_transfer_bound": [
        "pcie",
        "transfer-bound",
        "transfer bound",
        "memcpy",
        "h2d",
        "d2h",
        "host-to-device",
        "device-to-host",
        "data transfer",
        "bandwidth-limited",
        "transfer dominated",
    ],
    "cpu_gpu_overlap_missing": [
        "overlap missing",
        "no overlap",
        "missing overlap",
        "pipeline",
        "concurrent transfer",
        "async memcpy",
        "serialized transfer",
        "transfer-compute overlap",
    ],
    "unnecessary_host_staging_intranode": [
        "host staging",
        "host-staged",
        "cuda-aware",
        "device pointer",
        "intra-node",
        "intranode",
        "p2p",
        "nvlink",
        "unnecessary copy",
        "host copy",
        "stage to host",
    ],
    "mpi_load_imbalance": [
        "load imbalance",
        "barrier stall",
        "load balance",
        "imbalanced",
        "mpi_barrier",
        "rank imbalance",
        "uneven work",
        "barrier wait",
    ],
    "host_staged_collective": [
        "allreduce",
        "host-staged",
        "host staged",
        "nccl",
        "collective",
        "mpi_allreduce",
        "device pointer",
        "stage.*allreduce",
    ],
    "host_staged_halo_exchange": [
        "halo",
        "host-staged",
        "host staged",
        "gpu-direct",
        "rdma",
        "sendrecv",
        "mpi_sendrecv",
        "inter-node",
        "internode",
        "staging halo",
    ],
}


def _hypothesis_text(h: dict[str, Any]) -> str:
    """Combine the most informative fields into a single lowercase text blob."""
    return " ".join(str(h.get(k, "")) for k in ("description", "evidence", "suggestion")).lower()


def score_bottleneck(
    hypotheses: list[dict[str, Any]],
    expected_bottleneck: str,
) -> tuple[bool, str | None, int | None]:
    """Return (detected, match_type, hypothesis_index).

    match_type is "enum" (exact type match), "keyword" (semantic text match),
    or None (not detected).
    """
    expected_enum_values = _BOTTLENECK_ENUM_MAP.get(expected_bottleneck, set())
    keywords = _BOTTLENECK_KEYWORDS.get(expected_bottleneck, [])

    # Pass 1: exact bottleneck_type enum match
    for i, h in enumerate(hypotheses):
        if h.get("bottleneck_type", "") in expected_enum_values:
            return True, "enum", i

    # Pass 2: keyword match across description + evidence + suggestion
    for i, h in enumerate(hypotheses):
        text = _hypothesis_text(h)
        if any(kw in text for kw in keywords):
            return True, "keyword", i

    return False, None, None


def false_positive_count(
    hypotheses: list[dict[str, Any]],
    expected_bottleneck: str,
) -> int:
    """Count hypotheses that address neither the expected bottleneck enum type
    nor any of its keywords — i.e., appear unrelated to the ground truth."""
    expected_enum_values = _BOTTLENECK_ENUM_MAP.get(expected_bottleneck, set())
    keywords = _BOTTLENECK_KEYWORDS.get(expected_bottleneck, [])

    count = 0
    for h in hypotheses:
        enum_ok = h.get("bottleneck_type", "") in expected_enum_values
        text = _hypothesis_text(h)
        keyword_ok = any(kw in text for kw in keywords)
        if not enum_ok and not keyword_ok:
            count += 1
    return count


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
    msg = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(_strip_fences(msg.content[0].text))


def _judge_openai(prompt: str, model: str) -> dict[str, Any]:
    import openai

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(_strip_fences(resp.choices[0].message.content))


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
        score = max(0, min(2, int(result.get("score", 0))))
        explanation = str(result.get("explanation", ""))
    except Exception as exc:
        score = -1
        explanation = f"Judge call failed: {exc}"

    return SuggestionScore(
        action=suggestion.get("action", ""),
        mechanism=suggestion.get("mechanism", ""),
        score=score,
        explanation=explanation,
    )


def suggestion_coverage_pct(scores: list[SuggestionScore]) -> float:
    """Return coverage as a percentage: sum(scores) / (2 × N) × 100.

    Skipped/errored scores (score == -1) are excluded from both numerator and
    denominator so a partial judge failure doesn't deflate the result.
    """
    valid = [s for s in scores if s.score >= 0]
    if not valid:
        return 0.0
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
    coverage_pct: float = 0.0  # 0–100
    # False positives
    false_positive_count: int = 0
    # Timing
    elapsed_s: float = 0.0
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
            "elapsed_s": self.elapsed_s,
            "error": self.error,
        }

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
            coverage_pct=d.get("coverage_pct", 0.0),
            false_positive_count=d.get("false_positive_count", 0),
            elapsed_s=d.get("elapsed_s", 0.0),
            error=d.get("error"),
        )


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
) -> RunScore:
    """Score one run's hypotheses against ground truth."""
    scenario = gt_runtime.get("scenario", "unknown")
    expected_bottleneck = gt_runtime.get("expected_bottleneck", "unknown")

    detected, match_type, matched_idx = score_bottleneck(hypotheses, expected_bottleneck)
    fp_count = false_positive_count(hypotheses, expected_bottleneck)

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
        elapsed_s=elapsed_s,
    )
