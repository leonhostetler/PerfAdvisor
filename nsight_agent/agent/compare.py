"""Single-shot LLM comparison agent for two Nsight Systems profiles.

Unlike the hypothesis agent (which uses a multi-turn tool-use loop), the
comparison agent is single-shot: both ProfileSummary objects and the
pre-computed ProfileDiff are serialized into one prompt and sent to the LLM,
which returns a structured ComparisonReport JSON.

Three comparison modes (determined by compute_profile_diff before this is called):
  phase_aware       — both summaries with full phase breakdown injected
  summary           — phases stripped; per-kernel diff included
  summary_no_kernel — phases stripped; per-kernel diff omitted; top-level only
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nsight_agent.analysis.models import ProfileDiff, ProfileSummary
from nsight_agent.agent.logger import LLMLogger

_COMPARE_SYSTEM_PROMPT = """\
You are a GPU performance engineer comparing two Nsight Systems profiles.
Your goal is to identify and describe the most significant differences in GPU behavior — \
utilization, kernel mix, memory bandwidth usage, MPI overhead, idle time — \
and characterize their magnitude.

Do not assume which profile is "better." State differences as factual observations \
grounded strictly in the provided data. A reader doing a before/after comparison should \
be able to infer what improved or worsened from the magnitudes and directions you report.

Output ONLY a JSON object (not an array, not wrapped in markdown fences) with this exact schema:
{
  "narrative": "2-4 sentence summary of the most important differences",
  "key_differences": [
    {
      "metric": "brief metric name",
      "profile_a": "formatted value for profile A",
      "profile_b": "formatted value for profile B",
      "magnitude_pct": <relative change as a float (positive = B larger), or null if not meaningful>,
      "note": "one sentence factual interpretation of this difference"
    }
  ]
}

Order key_differences by absolute magnitude (largest change first).
Include at least the top 5 differences and no more than 15.
"""

_MODE_DESCRIPTION: dict[str, str] = {
    "phase_aware": (
        "Both profiles have the same execution phase structure. "
        "Analyze differences within each matching phase, then summarize overall."
    ),
    "summary": (
        "The profiles have different phase structures so phase-level comparison is not possible. "
        "Analyze overall differences. Per-kernel diffs are included."
    ),
    "summary_no_kernel": (
        "The profiles have different phase structures and low kernel name overlap (<20%). "
        "Focus on top-level GPU behavior: utilization, total kernel time, memory bandwidth, "
        "idle time, and MPI overhead. Per-kernel comparison is not included."
    ),
}


def _build_prompt(
    summary_a: ProfileSummary,
    summary_b: ProfileSummary,
    diff: ProfileDiff,
) -> str:
    mode = diff.comparison_mode
    # Strip phases from summaries for non-phase-aware modes to keep the prompt compact
    if mode != "phase_aware":
        summary_a = summary_a.model_copy(update={"phases": []})
        summary_b = summary_b.model_copy(update={"phases": []})

    mode_desc = _MODE_DESCRIPTION[mode]
    return (
        f"Compare these two Nsight Systems profiles.\n\n"
        f"Profile A: {diff.profile_a_name}\n"
        f"Profile B: {diff.profile_b_name}\n\n"
        f"Comparison mode: {mode_desc}\n\n"
        f"## Profile A Summary\n"
        f"{summary_a.model_dump_json(indent=2)}\n\n"
        f"## Profile B Summary\n"
        f"{summary_b.model_dump_json(indent=2)}\n\n"
        f"## Pre-computed Structural Diff\n"
        f"{diff.model_dump_json(indent=2)}\n"
    )


def _extract_report(text: str) -> dict[str, Any]:
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"narrative": text, "key_differences": []}


# ---------------------------------------------------------------------------
# Provider backends (all single-shot, no tool-use loop)
# ---------------------------------------------------------------------------


def _call_anthropic(prompt: str, model: str) -> tuple[str, int, int]:
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_COMPARE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = ""
    for block in response.content:
        if hasattr(block, "text"):
            text = block.text
            break
    return text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(prompt: str, model: str) -> tuple[str, int, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required: pip install openai")
    client = OpenAI()
    # Try max_tokens first; fall back to max_completion_tokens for newer models
    _limit_param = "max_tokens"

    def _create() -> Any:
        nonlocal _limit_param
        from openai import BadRequestError

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=[
                {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            **{_limit_param: 4096},
        )
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if _limit_param == "max_tokens" and "max_completion_tokens" in str(e):
                _limit_param = "max_completion_tokens"
                kwargs[_limit_param] = kwargs.pop("max_tokens")
                return client.chat.completions.create(**kwargs)
            raise

    response = _create()
    text = response.choices[0].message.content or ""
    inp = response.usage.prompt_tokens if response.usage else 0
    out = response.usage.completion_tokens if response.usage else 0
    return text, inp, out


def _call_gemini(prompt: str, model: str) -> tuple[str, int, int]:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError("google-genai package required: pip install google-genai")
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set")
    client = genai.Client(api_key=api_key)
    config = genai_types.GenerateContentConfig(
        system_instruction=_COMPARE_SYSTEM_PROMPT,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = response.text or ""
    um = getattr(response, "usage_metadata", None)
    inp = getattr(um, "prompt_token_count", 0) or 0 if um else 0
    out = getattr(um, "candidates_token_count", 0) or 0 if um else 0
    return text, inp, out


def _call_claude_code(full_prompt: str) -> tuple[str, int, int, float | None]:
    result = subprocess.run(
        ["claude", "-p", full_prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    text = data.get("result", "")
    usage = data.get("usage", {})
    inp = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )
    out = usage.get("output_tokens") or 0
    cost_usd: float | None = data.get("total_cost_usd")
    return text, inp, out, cost_usd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_compare(
    profile_path_a: str | Path,
    profile_path_b: str | Path,
    *,
    summary_a: ProfileSummary,
    summary_b: ProfileSummary,
    diff: ProfileDiff,
    model: str | None = None,
    verbose: bool = True,
    token_usage: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> dict[str, Any]:
    """Compare two profiles using a single-shot LLM call.

    Callers are responsible for computing summaries and the diff before calling
    this function (cmd_compare does this so it can display phase tables and the
    mode status message before the LLM call starts).

    Returns the parsed ComparisonReport dict.
    """
    from nsight_agent.agent.loop import _parse_provider_and_model

    resolved_provider, resolved_model, _ = _parse_provider_and_model(model)

    prompt = _build_prompt(summary_a, summary_b, diff)

    if verbose:
        log(
            f"[local] Comparing {diff.profile_a_name} vs {diff.profile_b_name} "
            f"(provider={resolved_provider}, model={resolved_model})"
        )

    if logger:
        logger.write_request(
            1,
            {"system": _COMPARE_SYSTEM_PROMPT, "message": prompt},
        )

    cost_usd: float | None = None
    if resolved_provider == "anthropic":
        text, inp, out = _call_anthropic(prompt, resolved_model)
    elif resolved_provider == "openai":
        text, inp, out = _call_openai(prompt, resolved_model)
    elif resolved_provider == "gemini":
        text, inp, out = _call_gemini(prompt, resolved_model)
    else:
        text, inp, out, cost_usd = _call_claude_code(f"{_COMPARE_SYSTEM_PROMPT}\n\n{prompt}")

    if logger:
        logger.write_response(
            1,
            {
                "text": text,
                "usage": {"input_tokens": inp, "output_tokens": out},
                **({"total_cost_usd": cost_usd} if cost_usd is not None else {}),
            },
        )

    if verbose:
        from nsight_agent.agent.loop import _trunc

        log(f"[← llm] {_trunc(text, 200)}")

    if token_usage is not None:
        token_usage["input_tokens"] = inp
        token_usage["output_tokens"] = out
        if cost_usd is not None:
            token_usage["cost_usd"] = cost_usd

    return _extract_report(text)
