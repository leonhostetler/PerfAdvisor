"""Agent loop: drives Claude to analyze a profile and produce hypotheses.

Two backends are supported:
  - API backend (default when ANTHROPIC_API_KEY is set): multi-turn tool-use loop
    via the Anthropic SDK. Claude can issue follow-up SQL queries and drill down.
  - Claude Code backend (fallback): pre-computes the full ProfileSummary and sends
    it in a single prompt to `claude -p` via subprocess. No API key required.

Both backends save the prompt and raw response to files next to the profile:
  {profile_stem}_{timestamp}_prompt.txt
  {profile_stem}_{timestamp}_response.txt
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from nsight_agent.analysis.metrics import compute_profile_summary
from nsight_agent.analysis.models import ProfileSummary
from nsight_agent.agent.tools import dispatch, tool_schemas
from nsight_agent.ingestion.profile import NsysProfile

MODEL = "claude-opus-4-6"
MAX_TURNS = 20

_HYPOTHESIS_SCHEMA = """\
Each hypothesis object must have these fields:
  - bottleneck_type: one of [compute_bound, memory_bound, mpi_latency, mpi_imbalance,
                              cpu_launch_overhead, synchronization, io, other]
  - description: concise plain-English description of the bottleneck
  - evidence: specific numbers from the profile that support this hypothesis
  - suggestion: concrete, actionable recommendation
  - expected_impact: estimated relative improvement (high / medium / low)\
"""

_SYSTEM_PROMPT_API = f"""You are an expert GPU performance engineer analyzing an NVIDIA Nsight Systems profile.

Your goal is to identify the most significant performance bottlenecks and produce a ranked list of
actionable hypotheses.

{_HYPOTHESIS_SCHEMA}

Start by calling profile_summary to orient yourself, then use the other tools as needed.
Work systematically: time budget → kernels → memory → MPI (if present) → NVTX context → idle gaps.
Use sql_query for any targeted follow-up that the structured tools don't cover.

When you have gathered enough evidence, output your final answer as a JSON array of hypothesis
objects (not wrapped in markdown fences) and nothing else after it.
"""


def _format_summary_prompt(summary_json: str, profile_name: str) -> str:
    return f"""\
You are an expert GPU performance engineer. Analyze the following Nsight Systems profile summary
for '{profile_name}' and produce a ranked list of actionable performance hypotheses.

{_HYPOTHESIS_SCHEMA}

Output ONLY a JSON array of hypothesis objects — no prose, no markdown fences.

## Profile Summary (JSON)

{summary_json}
"""


def _extract_hypotheses(text: str) -> list[dict[str, Any]]:
    """Extract a JSON array of hypotheses from a text response."""
    text = text.strip()
    start = text.rfind("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return []


def _save_files(
    profile_path: Path,
    prompt: str,
    response: str,
    verbose: bool,
) -> tuple[Path, Path]:
    """Save prompt and response to files next to the profile. Returns (prompt_path, response_path)."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = profile_path.stem
    out_dir = profile_path.parent

    prompt_path = out_dir / f"{stem}_{ts}_prompt.txt"
    response_path = out_dir / f"{stem}_{ts}_response.txt"

    prompt_path.write_text(prompt, encoding="utf-8")
    response_path.write_text(response, encoding="utf-8")

    if verbose:
        print(f"[agent] Prompt saved to:   {prompt_path}")
        print(f"[agent] Response saved to: {response_path}")

    return prompt_path, response_path


# ---------------------------------------------------------------------------
# API backend (ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

def _run_api(
    profile: NsysProfile,
    *,
    model: str,
    max_turns: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    import anthropic

    client = anthropic.Anthropic()
    schemas = tool_schemas()
    messages: list[dict] = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT_API,
            tools=schemas,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if hasattr(block, "text"):
                    print(f"[agent] {block.text[:200]}{'...' if len(block.text) > 200 else ''}")
                elif hasattr(block, "name"):
                    print(f"[tool ] {block.name}({json.dumps(block.input)[:120]})")

        if response.stop_reason == "end_turn":
            for block in reversed(response.content):
                if hasattr(block, "text"):
                    hypotheses = _extract_hypotheses(block.text)
                    if hypotheses:
                        # Save: prompt = system prompt + full message history
                        prompt_record = (
                            f"=== SYSTEM PROMPT ===\n{_SYSTEM_PROMPT_API}\n\n"
                            f"=== TOOL SCHEMAS ===\n{json.dumps(schemas, indent=2)}\n\n"
                            f"=== MESSAGE HISTORY ===\n{json.dumps(messages, indent=2, default=str)}"
                        )
                        response_record = block.text
                        _save_files(profile.path, prompt_record, response_record, verbose)
                        return hypotheses
            return []

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_json = dispatch(profile, block.name, block.input)
            if verbose:
                print(f"[tool ] → {result_json[:200]}{'...' if len(result_json) > 200 else ''}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_json,
            })
        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")


# ---------------------------------------------------------------------------
# Claude Code backend (subprocess, no API key needed)
# ---------------------------------------------------------------------------

def _run_claude_code(
    profile: NsysProfile,
    *,
    verbose: bool,
    summary: ProfileSummary | None = None,
) -> list[dict[str, Any]]:
    if verbose:
        print("[agent] No ANTHROPIC_API_KEY found — falling back to Claude Code (claude -p)")

    if summary is None:
        if verbose:
            print("[agent] Computing profile summary...")
        summary = compute_profile_summary(profile)

    summary_json = summary.model_dump_json(indent=2)
    prompt = _format_summary_prompt(summary_json, profile.path.name)

    if verbose:
        print("[agent] Sending to Claude Code...")

    result = subprocess.run(
        ["claude", "-p", prompt, "--output-format", "json"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    response_text = data.get("result", "")

    _save_files(profile.path, prompt, response_text, verbose)

    if verbose:
        print(f"[agent] {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    return _extract_hypotheses(response_text)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(
    profile_path: str | Path,
    *,
    model: str = MODEL,
    max_turns: int = MAX_TURNS,
    verbose: bool = True,
    summary: ProfileSummary | None = None,
) -> list[dict[str, Any]]:
    """Analyze a profile and return a list of hypothesis dicts.

    Uses the Anthropic API (multi-turn tool-use) when ANTHROPIC_API_KEY is set,
    otherwise falls back to `claude -p` via subprocess.

    Pass a pre-computed `summary` to avoid recomputing it (e.g. when the caller
    already computed it for display purposes).
    """
    profile = NsysProfile(profile_path)

    if verbose:
        print(f"[agent] Analyzing {profile.path.name}")

    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            hypotheses = _run_api(profile, model=model, max_turns=max_turns, verbose=verbose)
        else:
            hypotheses = _run_claude_code(profile, verbose=verbose, summary=summary)
    finally:
        profile.close()

    return hypotheses
