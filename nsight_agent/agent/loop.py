"""Agent loop: drives an LLM to analyze a profile and produce hypotheses.

Three backends are supported:
  - anthropic (default when ANTHROPIC_API_KEY is set): multi-turn tool-use loop
    via the Anthropic SDK. Supports pre-seeding to skip 2 round-trips.
  - openai (when OPENAI_API_KEY is set): OpenAI function-calling format.
    Uses the same tool schemas, translated to OpenAI's "type":"function" wrapper.
  - gemini (when GOOGLE_API_KEY is set): Google GenerativeAI function declarations.
    Summary is injected into the initial message rather than pre-seeded.
  - claude_code (fallback): pre-computes the full ProfileSummary and sends it to
    `claude -p` via subprocess. No API key required.

Select a provider explicitly with --provider, or prefix the model ID:
  openai:gpt-4o, gemini:gemini-2.0-flash, anthropic:claude-opus-4-6

Both prompt and raw response are saved next to the profile:
  {profile_stem}_{timestamp}_prompt.txt
  {profile_stem}_{timestamp}_response.txt
"""

from __future__ import annotations

import importlib
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

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic": "claude-opus-4-6",
    "openai": "gpt-4o",
    "gemini": "gemini-2.0-flash",
    "claude_code": "claude-opus-4-6",
}
MAX_TURNS = 20

_KNOWN_PROVIDERS = ("anthropic", "openai", "gemini")

# Maps provider name -> (importable module, install hint).
# Providers not listed here (e.g. anthropic, claude_code) are always available.
_PROVIDER_PACKAGES: dict[str, tuple[str, str]] = {
    "openai": ("openai", "pip install openai"),
    "gemini": ("google.generativeai", "pip install google-generativeai"),
}


def check_provider_available(provider: str) -> str | None:
    """Return an install hint string if the provider's package is missing, else None."""
    if provider not in _PROVIDER_PACKAGES:
        return None
    module, hint = _PROVIDER_PACKAGES[provider]
    try:
        importlib.import_module(module)
        return None
    except ImportError:
        return hint


def get_provider_availability() -> dict[str, str | None]:
    """Return {provider: None} if available or {provider: install_hint} if missing."""
    return {p: check_provider_available(p) for p in _KNOWN_PROVIDERS}

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

The profile_summary and phase_summary results have already been pre-loaded for you as the first
tool-result exchange in this conversation — do not call those tools again.

Work systematically within the dominant phases: kernels → memory → MPI (if present) → idle gaps.
Use sql_query for any targeted follow-up that the structured tools don't cover.

You may call multiple tools in a single response to gather data in parallel.

When you have gathered enough evidence, output your final answer as a JSON array of hypothesis
objects (not wrapped in markdown fences) and nothing else after it.
"""


def _format_summary_prompt(summary_json: str, profile_name: str) -> str:
    return f"""\
You are an expert GPU performance engineer. Analyze the following Nsight Systems profile summary
for '{profile_name}' and produce a ranked list of actionable performance hypotheses.

{_HYPOTHESIS_SCHEMA}

The summary includes a 'phases' field that partitions the profile into sequential, non-overlapping
execution phases (e.g., initialization, solvers, teardown). Each phase has its own GPU utilization,
top kernels, and MPI breakdown. Analyze phases independently — global averages can be misleading
when phases have very different performance characteristics. Focus hypotheses on the phases that
dominate execution time.

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
    """Save prompt and response to files next to the profile."""
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


def _parse_provider_and_model(provider: str | None, model: str | None) -> tuple[str, str, str]:
    """Resolve (provider, model_id, reason) from explicit --provider and model string.

    Model strings may carry a provider prefix: "openai:gpt-4o", "gemini:gemini-2.0-flash".
    Resolution order:
      1. Provider prefix in model string (e.g. "openai:gpt-4o")
      2. Explicit --provider flag
      3. Auto-detect from available API keys (ANTHROPIC > OPENAI > GOOGLE)
      4. Fall back to claude_code subprocess

    If model is None, a sensible default is chosen for the resolved provider.
    """
    if model:
        for p in _KNOWN_PROVIDERS:
            if model.startswith(f"{p}:"):
                return p, model[len(p) + 1:], "provider prefix in --model"

    def _default(p: str) -> str:
        return model if model else _DEFAULT_MODELS[p]

    if provider:
        return provider, _default(provider), "explicit --provider flag"

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", _default("anthropic"), "presence of ANTHROPIC_API_KEY"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", _default("openai"), "presence of OPENAI_API_KEY"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini", _default("gemini"), "presence of GOOGLE_API_KEY"

    return "claude_code", _default("claude_code"), "no API keys found — falling back to claude subprocess"


# ---------------------------------------------------------------------------
# Schema translation helpers
# ---------------------------------------------------------------------------

def _schemas_to_openai(schemas: list[dict]) -> list[dict]:
    """Wrap Anthropic tool schemas in OpenAI's {"type":"function","function":{...}} envelope."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _preseed_messages(profile: NsysProfile, summary: ProfileSummary) -> list[dict]:
    """Inject profile_summary and phase_summary results as the opening exchange.

    Saves 2 API round-trips by reusing the already-computed ProfileSummary.
    """
    profile_result = json.dumps({
        "profile_span_s": summary.profile_span_s,
        "gpu_kernel_s": summary.gpu_kernel_s,
        "gpu_utilization_pct": summary.gpu_utilization_pct,
        "mpi_present": summary.mpi_present,
        "nvtx_present": bool(summary.nvtx_ranges),
        "tables": sorted(profile.tables),
    })
    phase_result = json.dumps({
        "phases": [p.model_dump() for p in summary.phases]
    })
    return [
        {"role": "user", "content": "Begin analysis."},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "pre_1", "name": "profile_summary", "input": {}},
                {"type": "tool_use", "id": "pre_2", "name": "phase_summary", "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "pre_1", "content": profile_result},
                {"type": "tool_result", "tool_use_id": "pre_2", "content": phase_result},
            ],
        },
    ]


def _run_api(
    profile: NsysProfile,
    *,
    model: str,
    max_turns: int,
    verbose: bool,
    summary: ProfileSummary | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    import anthropic

    client = anthropic.Anthropic()
    schemas = tool_schemas()
    messages: list[dict] = _preseed_messages(profile, summary) if summary is not None else []
    input_tokens = 0
    output_tokens = 0

    for _ in range(max_turns):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_SYSTEM_PROMPT_API,
            tools=schemas,
            messages=messages,
        )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
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
                        prompt_record = (
                            f"=== SYSTEM PROMPT ===\n{_SYSTEM_PROMPT_API}\n\n"
                            f"=== TOOL SCHEMAS ===\n{json.dumps(schemas, indent=2)}\n\n"
                            f"=== MESSAGE HISTORY ===\n{json.dumps(messages, indent=2, default=str)}"
                        )
                        _save_files(profile.path, prompt_record, block.text, verbose)
                        return hypotheses, input_tokens, output_tokens
            return [], input_tokens, output_tokens

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
# OpenAI backend
# ---------------------------------------------------------------------------

def _preseed_messages_openai(profile: NsysProfile, summary: ProfileSummary) -> list[dict]:
    """Inject pre-computed profile/phase summaries in OpenAI's tool-call format."""
    profile_result = json.dumps({
        "profile_span_s": summary.profile_span_s,
        "gpu_kernel_s": summary.gpu_kernel_s,
        "gpu_utilization_pct": summary.gpu_utilization_pct,
        "mpi_present": summary.mpi_present,
        "nvtx_present": bool(summary.nvtx_ranges),
        "tables": sorted(profile.tables),
    })
    phase_result = json.dumps({"phases": [p.model_dump() for p in summary.phases]})
    return [
        {"role": "user", "content": "Begin analysis."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "pre_1", "type": "function",
                 "function": {"name": "profile_summary", "arguments": "{}"}},
                {"id": "pre_2", "type": "function",
                 "function": {"name": "phase_summary", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "pre_1", "content": profile_result},
        {"role": "tool", "tool_call_id": "pre_2", "content": phase_result},
    ]


def _run_openai(
    profile: NsysProfile,
    *,
    model: str,
    max_turns: int,
    verbose: bool,
    summary: ProfileSummary | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for the OpenAI backend: pip install openai"
        )

    client = OpenAI()
    schemas = tool_schemas()
    openai_tools = _schemas_to_openai(schemas)
    messages: list[dict] = (
        _preseed_messages_openai(profile, summary) if summary is not None
        else [{"role": "user", "content": "Begin analysis."}]
    )
    # Prepend system prompt as a system message
    messages = [{"role": "system", "content": _SYSTEM_PROMPT_API}] + messages
    input_tokens = 0
    output_tokens = 0

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            tool_choice="auto",
            max_tokens=4096,
        )
        if response.usage:
            input_tokens += response.usage.prompt_tokens
            output_tokens += response.usage.completion_tokens
        choice = response.choices[0]
        msg = choice.message

        # Serialize to dict for the history (SDK objects aren't JSON-serializable)
        msg_dict: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if verbose:
            if msg.content:
                print(f"[agent] {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"[tool ] {tc.function.name}({tc.function.arguments[:120]})")

        if choice.finish_reason == "stop":
            hypotheses = _extract_hypotheses(msg.content or "")
            if hypotheses:
                prompt_record = (
                    f"=== SYSTEM PROMPT ===\n{_SYSTEM_PROMPT_API}\n\n"
                    f"=== MESSAGE HISTORY ===\n{json.dumps(messages, indent=2, default=str)}"
                )
                _save_files(profile.path, prompt_record, msg.content or "", verbose)
                return hypotheses, input_tokens, output_tokens
            return [], input_tokens, output_tokens

        # finish_reason == "tool_calls"
        for tc in (msg.tool_calls or []):
            result_json = dispatch(profile, tc.function.name, json.loads(tc.function.arguments))
            if verbose:
                print(f"[tool ] → {result_json[:200]}{'...' if len(result_json) > 200 else ''}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_json,
            })

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")


# ---------------------------------------------------------------------------
# Gemini backend
# ---------------------------------------------------------------------------

def _run_gemini(
    profile: NsysProfile,
    *,
    model: str,
    max_turns: int,
    verbose: bool,
    summary: ProfileSummary | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    try:
        import google.generativeai as genai
        from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
    except ImportError:
        raise ImportError(
            "google-generativeai package is required for the Gemini backend: pip install google-generativeai"
        )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set for Gemini backend")
    genai.configure(api_key=api_key)

    schemas = tool_schemas()
    declarations = [
        FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters=s["input_schema"],
        )
        for s in schemas
    ]
    gemini_model = genai.GenerativeModel(
        model_name=model,
        tools=[GeminiTool(function_declarations=declarations)],
        system_instruction=_SYSTEM_PROMPT_API,
    )
    chat = gemini_model.start_chat()

    # Gemini doesn't support injecting pre-computed tool results into history;
    # include the pre-computed summary directly in the initial user message.
    if summary is not None:
        profile_result = json.dumps({
            "profile_span_s": summary.profile_span_s,
            "gpu_kernel_s": summary.gpu_kernel_s,
            "gpu_utilization_pct": summary.gpu_utilization_pct,
            "mpi_present": summary.mpi_present,
            "nvtx_present": bool(summary.nvtx_ranges),
            "tables": sorted(profile.tables),
        })
        phase_result = json.dumps({"phases": [p.model_dump() for p in summary.phases]})
        init_msg = (
            "Begin analysis. Pre-computed profile data is provided below.\n\n"
            f"profile_summary:\n{profile_result}\n\n"
            f"phase_summary:\n{phase_result}"
        )
    else:
        init_msg = "Begin analysis."

    input_tokens = 0
    output_tokens = 0

    def _add_gemini_usage(r: Any) -> None:
        nonlocal input_tokens, output_tokens
        um = getattr(r, "usage_metadata", None)
        if um:
            input_tokens += getattr(um, "prompt_token_count", 0) or 0
            output_tokens += getattr(um, "candidates_token_count", 0) or 0

    response = chat.send_message(init_msg)
    _add_gemini_usage(response)

    for _ in range(max_turns):
        function_calls = [
            p.function_call for p in response.parts
            if getattr(p, "function_call", None) and p.function_call.name
        ]

        if verbose:
            for p in response.parts:
                if getattr(p, "text", None):
                    print(f"[agent] {p.text[:200]}{'...' if len(p.text) > 200 else ''}")
                fc = getattr(p, "function_call", None)
                if fc and fc.name:
                    print(f"[tool ] {fc.name}({dict(fc.args)})")

        if not function_calls:
            text = "".join(
                p.text for p in response.parts if getattr(p, "text", None)
            )
            hypotheses = _extract_hypotheses(text)
            if hypotheses:
                prompt_record = (
                    f"=== SYSTEM PROMPT ===\n{_SYSTEM_PROMPT_API}\n\n"
                    f"=== INITIAL MESSAGE ===\n{init_msg}"
                )
                _save_files(profile.path, prompt_record, text, verbose)
                return hypotheses, input_tokens, output_tokens
            return [], input_tokens, output_tokens

        function_responses = []
        for fc in function_calls:
            result_json = dispatch(profile, fc.name, dict(fc.args))
            if verbose:
                print(f"[tool ] → {result_json[:200]}{'...' if len(result_json) > 200 else ''}")
            function_responses.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=fc.name,
                        response={"result": json.loads(result_json)},
                    )
                )
            )
        response = chat.send_message(function_responses)
        _add_gemini_usage(response)

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")


# ---------------------------------------------------------------------------
# Claude Code backend (subprocess, no API key needed)
# ---------------------------------------------------------------------------

def _run_claude_code(
    profile: NsysProfile,
    *,
    verbose: bool,
    summary: ProfileSummary | None = None,
) -> tuple[list[dict[str, Any]], int, int, float | None]:
    if verbose:
        print("[agent] No API key found — falling back to Claude Code (claude -p)")

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

    usage = data.get("usage", {})
    # Include cache tokens in the input total — they drive cost and context size
    inp = (
        (usage.get("input_tokens") or 0)
        + (usage.get("cache_creation_input_tokens") or 0)
        + (usage.get("cache_read_input_tokens") or 0)
    )
    out = usage.get("output_tokens") or 0
    cost_usd: float | None = data.get("total_cost_usd")

    _save_files(profile.path, prompt, response_text, verbose)

    if verbose:
        print(f"[agent] {response_text[:300]}{'...' if len(response_text) > 300 else ''}")

    return _extract_hypotheses(response_text), inp, out, cost_usd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_agent(
    profile_path: str | Path,
    *,
    model: str = MODEL,
    provider: str | None = None,
    max_turns: int = MAX_TURNS,
    verbose: bool = True,
    summary: ProfileSummary | None = None,
    token_usage: dict[str, int | None] | None = None,
) -> list[dict[str, Any]]:
    """Analyze a profile and return a list of hypothesis dicts.

    Provider selection order:
      1. Provider prefix in model string (e.g. "openai:gpt-4o")
      2. Explicit `provider` argument (or --provider CLI flag)
      3. Auto-detect from ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
      4. Fall back to `claude -p` subprocess (no API key required)

    Pass a pre-computed `summary` to avoid recomputing it (e.g. when the caller
    already computed it for display purposes).

    If `token_usage` is provided, it will be populated with `input_tokens` and
    `output_tokens` after the run (both None for the claude_code fallback).
    """
    resolved_provider, resolved_model, _ = _parse_provider_and_model(provider, model)

    profile = NsysProfile(profile_path)

    if verbose:
        print(f"[agent] Analyzing {profile.path.name} (provider={resolved_provider}, model={resolved_model})")

    try:
        if resolved_provider == "anthropic":
            hypotheses, inp, out = _run_api(
                profile, model=resolved_model, max_turns=max_turns,
                verbose=verbose, summary=summary,
            )
        elif resolved_provider == "openai":
            hypotheses, inp, out = _run_openai(
                profile, model=resolved_model, max_turns=max_turns,
                verbose=verbose, summary=summary,
            )
        elif resolved_provider == "gemini":
            hypotheses, inp, out = _run_gemini(
                profile, model=resolved_model, max_turns=max_turns,
                verbose=verbose, summary=summary,
            )
        else:
            hypotheses, inp, out, cost_usd = _run_claude_code(profile, verbose=verbose, summary=summary)
            if token_usage is not None and cost_usd is not None:
                token_usage["cost_usd"] = cost_usd
    finally:
        profile.close()

    if token_usage is not None:
        token_usage["input_tokens"] = inp
        token_usage["output_tokens"] = out

    return hypotheses
