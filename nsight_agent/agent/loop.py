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

Select a provider via --model:
  openai:gpt-4o          (provider prefix + model)
  openai                 (provider only, uses default model)
  gemini:gemini-2.0-flash
  anthropic:claude-opus-4-6

LLM interaction logging (opt-in via --log / --log-file):
  {profile_stem}_{timestamp}_log.txt
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nsight_agent.analysis.metrics import compute_profile_summary
from nsight_agent.analysis.models import ProfileSummary
from nsight_agent.agent.logger import LLMLogger
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
# Inject a wrap-up warning this many turns before the limit so the model has
# a chance to produce output before it's cut off.
WARN_TURNS_BEFORE_LIMIT = 3

_WRAP_UP_WARNING = (
    "You have {remaining} turns remaining. "
    "If you have gathered sufficient evidence, output your final hypothesis JSON array now. "
    "Avoid calling tools unless they would meaningfully change your conclusions."
)
_FINAL_FORCED_PROMPT = (
    "Turn limit reached. Output your final hypothesis JSON array immediately "
    "based on all evidence gathered so far. Do not call any more tools."
)

_KNOWN_PROVIDERS = ("anthropic", "openai", "gemini")

# Maps provider name -> (importable module, install hint).
# Providers not listed here (e.g. anthropic, claude_code) are always available.
_PROVIDER_PACKAGES: dict[str, tuple[str, str]] = {
    "openai": ("openai", "pip install openai"),
    "gemini": ("google.genai", "pip install google-genai"),
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
  - phase: name of the execution phase this bottleneck belongs to. Use the name exactly as it
           appears in the phase_summary results (e.g. "ev0", "ev1024", "profile"). If the
           profile has no phase structure or only one phase, use "whole_profile".
           The phase names are already present in the pre-seeded phase_summary data — use
           those names directly; do not issue SQL queries to determine phase membership.
  - description: concise plain-English description of the bottleneck
  - evidence: specific numbers from the profile that support this hypothesis
  - suggestion: concrete, actionable recommendation
  - expected_impact: estimated relative improvement (high / medium / low)
  - action_category: effort level required to act on this suggestion; one of:
      runtime_config    — env vars, MPI params, driver flags, library options (no rebuild)
      launch_config     — block/grid dims, shared memory, occupancy tuning (recompile only)
      code_optimization — kernel rewrites, memory layout, stream pipelining, async transfers
      algorithm         — solver change, preconditioner, deflation, mathematical reformulation
    Boundary rule: if the change alters *what* is computed → algorithm;
    if it changes only *how* → code_optimization or lower.\
"""

_GROUNDING_CONSTRAINT = """\
Ground all suggestions strictly in the profile data provided.
Do not suggest application-specific environment variables, compiler flags, library options, or \
algorithmic changes unless the profile data explicitly demonstrates the relevant behavior \
(e.g. do not infer the application's name and then suggest configuration options from memory). \
If you are uncertain whether a specific option exists or applies, omit it or phrase the \
suggestion in generic terms that any engineer could verify."""

_SQL_SCHEMA_REFERENCE = """\
## SQLite Schema Reference

The profile is a SQLite database — only standard SQLite functions are available.
Do NOT use PERCENTILE_CONT, MEDIAN, or STDDEV (they are PostgreSQL/standard SQL extensions
and will fail). Use AVG, MIN, MAX, and manual percentile approximations instead.

Key column names for sql_query (all timestamps are in nanoseconds):

  CUPTI_ACTIVITY_KIND_KERNEL:
    start, end, shortName (FK→StringIds), demangledName (FK→StringIds, may be absent),
    gridX, gridY, gridZ, blockX, blockY, blockZ, registersPerThread,
    sharedMemoryExecuted, staticSharedMemory, dynamicSharedMemory, streamId, correlationId

  CUPTI_ACTIVITY_KIND_MEMCPY:
    start, end, bytes, copyKind (FK→ENUM_CUDA_MEMCPY_OPER)

  CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
    start, end

  CUPTI_ACTIVITY_KIND_RUNTIME:
    start, end, nameId (FK→StringIds), correlationId

  MPI_COLLECTIVES_EVENTS, MPI_P2P_EVENTS, MPI_START_WAIT_EVENTS:
    start, end, textId (FK→StringIds)   ← textId holds the MPI operation name

  NVTX_EVENTS:
    start, end, text (literal string, not a FK), eventType (use eventType = 59 for ranges)

  StringIds: id, value   ← join with: JOIN StringIds s ON s.id = <fk_column>

Not all tables are present in every profile. The profile_summary tool lists available tables.
Use the get_table_schema tool (or PRAGMA table_info(<table>)) to inspect an unfamiliar table.\
"""

_SYSTEM_PROMPT_API_BASE = f"""You are an expert GPU performance engineer analyzing an NVIDIA Nsight Systems profile.

Your goal is to identify the most significant performance bottlenecks and produce a ranked list of
actionable hypotheses.

{_HYPOTHESIS_SCHEMA}

The profile_summary and phase_summary results have already been pre-loaded for you as the first
tool-result exchange in this conversation — do not call those tools again.

Work systematically within the dominant phases: kernels → memory → MPI (if present) → idle gaps.
Use sql_query for any targeted follow-up that the structured tools don't cover.

You may call multiple tools in a single response to gather data in parallel.
Before calling a tool, check whether you have already called it with the same arguments earlier
in this conversation — do not issue duplicate tool calls.

When you have gathered enough evidence, output your final answer as a JSON array of hypothesis
objects (not wrapped in markdown fences) and nothing else after it.

{_SQL_SCHEMA_REFERENCE}
"""


def _build_system_prompt(grounded: bool = True) -> str:
    if grounded:
        return _SYSTEM_PROMPT_API_BASE + "\n" + _GROUNDING_CONSTRAINT
    return _SYSTEM_PROMPT_API_BASE


def _format_summary_prompt(summary_json: str, profile_name: str, grounded: bool = True) -> str:
    grounding_section = f"\n{_GROUNDING_CONSTRAINT}\n" if grounded else ""
    return f"""\
You are an expert GPU performance engineer. Analyze the following Nsight Systems profile summary
for '{profile_name}' and produce a ranked list of actionable performance hypotheses.

{_HYPOTHESIS_SCHEMA}
{grounding_section}
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


def _trunc(s: str, n: int = 200) -> str:
    return s[:n] + "..." if len(s) > n else s


def _turn_header(turn: int, max_turns: int, log: Callable[[str], None] = print) -> None:
    label = f" Turn {turn} / {max_turns} "
    dashes = max(0, 60 - len(label))
    left = dashes // 2
    right = dashes - left
    log(f"\n{'─' * left}{label}{'─' * right}")



def _serialize_anthropic_content(content: list) -> list[dict]:
    """Convert Anthropic SDK content blocks to plain dicts for logging."""
    result = []
    for block in content:
        block_type = getattr(block, "type", "unknown")
        if block_type == "text":
            result.append({"type": "text", "text": block.text})
        elif block_type == "tool_use":
            result.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
        else:
            result.append({"type": str(block_type)})
    return result


def _parse_provider_and_model(model: str | None) -> tuple[str, str, str]:
    """Resolve (provider, model_id, reason) from the --model string.

    Model strings may carry a provider prefix or be a bare provider name:
      "openai:gpt-4o"     → provider=openai,   model=gpt-4o
      "openai"            → provider=openai,   model=<default>
      "claude-opus-4-6"   → provider auto-detected from env vars
      None                → provider auto-detected from env vars

    Resolution order:
      1. Provider prefix in model string (e.g. "openai:gpt-4o")
      2. Bare provider name in model string (e.g. "openai")
      3. Auto-detect from available API keys (ANTHROPIC > OPENAI > GOOGLE)
      4. Fall back to claude_code subprocess
    """
    if model:
        for p in _KNOWN_PROVIDERS:
            if model.startswith(f"{p}:"):
                return p, model[len(p) + 1 :], "provider prefix in --model"
        if model in _KNOWN_PROVIDERS:
            return model, _DEFAULT_MODELS[model], "provider name in --model"

    def _default(p: str) -> str:
        return model if model else _DEFAULT_MODELS[p]

    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", _default("anthropic"), "presence of ANTHROPIC_API_KEY"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai", _default("openai"), "presence of OPENAI_API_KEY"
    if os.environ.get("GOOGLE_API_KEY"):
        return "gemini", _default("gemini"), "presence of GOOGLE_API_KEY"

    return (
        "claude_code",
        _default("claude_code"),
        "no API keys found — falling back to claude subprocess",
    )


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
    profile_result = json.dumps(
        {
            "profile_span_s": summary.profile_span_s,
            "gpu_kernel_s": summary.gpu_kernel_s,
            "gpu_utilization_pct": summary.gpu_utilization_pct,
            "mpi_present": summary.mpi_present,
            "nvtx_present": bool(summary.nvtx_ranges),
            "tables": sorted(profile.tables),
        }
    )
    phase_result = json.dumps({"phases": [p.model_dump() for p in summary.phases]})
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
                {
                    "type": "tool_result",
                    "tool_use_id": "pre_2",
                    "content": phase_result,
                    "cache_control": {"type": "ephemeral"},
                },
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
    grounded: bool = True,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    import anthropic

    from nsight_agent.agent.preflight import estimate_json_tokens, estimate_prose_tokens

    client = anthropic.Anthropic()
    schemas = tool_schemas()
    messages: list[dict] = _preseed_messages(profile, summary) if summary is not None else []
    _system = [
        {
            "type": "text",
            "text": _build_system_prompt(grounded),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    input_tokens = 0
    output_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    # Anthropic allows at most 4 cache_control blocks per request.
    # We use 2 permanent markers (system + preseed) and keep the 2 most-recent
    # per-turn user messages marked, giving true sliding cache:
    #   turn N reads everything through turn N-1 from cache (0.10×)
    #   turn N writes only the new increment to cache (1.25×)
    # When a third per-turn message is about to be added, the oldest is stripped.
    _cache_prev: dict | None = None   # turn N-1 (keep marker)
    _cache_pprev: dict | None = None  # turn N-2 (strip marker on next advance)

    # Estimated total and cached context sizes for the upcoming turn.
    # Turn 1: full heuristic (no prior usage data); subsequent turns: exact from usage.
    _next_ctx_estimate = (
        estimate_prose_tokens(_build_system_prompt(grounded))
        + estimate_json_tokens(json.dumps(messages, default=str))
        + estimate_json_tokens(json.dumps(schemas))
    )
    _next_cached_estimate = 0  # nothing cached before the first call

    for turn in range(1, max_turns + 1):
        if verbose:
            _turn_header(turn, max_turns, log)
            if _next_cached_estimate:
                log(
                    f"[local] Context size ≈ {_next_ctx_estimate:,} tokens"
                    f" ({_next_cached_estimate:,} cached)"
                )
            else:
                log(f"[local] Context size ≈ {_next_ctx_estimate:,} tokens")

        if logger:
            logger.write_request(
                turn,
                {"system": _system, "tools": schemas, "messages": messages},
            )
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=_system,
            tools=schemas,
            messages=messages,
        )
        if logger:
            logger.write_response(
                turn,
                {
                    "stop_reason": response.stop_reason,
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                        "cache_creation_input_tokens": getattr(
                            response.usage, "cache_creation_input_tokens", None
                        ),
                        "cache_read_input_tokens": getattr(
                            response.usage, "cache_read_input_tokens", None
                        ),
                    },
                    "content": _serialize_anthropic_content(response.content),
                },
            )
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        cache_creation_tokens += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0) or 0

        _ctx_total = (
            response.usage.input_tokens
            + (getattr(response.usage, "cache_creation_input_tokens", 0) or 0)
            + (getattr(response.usage, "cache_read_input_tokens", 0) or 0)
        )

        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if hasattr(block, "text"):
                    log(f"[← llm] {_trunc(block.text)}")
                elif hasattr(block, "name"):
                    log(f"[← llm:tool] {block.name}({_trunc(json.dumps(block.input), 120)})")

        if response.stop_reason == "end_turn":
            for block in reversed(response.content):
                if hasattr(block, "text"):
                    hypotheses = _extract_hypotheses(block.text)
                    if hypotheses:
                        return (
                            hypotheses,
                            input_tokens,
                            cache_creation_tokens,
                            cache_read_tokens,
                            output_tokens,
                        )
            return [], input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_json = dispatch(profile, block.name, block.input)
            if verbose:
                log(f"[local] {block.name} → {_trunc(result_json)}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_json,
                }
            )

        turns_left = max_turns - turn
        if turns_left == WARN_TURNS_BEFORE_LIMIT:
            tool_results.append(
                {
                    "type": "text",
                    "text": _WRAP_UP_WARNING.format(remaining=turns_left),
                }
            )
            if verbose:
                log(f"[local] ({turns_left} turns remaining — wrap-up warning injected)")
        elif turns_left == 0:
            tool_results.append({"type": "text", "text": _FINAL_FORCED_PROMPT})

        # Estimate context and cached size for the next turn.
        # All of _ctx_total will be cache_read next turn; the new increment
        # (assistant response + tool results) will be cache_creation.
        _tool_results_tokens = estimate_json_tokens(json.dumps(tool_results, default=str))
        _next_ctx_estimate = _ctx_total + response.usage.output_tokens + _tool_results_tokens
        _next_cached_estimate = _ctx_total

        # Advance the sliding cache window: strip the oldest floating marker
        # (_cache_pprev), keep the previous turn's marker (_cache_prev), and
        # mark the new turn.  Total markers stay at 4 (system + preseed + 2
        # floating), which is Anthropic's maximum.
        if _cache_pprev is not None:
            _pprev_content = _cache_pprev.get("content", [])
            if isinstance(_pprev_content, list) and _pprev_content:
                _pprev_content[-1].pop("cache_control", None)
        if tool_results:
            tool_results[-1]["cache_control"] = {"type": "ephemeral"}
        _new_user_msg: dict = {"role": "user", "content": tool_results}
        messages.append(_new_user_msg)
        _cache_pprev, _cache_prev = _cache_prev, _new_user_msg

    # All turns exhausted — make one final call without tools to force text output.
    if verbose:
        log("[local] Turn limit reached — forcing final output (no tool calls).")
    _forced_turn = max_turns + 1
    if logger:
        logger.write_request(
            _forced_turn,
            {"system": _system, "messages": messages, "(forced_output_no_tools)": True},
        )
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_system,
        messages=messages,
    )
    if logger:
        logger.write_response(
            _forced_turn,
            {
                "stop_reason": response.stop_reason,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_creation_input_tokens": getattr(
                        response.usage, "cache_creation_input_tokens", None
                    ),
                    "cache_read_input_tokens": getattr(
                        response.usage, "cache_read_input_tokens", None
                    ),
                },
                "content": _serialize_anthropic_content(response.content),
            },
        )
    input_tokens += response.usage.input_tokens
    output_tokens += response.usage.output_tokens
    cache_creation_tokens += getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cache_read_tokens += getattr(response.usage, "cache_read_input_tokens", 0) or 0
    for block in reversed(response.content):
        if hasattr(block, "text"):
            hypotheses = _extract_hypotheses(block.text)
            if hypotheses:
                return (
                    hypotheses,
                    input_tokens,
                    cache_creation_tokens,
                    cache_read_tokens,
                    output_tokens,
                )
    if verbose:
        log("[local] Warning: no hypotheses extracted after forced output turn.")
    return [], input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


def _preseed_messages_openai(profile: NsysProfile, summary: ProfileSummary) -> list[dict]:
    """Inject pre-computed profile/phase summaries in OpenAI's tool-call format."""
    profile_result = json.dumps(
        {
            "profile_span_s": summary.profile_span_s,
            "gpu_kernel_s": summary.gpu_kernel_s,
            "gpu_utilization_pct": summary.gpu_utilization_pct,
            "mpi_present": summary.mpi_present,
            "nvtx_present": bool(summary.nvtx_ranges),
            "tables": sorted(profile.tables),
        }
    )
    phase_result = json.dumps({"phases": [p.model_dump() for p in summary.phases]})
    return [
        {"role": "user", "content": "Begin analysis."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "pre_1",
                    "type": "function",
                    "function": {"name": "profile_summary", "arguments": "{}"},
                },
                {
                    "id": "pre_2",
                    "type": "function",
                    "function": {"name": "phase_summary", "arguments": "{}"},
                },
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
    grounded: bool = True,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is required for the OpenAI backend: pip install openai")

    client = OpenAI()
    schemas = tool_schemas()
    openai_tools = _schemas_to_openai(schemas)
    messages: list[dict] = (
        _preseed_messages_openai(profile, summary)
        if summary is not None
        else [{"role": "user", "content": "Begin analysis."}]
    )
    # Prepend system prompt as a system message
    messages = [{"role": "system", "content": _build_system_prompt(grounded)}] + messages
    input_tokens = 0
    output_tokens = 0
    # Older models use max_tokens; newer models (o-series, gpt-4.1+) require max_completion_tokens.
    # Detect which parameter to use on the first call and reuse that for subsequent turns.
    _token_limit_param = "max_tokens"

    def _create(msgs: list[dict]) -> Any:
        nonlocal _token_limit_param
        from openai import BadRequestError

        kwargs = dict(
            model=model,
            messages=msgs,
            tools=openai_tools,
            tool_choice="auto",
            **{_token_limit_param: 4096},
        )
        try:
            return client.chat.completions.create(**kwargs)
        except BadRequestError as e:
            if _token_limit_param == "max_tokens" and "max_completion_tokens" in str(e):
                _token_limit_param = "max_completion_tokens"
                kwargs[_token_limit_param] = kwargs.pop("max_tokens")
                return client.chat.completions.create(**kwargs)
            raise

    for turn in range(1, max_turns + 1):
        if logger:
            logger.write_request(
                turn,
                {"messages": messages, "tools": openai_tools},
            )
        response = _create(messages)
        if logger:
            _msg_dict_log: dict[str, Any] = {
                "role": "assistant",
                "content": response.choices[0].message.content,
            }
            if response.choices[0].message.tool_calls:
                _msg_dict_log["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in response.choices[0].message.tool_calls
                ]
            logger.write_response(
                turn,
                {
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                        "completion_tokens": response.usage.completion_tokens
                        if response.usage
                        else None,
                    },
                    "message": _msg_dict_log,
                },
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
            _turn_header(turn, max_turns, log)
            if response.usage:
                ctx_tokens = response.usage.prompt_tokens
                cached_tokens = getattr(
                    getattr(response.usage, "prompt_tokens_details", None),
                    "cached_tokens",
                    0,
                ) or 0
                if cached_tokens:
                    log(
                        f"[local] Context size ≈ {ctx_tokens:,} tokens"
                        f" ({cached_tokens:,} cached)"
                    )
                else:
                    log(f"[local] Context size ≈ {ctx_tokens:,} tokens")
            if msg.content:
                log(f"[← llm] {_trunc(msg.content)}")
            for tc in msg.tool_calls or []:
                log(f"[← llm:tool] {tc.function.name}({_trunc(tc.function.arguments, 120)})")

        if choice.finish_reason == "stop":
            hypotheses = _extract_hypotheses(msg.content or "")
            if hypotheses:
                return hypotheses, input_tokens, 0, 0, output_tokens
            # Model stopped without a valid JSON array — inject a recovery prompt and continue.
            if verbose:
                log("[local] stop with no JSON array — injecting recovery prompt")
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your response did not contain the required JSON array of hypothesis objects. "
                        "Output ONLY a JSON array now — no prose, no markdown fences."
                    ),
                }
            )
            continue

        # finish_reason == "tool_calls"
        for tc in msg.tool_calls or []:
            result_json = dispatch(profile, tc.function.name, json.loads(tc.function.arguments))
            if verbose:
                log(f"[local] {tc.function.name} → {_trunc(result_json)}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_json,
                }
            )

        turns_left = max_turns - turn
        if turns_left == WARN_TURNS_BEFORE_LIMIT:
            messages.append(
                {
                    "role": "user",
                    "content": _WRAP_UP_WARNING.format(remaining=turns_left),
                }
            )
            if verbose:
                log(f"[local] ({turns_left} turns remaining — wrap-up warning injected)")
        elif turns_left == 0:
            messages.append({"role": "user", "content": _FINAL_FORCED_PROMPT})

    # All turns exhausted — make one final call without tools to force text output.
    if verbose:
        log("[local] Turn limit reached — forcing final output (no tool calls).")
    _forced_turn = max_turns + 1
    try:
        if logger:
            logger.write_request(
                _forced_turn,
                {"messages": messages, "(forced_output_no_tools)": True},
            )
        response_forced = client.chat.completions.create(
            model=model,
            messages=messages,
            **{_token_limit_param: 4096},
        )
        if logger:
            logger.write_response(
                _forced_turn,
                {
                    "finish_reason": response_forced.choices[0].finish_reason,
                    "usage": {
                        "prompt_tokens": response_forced.usage.prompt_tokens
                        if response_forced.usage
                        else None,
                        "completion_tokens": response_forced.usage.completion_tokens
                        if response_forced.usage
                        else None,
                    },
                    "content": response_forced.choices[0].message.content,
                },
            )
        if response_forced.usage:
            input_tokens += response_forced.usage.prompt_tokens
            output_tokens += response_forced.usage.completion_tokens
        forced_content = response_forced.choices[0].message.content or ""
        hypotheses = _extract_hypotheses(forced_content)
        if hypotheses:
            return hypotheses, input_tokens, 0, 0, output_tokens
    except Exception as exc:
        if verbose:
            log(f"[local] Forced output turn failed: {exc}")
    if verbose:
        log("[local] Warning: no hypotheses extracted after forced output turn.")
    return [], input_tokens, 0, 0, output_tokens


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
    grounded: bool = True,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> tuple[list[dict[str, Any]], int, int, int, int]:
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise ImportError(
            "google-genai package is required for the Gemini backend: pip install google-genai"
        )

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY environment variable is not set for Gemini backend")

    client = genai.Client(api_key=api_key)

    schemas = tool_schemas()
    declarations = [
        genai_types.FunctionDeclaration(
            name=s["name"],
            description=s["description"],
            parameters=s["input_schema"],
        )
        for s in schemas
    ]
    config = genai_types.GenerateContentConfig(
        system_instruction=_build_system_prompt(grounded),
        tools=[genai_types.Tool(function_declarations=declarations)],
    )

    # Gemini doesn't support injecting pre-computed tool results into history;
    # include the pre-computed summary directly in the initial user message.
    if summary is not None:
        profile_result = json.dumps(
            {
                "profile_span_s": summary.profile_span_s,
                "gpu_kernel_s": summary.gpu_kernel_s,
                "gpu_utilization_pct": summary.gpu_utilization_pct,
                "mpi_present": summary.mpi_present,
                "nvtx_present": bool(summary.nvtx_ranges),
                "tables": sorted(profile.tables),
            }
        )
        phase_result = json.dumps({"phases": [p.model_dump() for p in summary.phases]})
        init_msg = (
            "Begin analysis. Pre-computed profile data is provided below.\n\n"
            f"profile_summary:\n{profile_result}\n\n"
            f"phase_summary:\n{phase_result}"
        )
    else:
        init_msg = "Begin analysis."

    chat = client.chats.create(model=model, config=config)
    input_tokens = 0
    output_tokens = 0
    _log_turn = 0

    def _add_gemini_usage(r: Any) -> None:
        nonlocal input_tokens, output_tokens
        um = getattr(r, "usage_metadata", None)
        if um:
            input_tokens += getattr(um, "prompt_token_count", 0) or 0
            output_tokens += getattr(um, "candidates_token_count", 0) or 0

    def _gemini_response_payload(r: Any) -> dict:
        um = getattr(r, "usage_metadata", None)
        return {
            "text": r.text,
            "function_calls": [
                {"name": fc.name, "args": dict(fc.args)}
                for fc in (r.function_calls or [])
            ],
            "usage_metadata": {
                "prompt_token_count": getattr(um, "prompt_token_count", None),
                "candidates_token_count": getattr(um, "candidates_token_count", None),
            }
            if um
            else None,
        }

    _log_turn += 1
    if logger:
        logger.write_request(
            _log_turn,
            {
                "system_instruction": _build_system_prompt(grounded),
                "tool_declarations": [s["name"] for s in schemas],
                "message": init_msg,
            },
        )
    response = chat.send_message(init_msg)
    _add_gemini_usage(response)
    if logger:
        logger.write_response(_log_turn, _gemini_response_payload(response))

    for turn in range(1, max_turns + 1):
        function_calls = response.function_calls or []

        if verbose:
            _turn_header(turn, max_turns, log)
            um = getattr(response, "usage_metadata", None)
            if um:
                ctx_tokens = getattr(um, "prompt_token_count", 0) or 0
                cached_tokens = getattr(um, "cached_content_token_count", 0) or 0
                if ctx_tokens:
                    if cached_tokens:
                        log(
                            f"[local] Context size ≈ {ctx_tokens:,} tokens"
                            f" ({cached_tokens:,} cached)"
                        )
                    else:
                        log(f"[local] Context size ≈ {ctx_tokens:,} tokens")
            text = response.text or ""
            if text:
                log(f"[← llm] {_trunc(text)}")
            for fc in function_calls:
                log(f"[← llm:tool] {fc.name}({_trunc(str(dict(fc.args)), 120)})")

        if not function_calls:
            text = response.text or ""
            hypotheses = _extract_hypotheses(text)
            if hypotheses:
                return hypotheses, input_tokens, 0, 0, output_tokens
            return [], input_tokens, 0, 0, output_tokens

        parts = []
        parts_data: list[dict] = []
        for fc in function_calls:
            result_json = dispatch(profile, fc.name, dict(fc.args))
            if verbose:
                log(f"[local] {fc.name} → {_trunc(result_json)}")
            parts_data.append(
                {"type": "function_response", "name": fc.name, "result": json.loads(result_json)}
            )
            parts.append(
                genai_types.Part.from_function_response(
                    name=fc.name,
                    response={"result": json.loads(result_json)},
                )
            )

        turns_left = max_turns - turn
        if turns_left == WARN_TURNS_BEFORE_LIMIT:
            parts.append(genai_types.Part.from_text(_WRAP_UP_WARNING.format(remaining=turns_left)))
            parts_data.append({"type": "text", "text": _WRAP_UP_WARNING.format(remaining=turns_left)})
            if verbose:
                log(f"[local] ({turns_left} turns remaining — wrap-up warning injected)")
        elif turns_left == 0:
            parts.append(genai_types.Part.from_text(_FINAL_FORCED_PROMPT))
            parts_data.append({"type": "text", "text": _FINAL_FORCED_PROMPT})

        _log_turn += 1
        if logger:
            logger.write_request(_log_turn, {"parts": parts_data})
        response = chat.send_message(parts)
        _add_gemini_usage(response)
        if logger:
            logger.write_response(_log_turn, _gemini_response_payload(response))

    # All turns exhausted — send one final text-only message to force output.
    if verbose:
        log("[local] Turn limit reached — forcing final output (no tool calls).")
    _log_turn += 1
    try:
        if logger:
            logger.write_request(
                _log_turn,
                {"parts": [{"type": "text", "text": _FINAL_FORCED_PROMPT}], "(forced_output_no_tools)": True},
            )
        response_forced = chat.send_message(_FINAL_FORCED_PROMPT)
        _add_gemini_usage(response_forced)
        if logger:
            logger.write_response(_log_turn, _gemini_response_payload(response_forced))
        text = response_forced.text or ""
        hypotheses = _extract_hypotheses(text)
        if hypotheses:
            return hypotheses, input_tokens, 0, 0, output_tokens
    except Exception as exc:
        if verbose:
            log(f"[local] Forced output turn failed: {exc}")
    if verbose:
        log("[local] Warning: no hypotheses extracted after forced output turn.")
    return [], input_tokens, 0, 0, output_tokens


# ---------------------------------------------------------------------------
# Claude Code backend (subprocess, no API key needed)
# ---------------------------------------------------------------------------


def _run_claude_code(
    profile: NsysProfile,
    *,
    verbose: bool,
    summary: ProfileSummary | None = None,
    grounded: bool = True,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> tuple[list[dict[str, Any]], int, int, float | None]:
    if verbose:
        log("[local] No API key found — falling back to Claude Code (claude -p)")

    if summary is None:
        if verbose:
            log("[local] Computing profile summary...")
        summary = compute_profile_summary(profile)

    summary_json = summary.model_dump_json(indent=2)
    prompt = _format_summary_prompt(summary_json, profile.path.name, grounded=grounded)

    if verbose:
        log("[→ llm] Sending profile summary to Claude Code...")

    if logger:
        logger.write_request(1, {"prompt": prompt})

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

    if logger:
        logger.write_response(
            1,
            {
                "result": response_text,
                "usage": {
                    "input_tokens": usage.get("input_tokens"),
                    "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                    "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
                    "output_tokens": out,
                },
                "total_cost_usd": cost_usd,
            },
        )

    if verbose:
        log(f"[← llm] {_trunc(response_text, 300)}")

    return _extract_hypotheses(response_text), inp, out, cost_usd


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
    token_usage: dict[str, int | None] | None = None,
    grounded: bool = True,
    log: Callable[[str], None] = print,
    logger: LLMLogger | None = None,
) -> list[dict[str, Any]]:
    """Analyze a profile and return a list of hypothesis dicts.

    Provider selection order:
      1. Provider prefix in model string (e.g. "openai:gpt-4o")
      2. Bare provider name in model string (e.g. "openai")
      3. Auto-detect from ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
      4. Fall back to `claude -p` subprocess (no API key required)

    Pass a pre-computed `summary` to avoid recomputing it (e.g. when the caller
    already computed it for display purposes).

    If `token_usage` is provided, it will be populated with `input_tokens` and
    `output_tokens` after the run (both None for the claude_code fallback).
    """
    resolved_provider, resolved_model, _ = _parse_provider_and_model(model)

    profile = NsysProfile(profile_path)

    if verbose:
        log(
            f"[local] Analyzing {profile.path.name} (provider={resolved_provider}, model={resolved_model})"
        )

    try:
        if resolved_provider == "anthropic":
            hypotheses, inp, cache_write, cache_read, out = _run_api(
                profile,
                model=resolved_model,
                max_turns=max_turns,
                verbose=verbose,
                summary=summary,
                grounded=grounded,
                log=log,
                logger=logger,
            )
        elif resolved_provider == "openai":
            hypotheses, inp, cache_write, cache_read, out = _run_openai(
                profile,
                model=resolved_model,
                max_turns=max_turns,
                verbose=verbose,
                summary=summary,
                grounded=grounded,
                log=log,
                logger=logger,
            )
        elif resolved_provider == "gemini":
            hypotheses, inp, cache_write, cache_read, out = _run_gemini(
                profile,
                model=resolved_model,
                max_turns=max_turns,
                verbose=verbose,
                summary=summary,
                grounded=grounded,
                log=log,
                logger=logger,
            )
        else:
            hypotheses, inp, out, cost_usd = _run_claude_code(
                profile,
                verbose=verbose,
                summary=summary,
                grounded=grounded,
                log=log,
                logger=logger,
            )
            cache_write = cache_read = 0
            if token_usage is not None and cost_usd is not None:
                token_usage["cost_usd"] = cost_usd
    finally:
        profile.close()

    if token_usage is not None:
        token_usage["input_tokens"] = inp
        token_usage["cache_creation_tokens"] = cache_write
        token_usage["cache_read_tokens"] = cache_read
        token_usage["output_tokens"] = out

    return hypotheses
