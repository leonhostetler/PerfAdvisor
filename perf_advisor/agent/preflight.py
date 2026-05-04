"""Pre-flight checks and token estimation helpers for analyze and compare."""

from __future__ import annotations

from collections.abc import Callable

# Empirically calibrated chars-per-token ratios.
# English prose (system prompts, descriptions): ~4 chars/token across all providers.
# JSON tokenization efficiency varies by provider tokenizer:
#   Anthropic (Claude tokenizer): ~2.5 chars/token — dense numerics and structural
#     punctuation tokenize less efficiently.
#   OpenAI cl100k_base, Gemini SentencePiece: larger vocabularies and better BPE
#     for technical content yield ~3.5 chars/token on profile JSON.
_PROSE_CHARS_PER_TOKEN = 4.0
_JSON_CHARS_PER_TOKEN: dict[str, float] = {
    "anthropic": 2.5,
    "openai": 3.5,
    "gemini": 3.5,
}
_JSON_CHARS_PER_TOKEN_DEFAULT = 2.5


def estimate_prose_tokens(text: str) -> int:
    """Estimate token count for English prose using the 4-chars/token heuristic."""
    return int(len(text) / _PROSE_CHARS_PER_TOKEN)


def estimate_json_tokens(text: str, provider: str = "anthropic") -> int:
    """Estimate token count for JSON using a provider-specific chars/token ratio.

    Ratio varies by tokenizer: Anthropic ~2.5, OpenAI/Gemini ~3.5.
    JSON tokenizes less efficiently than prose for Anthropic due to dense numeric
    values and structural punctuation, but OpenAI and Gemini tokenizers handle
    JSON more efficiently owing to larger vocabularies.
    """
    ratio = _JSON_CHARS_PER_TOKEN.get(provider, _JSON_CHARS_PER_TOKEN_DEFAULT)
    return int(len(text) / ratio)


def count_tokens_exact(
    provider: str,
    model: str,
    system_prompt: str,
    *user_texts: str,
    tools: list | None = None,
) -> int | None:
    """Return an exact input token count using the provider's counting API.

    Currently only Anthropic's ``client.messages.count_tokens`` is supported.
    Returns ``None`` for any other provider or if the API call fails; callers
    should fall back to the heuristic functions in that case.
    """
    if provider != "anthropic":
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        content = "\n\n".join(user_texts)
        kwargs: dict = dict(
            model=model,
            system=system_prompt,
            messages=[{"role": "user", "content": content}],
        )
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.count_tokens(**kwargs)
        return resp.input_tokens
    except Exception:
        return None


# Average tokens added to the conversation per agent turn (tool result + assistant reasoning).
# Range in practice: 500 (small structured call) – 3,000 (large sql_query result).
_AVG_INCREMENT_TOKENS = 1500


def estimate_total_session_tokens(
    initial_tokens: int,
    max_turns: int,
    avg_increment: int = _AVG_INCREMENT_TOKENS,
) -> int:
    """Estimate total input tokens across a multi-turn session without caching.

    All three provider backends (Anthropic, OpenAI, Gemini) are stateless: each
    API call resends the full conversation history from the beginning.  The total
    across N turns is:

        Σ(k=1..N) [initial + (k-1) * increment]
        = N * initial + increment * N * (N-1) / 2
    """
    return initial_tokens * max_turns + avg_increment * max_turns * (max_turns - 1) // 2


def estimate_gemini_cache_breakdown(
    initial_tokens: int,
    max_turns: int,
    avg_increment: int = _AVG_INCREMENT_TOKENS,
) -> dict[str, int]:
    """Estimate Gemini explicit context cache token breakdown for a multi-turn run.

    Gemini caches a fixed prefix (system instruction + tool declarations + profile
    summary) once at the start of the session.  Every turn reads this prefix at the
    cached-token rate (0.25× for Gemini 2.5 models).  Only the incremental per-turn
    history (tool results + assistant reasoning) is billed at the full input rate.

    Returns a dict with:
      cache_write     — tokens in the created cache object (billed at 1.0×, one-time)
      cache_read      — total cached tokens read across all turns (billed at 0.25× each)
      input           — non-cached input tokens (per-turn incremental history, summed)
      cost_equivalent — weighted total: write×1.0 + read×0.25 + input×1.0
    """
    cache_write = initial_tokens
    cache_read = initial_tokens * max_turns
    input_non_cached = avg_increment * max_turns * (max_turns - 1) // 2
    cost_equivalent = int(cache_write + cache_read * 0.25 + input_non_cached)
    return {
        "cache_write": cache_write,
        "cache_read": cache_read,
        "input": input_non_cached,
        "cost_equivalent": cost_equivalent,
    }


def estimate_cache_breakdown(
    initial_tokens: int,
    max_turns: int,
    avg_increment: int = _AVG_INCREMENT_TOKENS,
) -> dict[str, int]:
    """Estimate Anthropic prompt-cache token breakdown for a sliding-cache multi-turn run.

    With sliding cache, every turn extends the cache boundary to include the latest
    tool-result exchange.  The breakdown across N turns is:

      Turn 1:   cache_creation = initial_tokens,         cache_read = 0
      Turn k≥2: cache_creation = avg_increment,          cache_read = initial + (k-2)*increment

    Returns a dict with:
      cache_creation  — total tokens written to cache (billed at 1.25×)
      cache_read      — total tokens read from cache (billed at 0.10×)
      input           — non-cached input tokens (≈ 0 with full prefix coverage)
      cost_equivalent — weighted total: creation×1.25 + read×0.10
    """
    # Turn 1 writes initial_tokens; each subsequent turn writes avg_increment
    cache_creation = initial_tokens + max(0, max_turns - 1) * avg_increment
    # Sum of cache reads: Σ(k=2..N) [initial + (k-2)*increment]
    if max_turns >= 2:
        n = max_turns - 1
        cache_read = n * initial_tokens + avg_increment * n * (n - 1) // 2
    else:
        cache_read = 0
    cost_equivalent = int(cache_creation * 1.25 + cache_read * 0.10)
    return {
        "cache_creation": cache_creation,
        "cache_read": cache_read,
        "input": 0,
        "cost_equivalent": cost_equivalent,
    }


# ---------------------------------------------------------------------------
# Profile readiness checks
# ---------------------------------------------------------------------------

# Remediation messages keyed by symptom; printed by run_preflight and
# re-used by the tool_sql_query OperationalError path for consistency.

_ROCPD_MSG_WRITER_TRUNCATED = (
    "GPU activity tables are empty and a SQLite journal sidecar is present — the rocpd writer "
    "was likely killed before flushing. Re-run with longer walltime and a SIGTERM grace period "
    "(e.g. `--signal=SIGTERM@300`) so the writer can finalize. The HIP/HSA API regions on disk "
    "are still usable but kernel/memcpy analysis will be sparse."
)
_ROCPD_MSG_NARROW_FLAG_SET = (
    "GPU activity tables are empty. To enable kernel timing and memcpy analysis, re-capture with "
    "`rocprofv3 --sys-trace --output-format rocpd` (the bundle includes "
    "`--kernel-trace --memory-copy-trace --hip-trace --hsa-trace --marker-trace "
    "--rccl-trace --scratch-memory-trace`). Falling back to API-only metrics for this run."
)
_ROCPD_MSG_NO_MARKERS = (
    "No marker ranges (rocTX) found — phase detection will not run. "
    "To enable phase detection, instrument the app with `roctxRangePush/Pop` (or `roctxMark`) "
    "around the iterations / regions you want labeled, and re-capture with `--sys-trace` "
    "(or at least `--marker-trace`). PerfAdvisor will still produce kernel-level analysis "
    "without phases."
)
_ROCPD_MSG_NO_MPI = (
    "No MPI ranges found — rocprofv3 doesn't trace MPI. For cross-rank collective-imbalance "
    "analysis, capture with `rocprof-sys` (`ROCPROFSYS_USE_MPI=true`) or run a parallel "
    "CrayPat capture. PerfAdvisor will skip MPI-overlap hypotheses for this profile."
)


def run_preflight(profile: object, *, log: Callable[[str], None] = print) -> None:
    """Print readiness hints for the given profile.

    For NSYS profiles: notes on missing MPI or marker data.
    For ROCPD profiles: reads emptiness diagnostics and emits actionable
    remediation messages for each detected symptom.
    """
    from perf_advisor.ingestion.base import Format

    fmt = getattr(profile, "format", None)
    caps = getattr(profile, "capabilities", None)

    if fmt == Format.NSYS:
        if caps is not None and not caps.has_mpi:
            log(
                "[preflight] No MPI data in this Nsight Systems profile. "
                "Cross-rank MPI imbalance analysis will be skipped."
            )
        if caps is not None and not caps.has_markers:
            log(
                "[preflight] No NVTX marker ranges found. "
                "Phase detection requires NVTX annotations — analysis will use the whole "
                "profile as a single phase."
            )

    elif fmt == Format.ROCPD:
        emptiness = getattr(profile, "emptiness", None)
        if emptiness is None:
            return

        _key_tables = frozenset(
            {"rocpd_kernel_dispatch", "rocpd_memory_copy", "rocpd_memory_allocate"}
        )
        all_key_empty = _key_tables.issubset(emptiness.empty_tables)

        if all_key_empty and emptiness.writer_truncation_suspected:
            log(f"[preflight] WARNING: {_ROCPD_MSG_WRITER_TRUNCATED}")
        elif all_key_empty:
            log(f"[preflight] WARNING: {_ROCPD_MSG_NARROW_FLAG_SET}")

        if caps is not None and not caps.has_markers:
            log(f"[preflight] NOTE: {_ROCPD_MSG_NO_MARKERS}")

        if caps is not None and not caps.has_mpi:
            log(f"[preflight] NOTE: {_ROCPD_MSG_NO_MPI}")
