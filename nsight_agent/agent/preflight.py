"""Pre-flight token estimation helpers for analyze and compare."""

from __future__ import annotations

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
