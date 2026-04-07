"""Pre-flight token estimation helpers for analyze and compare."""

from __future__ import annotations

# Empirically calibrated chars-per-token ratios.
# English prose (system prompts, descriptions): ~4 chars/token.
# JSON (profile summaries, diffs, tool schemas): ~2.5 chars/token.
# JSON tokenizes less efficiently than prose because of dense numeric values,
# repeated structural characters, and underscore-heavy field names.
_PROSE_CHARS_PER_TOKEN = 4.0
_JSON_CHARS_PER_TOKEN = 2.5


def estimate_prose_tokens(text: str) -> int:
    """Estimate token count for English prose using the 4-chars/token heuristic."""
    return int(len(text) / _PROSE_CHARS_PER_TOKEN)


def estimate_json_tokens(text: str) -> int:
    """Estimate token count for JSON using the 2.5-chars/token heuristic.

    JSON tokenizes significantly less efficiently than prose (~2.5 chars/token
    vs ~4) due to dense numeric values, structural punctuation, and
    underscore-separated field names.
    """
    return int(len(text) / _JSON_CHARS_PER_TOKEN)


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
