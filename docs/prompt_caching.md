# Prompt Caching

## Motivation

PerfAdvisor's agent loop replays the full conversation history on every API call. For a typical 18-turn analysis run, the system prompt (≈1,500 tokens), tool schemas (≈3,000 tokens), and the pre-seeded profile summary (≈4,000–8,000 tokens) would otherwise be re-billed at full input cost on every turn. Without caching, 80–90% of the tokens in each request are identical to the previous turn.

Prompt caching addresses this: the static prefix is written to a provider-managed cache on the first call and re-read at a fraction of the normal input cost on all subsequent calls. For a typical 18-turn run with a 12,000-token prefix, this reduces billable input tokens by approximately 75–80%.

Pre-seeding (injecting the pre-computed `profile_summary` and `phase_summary` results as fake tool turns before the first API call) is a complementary optimization that saves 2–3 API round-trips. Caching makes pre-seeding even more cost-effective: the pre-seed block is written to cache once and read cheaply on every subsequent turn.

---

## Anthropic

### Caching mechanism

Anthropic's API supports explicit prompt caching via `cache_control: {"type": "ephemeral"}` markers attached to individual content blocks. A marked block — and everything before it — is written to cache on the first call and served at **0.10×** cost (10% of the normal input price) on subsequent calls within the cache TTL. Cache creation is billed at **1.25×** the normal input price. The Anthropic API allows at most **4 `cache_control` markers** per request.

### What we implemented

We use a **sliding cache window** that occupies three of the four marker slots (`_run_api` in `agent/loop.py`); the fourth is left in reserve:

1. **Permanent pre-seed marker** (`_preseed_messages`): The last tool result in the pre-seed block receives a permanent `cache_control: {"type": "ephemeral"}`. This covers the full static prefix — system prompt, tool schemas, and pre-seeded profile summary — as a single cache unit. It is set once and never moved.

   When pre-seeding is disabled (no summary provided), the marker falls back to the system prompt content block directly.

2. **Sliding per-turn markers**: Two additional "floating" markers track the two most recently added user messages. After each tool-call round-trip:
   - The newest user message (current tool results) receives a `cache_control` marker.
   - The previous floating marker is retained.
   - When adding the new marker would exceed the three-floating-marker budget (4 total minus the permanent one), the oldest floating marker is stripped from its message content.

   On turn N, the entire context through turn N−1 is served from cache; only the new tool-result exchange is billed as cache-creation input.

3. **Token accounting**: `cache_creation_input_tokens` and `cache_read_input_tokens` are read from each response's `usage` object and accumulated across turns. The CLI reports them in the post-run token summary alongside a computed cost-equivalent token count.

### Savings

For a typical 18-turn run:
- **Cache write**: ~75,000–100,000 tokens (system + tools + pre-seed + incremental turns), billed once at 1.25×
- **Cache read**: prefix-size × (turns − 1), billed at 0.10× per turn
- **Net reduction**: roughly 75–80% fewer billable input tokens compared to no caching

---

## OpenAI

### Caching mechanism

OpenAI applies **automatic prefix caching** to requests that share a common prefix. No developer action is required — cached tokens appear in `response.usage.prompt_tokens_details.cached_tokens`. Cached tokens are billed at approximately **0.50×** the standard input price.

### What we implemented

No explicit markers are needed. The multi-turn agent loop (`_run_openai`) gets automatic caching through its message structure:

- The system prompt is prepended as the first `{"role": "system"}` message on every turn, making the long static prefix eligible for prefix caching.
- Pre-seeding uses OpenAI's `tool_calls` / `role:"tool"` message format, keeping the prefix structurally stable across turns.

The verbose output reads `prompt_tokens_details.cached_tokens` from each response and reports it alongside the total context size per turn.

The `compare` subcommand's OpenAI backend (`_call_openai` in `agent/compare.py`) sends a single-turn request with no pre-seeding; prefix caching provides no benefit there.

---

## Gemini

### Caching mechanism

Gemini supports **explicit context caching** via the `client.caches.create()` API. A named `CachedContent` object is created once before the first turn, containing the system instruction, tool declarations, and an initial user message. Subsequent generation calls reference the cache by name via `GenerateContentConfig(cached_content=name)`. Cached tokens are billed at **0.25×** the normal input price for Gemini 2.5 models. The cache has a configurable TTL.

Unlike Anthropic's sliding window, the Gemini cache is a fixed object — the same prefix is reused across all turns without updating it.

### What we implemented

In `_run_gemini` (`agent/loop.py`), before the first message is sent:

1. A `CachedContent` is created with a 600-second TTL containing:
   - `system_instruction`: the full system prompt text
   - `tools`: a `Tool` object wrapping the `FunctionDeclaration` list
   - `contents`: a single user-role message with the pre-computed profile summary text (combining `profile_summary` + `phase_summary`, and `cross_rank_summary` if present)

2. All turns use `client.chats.create(model=..., config=GenerateContentConfig(cached_content=_cached_content.name))`. Every message in the session reads the fixed cached prefix at 0.25×.

3. **Fallback**: If cache creation fails — for example, because the profile summary is below Gemini's minimum-token threshold for caching, or the model does not support the caching API — the code falls back silently to injecting the summary into the first user message with no caching. The agent loop proceeds normally in this case.

4. **Token accounting**: `_cache_creation_tokens` is recorded from `_cached_content.usage_metadata.total_token_count` at creation time. Per-turn `cached_content_token_count` from `usage_metadata` is accumulated in `cache_read_tokens` and included in the post-run summary.

### Savings

For a typical 18-turn run, caching the prefix (~12,000 tokens) at 0.25× across 18 turns saves roughly 16× the one-time cache creation cost. The break-even point is a two-turn run; any run longer than that benefits from caching.

---

## Claude Code fallback

The `claude_code` backend invokes `claude -p <prompt> --output-format json` as a subprocess. It is a single-shot call with no multi-turn tool-use loop, so explicit prompt caching is not applicable. The code reads `cache_creation_input_tokens` and `cache_read_input_tokens` from the JSON output and folds them into the reported input token total; the subprocess may apply its own internal caching.
