# Other Ideas — nsight_agent

## 1. Grounding the model's output

Prevent the model from leaking training knowledge into hypotheses:

- **Grounding instruction in prompts**: add to both `_SYSTEM_PROMPT_API` and `_format_summary_prompt` in `nsight_agent/agent/loop.py`:
  
  > "Ground all hypotheses strictly in the provided numbers. Do not infer algorithm names, library internals, or solver types from prior knowledge. Describe only what the data shows."

- **Evidence validation in post-processing**: after `_extract_hypotheses`, scan each `evidence` string and flag hypotheses that cite no specific numbers from the profile data as low-confidence. This does not require an additional LLM call.

**Note:** This is ultimately a design choice. Leaving the model ungrounded may actually be preferable — it allows the model to draw on broader training knowledge to make larger connections (e.g., recognizing a known algorithm pattern or a common GPU bottleneck class) that the raw numbers alone would not suggest. The tradeoff is that some hypotheses may be hallucinated or over-confident. Grounding trades recall for precision; ungrounded trading precision for recall.
