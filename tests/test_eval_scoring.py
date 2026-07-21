"""Guards on the benchmark scorer itself.

The headline property: a profile-blind advisor must score near zero. Before the
conjunction fix, a constant that emitted one hypothesis of every ``bottleneck_type``
enum value — with no text and without opening a profile — scored 8/8 detection,
because the enum map alone cannot separate scenarios that share a coarse type.
Any future loosening of the matching rules should fail these tests first.
"""

from __future__ import annotations

import warnings
from typing import get_args

import pytest

from perf_advisor.analysis.models import BottleneckType
from perf_advisor.eval.scorer import (
    _BOTTLENECK_ENUM_MAP,
    _BOTTLENECK_KEYWORDS,
    _JUDGE_SCHEMA,
    OPTIMAL_PATH_BOTTLENECKS,
    RunScore,
    SuggestionScore,
    _coerce_judge_score,
    detection_at_k,
    detection_by_repeat,
    detection_stability,
    false_positive_count,
    matches_also_true,
    mean_reciprocal_rank,
    mrr_by_repeat,
    repeat_indices,
    score_bottleneck,
    secondary_true_count,
    suggestion_coverage_pct,
)

# Every value the advisor may emit as a bottleneck_type.
BOTTLENECK_TYPES = get_args(BottleneckType)

# The eight injected-deficiency scenarios in bench/ground_truth_meta.json.
ALL_EXPECTED = [
    "kernel_launch_overhead",
    "cpu_sync_stall",
    "pcie_transfer_bound",
    "cpu_gpu_overlap_missing",
    "unnecessary_host_staging_intranode",
    "mpi_load_imbalance",
    "host_staged_collective",
    "host_staged_halo_exchange",
]


def _hyp(bottleneck_type: str, text: str = "") -> dict[str, str]:
    return {
        "bottleneck_type": bottleneck_type,
        "description": text,
        "evidence": "",
        "suggestion": "",
    }


# ---------------------------------------------------------------------------
# Profile-blind baselines must not score
# ---------------------------------------------------------------------------


def test_enum_only_baseline_detects_nothing():
    """One empty hypothesis per enum value — the degenerate profile-blind model."""
    constant = [_hyp(t) for t in BOTTLENECK_TYPES]
    detected = [e for e in ALL_EXPECTED if score_bottleneck(constant, e)[0]]
    assert detected == [], f"enum-only baseline scored on {detected}"


def test_generic_text_baseline_scores_at_most_one():
    """A plausible-sounding constant that never reads a profile.

    It may land on at most one scenario by luck; it must not sweep the suite.
    """
    constant = [
        _hyp("cpu_launch_overhead", "many small kernel launches dominate the timeline"),
        _hyp("synchronization", "the host blocks on stream synchronization"),
        _hyp("memory_bound", "kernels appear limited by memory bandwidth"),
        _hyp("mpi_latency", "MPI communication overhead is significant"),
        _hyp("io", "host-device data transfers take substantial time"),
        _hyp("compute_bound", "kernels are compute bound"),
        _hyp("other", "consider general tuning"),
    ]
    detected = [e for e in ALL_EXPECTED if score_bottleneck(constant, e)[0]]
    assert len(detected) <= 1, f"generic baseline swept {detected}"


def test_generic_cuda_aware_answer_does_not_sweep_the_mpi_scenarios():
    """A bare "use CUDA-aware MPI" is the right fix for all three scenarios.

    Saying it without naming which one is present is not evidence the advisor
    distinguished them, so it must not score on all three.
    """
    generic = [
        _hyp("mpi_latency", "host staging detected; use CUDA-aware MPI with device pointers")
    ]
    mpi_scenarios = [
        "unnecessary_host_staging_intranode",
        "host_staged_collective",
        "host_staged_halo_exchange",
    ]
    detected = [e for e in mpi_scenarios if score_bottleneck(generic, e)[0]]
    assert detected == [], f"generic MPI answer scored on {detected}"


# ---------------------------------------------------------------------------
# Genuine diagnoses must still score
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expected", "bottleneck_type", "text"),
    [
        (
            "kernel_launch_overhead",
            "cpu_launch_overhead",
            "10k tiny kernel launches; launch overhead dominates. Use CUDA Graphs.",
        ),
        (
            "cpu_sync_stall",
            "synchronization",
            "cudaStreamSynchronize after every launch blocks the host.",
        ),
        (
            "pcie_transfer_bound",
            "memory_bound",
            "PCIe transfers dwarf compute; raise arithmetic intensity, keep data GPU-resident.",
        ),
        (
            "cpu_gpu_overlap_missing",
            "synchronization",
            "No overlap between copy and compute; use cudaMemcpyAsync on a transfer stream.",
        ),
        (
            "unnecessary_host_staging_intranode",
            "mpi_latency",
            "Ranks share a node; stage via NVLink peer access instead of host buffers.",
        ),
        (
            "mpi_load_imbalance",
            "mpi_imbalance",
            "Rank 3 does 4x the work; the others idle in MPI_Barrier. Load imbalance.",
        ),
        (
            "host_staged_collective",
            "mpi_latency",
            "MPI_Allreduce round-trips through host memory; use NCCL ncclAllReduce.",
        ),
        (
            "host_staged_halo_exchange",
            "mpi_latency",
            "Inter-node halo exchange staged through host; use GPU-Direct RDMA.",
        ),
    ],
)
def test_correct_diagnosis_is_detected(expected, bottleneck_type, text):
    detected, match_type, idx = score_bottleneck([_hyp(bottleneck_type, text)], expected)
    assert detected, f"missed a correct {expected} diagnosis"
    assert match_type == "enum+keyword"
    assert idx == 0


def test_keyword_sets_are_mutually_disjoint():
    """No keyword may be claimed by two scenarios.

    Load-bearing for the three ``mpi_latency`` scenarios, which share an enum and
    so are separated by nothing else — but asserted across all eight, since a
    duplicated keyword silently makes two scenarios interchangeable.
    """
    sets = {name: set(kws) for name, kws in _BOTTLENECK_KEYWORDS.items()}
    for a, b in ((x, y) for x in sets for y in sets if x < y):
        assert not sets[a] & sets[b], f"{a} and {b} share keywords: {sets[a] & sets[b]}"


# Rubric entries that legitimately carry another scenario's vocabulary, with why.
# Anything not listed here must not — see test_rubric_entries_are_scenario_specific.
_ALLOWED_RUBRIC_CROSS_MENTIONS = {
    # Naming the transports NCCL runs over is describing NCCL, not recommending
    # the intra-node or inter-node fix; no p2p_staged or halo answer satisfies
    # "use ncclAllReduce".
    ("mpi_allreduce", 2): "NCCL transport description mentions NVLink/RDMA",
}


def test_rubric_entries_are_scenario_specific():
    """A rubric entry must not be satisfiable by a different scenario's answer.

    The coverage metric's analogue of the detection conjunction. Before this was
    enforced, ``overlap_missing`` carried "audit for cudaDeviceSynchronize /
    cudaStreamSynchronize" — verbatim the ``sync_stall`` answer — so an advisor
    that diagnosed the wrong scenario still banked a third of the available
    coverage points on test_04.

    Vocabulary presence is a proxy for satisfiability, so genuine exceptions are
    allowlisted above rather than silently tolerated.
    """
    import json
    from pathlib import Path

    meta_path = Path(__file__).resolve().parents[1] / "bench" / "ground_truth_meta.json"
    if not meta_path.exists():  # bench/ pruned from the checkout
        pytest.skip("bench/ground_truth_meta.json not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    scenarios = {s: m for s, m in meta.items() if isinstance(m, dict)}

    offenders = []
    for scenario, entry in scenarios.items():
        own = entry["expected_bottleneck"]
        for i, sugg in enumerate(entry.get("suggestions", []), 1):
            if (scenario, i) in _ALLOWED_RUBRIC_CROSS_MENTIONS:
                continue
            text = f"{sugg['action']} {sugg['mechanism']}".lower()
            for other, keywords in _BOTTLENECK_KEYWORDS.items():
                if other == own:
                    continue
                hits = [k for k in keywords if k in text]
                if hits:
                    offenders.append(f"{scenario} #{i} reads as [{other}]: {hits}")

    assert not offenders, "rubric entries satisfiable by another scenario:\n  " + "\n  ".join(
        offenders
    )


def test_rubric_covers_every_scored_scenario():
    """Each scored scenario needs a rubric, or its coverage is silently n/a."""
    import json
    from pathlib import Path

    meta_path = Path(__file__).resolve().parents[1] / "bench" / "ground_truth_meta.json"
    if not meta_path.exists():
        pytest.skip("bench/ground_truth_meta.json not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    covered = {m["expected_bottleneck"] for m in meta.values() if isinstance(m, dict)}
    assert set(ALL_EXPECTED) <= covered, f"no rubric for {set(ALL_EXPECTED) - covered}"


def test_readme_expected_bottlenecks_match_the_scorer():
    """bench/README.md's per-test enum claims must match ``_BOTTLENECK_ENUM_MAP``.

    Each test description in § "Test descriptions" opens with a line naming the
    scenario's ``expected_bottleneck`` and the advisor ``bottleneck_type`` enum(s)
    the scorer accepts for it. That text is how a reader learns what a correct
    answer looks like without opening the scorer, so it is wrong in the most
    expensive way when it drifts: silently, and only discovered by someone acting
    on it.

    This is not hypothetical for the file. The same section described test_08's
    expected fix as "double-buffering" long after the rubric entry had been
    rewritten to the interior/boundary split, because nothing checked prose against
    the thing it described. Enum claims are the machine-checkable part of that
    surface, so they are checked here; the fix summaries still rely on review.
    """
    import json  # noqa: F401 — parity with the sibling rubric tests
    import re
    from pathlib import Path

    readme = Path(__file__).resolve().parents[1] / "bench" / "README.md"
    if not readme.exists():  # bench/ pruned from the checkout
        pytest.skip("bench/README.md not present")

    claims = re.findall(
        r"\*Expected bottleneck:\* `([a-z_]+)` — advisor enum (.+?)\.\n",
        readme.read_text(encoding="utf-8"),
    )
    assert claims, "no '*Expected bottleneck:*' lines found — did the format change?"

    mismatches = []
    for label, enum_text in claims:
        documented = set(re.findall(r"`([a-z_]+)`", enum_text))
        actual = _BOTTLENECK_ENUM_MAP.get(label)
        if actual is None:
            mismatches.append(f"{label}: documented in README, absent from the enum map")
        elif documented != actual:
            mismatches.append(
                f"{label}: README says {sorted(documented)}, scorer says {sorted(actual)}"
            )

    assert not mismatches, "bench/README.md disagrees with the scorer:\n  " + "\n  ".join(
        mismatches
    )

    # Both directions: a scenario the scorer knows about but the README never
    # describes is an undocumented test, which is the state § "Run Reference"
    # exists to prevent.
    undocumented = set(ALL_EXPECTED) - {label for label, _ in claims}
    assert not undocumented, f"no README test description for {sorted(undocumented)}"


def test_readme_lists_every_rubric_suggestion():
    """Each test description in bench/README.md must enumerate as many numbered
    fixes as the scenario has rubric suggestions.

    The README fix text is a deliberate summary of ground_truth_meta.json, not a
    copy, so only the count is machine-checkable — but the count is exactly what
    drifts when a suggestion is added or removed and the prose is hand-synced (this
    session rewrote several fix-3 entries). ground_truth_meta.json stays
    authoritative; this pins the README from silently falling a fix behind it.
    """
    import json
    import re
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "bench"
    readme_path, meta_path = bench / "README.md", bench / "ground_truth_meta.json"
    if not readme_path.exists() or not meta_path.exists():  # bench/ pruned
        pytest.skip("bench/ README or ground_truth_meta.json not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    doc = readme_path.read_text(encoding="utf-8")
    by_run = {
        v["run_id"]: v for v in meta.values() if isinstance(v, dict) and "run_id" in v
    }

    mismatches = []
    for run_id, entry in sorted(by_run.items()):
        n_rubric = len(entry.get("suggestions", []))
        block = re.search(
            rf"\*\*{re.escape(run_id)} \(.*?(?=\n\*\*test_|\n## )", doc, re.S
        )
        if not block:
            mismatches.append(f"{run_id}: no README description block")
            continue
        n_listed = len(re.findall(r"^\d+\.", block.group(0), re.M))
        if n_listed != n_rubric:
            mismatches.append(
                f"{run_id}: README lists {n_listed} fixes, rubric has {n_rubric}"
            )

    assert not mismatches, "bench/README.md out of sync with the rubric:\n  " + "\n  ".join(
        mismatches
    )


def test_readme_scenario_tags_match_the_sbatch_scripts():
    """The `scen_*` tag on each README test description must be what actually runs.

    The test -> scenario mapping is now written in three places: the sbatch scripts
    (which launch it), the scenario table, and the per-test description headings.
    The sbatch scripts are the only one of the three that is executable, so they are
    the source of truth here and the prose is checked against them.

    Also asserts the two platforms agree, which is the lockstep rule (CLAUDE.md,
    "Two benchmarks, kept in lockstep") applied to the launchers: a given test_NN
    must run the same scenario on CUDA and HIP, so a single README tag can be
    correct for both.
    """
    import re
    from pathlib import Path

    bench = Path(__file__).resolve().parents[1] / "bench"
    readme = bench / "README.md"
    scripts = sorted(bench.rglob("submit_*.sbatch"))
    if not readme.exists() or not scripts:  # bench/ pruned from the checkout
        pytest.skip("bench/ submit scripts or README.md not present")

    # What each script actually launches: a `run`/`profile_mpi` line, then the
    # --scenario flag within the same backslash-continued invocation.
    launched: dict[str, set[str]] = {}
    for script in scripts:
        lines = script.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            m = re.match(r"^(?:run|profile_mpi)\s+(test_\d+)\b", line)
            if not m:
                continue
            for cont in lines[i : i + 6]:
                sm = re.search(r"--scenario\s+(scen_[a-z])\b", cont)
                if sm:
                    launched.setdefault(m.group(1), set()).add(sm.group(1))
                    break
                if not cont.rstrip().endswith("\\"):  # invocation ended
                    break
    assert launched, "no scenario invocations parsed — did the sbatch format change?"

    split = {t: sorted(v) for t, v in launched.items() if len(v) > 1}
    assert not split, f"a test runs different scenarios per platform: {split}"

    documented = dict(
        re.findall(r"^\*\*(test_\d+) \(`(scen_[a-z])`", readme.read_text(encoding="utf-8"), re.M)
    )
    assert documented, "no tagged test descriptions found — did the heading format change?"

    mismatches = [
        f"{t}: README says {documented[t]}, sbatch launches {sorted(launched[t])[0]}"
        for t in sorted(set(documented) & set(launched))
        if {documented[t]} != launched[t]
    ]
    assert not mismatches, "bench/README.md disagrees with the submit scripts:\n  " + "\n  ".join(
        mismatches
    )

    assert not set(launched) - set(documented), (
        f"launched but undescribed in README: {sorted(set(launched) - set(documented))}"
    )
    assert not set(documented) - set(launched), (
        f"described in README but no sbatch launches it: {sorted(set(documented) - set(launched))}"
    )


def test_every_scored_scenario_has_both_an_enum_set_and_keywords():
    for expected in ALL_EXPECTED:
        assert _BOTTLENECK_ENUM_MAP.get(expected), f"{expected} has no enum set"
        assert _BOTTLENECK_KEYWORDS.get(expected), f"{expected} has no keywords"


def test_optimal_path_scenarios_are_never_detected():
    """No deficiency was injected, so there is nothing to find.

    Skipped while the benchmark has no optimal-path scenario: ``scen_h`` and
    ``scen_n`` were removed on 2026-07-20, leaving ``OPTIMAL_PATH_BOTTLENECKS``
    empty, and a loop over an empty set asserts nothing. Skipping says so out loud
    rather than reporting a pass that was never earned. Reinstating a clean-profile
    control (todo_list.md item 12) re-arms this automatically.
    """
    if not OPTIMAL_PATH_BOTTLENECKS:
        pytest.skip("no optimal-path scenarios defined — nothing to assert")
    anything = [_hyp(t, "nvlink allreduce halo cuda graph pcie") for t in BOTTLENECK_TYPES]
    for expected in OPTIMAL_PATH_BOTTLENECKS:
        assert not score_bottleneck(anything, expected)[0]


# ---------------------------------------------------------------------------
# Rank awareness
# ---------------------------------------------------------------------------


def test_detection_records_rank_of_first_match():
    noise = [_hyp("compute_bound", "kernels are compute bound")] * 4
    hit = _hyp("cpu_launch_overhead", "launch overhead dominates; use CUDA Graphs")
    detected, _, idx = score_bottleneck([*noise, hit], "kernel_launch_overhead")
    assert detected and idx == 4


def _run(idx: int | None, *, optimal: bool = False, error: str | None = None) -> RunScore:
    return RunScore(
        run_id="test_00",
        scenario="s",
        expected_bottleneck="kernel_launch_overhead",
        sqlite_paths=[],
        hypotheses=[],
        bottleneck_detected=idx is not None,
        match_type="enum+keyword" if idx is not None else None,
        matched_hypothesis_idx=idx,
        is_optimal_path=optimal,
        error=error,
    )


def test_detection_at_k_distinguishes_leading_from_burying():
    leads = [_run(0), _run(0)]
    buries = [_run(4), _run(4)]
    assert detection_at_k(leads, 1) == (2, 2)
    assert detection_at_k(buries, 1) == (0, 2)
    # Both look identical under the old any-hypothesis-anywhere tally:
    assert detection_at_k(leads, 99) == detection_at_k(buries, 99) == (2, 2)


def test_mean_reciprocal_rank():
    assert mean_reciprocal_rank([_run(0), _run(0)]) == pytest.approx(1.0)
    assert mean_reciprocal_rank([_run(1), _run(1)]) == pytest.approx(0.5)
    assert mean_reciprocal_rank([_run(None), _run(None)]) == pytest.approx(0.0)
    assert mean_reciprocal_rank([]) is None


def test_errored_and_optimal_runs_excluded_from_denominator():
    results = [_run(0), _run(None, optimal=True), _run(None, error="boom")]
    assert detection_at_k(results, 1) == (1, 1)


# ---------------------------------------------------------------------------
# Judge score coercion — malformed replies skip, they do not score zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [{"score": 0}, {"score": 1}, {"score": 2}])
def test_valid_judge_scores_pass_through(payload):
    score, err = _coerce_judge_score(payload)
    assert score == payload["score"] and err is None


def test_judge_score_accepts_numeric_strings_and_floats():
    assert _coerce_judge_score({"score": "2"})[0] == 2
    assert _coerce_judge_score({"score": 2.0})[0] == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"explanation": "covered"},  # missing key — the original bug
        {"rating": 2},  # wrong key name
        {"score": None},
        {"score": "full"},
        {"score": 5},  # outside the rubric
        {"score": -1},
        {"score": True},  # bool is not a rubric verdict
    ],
)
def test_malformed_judge_replies_are_skipped_not_zeroed(payload):
    score, err = _coerce_judge_score(payload)
    assert score == -1, f"{payload!r} was silently scored instead of skipped"
    assert err


# ---------------------------------------------------------------------------
# Coverage: "nothing judged" is not "covered nothing"
# ---------------------------------------------------------------------------


def _sugg(score: int) -> SuggestionScore:
    return SuggestionScore(action="a", mechanism="m", score=score, explanation="")


def test_coverage_is_none_when_nothing_was_judged():
    assert suggestion_coverage_pct([]) is None
    assert suggestion_coverage_pct([_sugg(-1), _sugg(-1)]) is None


def test_coverage_zero_is_distinct_from_none():
    assert suggestion_coverage_pct([_sugg(0), _sugg(0)]) == 0.0


def test_skipped_scores_leave_the_denominator_alone():
    # 2/2 on the one judged item -> 100%, not 50%.
    assert suggestion_coverage_pct([_sugg(2), _sugg(-1)]) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Structured outputs: schema shape, and the fallback when a model rejects it
# ---------------------------------------------------------------------------


def test_judge_schema_satisfies_both_providers_strict_mode():
    """Strict mode on Anthropic and OpenAI both require these exact properties."""
    schema = _JUDGE_SCHEMA
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    # Every declared property must be required, or strict mode rejects the schema.
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["score"]["enum"] == [0, 1, 2]


def test_schema_constrains_score_to_the_rubric_values():
    """The schema's enum and the coercion helper must agree on what is valid."""
    for value in _JUDGE_SCHEMA["properties"]["score"]["enum"]:
        assert _coerce_judge_score({"score": value, "explanation": "x"})[0] == value
    for value in (-1, 3, 99):
        assert value not in _JUDGE_SCHEMA["properties"]["score"]["enum"]
        assert _coerce_judge_score({"score": value, "explanation": "x"})[0] == -1


def _patch_anthropic_transport(monkeypatch, handler):
    """Point ``anthropic.Anthropic()`` at a mock transport driven by ``handler``.

    The real class is captured *before* patching — building the stub inside the
    replacement would recurse.
    """
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    real_cls = anthropic.Anthropic
    client = real_cls(
        api_key="test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
    return client


def _message_response(httpx, text: str):
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def test_judge_sends_the_schema_on_the_wire(monkeypatch):
    """The schema must reach the request body, not just the SDK call site.

    ``output_config`` goes through ``extra_body`` because it is not a typed
    parameter on older anthropic SDKs (0.50.0 in the runtime conda env), only on
    newer ones. This asserts the version-stable path actually serializes.
    """
    import json as _json

    from perf_advisor.eval import scorer

    httpx = pytest.importorskip("httpx")
    seen = {}

    def handler(request):
        seen.update(_json.loads(request.content))
        return _message_response(httpx, '{"score": 2, "explanation": "ok"}')

    monkeypatch.setattr(scorer, "_STRUCTURED_OUTPUT_WARNED", set())
    _patch_anthropic_transport(monkeypatch, handler)

    result = scorer._judge_anthropic("prompt", "claude-haiku-4-5-20251001")
    assert result == {"score": 2, "explanation": "ok"}
    assert seen["output_config"]["format"]["type"] == "json_schema"
    assert seen["output_config"]["format"]["schema"] == _JUDGE_SCHEMA


def test_judge_falls_back_when_the_model_rejects_structured_outputs(monkeypatch):
    """--judge-model accepts any model; an unsupported one must not fail the run."""
    from perf_advisor.eval import scorer

    httpx = pytest.importorskip("httpx")
    bodies = []

    def handler(request):
        import json as _json

        body = _json.loads(request.content)
        bodies.append(body)
        if "output_config" in body:
            return httpx.Response(
                400,
                json={"type": "error", "error": {
                    "type": "invalid_request_error",
                    "message": "output_config: unsupported for this model",
                }},
            )
        return _message_response(httpx, '{"score": 1, "explanation": "partial"}')

    monkeypatch.setattr(scorer, "_STRUCTURED_OUTPUT_WARNED", set())
    _patch_anthropic_transport(monkeypatch, handler)

    with pytest.warns(RuntimeWarning, match="rejected structured outputs"):
        result = scorer._judge_anthropic("prompt", "claude-legacy")

    assert result == {"score": 1, "explanation": "partial"}
    assert len(bodies) == 2, "expected a structured attempt then an unstructured retry"
    assert "output_config" in bodies[0] and "output_config" not in bodies[1]

    # Warned once per model, not once per suggestion.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert scorer._judge_anthropic("prompt", "claude-legacy")["score"] == 1


def test_transient_errors_are_not_retried_as_unsupported(monkeypatch):
    """A 429/500 must propagate, not be mistaken for 'model lacks the feature'."""
    from perf_advisor.eval import scorer

    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx")
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )

    monkeypatch.setattr(scorer, "_STRUCTURED_OUTPUT_WARNED", set())
    _patch_anthropic_transport(monkeypatch, handler)

    with pytest.raises(anthropic.RateLimitError):
        scorer._judge_anthropic("prompt", "claude-haiku-4-5-20251001")
    assert len(calls) == 1, "a rate limit must not trigger the unstructured retry"


# ---------------------------------------------------------------------------
# also_true: known-true secondary observations are excused from the FP count,
# but never count as detecting the injected bottleneck.
# ---------------------------------------------------------------------------

_BARRIER_ALSO_TRUE = [
    {
        "bottleneck_type": ["synchronization", "mpi_imbalance"],
        "keywords": ["mpi_barrier", "barrier"],
        "why": "harness barrier_sync() runs inside the capture window",
    }
]


def test_also_true_never_counts_as_detection():
    """The whole point: it excuses a false positive, it does not find the bug."""
    h = [_hyp("mpi_imbalance", "MPI_Barrier dominates; ranks wait at the barrier")]
    assert matches_also_true(h[0], _BARRIER_ALSO_TRUE)
    detected, _, _ = score_bottleneck(h, "host_staged_collective")
    assert not detected, "an also_true match must not satisfy detection"


def test_also_true_is_excluded_from_false_positives():
    h = [_hyp("mpi_imbalance", "MPI_Barrier dominates; ranks wait at the barrier")]
    assert false_positive_count(h, "host_staged_collective") == 1
    assert false_positive_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 0
    assert secondary_true_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 1


def test_unrelated_hypotheses_still_count_as_false_positives():
    """also_true must not become a blanket excuse."""
    h = [_hyp("compute_bound", "kernels are compute bound; increase occupancy")]
    assert false_positive_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 1
    assert secondary_true_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 0


def test_also_true_requires_enum_and_keyword():
    """Same conjunction discipline as detection — either alone is insufficient."""
    enum_only = _hyp("mpi_imbalance", "ranks are doing uneven amounts of work")
    kw_only = _hyp("compute_bound", "MPI_Barrier shows up prominently")
    assert not matches_also_true(enum_only, _BARRIER_ALSO_TRUE)
    assert not matches_also_true(kw_only, _BARRIER_ALSO_TRUE)


def test_primary_match_is_not_double_counted_as_secondary():
    """A hypothesis that finds the real bottleneck is neither FP nor secondary."""
    h = [_hyp("mpi_latency", "MPI_Allreduce round-trips through host; use NCCL")]
    assert false_positive_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 0
    assert secondary_true_count(h, "host_staged_collective", _BARRIER_ALSO_TRUE) == 0
    assert score_bottleneck(h, "host_staged_collective")[0]


def test_missing_also_true_is_inert():
    h = [_hyp("compute_bound", "unrelated")]
    assert false_positive_count(h, "host_staged_collective", None) == 1
    assert false_positive_count(h, "host_staged_collective", []) == 1


def test_rubric_also_true_entries_are_well_formed():
    """Every also_true entry must carry both matchers and a documented reason."""
    import json
    from pathlib import Path

    meta_path = Path(__file__).resolve().parents[1] / "bench" / "ground_truth_meta.json"
    if not meta_path.exists():
        pytest.skip("bench/ground_truth_meta.json not present")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    seen = 0
    for scenario, entry in meta.items():
        if not isinstance(entry, dict):
            continue
        own = entry["expected_bottleneck"]
        primary = _BOTTLENECK_ENUM_MAP.get(own, set())
        for e in entry.get("also_true", []):
            seen += 1
            assert e.get("bottleneck_type"), f"{scenario}: also_true needs bottleneck_type"
            assert e.get("keywords"), f"{scenario}: also_true needs keywords"
            assert e.get("why"), f"{scenario}: also_true needs a documented reason"
            # Keep secondary enums disjoint from the primary accept set, so a
            # single hypothesis can never be both the answer and an excuse.
            overlap = set(e["bottleneck_type"]) & primary
            assert not overlap, f"{scenario}: also_true overlaps primary enums {overlap}"
    assert seen, "expected at least one also_true entry in the rubric"


# ---------------------------------------------------------------------------
# Repeated runs: per-repetition aggregates and per-scenario stability
# ---------------------------------------------------------------------------


def _rrun(run_id, scenario, idx, rep, *, error=None, optimal=False):
    return RunScore(
        run_id=run_id,
        scenario=scenario,
        expected_bottleneck="kernel_launch_overhead",
        sqlite_paths=[],
        hypotheses=[],
        bottleneck_detected=idx is not None,
        match_type="enum+keyword" if idx is not None else None,
        matched_hypothesis_idx=idx,
        repeat=rep,
        error=error,
        is_optimal_path=optimal,
    )


def test_repeat_indices_are_deduped_and_sorted():
    runs = [_rrun("t1", "s", 0, 2), _rrun("t1", "s", 0, 0), _rrun("t2", "s", 0, 2)]
    assert repeat_indices(runs) == [0, 2]


def test_detection_is_computed_per_repetition_not_pooled():
    """One observation per repeat — pooling would understate the spread."""
    runs = [
        _rrun("t1", "a", 0, 0), _rrun("t2", "b", 0, 0),      # repeat 0: 2/2
        _rrun("t1", "a", 0, 1), _rrun("t2", "b", None, 1),   # repeat 1: 1/2
    ]
    assert detection_by_repeat(runs, 1) == [(2, 2), (1, 2)]


def test_mrr_by_repeat():
    runs = [_rrun("t1", "a", 0, 0), _rrun("t1", "a", 1, 1)]
    vals = mrr_by_repeat(runs)
    assert vals == pytest.approx([1.0, 0.5])


def test_stability_distinguishes_always_sometimes_never():
    """The three cases an aggregate averages into one number."""
    runs = []
    for rep in range(4):
        runs.append(_rrun("t1", "always", 0, rep))
        runs.append(_rrun("t2", "sometimes", 0 if rep % 2 else None, rep))
        runs.append(_rrun("t3", "never", None, rep))
    stability = {scen: (hits, obs) for _, scen, hits, obs in detection_stability(runs)}
    assert stability["always"] == (4, 4)
    assert stability["sometimes"] == (2, 4)
    assert stability["never"] == (0, 4)


def test_stability_excludes_errored_repeats_from_the_denominator():
    """A crashed repeat is missing data, not a miss."""
    runs = [
        _rrun("t1", "s", 0, 0),
        _rrun("t1", "s", None, 1, error="boom"),
        _rrun("t1", "s", 0, 2),
    ]
    assert detection_stability(runs) == [("t1", "s", 2, 2)]


def test_stability_excludes_optimal_path_runs():
    runs = [_rrun("t1", "s", 0, 0), _rrun("t9", "opt", None, 0, optimal=True)]
    assert [row[1] for row in detection_stability(runs)] == ["s"]


def test_stability_preserves_run_order():
    runs = [_rrun("t3", "c", 0, 0), _rrun("t1", "a", 0, 0), _rrun("t3", "c", 0, 1)]
    assert [row[0] for row in detection_stability(runs)] == ["t3", "t1"]
