"""Tests for Hypothesis model validation and canonicalisation.

The agent receives free-form JSON from an LLM; these tests pin down that
near-miss enum spellings are normalised rather than silently passed through
to the eval scorer (which compares against canonical enum values).
"""

from __future__ import annotations

from perf_advisor.agent.loop import _extract_hypotheses, _validate_hypotheses
from perf_advisor.analysis.models import Hypothesis


def test_canonical_values_pass_through_unchanged():
    h = Hypothesis.model_validate(
        {
            "bottleneck_type": "memory_bound",
            "expected_impact": "high",
            "action_category": "code_optimization",
            "confidence": "high",
        }
    )
    assert h.bottleneck_type == "memory_bound"
    assert h.expected_impact == "high"
    assert h.action_category == "code_optimization"
    assert h.confidence == "high"
    assert h.coercion_notes == []


def test_hyphenated_spelling_is_canonicalised():
    """The exact failure mode that made the eval scorer report a false miss."""
    h = Hypothesis.model_validate({"bottleneck_type": "memory-bound"})
    assert h.bottleneck_type == "memory_bound"
    assert h.coercion_notes == []


def test_case_and_whitespace_are_normalised():
    h = Hypothesis.model_validate(
        {"bottleneck_type": "  MPI_Imbalance ", "expected_impact": "HIGH"}
    )
    assert h.bottleneck_type == "mpi_imbalance"
    assert h.expected_impact == "high"


def test_known_aliases_map_to_canonical_values():
    assert Hypothesis.model_validate({"bottleneck_type": "load imbalance"}).bottleneck_type == (
        "mpi_imbalance"
    )
    assert Hypothesis.model_validate({"bottleneck_type": "sync"}).bottleneck_type == (
        "synchronization"
    )
    assert Hypothesis.model_validate(
        {"action_category": "kernel optimization"}
    ).action_category == ("code_optimization")


def test_unrecognised_value_falls_back_and_is_recorded():
    h = Hypothesis.model_validate({"bottleneck_type": "quantum_entanglement"})
    assert h.bottleneck_type == "other"
    assert len(h.coercion_notes) == 1
    assert "quantum_entanglement" in h.coercion_notes[0]


def test_out_of_range_runtime_fraction_is_clamped():
    h = Hypothesis.model_validate({"runtime_fraction_pct": 150.0})
    assert h.runtime_fraction_pct == 100.0
    assert h.coercion_notes and "clamped" in h.coercion_notes[0]

    h_neg = Hypothesis.model_validate({"runtime_fraction_pct": -5.0})
    assert h_neg.runtime_fraction_pct == 0.0


def test_non_numeric_runtime_fraction_is_dropped():
    h = Hypothesis.model_validate({"runtime_fraction_pct": "about half"})
    assert h.runtime_fraction_pct is None
    assert h.coercion_notes


def test_unknown_extra_fields_are_preserved():
    """Models sometimes add useful fields; validation must not discard them."""
    h = Hypothesis.model_validate({"bottleneck_type": "io", "affected_kernels": ["k1", "k2"]})
    dumped = h.model_dump()
    assert dumped["affected_kernels"] == ["k1", "k2"]


def test_validate_drops_non_object_entries():
    assert _validate_hypotheses([{"bottleneck_type": "io"}, "garbage", 42]) == [
        Hypothesis.model_validate({"bottleneck_type": "io"}).model_dump()
    ]


def test_extract_hypotheses_validates_output():
    text = '[{"bottleneck_type": "memory-bound", "description": "d"}]'
    out = _extract_hypotheses(text)
    assert len(out) == 1
    assert out[0]["bottleneck_type"] == "memory_bound"


def test_extract_hypotheses_handles_brackets_in_kernel_names():
    """Regression guard for the ROCm 'kernel [clone .kd]' case."""
    text = '[{"bottleneck_type": "compute_bound", "evidence": "kernel_a [clone .kd] hot"}]'
    out = _extract_hypotheses(text)
    assert len(out) == 1
    assert out[0]["evidence"] == "kernel_a [clone .kd] hot"


def test_missing_fields_get_safe_defaults():
    out = _extract_hypotheses('[{"description": "something slow"}]')
    assert out[0]["bottleneck_type"] == "other"
    assert out[0]["confidence"] == "low"
    assert out[0]["action_category"] is None


# --- Amdahl speedup bounds -------------------------------------------------


def test_speedup_bounds_are_derived_not_trusted():
    """Model-supplied bounds are discarded and recomputed from the fraction."""
    h = Hypothesis.model_validate(
        {
            "runtime_fraction_pct": 50.0,
            "estimated_speedup_pct_lower": 999.0,  # nonsense from the model
            "estimated_speedup_pct_upper": 999.0,
        }
    )
    # F=0.5: lower = 1/(1-0.25)-1 = 33.3%, upper = 1/(1-0.5)-1 = 100%
    assert h.estimated_speedup_pct_lower == 33.3
    assert h.estimated_speedup_pct_upper == 100.0


def test_speedup_bounds_known_values():
    h = Hypothesis.model_validate({"runtime_fraction_pct": 20.0})
    # F=0.2: lower = 1/0.9-1 = 11.1%, upper = 1/0.8-1 = 25%
    assert h.estimated_speedup_pct_lower == 11.1
    assert h.estimated_speedup_pct_upper == 25.0


def test_zero_fraction_gives_zero_speedup():
    h = Hypothesis.model_validate({"runtime_fraction_pct": 0.0})
    assert h.estimated_speedup_pct_lower == 0.0
    assert h.estimated_speedup_pct_upper == 0.0


def test_null_fraction_nulls_both_bounds():
    h = Hypothesis.model_validate(
        {"estimated_speedup_pct_lower": 50.0, "estimated_speedup_pct_upper": 80.0}
    )
    assert h.runtime_fraction_pct is None
    assert h.estimated_speedup_pct_lower is None
    assert h.estimated_speedup_pct_upper is None


def test_full_fraction_gives_unbounded_upper():
    """F=1 means eliminating the bottleneck leaves zero runtime: no finite bound."""
    h = Hypothesis.model_validate({"runtime_fraction_pct": 100.0})
    assert h.estimated_speedup_pct_upper is None
    assert h.estimated_speedup_pct_lower == 100.0
    assert any("unbounded" in n for n in h.coercion_notes)


def test_upper_bound_always_at_least_lower_bound():
    for pct in (1.0, 10.0, 33.3, 50.0, 75.0, 99.0):
        h = Hypothesis.model_validate({"runtime_fraction_pct": pct})
        assert h.estimated_speedup_pct_upper >= h.estimated_speedup_pct_lower


def test_bounds_derived_after_clamping():
    """An out-of-range fraction is clamped first, then bounds follow the clamp."""
    h = Hypothesis.model_validate({"runtime_fraction_pct": -10.0})
    assert h.runtime_fraction_pct == 0.0
    assert h.estimated_speedup_pct_upper == 0.0
