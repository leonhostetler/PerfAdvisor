"""Tests for capability-gap notes (analysis.diagnostics)."""

from __future__ import annotations

from perf_advisor.analysis.diagnostics import CapabilityNote, capability_notes
from perf_advisor.ingestion.base import Format, ProfileCapabilities


def _caps(**overrides) -> ProfileCapabilities:
    """Full-capability baseline; individual flags can be overridden."""
    defaults = dict(
        has_kernels=True,
        has_memcpy=True,
        has_runtime_api=True,
        has_markers=True,
        has_mpi=True,
        has_cpu_samples=False,
        has_pmc_counters=True,
        has_sysmetrics=False,
        schema_version="5.0",
    )
    defaults.update(overrides)
    return ProfileCapabilities(**defaults)


def codes(notes: list[CapabilityNote]) -> list[str]:
    return [n.code for n in notes]


# ---------------------------------------------------------------------------
# Full-capability profiles should produce no notes
# ---------------------------------------------------------------------------


def test_nsys_full_caps_no_notes():
    assert capability_notes(Format.NSYS, _caps()) == []


def test_rocpd_full_caps_no_notes():
    assert capability_notes(Format.ROCPD, _caps()) == []


# ---------------------------------------------------------------------------
# nsys notes — individual flags
# ---------------------------------------------------------------------------


def test_nsys_no_mpi():
    notes = capability_notes(Format.NSYS, _caps(has_mpi=False))
    assert codes(notes) == ["N1"]
    assert "nsys profile" in notes[0].message
    assert "--mpi-impl" in notes[0].message


def test_nsys_no_markers():
    notes = capability_notes(Format.NSYS, _caps(has_markers=False))
    assert codes(notes) == ["N2"]
    assert "nvtx" in notes[0].message.lower()


def test_nsys_no_memcpy_alone():
    notes = capability_notes(Format.NSYS, _caps(has_memcpy=False))
    assert "N3" in codes(notes)


def test_nsys_no_runtime_api_alone():
    notes = capability_notes(Format.NSYS, _caps(has_runtime_api=False))
    assert "N3" in codes(notes)


def test_nsys_no_memcpy_and_no_runtime_api_coalesced():
    notes = capability_notes(Format.NSYS, _caps(has_memcpy=False, has_runtime_api=False))
    assert codes(notes).count("N3") == 1


def test_nsys_no_pmc_counters():
    notes = capability_notes(Format.NSYS, _caps(has_pmc_counters=False))
    assert codes(notes) == ["N4"]
    assert "gpu-metrics-device" in notes[0].message


# ---------------------------------------------------------------------------
# nsys ordering: N1 > N2 > N3 > N4
# ---------------------------------------------------------------------------


def test_nsys_ordering_all_missing():
    notes = capability_notes(
        Format.NSYS,
        _caps(has_mpi=False, has_markers=False, has_memcpy=False, has_pmc_counters=False),
    )
    assert codes(notes) == ["N1", "N2", "N3", "N4"]


# ---------------------------------------------------------------------------
# rocpd notes — individual flags
# ---------------------------------------------------------------------------


def test_rocpd_no_mpi():
    notes = capability_notes(Format.ROCPD, _caps(has_mpi=False))
    assert codes(notes) == ["R1"]
    assert "rocprof-sys-sample" in notes[0].message
    assert "--mpi" in notes[0].message


def test_rocpd_no_memcpy_alone():
    notes = capability_notes(Format.ROCPD, _caps(has_memcpy=False))
    assert "R2" in codes(notes)
    assert "rocprof-sys-sample" in notes[0].message


def test_rocpd_no_runtime_api_alone():
    notes = capability_notes(Format.ROCPD, _caps(has_runtime_api=False))
    assert "R2" in codes(notes)


def test_rocpd_no_memcpy_and_no_runtime_api_coalesced():
    notes = capability_notes(Format.ROCPD, _caps(has_memcpy=False, has_runtime_api=False))
    assert codes(notes).count("R2") == 1


def test_rocpd_no_markers():
    notes = capability_notes(Format.ROCPD, _caps(has_markers=False))
    assert codes(notes) == ["R3"]
    assert "roctx" in notes[0].message.lower()


def test_rocpd_no_pmc_counters():
    notes = capability_notes(Format.ROCPD, _caps(has_pmc_counters=False))
    assert codes(notes) == ["R4"]
    assert "--hardware-counters" in notes[0].message


# ---------------------------------------------------------------------------
# rocpd ordering: R1 > R2 > R3 > R4
# ---------------------------------------------------------------------------


def test_rocpd_ordering_all_missing():
    notes = capability_notes(
        Format.ROCPD,
        _caps(has_mpi=False, has_memcpy=False, has_markers=False, has_pmc_counters=False),
    )
    assert codes(notes) == ["R1", "R2", "R3", "R4"]


# ---------------------------------------------------------------------------
# Fixture-backed smoke test: synthetic profiles produce no notes
# (they are built with full capabilities)
# ---------------------------------------------------------------------------


def test_synthetic_nsys_notes(synthetic_profile):
    # The synthetic nsys fixture has no PMC counters (minimal fixture) → N4 only.
    notes = capability_notes(synthetic_profile.format, synthetic_profile.capabilities)
    assert codes(notes) == ["N4"]


def test_synthetic_rocpd_notes(synthetic_rocpd_profile):
    # The synthetic rocpd fixture is minimal: no MPI, no markers, no PMC counters → R1, R3, R4.
    notes = capability_notes(synthetic_rocpd_profile.format, synthetic_rocpd_profile.capabilities)
    assert codes(notes) == ["R1", "R3", "R4"]
