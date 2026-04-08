"""Tests for the ingestion layer against the real test profile."""

import pytest

from perf_advisor.ingestion.profile import NsysProfile

TEST_PROFILE = "/home/ads.leonhost/Downloads/nsight/nsys_4864_cgdef_2node_1rhs/cg_4864_1rhs.sqlite"


@pytest.fixture
def profile():
    p = NsysProfile(TEST_PROFILE)
    yield p
    p.close()


def test_profile_opens(profile):
    assert profile.path.exists()


def test_tables_present(profile):
    assert "CUPTI_ACTIVITY_KIND_KERNEL" in profile.tables
    assert "CUPTI_ACTIVITY_KIND_MEMCPY" in profile.tables
    assert "StringIds" in profile.tables


def test_has_mpi(profile):
    assert profile.has_mpi()


def test_has_nvtx(profile):
    assert profile.has_nvtx()


def test_string_resolution(profile):
    # ShortName IDs should resolve to non-empty strings
    row = profile.query("SELECT shortName FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 1")[0]
    name = profile.resolve_string(row["shortName"])
    assert name and not name.startswith("<id:")


def test_context_manager():
    with NsysProfile(TEST_PROFILE) as p:
        assert "CUPTI_ACTIVITY_KIND_KERNEL" in p.tables


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        NsysProfile("/nonexistent/path.sqlite")
