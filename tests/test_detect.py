"""Tests for format detection and open_profile() factory."""

from __future__ import annotations

import sqlite3

import pytest

from perf_advisor.ingestion.base import Format
from perf_advisor.ingestion.detect import detect_format, open_profile
from perf_advisor.ingestion.nsys import NsysProfile
from perf_advisor.ingestion.rocpd import RocpdProfile


def test_detect_nsys(synthetic_profile_path):
    assert detect_format(synthetic_profile_path) == Format.NSYS


def test_detect_rocpd(synthetic_rocpd_path):
    assert detect_format(synthetic_rocpd_path) == Format.ROCPD


def test_open_profile_nsys(synthetic_profile_path):
    with open_profile(synthetic_profile_path) as p:
        assert isinstance(p, NsysProfile)
        assert p.format == Format.NSYS


def test_open_profile_rocpd(synthetic_rocpd_path):
    with open_profile(synthetic_rocpd_path) as p:
        assert isinstance(p, RocpdProfile)
        assert p.format == Format.ROCPD


def test_detect_unknown_schema(tmp_path):
    db = tmp_path / "unrecognized.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE Foo (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="Unrecognized"):
        detect_format(db)


def test_detect_non_sqlite(tmp_path):
    p = tmp_path / "not_a_db.bin"
    p.write_bytes(b"\xff\xfe" * 100)
    with pytest.raises(ValueError):
        detect_format(p)


def test_open_profile_nsys_rep_raises(tmp_path):
    p = tmp_path / "profile.nsys-rep"
    p.write_bytes(b"\x00" * 16)
    with pytest.raises(ValueError, match="export to SQLite"):
        open_profile(p)


def test_detect_rocpd_minimal(tmp_path):
    """rocpd_string alone (truncated/empty file) is sufficient for detection."""
    db = tmp_path / "minimal.rocpd"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE rocpd_string (id INTEGER, guid TEXT, string TEXT)")
    conn.commit()
    conn.close()
    assert detect_format(db) == Format.ROCPD
