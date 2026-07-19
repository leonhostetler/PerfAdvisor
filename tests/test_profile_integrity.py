"""Tests verifying capability flags use data-presence rather than table-presence.

An empty table (schema present, zero rows) must yield the same False capability
flags as a missing table, ensuring truncated profiles are detected correctly.
"""

from __future__ import annotations

import sqlite3

from perf_advisor.ingestion.nsys import NsysProfile
from perf_advisor.ingestion.rocpd import RocpdProfile


def _make_nsys_empty_tables(path) -> None:
    """Minimal nsys SQLite: all key tables present but empty."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO StringIds VALUES (1, 'placeholder')")
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY "
        "(start INTEGER, end INTEGER, bytes INTEGER, copyKind INTEGER)"
    )
    conn.execute(
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER)"
    )
    conn.execute(
        "CREATE TABLE NVTX_EVENTS (start INTEGER, end INTEGER, text TEXT, eventType INTEGER)"
    )
    conn.execute("CREATE TABLE MPI_COLLECTIVES_EVENTS (start INTEGER, end INTEGER, textId INTEGER)")
    conn.execute("CREATE TABLE MPI_P2P_EVENTS (start INTEGER, end INTEGER, textId INTEGER)")
    conn.commit()
    conn.close()


def _make_rocpd_empty_tables(path) -> None:
    """Minimal rocpd SQLite: key tables present but empty."""
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE rocpd_metadata (tag TEXT, value TEXT)")
    conn.execute("INSERT INTO rocpd_metadata VALUES ('schema_version', '3')")
    conn.execute("CREATE TABLE rocpd_string (id INTEGER, guid TEXT, string TEXT)")
    conn.execute("INSERT INTO rocpd_string VALUES (1, 'test', 'placeholder')")
    conn.execute(
        "CREATE TABLE rocpd_kernel_dispatch (id INTEGER, guid TEXT, start INTEGER, end INTEGER)"
    )
    conn.execute(
        "CREATE TABLE rocpd_memory_copy "
        "(id INTEGER, guid TEXT, start INTEGER, end INTEGER, size INTEGER)"
    )
    conn.execute("CREATE TABLE rocpd_sample (id INTEGER, guid TEXT)")
    conn.execute("CREATE TABLE rocpd_pmc_event (id INTEGER, guid TEXT)")
    conn.commit()
    conn.close()


def test_nsys_empty_tables_yield_false_capabilities(tmp_path):
    path = tmp_path / "empty.sqlite"
    _make_nsys_empty_tables(path)
    with NsysProfile(path) as p:
        caps = p.capabilities
        assert not caps.has_kernels
        assert not caps.has_memcpy
        assert not caps.has_runtime_api
        assert not caps.has_markers
        assert not caps.has_mpi


def test_rocpd_empty_tables_yield_false_capabilities(tmp_path):
    path = tmp_path / "empty.rocpd"
    _make_rocpd_empty_tables(path)
    with RocpdProfile(path) as p:
        caps = p.capabilities
        assert not caps.has_kernels
        assert not caps.has_memcpy
        assert not caps.has_pmc_counters
        assert not caps.has_cpu_samples


def test_nsys_populated_tables_still_yield_true(tmp_path):
    """Regression: data-presence check must not break profiles with actual data."""
    path = tmp_path / "populated.sqlite"
    _make_nsys_empty_tables(path)
    conn = sqlite3.connect(str(path))
    conn.execute("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (1000, 2000, 1)")
    conn.execute("INSERT INTO MPI_COLLECTIVES_EVENTS VALUES (500, 600, 1)")
    conn.commit()
    conn.close()
    with NsysProfile(path) as p:
        assert p.capabilities.has_kernels
        assert p.capabilities.has_mpi
        assert not p.capabilities.has_memcpy  # still empty
