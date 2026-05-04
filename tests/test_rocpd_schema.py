"""Schema-level tests for the synthetic rocpd SQLite fixture.

These run in CI and on any machine — no real rocpd profile required.
They validate the fixture builder in conftest.py against the rocpd
schema_version=3 contract, so later phases that add RocpdProfile can
assert behaviour against a known-good baseline.

Expected values are derived directly from _build_synthetic_rocpd_db():

  GPU kernel time = 4×2 ms + 5 ms = 13 ms = 0.013 s
  Inter-kernel gaps:
    3 × 5 µs  (<10 µs)
    1 × 2 ms  (1–10 ms)
  Top kernel: dslash_function<Dslash3D,int>  8 ms  ≈61.5 %
  2nd kernel: reduce_kernel<float>           5 ms  ≈38.5 %
  Memory copies: 1 D2D (1 MB), 1 H2D (256 KB), 1 D2H (256 KB)
  Regions: 2 per category × 4 categories = 8 total
  Agents: 1 CPU (id=1), 1 GPU (id=2, MI250X-like)
  GPU extdata: cu_count=110, gfx_target_version=90010
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _q(conn: sqlite3.Connection, sql: str, params=()) -> list[sqlite3.Row]:
    return conn.execute(sql, params).fetchall()


# ---------------------------------------------------------------------------
# Fixture structure
# ---------------------------------------------------------------------------


def test_synthetic_rocpd_file_exists(synthetic_rocpd_path):
    assert synthetic_rocpd_path.exists()
    assert synthetic_rocpd_path.stat().st_size > 0


def test_schema_version_is_3(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT value FROM rocpd_metadata WHERE tag='schema_version'")
    assert rows and rows[0]["value"] == "3"


def test_guid_in_metadata(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT value FROM rocpd_metadata WHERE tag='guid'")
    assert rows and rows[0]["value"] == "deadbeef-0000-0000-0000-000000000001"


def test_concrete_tables_exist(synthetic_rocpd_path):
    expected_tables = {
        "rocpd_string_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_node_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_process_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_thread_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_agent_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_queue_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_stream_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_code_object_deadbeef_0000_0000_0000_000000000001",
        "rocpd_info_kernel_symbol_deadbeef_0000_0000_0000_000000000001",
        "rocpd_event_deadbeef_0000_0000_0000_000000000001",
        "rocpd_region_deadbeef_0000_0000_0000_000000000001",
        "rocpd_kernel_dispatch_deadbeef_0000_0000_0000_000000000001",
        "rocpd_memory_copy_deadbeef_0000_0000_0000_000000000001",
        "rocpd_memory_allocate_deadbeef_0000_0000_0000_000000000001",
        "rocpd_metadata_deadbeef_0000_0000_0000_000000000001",
    }
    with _conn(synthetic_rocpd_path) as conn:
        names = {r[0] for r in _q(conn, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert expected_tables <= names


def test_unsuffixed_views_exist(synthetic_rocpd_path):
    expected_views = {
        "rocpd_string",
        "rocpd_info_node",
        "rocpd_info_process",
        "rocpd_info_thread",
        "rocpd_info_agent",
        "rocpd_info_queue",
        "rocpd_info_stream",
        "rocpd_info_code_object",
        "rocpd_info_kernel_symbol",
        "rocpd_event",
        "rocpd_region",
        "rocpd_kernel_dispatch",
        "rocpd_memory_copy",
        "rocpd_memory_allocate",
        "rocpd_metadata",
    }
    with _conn(synthetic_rocpd_path) as conn:
        names = {r[0] for r in _q(conn, "SELECT name FROM sqlite_master WHERE type='view'")}
    assert expected_views <= names


def test_convenience_views_exist(synthetic_rocpd_path):
    expected = {
        "kernels",
        "regions",
        "memory_copies",
        "memory_allocations",
        "kernel_symbols",
        "code_objects",
        "processes",
        "threads",
        "top_kernels",
    }
    with _conn(synthetic_rocpd_path) as conn:
        names = {r[0] for r in _q(conn, "SELECT name FROM sqlite_master WHERE type='view'")}
    assert expected <= names


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


def test_one_gpu_agent(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT * FROM rocpd_info_agent WHERE type='GPU'")
    assert len(rows) == 1


def test_gpu_agent_product_name(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT product_name FROM rocpd_info_agent WHERE type='GPU'")
    assert rows[0]["product_name"] == "AMD Instinct MI250X"


def test_gpu_agent_extdata_cu_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn,
            "SELECT JSON_EXTRACT(extdata,'$.cu_count') AS cu_count "
            "FROM rocpd_info_agent WHERE type='GPU'",
        )
    assert rows[0]["cu_count"] == 110


def test_gpu_agent_extdata_gfx_target_version(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn,
            "SELECT JSON_EXTRACT(extdata,'$.gfx_target_version') AS v "
            "FROM rocpd_info_agent WHERE type='GPU'",
        )
    assert rows[0]["v"] == 90010


def test_cpu_agent_present(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT * FROM rocpd_info_agent WHERE type='CPU'")
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# Kernel dispatches
# ---------------------------------------------------------------------------


def test_kernel_dispatch_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_kernel_dispatch")
    assert rows[0]["n"] == 5  # 4 dslash + 1 reduce


def test_kernel_dispatch_via_kernels_view(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM kernels")
    assert rows[0]["n"] == 5


def test_top_kernels_dslash_first(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name, total_calls FROM top_kernels ORDER BY total_duration DESC")
    assert rows[0]["name"] == "dslash_function<Dslash3D,int>"
    assert rows[0]["total_calls"] == 4


def test_top_kernels_reduce_second(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name, total_calls FROM top_kernels ORDER BY total_duration DESC")
    assert rows[1]["name"] == "reduce_kernel<float>"
    assert rows[1]["total_calls"] == 1


def test_total_gpu_kernel_time_ms(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT SUM(end - start) AS ns FROM rocpd_kernel_dispatch")
    total_ms = rows[0]["ns"] / 1_000_000
    assert abs(total_ms - 13.0) < 0.01  # 4×2 ms + 5 ms


def test_dslash_percentage_above_60(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn,
            "SELECT percentage FROM top_kernels WHERE name='dslash_function<Dslash3D,int>'",
        )
    assert rows[0]["percentage"] > 60.0


def test_kernel_symbol_display_name_strips_kd(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT display_name FROM rocpd_info_kernel_symbol")
    display_names = {r["display_name"] for r in rows}
    assert all(not n.endswith(".kd") for n in display_names)


def test_kernels_view_agent_type_is_gpu(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT DISTINCT agent_type FROM kernels")
    assert len(rows) == 1 and rows[0]["agent_type"] == "GPU"


def test_kernels_view_lds_size(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT lds_size FROM kernels WHERE name='reduce_kernel<float>'")
    assert rows[0]["lds_size"] == 16384


# ---------------------------------------------------------------------------
# Memory copies
# ---------------------------------------------------------------------------


def test_memory_copy_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_memory_copy")
    assert rows[0]["n"] == 3


def test_memory_copies_view_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM memory_copies")
    assert rows[0]["n"] == 3


def test_memory_copy_directions(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name, COUNT(*) AS cnt FROM memory_copies GROUP BY name")
    by_name = {r["name"]: r["cnt"] for r in rows}
    assert by_name.get("MEMORY_COPY_DEVICE_TO_DEVICE") == 1
    assert by_name.get("MEMORY_COPY_HOST_TO_DEVICE") == 1
    assert by_name.get("MEMORY_COPY_DEVICE_TO_HOST") == 1


def test_d2d_copy_size(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT size FROM memory_copies WHERE name='MEMORY_COPY_DEVICE_TO_DEVICE'")
    assert rows[0]["size"] == 1_048_576  # 1 MB


def test_h2d_dst_is_gpu(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn, "SELECT dst_agent_type FROM memory_copies WHERE name='MEMORY_COPY_HOST_TO_DEVICE'"
        )
    assert rows[0]["dst_agent_type"] == "GPU"


def test_d2h_src_is_gpu(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn, "SELECT src_agent_type FROM memory_copies WHERE name='MEMORY_COPY_DEVICE_TO_HOST'"
        )
    assert rows[0]["src_agent_type"] == "GPU"


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------


def test_region_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_region")
    assert rows[0]["n"] == 8


def test_regions_view_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM regions")
    assert rows[0]["n"] == 8


def test_all_four_categories_present(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT DISTINCT category FROM regions ORDER BY category")
    cats = {r["category"] for r in rows}
    assert cats == {
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HIP_RUNTIME_API_EXT",
        "HIP_COMPILER_API_EXT",
    }


def test_two_regions_per_category(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT category, COUNT(*) AS cnt FROM regions GROUP BY category")
    for row in rows:
        assert row["cnt"] == 2, f"{row['category']} has {row['cnt']} regions, expected 2"


def test_hsa_core_api_region_names(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name FROM regions WHERE category='HSA_CORE_API' ORDER BY name")
    names = {r["name"] for r in rows}
    assert names == {"hsa_queue_create", "hsa_signal_wait_scacquire"}


def test_hip_compiler_api_region_names(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(
            conn, "SELECT name FROM regions WHERE category='HIP_COMPILER_API_EXT' ORDER BY name"
        )
    names = {r["name"] for r in rows}
    assert names == {"__hipRegisterFatBinary", "hipModuleGetFunction"}


# ---------------------------------------------------------------------------
# Memory allocations
# ---------------------------------------------------------------------------


def test_memory_allocate_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_memory_allocate")
    assert rows[0]["n"] == 2


def test_memory_allocations_view_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM memory_allocations")
    assert rows[0]["n"] == 2


def test_alloc_and_free_present(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT type FROM memory_allocations ORDER BY type")
    types = {r["type"] for r in rows}
    assert types == {"ALLOC", "FREE"}


# ---------------------------------------------------------------------------
# Infrastructure — streams, queues, processes
# ---------------------------------------------------------------------------


def test_one_stream(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name FROM rocpd_info_stream")
    assert len(rows) == 1 and rows[0]["name"] == "Default Stream"


def test_one_queue(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT name FROM rocpd_info_queue")
    assert len(rows) == 1 and rows[0]["name"] == "Default Queue"


def test_processes_view_returns_row(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT command FROM processes")
    assert rows and rows[0]["command"] == "./synthetic_app"


def test_kernel_symbols_view_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM kernel_symbols")
    assert rows[0]["n"] == 2


def test_code_objects_view_count(synthetic_rocpd_path):
    with _conn(synthetic_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM code_objects")
    assert rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# Real-profile smoke tests (skipped in CI)
# ---------------------------------------------------------------------------


def test_real_rocpd_opens(real_rocpd_path):
    """Smoke: the real rank-0 DB opens and has expected table counts."""
    with _conn(real_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_kernel_dispatch")
    assert rows[0]["n"] == 401_777


def test_real_rocpd_four_categories(real_rocpd_path):
    with _conn(real_rocpd_path) as conn:
        rows = _q(conn, "SELECT DISTINCT category FROM regions ORDER BY category")
    cats = {r["category"] for r in rows}
    assert cats == {
        "HSA_CORE_API",
        "HSA_AMD_EXT_API",
        "HIP_RUNTIME_API_EXT",
        "HIP_COMPILER_API_EXT",
    }


def test_real_rocpd_schema_version_3(real_rocpd_path):
    with _conn(real_rocpd_path) as conn:
        rows = _q(conn, "SELECT value FROM rocpd_metadata WHERE tag='schema_version'")
    assert rows[0]["value"] == "3"


def test_real_rocpd_gpu_agent_count(real_rocpd_path):
    with _conn(real_rocpd_path) as conn:
        rows = _q(conn, "SELECT COUNT(*) AS n FROM rocpd_info_agent WHERE type='GPU'")
    assert rows[0]["n"] == 8  # 8 MI250X GCDs per rank
