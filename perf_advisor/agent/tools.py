"""Tool definitions exposed to the Claude agent.

Each tool function takes a Profile and structured arguments, and returns
a JSON-serializable dict. The tool schemas follow the Anthropic tool-use format.
"""

from __future__ import annotations

import json
import re
import signal
import sqlite3
import threading
from typing import Any

from perf_advisor.analysis.metrics import (
    _compute_launch_overhead,
    _window_idle_time,
    _window_kernel_time,
    _window_marker_ranges,
    _window_memcpy_by_kind,
    _window_mpi_ops,
    _window_streams,
    _window_top_kernels,
    compute_device_info,
    compute_gap_histogram,
    compute_gpu_kernel_time,
    compute_marker_ranges,
    compute_memcpy_by_kind,
    compute_mpi_ops,
    compute_profile_span,
    compute_streams,
    compute_top_kernels,
)
from perf_advisor.ingestion import Format
from perf_advisor.ingestion.base import Profile

# ---------------------------------------------------------------------------
# Cross-vendor table redirect hints for the sql_query OperationalError handler
# ---------------------------------------------------------------------------

_NSYS_TO_ROCPD_TABLE: dict[str, str] = {
    "CUPTI_ACTIVITY_KIND_KERNEL": "rocpd_kernel_dispatch (join rocpd_info_kernel_symbol for names)",
    "CUPTI_ACTIVITY_KIND_MEMCPY": "rocpd_memory_copy",
    "CUPTI_ACTIVITY_KIND_RUNTIME": "rocpd_region (categories: HIP_RUNTIME_API_EXT, HSA_CORE_API)",
    "NVTX_EVENTS": "rocpd_region (filter out API categories to get user markers)",
    "MPI_COLLECTIVES_EVENTS": "rocpd_region WHERE category='MPI' (rocprof-sys only)",
    "MPI_P2P_EVENTS": "rocpd_region WHERE category='MPI' (rocprof-sys only)",
    "MPI_START_WAIT_EVENTS": "rocpd_region WHERE category='MPI' (rocprof-sys only)",
    "StringIds": "rocpd_string (columns: id, guid, string)",
}
_ROCPD_TO_NSYS_TABLE: dict[str, str] = {
    "rocpd_kernel_dispatch": "CUPTI_ACTIVITY_KIND_KERNEL",
    "rocpd_memory_copy": "CUPTI_ACTIVITY_KIND_MEMCPY",
    "rocpd_region": "NVTX_EVENTS (markers) or MPI_COLLECTIVES_EVENTS/MPI_P2P_EVENTS (MPI)",
    "rocpd_string": "StringIds (columns: id, value)",
    "rocpd_event": "(no direct equivalent; category info is in ENUM_* tables)",
    "rocpd_info_agent": "TARGET_INFO_GPU",
    "rocpd_info_kernel_symbol": "(kernel name info embedded in CUPTI_ACTIVITY_KIND_KERNEL)",
}

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_profile_summary(profile: Profile, _args: dict[str, Any]) -> dict:
    """Return the top-level time budget for the profile."""
    span_s = compute_profile_span(profile)
    kernel_s = compute_gpu_kernel_time(profile)
    return {
        "format": profile.format.value,
        "profile_span_s": round(span_s, 3),
        "gpu_kernel_s": round(kernel_s, 3),
        "gpu_utilization_pct": round(100.0 * kernel_s / span_s, 1) if span_s else 0.0,
        "mpi_present": profile.capabilities.has_mpi,
        "markers_present": profile.capabilities.has_markers,
        "tables": sorted(profile.tables),
    }


def tool_top_kernels(profile: Profile, args: dict[str, Any]) -> dict:
    """Return the top GPU kernels by total execution time."""
    limit = int(args.get("limit", 15))
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    device_info = compute_device_info(profile)
    launch_overhead = _compute_launch_overhead(profile)
    if start_ns is not None and end_ns is not None:
        start_ns, end_ns = int(start_ns), int(end_ns)
        total_kernel_s = _window_kernel_time(profile, start_ns, end_ns)
        kernels = _window_top_kernels(
            profile,
            start_ns,
            end_ns,
            total_kernel_s,
            limit=limit,
            device_info=device_info,
            launch_overhead=launch_overhead,
        )
    else:
        kernels = compute_top_kernels(
            profile, limit=limit, device_info=device_info, launch_overhead=launch_overhead
        )
    return {"kernels": [k.model_dump() for k in kernels]}


def tool_gap_histogram(profile: Profile, args: dict[str, Any]) -> dict:
    """Return a histogram of GPU idle gaps between kernel launches."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        total_idle_s, buckets = _window_idle_time(profile, int(start_ns), int(end_ns))
    else:
        total_idle_s, buckets = compute_gap_histogram(profile)
    return {
        "total_idle_s": round(total_idle_s, 3),
        "buckets": [b.model_dump() for b in buckets],
    }


def tool_memcpy_summary(profile: Profile, args: dict[str, Any]) -> dict:
    """Return memory transfer summary broken down by direction/kind."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        transfers = _window_memcpy_by_kind(profile, int(start_ns), int(end_ns))
    else:
        transfers = compute_memcpy_by_kind(profile)
    return {"transfers": [t.model_dump() for t in transfers]}


def tool_mpi_summary(profile: Profile, args: dict[str, Any]) -> dict:
    """Return MPI operation summary. Returns empty list if no MPI data."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        ops = _window_mpi_ops(profile, int(start_ns), int(end_ns))
    else:
        ops = compute_mpi_ops(profile)
    return {"mpi_present": profile.capabilities.has_mpi, "ops": [o.model_dump() for o in ops]}


def tool_marker_ranges(profile: Profile, args: dict[str, Any]) -> dict:
    """Return top marker annotation ranges (NVTX or rocTX) by total time."""
    limit = int(args.get("limit", 20))
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        ranges = _window_marker_ranges(profile, int(start_ns), int(end_ns), limit=limit)
    else:
        ranges = compute_marker_ranges(profile, limit=limit)
    return {
        "markers_present": profile.capabilities.has_markers,
        "ranges": [r.model_dump() for r in ranges],
    }


def tool_stream_summary(profile: Profile, args: dict[str, Any]) -> dict:
    """Return per-stream GPU utilization."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        streams = _window_streams(profile, int(start_ns), int(end_ns))
    else:
        streams = compute_streams(profile)
    return {"streams": [s.model_dump() for s in streams]}


def tool_phase_summary(profile: Profile, args: dict[str, Any]) -> dict:
    """Return the profile segmented into sequential execution phases."""
    from perf_advisor.analysis.metrics import compute_phase_summary
    from perf_advisor.analysis.phases import detect_phases

    max_phases = int(args.get("max_phases", 6))
    phases = detect_phases(profile, max_phases=max_phases)
    if not phases:
        return {"phases": []}

    device_info = compute_device_info(profile)
    launch_overhead = _compute_launch_overhead(profile)
    profile_start_ns = phases[0].start_ns
    summaries = [
        compute_phase_summary(
            profile,
            p,
            profile_start_ns,
            device_info=device_info,
            launch_overhead=launch_overhead,
        )
        for p in phases
    ]
    return {"phases": [s.model_dump() for s in summaries]}


_SQL_RESULT_BYTE_LIMIT = 100_000  # 100 KB
_SQL_TIMEOUT_S = 30  # seconds before a query is automatically interrupted


def tool_sql_query(profile: Profile, args: dict[str, Any]) -> dict:
    """Execute an arbitrary read-only SQL query against the profile SQLite database.

    Use this for targeted follow-up questions not covered by other tools.
    The database is opened read-only; any write attempt will fail.
    Returns up to 200 rows.
    Note: SQLite only — PERCENTILE_CONT, MEDIAN, STDDEV are not supported.
    """
    sql = args.get("sql", "").strip()
    if not sql:
        return {"error": "No SQL provided"}

    # Only SELECT / WITH (CTE) statements are permitted. Writes are already
    # blocked by the read-only connection, but ATTACH, PRAGMA with side-effects,
    # etc. are also disallowed here.
    # Strip -- and /* */ comments before checking so leading comments don't
    # cause legitimate SELECT queries to be rejected.
    sql_norm = re.sub(r"--[^\n]*|/\*.*?\*/", "", sql, flags=re.DOTALL).lstrip().upper()
    if not (sql_norm.startswith("SELECT") or sql_norm.startswith("WITH")):
        return {
            "error": (
                "Only SELECT queries are permitted. Use get_table_schema for schema inspection."
            )
        }

    # Inject LIMIT 200 if none is present so the query cannot return a
    # pathologically large result set.
    if not re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
        sql = sql.rstrip(";") + " LIMIT 200"

    # stop event shared by the timeout timer and the SIGINT handler.
    # timed_out flag distinguishes which source fired so we can give the LLM
    # a more actionable error message.
    stop = threading.Event()
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        stop.set()

    timer = threading.Timer(_SQL_TIMEOUT_S, _on_timeout)

    # Wire SIGINT to the same stop event so Ctrl-C also interrupts SQLite.
    # Only installed on the main thread; worker threads are left alone.
    orig_handler = signal.getsignal(signal.SIGINT)
    on_main_thread = threading.current_thread() is threading.main_thread()

    def _on_sigint(sig: int, frame: object) -> None:
        stop.set()
        signal.signal(signal.SIGINT, orig_handler)  # restore so next Ctrl-C terminates

    if on_main_thread:
        signal.signal(signal.SIGINT, _on_sigint)
    timer.start()
    try:
        rows = profile.query_safe(sql, stop_event=stop, row_limit=200)
    except sqlite3.OperationalError as e:
        if timed_out.is_set():
            return {
                "error": (
                    f"Query timed out after {_SQL_TIMEOUT_S}s. "
                    "The query was too expensive to complete. "
                    "Simplify it: avoid self-joins and correlated subqueries on large tables, "
                    "scope with a WHERE clause on start/end timestamps, "
                    "or use a structured tool (top_kernels, mpi_summary, etc.) instead."
                )
            }
        err_msg = str(e)
        if "no such table" in err_msg:
            wrong_table = err_msg.split("no such table:")[-1].strip()
            if profile.format == Format.ROCPD and wrong_table in _NSYS_TO_ROCPD_TABLE:
                equiv = _NSYS_TO_ROCPD_TABLE[wrong_table]
                return {
                    "error": (
                        f"Table '{wrong_table}' does not exist in this rocpd profile. "
                        f"The rocpd equivalent is: {equiv}"
                    )
                }
            if profile.format == Format.NSYS and wrong_table in _ROCPD_TO_NSYS_TABLE:
                equiv = _ROCPD_TO_NSYS_TABLE[wrong_table]
                return {
                    "error": (
                        f"Table '{wrong_table}' does not exist in this Nsight Systems profile. "
                        f"The NSYS equivalent is: {equiv}"
                    )
                }
        return {"error": err_msg}
    except Exception as e:
        if timed_out.is_set():
            return {
                "error": (
                    f"Query timed out after {_SQL_TIMEOUT_S}s. "
                    "Simplify it or use a structured tool instead."
                )
            }
        return {"error": str(e)}
    finally:
        timer.cancel()
        if on_main_thread:
            signal.signal(signal.SIGINT, orig_handler)

    result = {"rows": [dict(r) for r in rows], "count": len(rows)}
    result_bytes = len(json.dumps(result, default=str))
    if result_bytes > _SQL_RESULT_BYTE_LIMIT:
        return {
            "error": (
                f"Result too large ({len(rows)} rows, {result_bytes:,} bytes after serialization). "
                "Narrow your query: add a WHERE clause scoped to a time window, "
                "select specific columns instead of *, or use the structured tools "
                "(top_kernels, memcpy_summary, mpi_summary, etc.) for pre-aggregated data."
            )
        }
    return result


def tool_get_table_schema(profile: Profile, args: dict[str, Any]) -> dict:
    """Return the column names for a table in the profile SQLite database."""
    table = args.get("table", "").strip()
    if not table:
        return {"error": "No table name provided"}
    if not profile.has_table(table):
        return {
            "error": f"Table '{table}' not found.",
            "available_tables": sorted(profile.tables),
        }
    return {"table": table, "columns": profile.columns(table)}


# ---------------------------------------------------------------------------
# Registry: maps tool name -> (function, schema)
# ---------------------------------------------------------------------------

# Optional time-window parameters shared by most tools.
_WINDOW_SCHEMA_PROPS: dict[str, Any] = {
    "start_ns": {
        "type": "integer",
        "description": (
            "Start of time window as an absolute nanosecond timestamp. "
            "Each phase in the pre-seeded phase summary includes start_ns and end_ns "
            "fields — pass them directly to scope this tool to that phase."
        ),
    },
    "end_ns": {
        "type": "integer",
        "description": (
            "End of time window as an absolute nanosecond timestamp. "
            "Must be provided together with start_ns."
        ),
    },
}

TOOL_REGISTRY: dict[str, tuple[Any, dict]] = {
    "profile_summary": (
        tool_profile_summary,
        {
            "name": "profile_summary",
            "description": (
                "Return the top-level time budget for the profile: format, wall-clock span, "
                "GPU kernel time, GPU utilization, and which optional data sources are present "
                "(MPI, markers)."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ),
    "top_kernels": (
        tool_top_kernels,
        {
            "name": "top_kernels",
            "description": (
                "Return the top GPU kernels ranked by total execution time. "
                "Pass start_ns/end_ns to scope the results to a specific phase."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of kernels to return (default 15).",
                    },
                    **_WINDOW_SCHEMA_PROPS,
                },
                "required": [],
            },
        },
    ),
    "gap_histogram": (
        tool_gap_histogram,
        {
            "name": "gap_histogram",
            "description": (
                "Return a histogram of GPU idle gaps between kernel launches. "
                "Large gaps (>1ms) often indicate CPU-side bottlenecks or MPI wait time. "
                "Pass start_ns/end_ns to scope the results to a specific phase."
            ),
            "input_schema": {
                "type": "object",
                "properties": {**_WINDOW_SCHEMA_PROPS},
                "required": [],
            },
        },
    ),
    "memcpy_summary": (
        tool_memcpy_summary,
        {
            "name": "memcpy_summary",
            "description": (
                "Return memory transfer summary by kind (H2D, D2H, P2P, etc.). "
                "Pass start_ns/end_ns to scope the results to a specific phase."
            ),
            "input_schema": {
                "type": "object",
                "properties": {**_WINDOW_SCHEMA_PROPS},
                "required": [],
            },
        },
    ),
    "mpi_summary": (
        tool_mpi_summary,
        {
            "name": "mpi_summary",
            "description": (
                "Return MPI operation breakdown by total time and call count. "
                "Returns empty if the profile has no MPI instrumentation. "
                "Pass start_ns/end_ns to scope the results to a specific phase."
            ),
            "input_schema": {
                "type": "object",
                "properties": {**_WINDOW_SCHEMA_PROPS},
                "required": [],
            },
        },
    ),
    "marker_ranges": (
        tool_marker_ranges,
        {
            "name": "marker_ranges",
            "description": (
                "Return top marker annotation ranges (NVTX or rocTX) by total wall-clock time. "
                "These are application-defined labels and give semantic context "
                "to what the GPU and CPU were doing. "
                "Pass start_ns/end_ns to scope the results to a specific phase."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of ranges to return (default 20).",
                    },
                    **_WINDOW_SCHEMA_PROPS,
                },
                "required": [],
            },
        },
    ),
    "stream_summary": (
        tool_stream_summary,
        {
            "name": "stream_summary",
            "description": (
                "Return per-CUDA-stream GPU utilization. "
                "A single dominant stream indicates no concurrent kernel execution. "
                "Pass start_ns/end_ns to scope the results to a specific phase. "
                "Note: stream IDs may not be present in all profile formats."
            ),
            "input_schema": {
                "type": "object",
                "properties": {**_WINDOW_SCHEMA_PROPS},
                "required": [],
            },
        },
    ),
    "phase_summary": (
        tool_phase_summary,
        {
            "name": "phase_summary",
            "description": (
                "Segment the profile into sequential, non-overlapping execution phases "
                "(e.g., initialization, main computation, teardown) and return per-phase "
                "metrics. Use this early in analysis — phases can have very different "
                "performance characteristics and global averages can be misleading."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_phases": {
                        "type": "integer",
                        "description": "Maximum number of phases to return (default 6).",
                    }
                },
                "required": [],
            },
        },
    ),
    "sql_query": (
        tool_sql_query,
        {
            "name": "sql_query",
            "description": (
                "Execute a read-only SQL query against the profile SQLite database. "
                "Use this for targeted follow-up analysis not covered by other tools. "
                "Table names and column layouts are in the SQL schema reference in the system "
                "prompt — use get_table_schema to verify column names before writing SQL. "
                "Constraints: only SELECT/WITH queries are accepted; LIMIT 200 is injected "
                "automatically if absent; results over 100 KB are blocked with an error. "
                "SQLite only — PERCENTILE_CONT, MEDIAN, and STDDEV are not supported. "
                "When a table is not found, a redirect hint is returned naming the correct "
                "table for this profile format."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "The SQL SELECT statement to execute.",
                    }
                },
                "required": ["sql"],
            },
        },
    ),
    "get_table_schema": (
        tool_get_table_schema,
        {
            "name": "get_table_schema",
            "description": (
                "Return the column names for a specific table in the profile SQLite database. "
                "Use this *instead of* SELECT * LIMIT 1 whenever you need to inspect column "
                "names — never issue a SELECT * query just to discover the schema."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "table": {
                        "type": "string",
                        "description": (
                            "The table or view name to inspect. "
                            "Call this before writing sql_query to avoid column name errors. "
                            "Table names depend on the profile format — see the SQL schema "
                            "reference in the system prompt."
                        ),
                    }
                },
                "required": ["table"],
            },
        },
    ),
}


def dispatch(profile: Profile, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call from the agent loop and return a JSON string result."""
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    fn, _ = TOOL_REGISTRY[tool_name]
    result = fn(profile, tool_input)
    return json.dumps(result, default=str)


def tool_schemas() -> list[dict]:
    """Return all tool schemas in Anthropic tool-use format."""
    return [schema for _, (_, schema) in TOOL_REGISTRY.items()]
