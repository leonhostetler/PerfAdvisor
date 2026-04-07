"""Tool definitions exposed to the Claude agent.

Each tool function takes a NsysProfile and structured arguments, and returns
a JSON-serializable dict. The tool schemas follow the Anthropic tool-use format.
"""

from __future__ import annotations

import json
from typing import Any

from nsight_agent.analysis.metrics import (
    compute_gap_histogram,
    compute_gpu_kernel_time,
    compute_memcpy_by_kind,
    compute_mpi_ops,
    compute_nvtx_ranges,
    compute_profile_span,
    compute_streams,
    compute_top_kernels,
    _window_idle_time,
    _window_kernel_time,
    _window_memcpy_by_kind,
    _window_mpi_ops,
    _window_nvtx_ranges,
    _window_streams,
    _window_top_kernels,
)
from nsight_agent.ingestion.profile import NsysProfile


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_profile_summary(profile: NsysProfile, _args: dict[str, Any]) -> dict:
    """Return the top-level time budget for the profile."""
    span_s = compute_profile_span(profile)
    kernel_s = compute_gpu_kernel_time(profile)
    return {
        "profile_span_s": round(span_s, 3),
        "gpu_kernel_s": round(kernel_s, 3),
        "gpu_utilization_pct": round(100.0 * kernel_s / span_s, 1) if span_s else 0.0,
        "mpi_present": profile.has_mpi(),
        "nvtx_present": profile.has_nvtx(),
        "tables": sorted(profile.tables),
    }


def tool_top_kernels(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return the top GPU kernels by total execution time."""
    limit = int(args.get("limit", 15))
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        start_ns, end_ns = int(start_ns), int(end_ns)
        total_kernel_s = _window_kernel_time(profile, start_ns, end_ns)
        kernels = _window_top_kernels(profile, start_ns, end_ns, total_kernel_s, limit=limit)
    else:
        kernels = compute_top_kernels(profile, limit=limit)
    return {"kernels": [k.model_dump() for k in kernels]}


def tool_gap_histogram(profile: NsysProfile, args: dict[str, Any]) -> dict:
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


def tool_memcpy_summary(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return memory transfer summary broken down by direction/kind."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        transfers = _window_memcpy_by_kind(profile, int(start_ns), int(end_ns))
    else:
        transfers = compute_memcpy_by_kind(profile)
    return {"transfers": [t.model_dump() for t in transfers]}


def tool_mpi_summary(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return MPI operation summary. Returns empty list if no MPI data."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        ops = _window_mpi_ops(profile, int(start_ns), int(end_ns))
    else:
        ops = compute_mpi_ops(profile)
    return {"mpi_present": profile.has_mpi(), "ops": [o.model_dump() for o in ops]}


def tool_nvtx_ranges(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return top NVTX annotation ranges by total time."""
    limit = int(args.get("limit", 20))
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        ranges = _window_nvtx_ranges(profile, int(start_ns), int(end_ns), limit=limit)
    else:
        ranges = compute_nvtx_ranges(profile, limit=limit)
    return {"nvtx_present": profile.has_nvtx(), "ranges": [r.model_dump() for r in ranges]}


def tool_stream_summary(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return per-stream GPU utilization."""
    start_ns = args.get("start_ns")
    end_ns = args.get("end_ns")
    if start_ns is not None and end_ns is not None:
        streams = _window_streams(profile, int(start_ns), int(end_ns))
    else:
        streams = compute_streams(profile)
    return {"streams": [s.model_dump() for s in streams]}


def tool_phase_summary(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return the profile segmented into sequential execution phases."""
    from nsight_agent.analysis.metrics import compute_phase_summary
    from nsight_agent.analysis.phases import detect_phases

    max_phases = int(args.get("max_phases", 6))
    phases = detect_phases(profile, max_phases=max_phases)
    if not phases:
        return {"phases": []}

    profile_start_ns = phases[0].start_ns
    summaries = [compute_phase_summary(profile, p, profile_start_ns) for p in phases]
    return {"phases": [s.model_dump() for s in summaries]}


def tool_sql_query(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Execute an arbitrary read-only SQL query against the profile SQLite database.

    Use this for targeted follow-up questions not covered by other tools.
    The database is opened read-only; any write attempt will fail.
    Returns up to 200 rows.
    """
    sql = args.get("sql", "").strip()
    if not sql:
        return {"error": "No SQL provided"}
    try:
        rows = profile.query(sql)[:200]
        return {"rows": [dict(r) for r in rows], "count": len(rows)}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Registry: maps tool name -> (function, schema)
# ---------------------------------------------------------------------------

# Optional time-window parameters shared by most tools.
_WINDOW_SCHEMA_PROPS: dict[str, Any] = {
    "start_ns": {
        "type": "integer",
        "description": (
            "Start of time window as an absolute CUPTI timestamp in nanoseconds. "
            "Each phase in the pre-seeded phase summary includes start_ns and end_ns "
            "fields — pass them directly to scope this tool to that phase."
        ),
    },
    "end_ns": {
        "type": "integer",
        "description": (
            "End of time window as an absolute CUPTI timestamp in nanoseconds. "
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
                "Return the top-level time budget for the profile: wall-clock span, "
                "GPU kernel time, GPU utilization, and which optional tables are present "
                "(MPI, NVTX)."
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
    "nvtx_ranges": (
        tool_nvtx_ranges,
        {
            "name": "nvtx_ranges",
            "description": (
                "Return top NVTX annotation ranges by total wall-clock time. "
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
                "Pass start_ns/end_ns to scope the results to a specific phase."
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
                "Execute a read-only SQL query directly against the Nsight Systems SQLite "
                "database. Use this for targeted follow-up analysis not covered by other tools. "
                "Tables include: CUPTI_ACTIVITY_KIND_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY, "
                "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, NVTX_EVENTS, MPI_COLLECTIVES_EVENTS, "
                "MPI_P2P_EVENTS, StringIds (resolves integer name IDs), ENUM_* (resolve "
                "integer kind/type codes). Returns up to 200 rows."
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
}


def dispatch(profile: NsysProfile, tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call from the agent loop and return a JSON string result."""
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    fn, _ = TOOL_REGISTRY[tool_name]
    result = fn(profile, tool_input)
    return json.dumps(result, default=str)


def tool_schemas() -> list[dict]:
    """Return all tool schemas in Anthropic tool-use format."""
    return [schema for _, (_, schema) in TOOL_REGISTRY.items()]
