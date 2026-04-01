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
    kernels = compute_top_kernels(profile, limit=limit)
    return {"kernels": [k.model_dump() for k in kernels]}


def tool_gap_histogram(profile: NsysProfile, _args: dict[str, Any]) -> dict:
    """Return a histogram of GPU idle gaps between kernel launches."""
    total_idle_s, buckets = compute_gap_histogram(profile)
    return {
        "total_idle_s": round(total_idle_s, 3),
        "buckets": [b.model_dump() for b in buckets],
    }


def tool_memcpy_summary(profile: NsysProfile, _args: dict[str, Any]) -> dict:
    """Return memory transfer summary broken down by direction/kind."""
    transfers = compute_memcpy_by_kind(profile)
    return {"transfers": [t.model_dump() for t in transfers]}


def tool_mpi_summary(profile: NsysProfile, _args: dict[str, Any]) -> dict:
    """Return MPI operation summary. Returns empty list if no MPI data."""
    ops = compute_mpi_ops(profile)
    return {"mpi_present": profile.has_mpi(), "ops": [o.model_dump() for o in ops]}


def tool_nvtx_ranges(profile: NsysProfile, args: dict[str, Any]) -> dict:
    """Return top NVTX annotation ranges by total time."""
    limit = int(args.get("limit", 20))
    ranges = compute_nvtx_ranges(profile, limit=limit)
    return {"nvtx_present": profile.has_nvtx(), "ranges": [r.model_dump() for r in ranges]}


def tool_stream_summary(profile: NsysProfile, _args: dict[str, Any]) -> dict:
    """Return per-stream GPU utilization."""
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
            "description": "Return the top GPU kernels ranked by total execution time.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of kernels to return (default 15).",
                    }
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
                "Large gaps (>1ms) often indicate CPU-side bottlenecks or MPI wait time."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ),
    "memcpy_summary": (
        tool_memcpy_summary,
        {
            "name": "memcpy_summary",
            "description": "Return memory transfer summary by kind (H2D, D2H, P2P, etc.).",
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ),
    "mpi_summary": (
        tool_mpi_summary,
        {
            "name": "mpi_summary",
            "description": (
                "Return MPI operation breakdown by total time and call count. "
                "Returns empty if the profile has no MPI instrumentation."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        },
    ),
    "nvtx_ranges": (
        tool_nvtx_ranges,
        {
            "name": "nvtx_ranges",
            "description": (
                "Return top NVTX annotation ranges by total wall-clock time. "
                "These are application-defined labels and give semantic context "
                "to what the GPU and CPU were doing."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of ranges to return (default 20).",
                    }
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
                "A single dominant stream indicates no concurrent kernel execution."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
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
