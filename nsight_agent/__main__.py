"""CLI entry point: python -m nsight_agent analyze <profile.sqlite>"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def cmd_analyze(args: argparse.Namespace) -> None:
    from nsight_agent.agent.loop import run_agent

    hypotheses = run_agent(args.profile, verbose=not args.quiet)

    if args.json:
        print(json.dumps(hypotheses, indent=2))
        return

    table = Table(title=f"Hypotheses — {Path(args.profile).name}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan")
    table.add_column("Impact", style="bold")
    table.add_column("Description")
    table.add_column("Suggestion")

    for i, h in enumerate(hypotheses, 1):
        impact_color = {"high": "red", "medium": "yellow", "low": "green"}.get(
            str(h.get("expected_impact", "")).lower(), "white"
        )
        table.add_row(
            str(i),
            h.get("bottleneck_type", "—"),
            f"[{impact_color}]{h.get('expected_impact', '—')}[/{impact_color}]",
            h.get("description", ""),
            h.get("suggestion", ""),
        )

    console.print(table)


def cmd_summary(args: argparse.Namespace) -> None:
    from nsight_agent.analysis.metrics import compute_profile_summary
    from nsight_agent.ingestion.profile import NsysProfile

    with NsysProfile(args.profile) as profile:
        summary = compute_profile_summary(profile)

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]Profile:[/bold] {summary.profile_path}")
    console.print(f"  Span:            {summary.profile_span_s:.1f}s")
    console.print(f"  GPU kernel time: {summary.gpu_kernel_s:.1f}s  ({summary.gpu_utilization_pct:.1f}% utilization)")
    console.print(f"  GPU memcpy time: {summary.gpu_memcpy_s:.1f}s")
    console.print(f"  GPU sync time:   {summary.gpu_sync_s:.1f}s")
    console.print(f"  GPU idle time:   {summary.total_gpu_idle_s:.1f}s")
    console.print(f"  MPI present:     {summary.mpi_present}")

    t = Table(title="Top Kernels")
    for col in ("Kernel", "Calls", "Total (s)", "Avg (ms)", "% GPU"):
        t.add_column(col)
    for k in summary.top_kernels:
        t.add_row(k.name, str(k.calls), str(k.total_s), str(k.avg_ms), f"{k.pct_of_gpu_time}%")
    console.print(t)

    if summary.mpi_ops:
        m = Table(title="MPI Operations")
        for col in ("Op", "Calls", "Total (s)", "Avg (ms)"):
            m.add_column(col)
        for op in summary.mpi_ops:
            m.add_row(op.op, str(op.calls), str(op.total_s), str(op.avg_ms))
        console.print(m)


def main() -> None:
    parser = argparse.ArgumentParser(prog="nsight-agent")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Run agent hypothesis generation on a profile")
    p_analyze.add_argument("profile", help="Path to .sqlite profile")
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON")
    p_analyze.add_argument("--quiet", action="store_true", help="Suppress agent turn logging")

    p_summary = sub.add_parser("summary", help="Print structured metrics summary")
    p_summary.add_argument("profile", help="Path to .sqlite profile")
    p_summary.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
