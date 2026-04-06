"""CLI entry point: python -m nsight_agent analyze <profile.sqlite>"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def _print_timings(timings: dict[str, float]) -> None:
    total = sum(timings.values())
    if total <= 0:
        return
    t = Table(title="Timing breakdown", show_header=True, show_footer=False)
    t.add_column("Stage")
    t.add_column("Time (s)", justify="right")
    t.add_column("%", justify="right", style="dim")
    labels = [
        ("phase_detection_s", "Phase detection"),
        ("metrics_s", "Metrics / analysis"),
        ("agent_s", "AI correspondence"),
    ]
    for key, label in labels:
        if key in timings:
            s = timings[key]
            t.add_row(label, f"{s:.1f}", f"{100 * s / total:.0f}%")
    t.add_row("[bold]Total[/bold]", f"[bold]{total:.1f}[/bold]", "")
    console.print(t)


def cmd_analyze(args: argparse.Namespace) -> None:
    from nsight_agent.agent.loop import (
        _parse_provider_and_model,
        check_provider_available,
        get_provider_availability,
        run_agent,
    )
    from nsight_agent.analysis.metrics import compute_profile_summary
    from nsight_agent.ingestion.profile import NsysProfile

    # Resolve provider early so we can fail fast before any expensive work.
    resolved_provider, resolved_model, reason = _parse_provider_and_model(args.provider, args.model)

    if not args.quiet:
        console.print(f"Using AI provider = [cyan]{resolved_provider}[/cyan], model = [cyan]{resolved_model}[/cyan] (selected based on {reason})")

    missing = check_provider_available(resolved_provider)
    if missing:
        console.print(f"[red]Error:[/red] {resolved_provider} provider requires a missing package: {missing}")
        sys.exit(1)

    # Warn about any other unavailable providers so the user knows what's on offer.
    if not args.quiet:
        availability = get_provider_availability()
        unavailable = [(p, hint) for p, hint in availability.items() if hint and p != resolved_provider]
        for p, hint in unavailable:
            console.print(f"[yellow]Warning:[/yellow] {p} provider unavailable ({hint})", highlight=False)

    timings: dict[str, float] = {}
    with NsysProfile(args.profile) as profile:
        summary = compute_profile_summary(profile, max_phases=args.max_phases, timings=timings)

    if not args.quiet and summary.phases:
        ph = Table(title="Detected Execution Phases")
        ph.add_column("Phase")
        ph.add_column("Start (s)", justify="right")
        ph.add_column("End (s)", justify="right")
        ph.add_column("Duration (s)", justify="right")
        ph.add_column("GPU Util %", justify="right")
        ph.add_column("GPU Kernel (s)", justify="right")
        ph.add_column("GPU Idle (s)", justify="right")
        ph.add_column("Top Kernel")
        if summary.mpi_present:
            ph.add_column("Top MPI Op")
        for p in summary.phases:
            top_kernel = (
                f"{p.top_kernels[0].name} ({p.top_kernels[0].pct_of_gpu_time}%)"
                if p.top_kernels else "—"
            )
            row = [
                p.name,
                str(p.start_s),
                str(p.end_s),
                str(p.duration_s),
                f"{p.gpu_utilization_pct}%",
                str(p.gpu_kernel_s),
                str(p.total_gpu_idle_s),
                top_kernel,
            ]
            if summary.mpi_present:
                top_mpi = (
                    f"{p.mpi_ops[0].op} ({p.mpi_ops[0].total_s}s)"
                    if p.mpi_ops else "—"
                )
                row.append(top_mpi)
            ph.add_row(*row)
        console.print(ph)

    if args.verbose:
        console.print("\n[bold]── ProfileSummary sent to AI ──[/bold]")
        console.print(summary.model_dump_json(indent=2))
        console.print("[bold]── End ProfileSummary ──[/bold]\n")

    token_usage: dict[str, int | None] = {}
    t_agent = time.perf_counter()
    hypotheses = run_agent(
        args.profile, summary=summary, verbose=not args.quiet,
        model=args.model, provider=args.provider, token_usage=token_usage,
    )
    timings["agent_s"] = time.perf_counter() - t_agent

    if args.json:
        print(json.dumps(hypotheses, indent=2))
        return

    table = Table(title=f"Hypotheses — {Path(args.profile).name}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan")
    table.add_column("Impact", style="bold")
    table.add_column("Action", style="magenta")
    table.add_column("Description")
    table.add_column("Suggestion")

    _ACTION_ABBREV = {
        "runtime_config": "runtime",
        "launch_config": "launch",
        "code_optimization": "code",
        "algorithm": "algorithm",
    }

    for i, h in enumerate(hypotheses, 1):
        impact_color = {"high": "red", "medium": "yellow", "low": "green"}.get(
            str(h.get("expected_impact", "")).lower(), "white"
        )
        action_raw = str(h.get("action_category", "")).lower()
        action_str = _ACTION_ABBREV.get(action_raw, action_raw or "—")
        table.add_row(
            str(i),
            h.get("bottleneck_type", "—"),
            f"[{impact_color}]{h.get('expected_impact', '—')}[/{impact_color}]",
            action_str,
            h.get("description", ""),
            h.get("suggestion", ""),
        )

    console.print(table)

    if not args.quiet:
        _print_timings(timings)
        inp = token_usage.get("input_tokens")
        out = token_usage.get("output_tokens")
        if inp is not None and out is not None:
            cost = token_usage.get("cost_usd")
            cost_str = f"  Cost: [yellow]${cost:.4f}[/yellow]" if cost is not None else ""
            console.print(
                f"  Tokens: [cyan]{inp:,}[/cyan] in / [cyan]{out:,}[/cyan] out"
                f"  ([dim]{inp + out:,} total[/dim]){cost_str}"
            )
        else:
            console.print("  Tokens: [dim]N/A[/dim]")


def cmd_summary(args: argparse.Namespace) -> None:
    from nsight_agent.analysis.metrics import compute_profile_summary
    from nsight_agent.ingestion.profile import NsysProfile

    with NsysProfile(args.profile) as profile:
        summary = compute_profile_summary(profile, max_phases=args.max_phases)

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

    if summary.phases:
        ph = Table(title="Execution Phases")
        for col in ("Phase", "Start (s)", "End (s)", "Duration (s)", "GPU Util %", "Top Kernel"):
            ph.add_column(col)
        for p in summary.phases:
            top = p.top_kernels[0].name if p.top_kernels else "—"
            ph.add_row(
                p.name,
                str(p.start_s),
                str(p.end_s),
                str(p.duration_s),
                f"{p.gpu_utilization_pct}%",
                top,
            )
        console.print(ph)


def main() -> None:
    parser = argparse.ArgumentParser(prog="nsight-agent")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser("analyze", help="Run agent hypothesis generation on a profile")
    p_analyze.add_argument("profile", help="Path to .sqlite profile")
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON")
    p_analyze.add_argument("--verbose", action="store_true", help="Print the ProfileSummary sent to the AI")
    p_analyze.add_argument("--quiet", action="store_true", help="Suppress agent turn logging")
    p_analyze.add_argument(
        "--max-phases", type=int, default=6,
        help="Maximum number of execution phases to detect (default: 6; use 1 to disable)",
    )
    p_analyze.add_argument(
        "--model", default=None,
        help=(
            "Model for hypothesis generation (default: per-provider — "
            "claude-opus-4-6 for anthropic, gpt-4o for openai, gemini-2.0-flash for gemini). "
            "Use a provider prefix to select the backend implicitly: "
            "openai:gpt-4o, gemini:gemini-2.0-flash, anthropic:claude-opus-4-6."
        ),
    )
    p_analyze.add_argument(
        "--provider", default=None,
        choices=["anthropic", "openai", "gemini"],
        help=(
            "LLM provider to use (default: auto-detected from available API keys). "
            "Overridden by a provider prefix in --model."
        ),
    )

    p_summary = sub.add_parser("summary", help="Print structured metrics summary")
    p_summary.add_argument("profile", help="Path to .sqlite profile")
    p_summary.add_argument("--json", action="store_true", help="Output raw JSON")
    p_summary.add_argument(
        "--max-phases", type=int, default=6,
        help="Maximum number of execution phases to detect (default: 6; use 1 to disable)",
    )

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
