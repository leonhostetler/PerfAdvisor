"""CLI entry point: python -m nsight_agent analyze <profile.sqlite>"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table

from nsight_agent.agent.loop import MAX_TURNS, WARN_TURNS_BEFORE_LIMIT
from nsight_agent.agent.logger import LLMLogger

console = Console(record=True)


class _null_context:
    """Minimal no-op context manager used when logging is not requested."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_) -> None:
        pass


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
        _build_system_prompt,
        _parse_provider_and_model,
        check_provider_available,
        get_provider_availability,
        run_agent,
    )
    from nsight_agent.agent.preflight import (
        count_tokens_exact,
        estimate_cache_breakdown,
        estimate_json_tokens,
        estimate_prose_tokens,
        estimate_total_session_tokens,
    )
    from nsight_agent.agent.tools import tool_schemas
    from nsight_agent.analysis.metrics import compute_profile_summary
    from nsight_agent.ingestion.profile import NsysProfile

    # Resolve provider early so we can fail fast before any expensive work.
    resolved_provider, resolved_model, reason = _parse_provider_and_model(args.model)

    if not args.quiet:
        console.print(
            f"Using AI provider = [cyan]{resolved_provider}[/cyan], model = [cyan]{resolved_model}[/cyan] (selected based on {reason})"
        )

    missing = check_provider_available(resolved_provider)
    if missing:
        console.print(
            f"[red]Error:[/red] {resolved_provider} provider requires a missing package: {missing}"
        )
        sys.exit(1)

    # Warn about any other unavailable providers so the user knows what's on offer.
    if not args.quiet:
        availability = get_provider_availability()
        unavailable = [
            (p, hint) for p, hint in availability.items() if hint and p != resolved_provider
        ]
        for p, hint in unavailable:
            console.print(
                f"[yellow]Warning:[/yellow] {p} provider unavailable ({hint})", highlight=False
            )

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
                f"{p.top_kernels[0].short_name or p.top_kernels[0].name} ({p.top_kernels[0].pct_of_gpu_time}%)"
                if p.top_kernels
                else "—"
            )
            row: list[str] = [
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
                top_mpi = f"{p.mpi_ops[0].op} ({p.mpi_ops[0].total_s}s)" if p.mpi_ops else "—"
                row.append(top_mpi)
            ph.add_row(*row)
        console.print(ph)

    if args.verbose:
        console.print("\n[bold]── ProfileSummary sent to AI ──[/bold]")
        console.print(summary.model_dump_json(indent=2))
        console.print("[bold]── End ProfileSummary ──[/bold]\n")

    # Pre-flight token estimate
    _system_prompt = _build_system_prompt(grounded=not args.allow_app_knowledge)
    _summary_json = summary.model_dump_json(indent=2)
    _schemas = tool_schemas()
    _schemas_json = json.dumps(_schemas)

    if args.exact_token_count:
        _input_tokens = count_tokens_exact(
            resolved_provider,
            resolved_model,
            _system_prompt,
            _summary_json,
            tools=_schemas,
        )
        if _input_tokens is None:
            if not args.quiet and not args.json:
                console.print(
                    f"[dim](exact count unavailable for {resolved_provider}"
                    " — using heuristic)[/dim]"
                )
            _input_tokens = (
                estimate_prose_tokens(_system_prompt)
                + estimate_json_tokens(_summary_json, resolved_provider)
                + estimate_json_tokens(_schemas_json, resolved_provider)
            )
            _input_label = "heuristic"
        else:
            _input_label = "exact"
    else:
        _input_tokens = (
            estimate_prose_tokens(_system_prompt)
            + estimate_json_tokens(_summary_json, resolved_provider)
            + estimate_json_tokens(_schemas_json, resolved_provider)
        )
        _input_label = "heuristic"

    _max_turns = args.max_turns
    _output_lo = 5 * 600 + 800  # 3,800 — optimistic (5 turns)
    _output_hi = _max_turns * 600 + 800  # pessimistic (max turns)

    if not args.quiet and not args.json:
        if resolved_provider == "anthropic":
            _cache_est = estimate_cache_breakdown(_input_tokens, _max_turns)
            console.print(
                f"\n[bold]Token estimate (Anthropic, sliding prompt cache):[/bold]\n"
                f"  Cache write: ~{_cache_est['cache_creation']:,} tokens  (billed at 1.25×)\n"
                f"  Cache read:  ~{_cache_est['cache_read']:,} tokens  (billed at 0.10×)\n"
                f"  Non-cached:  ~0 tokens\n"
                f"  Output:      ~{_output_lo:,} – {_output_hi:,}"
                f" (5 – {_max_turns} turns)\n"
                f"  Cost-equiv:  ~{_cache_est['cost_equivalent']:,} tokens  ({_input_label})\n"
                f"  Model:       {resolved_model} ({resolved_provider})"
            )
        else:
            _lo_turns = min(5, _max_turns)
            _total_lo = estimate_total_session_tokens(_input_tokens, _lo_turns)
            _total_hi = estimate_total_session_tokens(_input_tokens, _max_turns)
            _cache_note = (
                "\n  [dim](OpenAI applies automatic ~50% caching to repeated prefixes)[/dim]"
                if resolved_provider == "openai"
                else ""
            )
            if _lo_turns < _max_turns:
                _input_range = (
                    f"~{_total_lo:,} – ~{_total_hi:,}"
                    f" ({_lo_turns} – {_max_turns} turns, {_input_label})"
                )
            else:
                _input_range = f"~{_total_hi:,} ({_input_label})"
            console.print(
                f"\n[bold]Token estimate (total across up to {_max_turns} turns):[/bold]\n"
                f"  Input:  {_input_range}{_cache_note}\n"
                f"  Output: ~{_output_lo:,} – {_output_hi:,}\n"
                f"  Model:  {resolved_model} ({resolved_provider})"
            )
        if not args.yes and sys.stdin.isatty():
            try:
                _answer = input("\nProceed? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if _answer in ("n", "no"):
                console.print("[yellow]Aborted.[/yellow]")
                sys.exit(0)

    log = lambda msg: console.print(msg, markup=False, highlight=False)

    # Compute timestamp now (start of run) for consistent log/transcript filenames.
    _start_time = datetime.now()
    _ts = _start_time.strftime("%Y%m%d_%H%M%S")
    _profile_stem = Path(args.profile).stem
    _out_dir = Path(args.profile).parent

    # Resolve log file path if requested.
    _log_requested = args.log or bool(args.log_file)
    _log_path = (
        Path(args.log_file)
        if args.log_file
        else (_out_dir / f"{_profile_stem}_{_ts}_log.txt" if _log_requested else None)
    )

    token_usage: dict[str, int | None] = {}
    t_agent = time.perf_counter()
    with (
        LLMLogger(_log_path) if _log_path else _null_context()
    ) as _logger:
        if _log_path and _logger is not None:
            _logger.write_header(
                command="analyze",
                argv=sys.argv,
                provider=resolved_provider,
                model=resolved_model,
                profile_path=args.profile,
                start_time=_start_time,
            )
            if not args.quiet:
                console.print(f"  LLM log:    {_log_path}")
        hypotheses = run_agent(
            args.profile,
            summary=summary,
            verbose=not args.quiet,
            model=args.model,
            max_turns=args.max_turns,
            token_usage=token_usage,
            grounded=not args.allow_app_knowledge,
            log=log,
            logger=_logger,
        )
    timings["agent_s"] = time.perf_counter() - t_agent

    if args.json:
        print(json.dumps(hypotheses, indent=2))
        return

    table = Table(title=f"Hypotheses — {Path(args.profile).name}", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Type", style="cyan")
    table.add_column("Phase", style="dim")
    table.add_column("Impact", style="bold")
    table.add_column("Action", style="magenta")
    table.add_column("Description")
    table.add_column("Evidence", style="dim")
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
            h.get("phase", "—"),
            f"[{impact_color}]{h.get('expected_impact', '—')}[/{impact_color}]",
            action_str,
            h.get("description", ""),
            h.get("evidence", ""),
            h.get("suggestion", ""),
        )

    console.print(table)

    if not args.quiet:
        _print_timings(timings)
        inp = token_usage.get("input_tokens")
        out = token_usage.get("output_tokens")
        cache_write = token_usage.get("cache_creation_tokens") or 0
        cache_read = token_usage.get("cache_read_tokens") or 0
        if inp is not None and out is not None:
            cost = token_usage.get("cost_usd")
            cost_str = f"  Cost: [yellow]${cost:.4f}[/yellow]" if cost is not None else ""
            if cache_write or cache_read:
                _total = inp + cache_write + cache_read + out
                _cost_equiv = int(cache_write * 1.25 + cache_read * 0.10 + inp)
                console.print(
                    f"  Tokens: [cyan]{inp:,}[/cyan] non-cached"
                    f" / [cyan]{cache_write:,}[/cyan] cache-write"
                    f" / [cyan]{cache_read:,}[/cyan] cache-read"
                    f" / [cyan]{out:,}[/cyan] out"
                    f"  ([dim]{_total:,} total |"
                    f" ~{_cost_equiv:,} cost-equiv input[/dim]){cost_str}"
                )
            else:
                console.print(
                    f"  Tokens: [cyan]{inp:,}[/cyan] in / [cyan]{out:,}[/cyan] out"
                    f"  ([dim]{inp + out:,} total[/dim]){cost_str}"
                )
        else:
            console.print("  Tokens: [dim]N/A[/dim]")

    _transcript_requested = args.transcript or bool(args.transcript_file)
    if _transcript_requested:
        _transcript_path = (
            Path(args.transcript_file)
            if args.transcript_file
            else _out_dir / f"{_profile_stem}_{_ts}_transcript.txt"
        )
        _transcript_path.write_text(console.export_text(clear=False), encoding="utf-8")
        if not args.quiet:
            console.print(f"  Transcript: {_transcript_path}")


def _print_phase_table(summary, title: str) -> None:
    """Print a phases table for one profile, or a note if no phases were detected."""
    if not summary.phases:
        console.print(f"[dim]{title}: no phases detected[/dim]")
        return
    ph = Table(title=title)
    ph.add_column("Phase")
    ph.add_column("Start (s)", justify="right")
    ph.add_column("End (s)", justify="right")
    ph.add_column("Duration (s)", justify="right")
    ph.add_column("GPU Util %", justify="right")
    ph.add_column("Top Kernel")
    for p in summary.phases:
        top_kernel = (
            f"{p.top_kernels[0].short_name or p.top_kernels[0].name} ({p.top_kernels[0].pct_of_gpu_time}%)"
            if p.top_kernels
            else "—"
        )
        ph.add_row(
            p.name,
            str(p.start_s),
            str(p.end_s),
            str(p.duration_s),
            f"{p.gpu_utilization_pct}%",
            top_kernel,
        )
    console.print(ph)


def cmd_compare(args: argparse.Namespace) -> None:
    from nsight_agent.agent.compare import _COMPARE_SYSTEM_PROMPT, _build_prompt, run_compare
    from nsight_agent.agent.loop import (
        _parse_provider_and_model,
        check_provider_available,
        get_provider_availability,
    )
    from nsight_agent.agent.preflight import (
        count_tokens_exact,
        estimate_json_tokens,
        estimate_prose_tokens,
    )
    from nsight_agent.analysis.diff import compute_profile_diff
    from nsight_agent.analysis.metrics import compute_profile_summary
    from nsight_agent.ingestion.profile import NsysProfile

    resolved_provider, resolved_model, reason = _parse_provider_and_model(args.model)

    if not args.quiet:
        console.print(
            f"Using AI provider = [cyan]{resolved_provider}[/cyan], "
            f"model = [cyan]{resolved_model}[/cyan] (selected based on {reason})"
        )

    missing = check_provider_available(resolved_provider)
    if missing:
        console.print(
            f"[red]Error:[/red] {resolved_provider} provider requires a missing package: {missing}"
        )
        sys.exit(1)

    if not args.quiet:
        availability = get_provider_availability()
        unavailable = [
            (p, hint) for p, hint in availability.items() if hint and p != resolved_provider
        ]
        for p, hint in unavailable:
            console.print(
                f"[yellow]Warning:[/yellow] {p} provider unavailable ({hint})", highlight=False
            )

    with NsysProfile(args.profile_a) as pa:
        summary_a = compute_profile_summary(pa, max_phases=args.max_phases)
    with NsysProfile(args.profile_b) as pb:
        summary_b = compute_profile_summary(pb, max_phases=args.max_phases)

    # Always print both phase tables first
    _print_phase_table(summary_a, f"Phases — {Path(args.profile_a).name}")
    _print_phase_table(summary_b, f"Phases — {Path(args.profile_b).name}")

    # Compute diff — also determines comparison_mode
    diff = compute_profile_diff(summary_a, summary_b)

    # Print status message
    n_a = len(summary_a.phases)
    n_b = len(summary_b.phases)
    if diff.comparison_mode == "phase_aware":
        console.print(
            f"\n[green]Phases match ({n_a} phases) — "
            f"proceeding with in-depth per-phase analysis.[/green]"
        )
    elif diff.comparison_mode == "summary":
        console.print(
            f"\n[yellow]Phases do not match (Profile A has {n_a}, Profile B has {n_b}) — "
            f"proceeding with summary comparison "
            f"(kernel overlap: {diff.kernel_overlap_pct:.0f}%).[/yellow]"
        )
    else:  # summary_no_kernel
        console.print(
            f"\n[yellow]Phases do not match (Profile A has {n_a}, Profile B has {n_b}) "
            f"and kernel name overlap is {diff.kernel_overlap_pct:.0f}% — "
            f"proceeding with summary analysis and skipping per-kernel comparison.[/yellow]"
        )

    # Pre-flight token estimate
    _compare_prompt = _build_prompt(summary_a, summary_b, diff)

    if args.exact_token_count:
        _input_tokens = count_tokens_exact(
            resolved_provider,
            resolved_model,
            _COMPARE_SYSTEM_PROMPT,
            _compare_prompt,
        )
        if _input_tokens is None:
            if not args.quiet and not args.json:
                console.print(
                    f"[dim](exact count unavailable for {resolved_provider}"
                    " — using heuristic)[/dim]"
                )
            _input_tokens = estimate_prose_tokens(_COMPARE_SYSTEM_PROMPT) + estimate_json_tokens(
                _compare_prompt, resolved_provider
            )
            _input_label = "heuristic"
        else:
            _input_label = "exact"
    else:
        _input_tokens = estimate_prose_tokens(_COMPARE_SYSTEM_PROMPT) + estimate_json_tokens(
            _compare_prompt, resolved_provider
        )
        _input_label = "heuristic"

    if not args.quiet and not args.json:
        console.print(
            f"\n[bold]Token estimate:[/bold]\n"
            f"  Input:  ~{_input_tokens:,} ({_input_label})\n"
            f"  Output: ~1,000 – 3,000 (single LLM call)\n"
            f"  Model:  {resolved_model} ({resolved_provider})"
        )
        if not args.yes and sys.stdin.isatty():
            try:
                _answer = input("\nProceed? [Y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                sys.exit(0)
            if _answer in ("n", "no"):
                console.print("[yellow]Aborted.[/yellow]")
                sys.exit(0)

    log = lambda msg: console.print(msg, markup=False, highlight=False)
    token_usage: dict[str, int | None] = {}

    _start_time = datetime.now()
    _ts = _start_time.strftime("%Y%m%d_%H%M%S")
    _stem_a = Path(args.profile_a).stem
    _stem_b = Path(args.profile_b).stem
    _out_dir = Path(args.profile_a).parent
    _file_stem = f"{_stem_a}_vs_{_stem_b}_{_ts}"

    _log_requested = args.log or bool(args.log_file)
    _log_path = (
        Path(args.log_file)
        if args.log_file
        else (_out_dir / f"{_file_stem}_log.txt" if _log_requested else None)
    )

    with (
        LLMLogger(_log_path) if _log_path else _null_context()
    ) as _logger:
        if _log_path and _logger is not None:
            _logger.write_header(
                command="compare",
                argv=sys.argv,
                provider=resolved_provider,
                model=resolved_model,
                start_time=_start_time,
            )
            if not args.quiet:
                console.print(f"  LLM log:    {_log_path}")
        report = run_compare(
            args.profile_a,
            args.profile_b,
            summary_a=summary_a,
            summary_b=summary_b,
            diff=diff,
            model=args.model,
            verbose=not args.quiet,
            token_usage=token_usage,
            log=log,
            logger=_logger,
        )

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Display narrative
    narrative = report.get("narrative", "")
    if narrative:
        console.print(f"\n[bold]Analysis[/bold]\n{narrative}")

    # Display key differences table
    key_diffs = report.get("key_differences", [])
    if key_diffs:
        name_a = Path(args.profile_a).name
        name_b = Path(args.profile_b).name
        t = Table(title="Key Differences", show_lines=True)
        t.add_column("Metric")
        t.add_column(f"A: {name_a}", justify="right")
        t.add_column(f"B: {name_b}", justify="right")
        t.add_column("Delta %", justify="right")
        t.add_column("Note")
        for d in key_diffs:
            mag = d.get("magnitude_pct")
            if mag is None:
                delta_str = "—"
                delta_color = "dim"
            else:
                delta_str = f"{mag:+.1f}%"
                abs_mag = abs(mag)
                delta_color = "bold" if abs_mag >= 50 else "yellow" if abs_mag >= 20 else "dim"
            t.add_row(
                d.get("metric", ""),
                d.get("profile_a", ""),
                d.get("profile_b", ""),
                f"[{delta_color}]{delta_str}[/{delta_color}]",
                d.get("note", ""),
            )
        console.print(t)

    if not args.quiet:
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

    _transcript_requested = args.transcript or bool(args.transcript_file)
    if _transcript_requested:
        _transcript_path = (
            Path(args.transcript_file)
            if args.transcript_file
            else _out_dir / f"{_file_stem}_transcript.txt"
        )
        _transcript_path.write_text(console.export_text(clear=False), encoding="utf-8")
        if not args.quiet:
            console.print(f"  Transcript: {_transcript_path}")


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
    console.print(
        f"  GPU kernel time: {summary.gpu_kernel_s:.1f}s  ({summary.gpu_utilization_pct:.1f}% utilization)"
    )
    console.print(f"  GPU memcpy time: {summary.gpu_memcpy_s:.1f}s")
    console.print(f"  GPU sync time:   {summary.gpu_sync_s:.1f}s")
    console.print(f"  GPU idle time:   {summary.total_gpu_idle_s:.1f}s")
    console.print(f"  MPI present:     {summary.mpi_present}")

    t = Table(title="Top Kernels")
    for col in ("Kernel", "Calls", "Total (s)", "Avg (ms)", "% GPU"):
        t.add_column(col)
    for k in summary.top_kernels:
        t.add_row(
            k.short_name or k.name,
            str(k.calls),
            str(k.total_s),
            str(k.avg_ms),
            f"{k.pct_of_gpu_time}%",
        )
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
            top = (p.top_kernels[0].short_name or p.top_kernels[0].name) if p.top_kernels else "—"
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
    p_analyze.add_argument(
        "--verbose", action="store_true", help="Print the ProfileSummary sent to the AI"
    )
    p_analyze.add_argument("--quiet", action="store_true", help="Suppress agent turn logging")
    p_analyze.add_argument(
        "--max-phases",
        type=int,
        default=6,
        help="Maximum number of execution phases to detect (default: 6; use 1 to disable)",
    )
    p_analyze.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=(
            f"Maximum number of agent tool-call turns before forcing output "
            f"(default: {MAX_TURNS}). A wrap-up warning is injected {WARN_TURNS_BEFORE_LIMIT} "
            f"turns before the limit; a forced no-tool output turn fires if the limit is hit."
        ),
    )
    p_analyze.add_argument(
        "--model",
        default=None,
        help=(
            "Model for hypothesis generation. Examples: "
            "'openai:gpt-4o' (provider prefix + model), "
            "'openai' (provider only, uses default model), "
            "'claude-haiku-4-5-20251001' (model only, provider auto-detected). "
            "Defaults: claude-opus-4-6 (anthropic), gpt-4o (openai), gemini-2.0-flash (gemini)."
        ),
    )
    p_analyze.add_argument(
        "--allow-app-knowledge",
        action="store_true",
        help=(
            "Allow the model to draw on application-specific knowledge from training data "
            "(e.g. suggest named environment variables or library options inferred from kernel names). "
            "By default suggestions are grounded strictly in the profile data."
        ),
    )
    p_analyze.add_argument(
        "--yes",
        action="store_true",
        help="Skip the pre-flight confirmation prompt and proceed automatically.",
    )
    p_analyze.add_argument(
        "--exact-token-count",
        action="store_true",
        help=(
            "Use the provider's token counting API for an exact input token count instead of "
            "the character-count heuristic (Anthropic only; adds one small API call before the "
            "main run). Falls back to heuristic for other providers."
        ),
    )
    p_analyze.add_argument(
        "--log",
        action="store_true",
        help=(
            "Save a full LLM interaction log (every request and response) next to the profile. "
            "The file is written in real time so a partial log is available if the run fails."
        ),
    )
    p_analyze.add_argument(
        "--log-file",
        metavar="PATH",
        help="Save the LLM interaction log to PATH (implies --log).",
    )
    p_analyze.add_argument(
        "--transcript",
        action="store_true",
        help="Save a transcript of everything printed to the terminal next to the profile.",
    )
    p_analyze.add_argument(
        "--transcript-file",
        metavar="PATH",
        help="Save the terminal transcript to PATH (implies --transcript).",
    )
    p_compare = sub.add_parser("compare", help="Compare two profiles and summarize differences")
    p_compare.add_argument("profile_a", help="Path to first .sqlite profile")
    p_compare.add_argument("profile_b", help="Path to second .sqlite profile")
    p_compare.add_argument("--json", action="store_true", help="Output raw JSON")
    p_compare.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    p_compare.add_argument(
        "--max-phases",
        type=int,
        default=6,
        help="Maximum number of execution phases to detect per profile (default: 6; use 1 to disable)",
    )
    p_compare.add_argument(
        "--model",
        default=None,
        help=(
            "Model for comparison. Same format as analyze --model: "
            "'openai:gpt-4o', 'openai', or a bare model ID."
        ),
    )
    p_compare.add_argument(
        "--yes",
        action="store_true",
        help="Skip the pre-flight confirmation prompt and proceed automatically.",
    )
    p_compare.add_argument(
        "--exact-token-count",
        action="store_true",
        help=(
            "Use the provider's token counting API for an exact input token count "
            "(Anthropic only; adds one small API call). "
            "Falls back to heuristic for other providers."
        ),
    )
    p_compare.add_argument(
        "--log",
        action="store_true",
        help="Save a full LLM interaction log (request and response) next to profile A.",
    )
    p_compare.add_argument(
        "--log-file",
        metavar="PATH",
        help="Save the LLM interaction log to PATH (implies --log).",
    )
    p_compare.add_argument(
        "--transcript",
        action="store_true",
        help="Save a transcript of everything printed to the terminal next to profile A.",
    )
    p_compare.add_argument(
        "--transcript-file",
        metavar="PATH",
        help="Save the terminal transcript to PATH (implies --transcript).",
    )

    p_summary = sub.add_parser("summary", help="Print structured metrics summary")
    p_summary.add_argument("profile", help="Path to .sqlite profile")
    p_summary.add_argument("--json", action="store_true", help="Output raw JSON")
    p_summary.add_argument(
        "--max-phases",
        type=int,
        default=6,
        help="Maximum number of execution phases to detect (default: 6; use 1 to disable)",
    )

    args = parser.parse_args()
    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "summary":
        cmd_summary(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
