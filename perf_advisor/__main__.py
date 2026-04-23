"""CLI entry point: python -m perf_advisor analyze <profile.sqlite>"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from perf_advisor.agent.logger import LLMLogger
from perf_advisor.agent.loop import MAX_TURNS, WARN_TURNS_BEFORE_LIMIT

console = Console(record=True)

_PHASE_WARNING = (
    "[orange1]Warning: automated phase detection has no semantic understanding of the "
    "application and can produce logically incorrect segmentations. Cross-reference results "
    "against your knowledge of the application's natural compute stages and the Nsight Systems "
    "GUI timeline; adjust --max-phases if the segmentation does not match expectations.[/orange1]"
)


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


def _print_cross_rank_tables(cross_rank_summary) -> None:
    """Print per-rank overview and per-phase imbalance tables."""
    from perf_advisor.analysis.models import CrossRankSummary

    crs: CrossRankSummary = cross_rank_summary

    # Table 1 — per-rank overview
    t1 = Table(title=f"Multi-rank Overview ({crs.num_ranks} ranks)", show_header=True)
    t1.add_column("Rank", justify="right")
    t1.add_column("GPU Kernel (s)", justify="right")
    t1.add_column("GPU Idle (s)", justify="right")
    t1.add_column("MPI Wait (s)", justify="right")
    t1.add_column("GPU Util %", justify="right")
    t1.add_column("Primary", justify="center")
    for ov in crs.per_rank_overview:
        marker = "[bold green]★[/bold green]" if ov.rank_id == crs.primary_rank_id else ""
        t1.add_row(
            str(ov.rank_id),
            f"{ov.gpu_kernel_s:.2f}",
            f"{ov.gpu_idle_s:.2f}",
            f"{ov.mpi_wait_s:.2f}",
            f"{ov.gpu_utilization_pct:.1f}%",
            marker,
        )
    console.print(t1)

    if not crs.phases:
        return

    # Table 2 — per-phase imbalance
    t2 = Table(title="Cross-rank Phase Imbalance", show_header=True)
    t2.add_column("Phase")
    t2.add_column("GPU Kernel Imbalance", justify="right")
    t2.add_column("Slowest Rank", justify="right")
    t2.add_column("MPI Wait Imbalance", justify="right")
    t2.add_column("Slowest Rank", justify="right")
    t2.add_column("Top Collective Imbalance")

    def _imbalance_color(score: float) -> str:
        if score >= 0.5:
            return "red"
        if score >= 0.2:
            return "yellow"
        return "green"

    for ph in crs.phases:
        gpu_color = _imbalance_color(ph.gpu_kernel_imbalance)
        mpi_color = _imbalance_color(ph.mpi_wait_imbalance)
        top_coll = (
            f"{ph.collective_imbalance[0].op} "
            f"[{_imbalance_color(ph.collective_imbalance[0].imbalance_score)}]"
            f"{ph.collective_imbalance[0].imbalance_score:.0%}[/]"
            if ph.collective_imbalance
            else "—"
        )
        t2.add_row(
            ph.phase_name,
            f"[{gpu_color}]{ph.gpu_kernel_imbalance:.0%}[/{gpu_color}]",
            str(ph.gpu_kernel_slowest_rank_id),
            f"[{mpi_color}]{ph.mpi_wait_imbalance:.0%}[/{mpi_color}]",
            str(ph.mpi_wait_slowest_rank_id),
            top_coll,
        )
    console.print(t2)

    if crs.phase_alignment == "index_order":
        console.print(
            Panel(
                "[yellow]Phase names differed across ranks — phases aligned by index order "
                "based on duration agreement. Cross-rank phase labels may not be exact.[/yellow]",
                title="[yellow]Warning: Phase Alignment[/yellow]",
                border_style="yellow",
            )
        )


def cmd_analyze(args: argparse.Namespace) -> None:
    from perf_advisor.agent.loop import (
        _build_system_prompt,
        _parse_provider_and_model,
        check_provider_available,
        get_provider_availability,
        run_agent,
    )
    from perf_advisor.agent.preflight import (
        count_tokens_exact,
        estimate_cache_breakdown,
        estimate_gemini_cache_breakdown,
        estimate_json_tokens,
        estimate_prose_tokens,
        estimate_total_session_tokens,
    )
    from perf_advisor.agent.tools import tool_schemas
    from perf_advisor.analysis.metrics import compute_profile_summary
    from perf_advisor.ingestion.profile import NsysProfile

    # Resolve provider early so we can fail fast before any expensive work.
    resolved_provider, resolved_model, reason = _parse_provider_and_model(args.model)

    if not args.quiet:
        console.print(
            f"Using AI provider = [cyan]{resolved_provider}[/cyan],"
            f" model = [cyan]{resolved_model}[/cyan] (selected based on {reason})"
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

    # -----------------------------------------------------------------------
    # Multi-rank path: more than one profile file provided
    # -----------------------------------------------------------------------
    cross_rank_summary = None
    primary_profile: str

    if len(args.profile) > 1:
        from pathlib import Path as _Path

        from perf_advisor.analysis.cross_rank import (
            align_phases,
            compute_cross_rank_summary,
            parse_rank_ids,
            select_consensus_k,
            select_primary_rank,
        )
        from perf_advisor.analysis.phases import compute_phase_cost_curve

        profile_paths = [_Path(p) for p in args.profile]
        rank_ids, parsed_ok = parse_rank_ids(profile_paths)
        if not parsed_ok and not args.quiet:
            console.print(
                "[yellow]Warning:[/yellow] Could not determine rank IDs from filenames "
                "— using index order (0, 1, 2, …).",
                highlight=False,
            )

        rank_id_to_path = dict(zip(rank_ids, profile_paths))

        if not args.quiet:
            console.print(
                f"  Multi-rank mode: {len(profile_paths)} profiles, rank IDs {sorted(rank_ids)}"
            )

        if args.verbose:
            print("[phase] Rank → file mapping:")
            for rid, path in sorted(rank_id_to_path.items()):
                print(f"         rank={rid}  {path}")

        # Pass 1 — lightweight cost-curve extraction for consensus k selection
        _cost_curves: dict[int, dict[int, float]] = {}
        _selected_ks: dict[int, int] = {}
        for rid, path in sorted(rank_id_to_path.items()):
            with NsysProfile(path) as _prof:
                _selected_ks[rid], _cost_curves[rid] = compute_phase_cost_curve(
                    _prof, max_phases=args.max_phases, rank=rid
                )

        _consensus_k, _consensus_abort = select_consensus_k(
            _cost_curves, _selected_ks, args.max_phases, verbose=args.verbose
        )

        if not args.quiet and _consensus_abort is None and len(set(_selected_ks.values())) > 1:
            _ks_str = ", ".join(f"rank {r}: k={_selected_ks[r]}" for r in sorted(_selected_ks))
            console.print(
                f"  Phase consensus: k={_consensus_k} (per-rank selections differed — {_ks_str})"
            )

        # Pass 2 — full pipeline; forced_k=None if consensus aborted (each rank uses its elbow)
        timings: dict[str, float] = {}
        summaries: dict = {}
        for rid, path in sorted(rank_id_to_path.items()):
            _rank_timings: dict[str, float] = {}
            with NsysProfile(path) as _prof:
                summaries[rid] = compute_profile_summary(
                    _prof,
                    max_phases=args.max_phases,
                    forced_k=_consensus_k,
                    timings=_rank_timings,
                    verbose=args.verbose,
                    rank=rid,
                )
            for k, v in _rank_timings.items():
                timings[k] = timings.get(k, 0.0) + v

        # Primary rank selection
        if args.primary_rank is not None:
            if args.primary_rank not in rank_ids:
                console.print(
                    f"[red]Error:[/red] --primary-rank {args.primary_rank} is not in the "
                    f"set of parsed rank IDs {sorted(rank_ids)}."
                )
                sys.exit(1)
            primary_rank_id = args.primary_rank
            if not args.quiet:
                console.print(
                    f"  Primary rank: [cyan]{primary_rank_id}[/cyan] (set via --primary-rank)"
                )
        else:
            primary_rank_id, outlier_reason = select_primary_rank(summaries)
            if not args.quiet:
                console.print(f"  Primary rank: [cyan]{primary_rank_id}[/cyan] ({outlier_reason})")

        primary_profile = str(rank_id_to_path[primary_rank_id])

        if _consensus_abort is not None:
            # Phase count diverged across ranks — skip cross-rank analysis entirely
            console.print(
                Panel(
                    f"[red]{_consensus_abort}[/red]\n\n"
                    f"Falling back to single-rank analysis on rank {primary_rank_id}.",
                    title="[red]Error: Cross-rank Phase Detection Diverged[/red]",
                    border_style="red",
                )
            )
            summary = summaries[primary_rank_id]
        else:
            # Phase alignment check (names / durations)
            alignment, alignment_msg = align_phases(summaries)

            if alignment == "failed":
                console.print(
                    Panel(
                        f"[red]{alignment_msg}[/red]\n\n"
                        f"Falling back to single-rank analysis on rank {primary_rank_id}.",
                        title="[red]Error: Cross-rank Phase Alignment Failed[/red]",
                        border_style="red",
                    )
                )
                summary = summaries[primary_rank_id]
            else:
                if alignment_msg and not args.quiet:
                    console.print(
                        Panel(
                            f"[yellow]{alignment_msg}[/yellow]",
                            title="[yellow]Warning: Phase Alignment[/yellow]",
                            border_style="yellow",
                        )
                    )
                cross_rank_summary = compute_cross_rank_summary(
                    summaries, primary_rank_id, alignment
                )
                summary = summaries[primary_rank_id]

                if not args.quiet:
                    _print_cross_rank_tables(cross_rank_summary)

    else:
        # Single-rank path (original behaviour)
        primary_profile = args.profile[0]
        timings = {}
        with NsysProfile(primary_profile) as profile:
            summary = compute_profile_summary(
                profile,
                max_phases=args.max_phases,
                timings=timings,
                verbose=args.verbose,
            )

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
                f"{p.top_kernels[0].short_name or p.top_kernels[0].name}"
                f" ({p.top_kernels[0].pct_of_gpu_time}%)"
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
        if args.max_phases > 1:
            console.print(_PHASE_WARNING)

    if args.verbose:
        console.print("\n[bold]── ProfileSummary sent to AI ──[/bold]")
        console.print(summary.model_dump_json(indent=2))
        console.print("[bold]── End ProfileSummary ──[/bold]\n")

    # Pre-flight token estimate
    _system_prompt = _build_system_prompt(
        grounded=not args.allow_app_knowledge,
        device_info=summary.device_info,
    )
    _summary_json = summary.model_dump_json(indent=2)
    _schemas = tool_schemas()
    _schemas_json = json.dumps(_schemas)
    _cross_rank_json = (
        json.dumps(cross_rank_summary.model_dump()) if cross_rank_summary is not None else ""
    )

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
                + estimate_json_tokens(_cross_rank_json, resolved_provider)
            )
            _input_label = "heuristic"
        else:
            _input_label = "exact"
    else:
        _input_tokens = (
            estimate_prose_tokens(_system_prompt)
            + estimate_json_tokens(_summary_json, resolved_provider)
            + estimate_json_tokens(_schemas_json, resolved_provider)
            + estimate_json_tokens(_cross_rank_json, resolved_provider)
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
        elif resolved_provider == "gemini":
            _gemini_est = estimate_gemini_cache_breakdown(_input_tokens, _max_turns)
            console.print(
                f"\n[bold]Token estimate (Gemini, explicit context cache):[/bold]\n"
                f"  Cached prefix: ~{_gemini_est['cache_write']:,} tokens"
                f"  (0.25× each of {_max_turns} turns)\n"
                f"  Cache reads:   ~{_gemini_est['cache_read']:,} tokens"
                f"  (total across all turns)\n"
                f"  Non-cached:    ~{_gemini_est['input']:,} tokens"
                f"  (incremental per-turn history)\n"
                f"  Output:        ~{_output_lo:,} – {_output_hi:,}"
                f" (5 – {_max_turns} turns)\n"
                f"  Cost-equiv:    ~{_gemini_est['cost_equivalent']:,} tokens"
                f"  ({_input_label})\n"
                f"  Model:         {resolved_model} ({resolved_provider})"
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

    def log(msg):
        return console.print(msg, markup=False, highlight=False)

    # Compute timestamp now (start of run) for consistent log/transcript filenames.
    _start_time = datetime.now()
    _ts = _start_time.strftime("%Y%m%d_%H%M%S")
    _profile_stem = Path(primary_profile).stem
    _out_dir = Path(primary_profile).parent

    # Resolve log file path if requested.
    _log_requested = args.log or bool(args.log_file)
    _log_path = (
        Path(args.log_file)
        if args.log_file
        else (_out_dir / f"{_profile_stem}_{_ts}_log.txt" if _log_requested else None)
    )

    token_usage: dict[str, int | None] = {}
    t_agent = time.perf_counter()
    with LLMLogger(_log_path) if _log_path else _null_context() as _logger:
        if _log_path and _logger is not None:
            _logger.write_header(
                command="analyze",
                argv=sys.argv,
                provider=resolved_provider,
                model=resolved_model,
                profile_path=primary_profile,
                start_time=_start_time,
            )
            if not args.quiet:
                console.print(f"  LLM log:    {_log_path}")
        try:
            hypotheses = run_agent(
                primary_profile,
                summary=summary,
                cross_rank_summary=cross_rank_summary,
                verbose=not args.quiet,
                model=args.model,
                max_turns=args.max_turns,
                token_usage=token_usage,
                grounded=not args.allow_app_knowledge,
                log=log,
                logger=_logger,
            )
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)
    timings["agent_s"] = time.perf_counter() - t_agent

    # Save hypothesis JSON next to the log file when logging is enabled.
    if _log_requested and _log_path is not None:
        from perf_advisor.analysis.models import HypothesisReport

        _hyp_path = _log_path.parent / f"{_profile_stem}_{_ts}_hypotheses.json"
        _report = HypothesisReport(
            profile_path=primary_profile,
            generated_at=_start_time.isoformat(),
            provider=resolved_provider,
            model=resolved_model,
            hypotheses=hypotheses,
        )
        _hyp_path.write_text(_report.model_dump_json(indent=2), encoding="utf-8")
        if not args.quiet:
            console.print(f"  Hypotheses: {_hyp_path}")

    if args.json:
        print(json.dumps(hypotheses, indent=2))
        return

    _ACTION_ABBREV = {
        "runtime_config": "runtime",
        "launch_config": "launch",
        "code_optimization": "code",
        "algorithm": "algorithm",
    }
    _IMPACT_COLOR = {"high": "red", "medium": "yellow", "low": "green"}
    _CONF_COLOR = {"high": "green", "medium": "yellow", "low": "dim"}
    _PANEL_BORDER = {"high": "red", "medium": "yellow", "low": "blue"}

    console.print(
        f"\n[bold]Hypotheses — {Path(primary_profile).name}[/bold]",
        justify="center",
    )

    for i, h in enumerate(hypotheses, 1):
        impact_raw = str(h.get("expected_impact", "")).lower()
        impact_color = _IMPACT_COLOR.get(impact_raw, "white")

        action_raw = str(h.get("action_category", "")).lower()
        action_str = _ACTION_ABBREV.get(action_raw, action_raw or "—")

        conf_raw = str(h.get("confidence", "")).lower()
        conf_color = _CONF_COLOR.get(conf_raw, "")

        rt_pct = h.get("runtime_fraction_pct")
        rt_str = f"{rt_pct:.1f}%" if rt_pct is not None else "—"

        spd_lo = h.get("estimated_speedup_pct_lower")
        spd_hi = h.get("estimated_speedup_pct_upper")
        if spd_lo is not None and spd_hi is not None:
            speedup_str = f"{spd_lo:.0f}–{spd_hi:.0f}%"
        elif spd_lo is not None:
            speedup_str = f"≥{spd_lo:.0f}%"
        elif spd_hi is not None:
            speedup_str = f"≤{spd_hi:.0f}%"
        else:
            speedup_str = "—"

        # Header bar: one compact line of metadata.
        # Use style= on each append — Text.append() treats its string as plain
        # text, so embedding Rich markup tags produces literal output.
        header = Text()
        header.append(f"[{i}] {h.get('bottleneck_type', '—')}", style="bold cyan")
        header.append(f"  ·  phase: {h.get('phase', '—')}", style="dim")
        header.append("  ·  impact: ", style="dim")
        header.append(impact_raw or "—", style=f"bold {impact_color}")
        header.append("  ·  action: ", style="dim")
        header.append(action_str, style="magenta")
        header.append("  ·  conf: ", style="dim")
        header.append(conf_raw or "—", style=conf_color or "")
        header.append(f"  ·  runtime {rt_str}", style="dim")
        if speedup_str != "—":
            header.append(f"  ·  speedup {speedup_str}", style="dim")

        # Body: three labeled text blocks
        body = Text()
        body.append("Description\n", style="bold")
        body.append(h.get("description", "—"))
        body.append("\n\nEvidence\n", style="bold dim")
        body.append(h.get("evidence", "—"), style="dim")
        body.append("\n\nSuggestion\n", style="bold")
        body.append(h.get("suggestion", "—"))

        panel_content = Text.assemble(header, "\n\n", body)
        border_color = _PANEL_BORDER.get(impact_raw, "white")
        console.print(Panel(panel_content, border_style=border_color, expand=True))

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
                _run_provider = token_usage.get("provider", "anthropic")
                if _run_provider == "gemini":
                    # Gemini: creation at 1.0×, reads at 0.25×
                    _cost_equiv = int(cache_write * 1.0 + cache_read * 0.25 + inp)
                else:
                    # Anthropic: creation at 1.25×, reads at 0.10×
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


def _print_phase_table(summary, title: str, max_phases: int = 10) -> None:
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
            f"{p.top_kernels[0].short_name or p.top_kernels[0].name}"
            f" ({p.top_kernels[0].pct_of_gpu_time}%)"
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
    if max_phases > 1:
        console.print(_PHASE_WARNING)


def cmd_compare(args: argparse.Namespace) -> None:
    from perf_advisor.agent.compare import _build_compare_system_prompt, _build_prompt, run_compare
    from perf_advisor.agent.loop import (
        _parse_provider_and_model,
        check_provider_available,
        get_provider_availability,
    )
    from perf_advisor.agent.preflight import (
        count_tokens_exact,
        estimate_json_tokens,
        estimate_prose_tokens,
    )
    from perf_advisor.analysis.diff import compute_profile_diff
    from perf_advisor.analysis.metrics import compute_profile_summary
    from perf_advisor.ingestion.profile import NsysProfile

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
    if not args.quiet:
        console.print(
            f"\n  Profile A: [cyan]{Path(args.profile_a).name}[/cyan]"
            f"\n  Profile B: [cyan]{Path(args.profile_b).name}[/cyan]"
        )
    _print_phase_table(summary_a, f"Phases — {Path(args.profile_a).name}", args.max_phases)
    _print_phase_table(summary_b, f"Phases — {Path(args.profile_b).name}", args.max_phases)

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
    _compare_system_prompt = _build_compare_system_prompt(grounded=not args.allow_app_knowledge)

    if args.exact_token_count:
        _input_tokens = count_tokens_exact(
            resolved_provider,
            resolved_model,
            _compare_system_prompt,
            _compare_prompt,
        )
        if _input_tokens is None:
            if not args.quiet and not args.json:
                console.print(
                    f"[dim](exact count unavailable for {resolved_provider}"
                    " — using heuristic)[/dim]"
                )
            _input_tokens = estimate_prose_tokens(_compare_system_prompt) + estimate_json_tokens(
                _compare_prompt, resolved_provider
            )
            _input_label = "heuristic"
        else:
            _input_label = "exact"
    else:
        _input_tokens = estimate_prose_tokens(_compare_system_prompt) + estimate_json_tokens(
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

    def log(msg):
        return console.print(msg, markup=False, highlight=False)

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

    with LLMLogger(_log_path) if _log_path else _null_context() as _logger:
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
        try:
            report = run_compare(
                args.profile_a,
                args.profile_b,
                summary_a=summary_a,
                summary_b=summary_b,
                diff=diff,
                model=args.model,
                verbose=not args.quiet,
                grounded=not args.allow_app_knowledge,
                token_usage=token_usage,
                log=log,
                logger=_logger,
            )
        except RuntimeError as e:
            console.print(f"[red]Error:[/red] {e}")
            sys.exit(1)

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
        t.add_column("Phase")
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
            phase_val = d.get("phase", "whole_profile")
            phase_str = (
                "[dim]—[/dim]" if phase_val == "whole_profile" else f"[cyan]{phase_val}[/cyan]"
            )
            t.add_row(
                d.get("metric", ""),
                phase_str,
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
    from perf_advisor.analysis.metrics import compute_profile_summary
    from perf_advisor.ingestion.profile import NsysProfile

    with NsysProfile(args.profile) as profile:
        summary = compute_profile_summary(profile, max_phases=args.max_phases, verbose=args.verbose)

    if args.json:
        print(summary.model_dump_json(indent=2))
        return

    console.print(f"\n[bold]Profile:[/bold] {summary.profile_path}")
    console.print(f"  Span:            {summary.profile_span_s:.1f}s")
    console.print(
        f"  GPU kernel time: {summary.gpu_kernel_s:.1f}s"
        f"  ({summary.gpu_utilization_pct:.1f}% utilization)"
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
        if args.max_phases > 1:
            console.print(_PHASE_WARNING)


def cmd_evaluate(args: argparse.Namespace) -> None:  # noqa: C901 — complexity is inherent
    import time
    from datetime import datetime
    from pathlib import Path as _Path

    from perf_advisor.agent.loop import (
        _parse_provider_and_model,
        check_provider_available,
    )
    from perf_advisor.analysis.metrics import compute_profile_summary
    from perf_advisor.eval.discover import discover_runs, load_ground_truth_meta
    from perf_advisor.eval.report import (
        load_results,
        print_run_details,
        print_summary_table,
        save_results,
    )
    from perf_advisor.eval.scorer import RunScore, score_run
    from perf_advisor.ingestion.profile import NsysProfile

    _start_time = datetime.now()
    _ts = _start_time.strftime("%Y%m%d_%H%M%S")

    # ── Resolve hypothesis-generation model ─────────────────────────────────
    resolved_provider, resolved_model, reason = _parse_provider_and_model(args.model)
    missing = check_provider_available(resolved_provider)
    if missing:
        console.print(
            f"[red]Error:[/red] {resolved_provider} provider requires a missing package: {missing}"
        )
        sys.exit(1)
    console.print(
        f"Hypothesis model: [cyan]{resolved_model}[/cyan]"
        f" ({resolved_provider}, selected based on {reason})"
    )

    # ── Resolve judge model ──────────────────────────────────────────────────
    judge_model_arg = args.judge_model or "claude-haiku-4-5-20251001"
    judge_provider, judge_model, _ = _parse_provider_and_model(judge_model_arg)
    if not args.skip_judge:
        judge_missing = check_provider_available(judge_provider)
        if judge_missing:
            console.print(
                f"[yellow]Warning:[/yellow] judge provider '{judge_provider}'"
                f" requires {judge_missing}. Suggestion scoring will be skipped."
            )
            args.skip_judge = True
        else:
            console.print(f"Judge model:      [cyan]{judge_model}[/cyan] ({judge_provider})")

    # ── Cached mode: load hypotheses from a previous run ────────────────────
    if args.cached:
        cached_path = _Path(args.cached)
        if not cached_path.exists():
            console.print(f"[red]Error:[/red] --cached file not found: {cached_path}")
            sys.exit(1)
        console.print(f"\nLoading cached hypotheses from {cached_path}")
        _cached_meta, cached_results = load_results(cached_path)

        # Re-score using current ground_truth_meta.json (rubric may have changed)
        bench_dir = _Path(args.ground_truth) if args.ground_truth else None
        if bench_dir:
            try:
                gt_meta_all = load_ground_truth_meta(bench_dir)
            except FileNotFoundError as e:
                console.print(f"[red]Error:[/red] {e}")
                sys.exit(1)
        else:
            gt_meta_all = {}

        results: list[RunScore] = []
        for cached in cached_results:
            gt_meta = gt_meta_all.get(cached.scenario)
            gt_runtime = {
                "scenario": cached.scenario,
                "expected_bottleneck": cached.expected_bottleneck,
            }
            console.print(f"  Rescoring {cached.run_id} ({cached.scenario})…")
            r = score_run(
                run_id=cached.run_id,
                gt_runtime=gt_runtime,
                gt_meta=gt_meta,
                hypotheses=cached.hypotheses,
                sqlite_paths=cached.sqlite_paths,
                judge_model=judge_model,
                judge_provider=judge_provider,
                skip_judge=args.skip_judge,
                elapsed_s=cached.elapsed_s,
            )
            results.append(r)

        console.print()
        print_summary_table(results, console)
        if args.verbose:
            for r in results:
                print_run_details(r, console)

        if args.output:
            _out = _Path(args.output)
            save_results(
                results,
                _out,
                metadata={
                    "model": "cached",
                    "judge_model": judge_model,
                    "judge_provider": judge_provider,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "cached_from": str(cached_path),
                },
            )
            console.print(f"\n  Results saved to {_out}")

        _transcript_requested = args.transcript or bool(args.transcript_file)
        if _transcript_requested:
            _transcript_dir = _Path(args.output).parent if args.output else _Path.cwd()
            _transcript_path = (
                _Path(args.transcript_file)
                if args.transcript_file
                else _transcript_dir / f"eval_{_ts}_transcript.txt"
            )
            _transcript_path.write_text(console.export_text(clear=False), encoding="utf-8")
            console.print(f"  Transcript: {_transcript_path}")
        return

    # ── Discovery mode ───────────────────────────────────────────────────────
    if not args.profiles_dir:
        console.print("[red]Error:[/red] profiles_dir is required unless --cached is given.")
        sys.exit(1)

    profiles_dir = _Path(args.profiles_dir)
    bench_dir = _Path(args.ground_truth) if args.ground_truth else profiles_dir.parent

    if not profiles_dir.is_dir():
        console.print(f"[red]Error:[/red] profiles directory not found: {profiles_dir}")
        sys.exit(1)

    try:
        runs = discover_runs(profiles_dir, bench_dir)
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if not runs:
        console.print(
            f"[yellow]Warning:[/yellow] no runs found under {profiles_dir}."
            " Check that benchmark profiles have been generated."
        )
        sys.exit(0)

    console.print(
        f"\nFound [bold]{len(runs)}[/bold] run(s) under {profiles_dir}"
        f" — subdirs: {', '.join(sorted({r.subdir for r in runs}))}"
    )
    for r in runs:
        n = len(r.sqlite_paths)
        rank_note = f" ({n} ranks)" if n > 1 else ""
        meta_note = "" if r.gt_meta else " [yellow](no meta)[/yellow]"
        console.print(
            f"  {r.run_id:8s}  {r.scenario:35s}  {r.expected_bottleneck}{rank_note}{meta_note}"
        )

    judge_note = " + judge scoring" if not args.skip_judge else " (judge skipped)"
    console.print(f"\nWill run PerfAdvisor on each profile{judge_note}.")
    if not args.yes and sys.stdin.isatty():
        try:
            answer = input("Proceed? [Y/n] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            sys.exit(0)
        if answer in ("n", "no"):
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)

    # ── Per-run analysis + scoring ───────────────────────────────────────────
    from perf_advisor.agent.loop import run_agent

    # Resolve per-run log directory (None = logging disabled).
    _log_requested = args.log or bool(args.log_file)
    _log_dir: _Path | None = None
    if _log_requested:
        if args.log_file:
            _log_dir = _Path(args.log_file)
            _log_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Default: place logs next to each run's primary sqlite
            _log_dir = None  # resolved per-run below using the sqlite parent

    results = []
    for run_cfg in runs:
        console.print(
            f"\n[bold]── {run_cfg.run_id}[/bold]"
            f"  scenario={run_cfg.scenario}"
            f"  ({len(run_cfg.sqlite_paths)} profile(s))"
        )
        t0 = time.perf_counter()
        hypotheses: list[dict] = []
        error: str | None = None

        try:
            # Build per-rank summaries (mirrors cmd_analyze multi-rank path)
            if run_cfg.is_multi_rank:
                from perf_advisor.analysis.cross_rank import (
                    align_phases,
                    compute_cross_rank_summary,
                    parse_rank_ids,
                    select_consensus_k,
                    select_primary_rank,
                )
                from perf_advisor.analysis.phases import compute_phase_cost_curve

                profile_paths = run_cfg.sqlite_paths
                rank_ids, _ = parse_rank_ids(profile_paths)
                rank_id_to_path = dict(zip(rank_ids, profile_paths))

                # Pass 1 — cost curves for consensus k
                _eval_cost_curves: dict[int, dict[int, float]] = {}
                _eval_selected_ks: dict[int, int] = {}
                for rid, path in sorted(rank_id_to_path.items()):
                    with NsysProfile(path) as _prof:
                        _eval_selected_ks[rid], _eval_cost_curves[rid] = compute_phase_cost_curve(
                            _prof, max_phases=args.max_phases
                        )

                _eval_consensus_k, _eval_consensus_abort = select_consensus_k(
                    _eval_cost_curves, _eval_selected_ks, args.max_phases
                )

                # Pass 2 — full pipeline
                summaries: dict = {}
                for rid, path in sorted(rank_id_to_path.items()):
                    with NsysProfile(path) as _prof:
                        summaries[rid] = compute_profile_summary(
                            _prof,
                            max_phases=args.max_phases,
                            forced_k=_eval_consensus_k,
                            verbose=False,
                        )

                primary_rank_id, _ = select_primary_rank(summaries)
                primary_profile = str(rank_id_to_path[primary_rank_id])

                if _eval_consensus_abort is not None:
                    cross_rank_summary = None
                    summary = summaries[primary_rank_id]
                    console.print(
                        "  [yellow]Warning:[/yellow] cross-rank phase detection diverged;"
                        " using primary rank only."
                    )
                else:
                    alignment, alignment_msg = align_phases(summaries)
                    if alignment == "failed":
                        cross_rank_summary = None
                        summary = summaries[primary_rank_id]
                        console.print(
                            "  [yellow]Warning:[/yellow] cross-rank phase alignment"
                            " failed; using primary rank only."
                        )
                    else:
                        if alignment_msg:
                            console.print(
                                f"  [yellow]Warning: phase alignment — {alignment_msg}[/yellow]"
                            )
                        cross_rank_summary = compute_cross_rank_summary(
                            summaries, primary_rank_id, alignment
                        )
                        summary = summaries[primary_rank_id]
            else:
                primary_profile = str(run_cfg.sqlite_paths[0])
                cross_rank_summary = None
                with NsysProfile(primary_profile) as _prof:
                    summary = compute_profile_summary(
                        _prof,
                        max_phases=args.max_phases,
                        verbose=False,
                    )

            # Resolve per-run log path
            _run_log_path: _Path | None = None
            if _log_requested:
                _run_log_dir = _log_dir or _Path(primary_profile).parent
                _run_log_path = _run_log_dir / f"{run_cfg.run_id}_{_ts}_log.txt"
                console.print(f"  LLM log:  {_run_log_path}")

            with LLMLogger(_run_log_path) if _run_log_path else _null_context() as _logger:
                if _run_log_path and _logger is not None:
                    _logger.write_header(
                        command="evaluate",
                        argv=sys.argv,
                        provider=resolved_provider,
                        model=resolved_model,
                        profile_path=primary_profile,
                        start_time=_start_time,
                    )
                hypotheses = run_agent(
                    primary_profile,
                    summary=summary,
                    cross_rank_summary=cross_rank_summary,
                    verbose=False,
                    model=args.model,
                    max_turns=args.max_turns,
                    grounded=not args.allow_app_knowledge,
                    log=lambda msg: console.print(f"  {msg}", markup=False, highlight=False),
                    logger=_logger,
                )
            console.print(
                f"  [green]✓[/green] {len(hypotheses)} hypothesis/hypotheses"
                f" in {time.perf_counter() - t0:.1f}s"
            )
        except Exception as exc:
            error = str(exc)
            console.print(f"  [red]✗ ERROR:[/red] {error}")

        elapsed = time.perf_counter() - t0
        r = score_run(
            run_id=run_cfg.run_id,
            gt_runtime=run_cfg.gt_runtime,
            gt_meta=run_cfg.gt_meta,
            hypotheses=hypotheses,
            sqlite_paths=[str(p) for p in run_cfg.sqlite_paths],
            judge_model=judge_model,
            judge_provider=judge_provider,
            skip_judge=args.skip_judge,
            elapsed_s=elapsed,
        )
        if error:
            r.error = error

        results.append(r)

        # Persist after every run so a partial eval is still useful
        if args.output:
            save_results(
                results,
                _Path(args.output),
                metadata={
                    "model": resolved_model,
                    "provider": resolved_provider,
                    "judge_model": judge_model,
                    "judge_provider": judge_provider,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "profiles_dir": str(profiles_dir),
                    "ground_truth_dir": str(bench_dir),
                },
            )

    # ── Final report ─────────────────────────────────────────────────────────
    console.print()
    print_summary_table(results, console)

    if args.verbose:
        for r in results:
            print_run_details(r, console)

    if args.output:
        console.print(f"\n  Results saved to {args.output}")

    _transcript_requested = args.transcript or bool(args.transcript_file)
    if _transcript_requested:
        _transcript_dir = _Path(args.output).parent if args.output else profiles_dir
        _transcript_path = (
            _Path(args.transcript_file)
            if args.transcript_file
            else _transcript_dir / f"eval_{_ts}_transcript.txt"
        )
        _transcript_path.write_text(console.export_text(clear=False), encoding="utf-8")
        console.print(f"  Transcript: {_transcript_path}")


class _FullHelpParser(argparse.ArgumentParser):
    """ArgumentParser that appends each subcommand's full help to the top-level -h output."""

    def format_help(self) -> str:
        text = super().format_help()
        for action in self._actions:
            if isinstance(action, argparse._SubParsersAction):
                parts = [text]
                for name, subparser in action.choices.items():
                    sep = "─" * 60
                    parts.append(f"\n{sep}\nSubcommand: {name}\n{sep}")
                    parts.append(subparser.format_help())
                return "\n".join(parts)
        return text


def main() -> None:
    parser = _FullHelpParser(prog="perf-advisor")
    sub = parser.add_subparsers(dest="command")

    p_analyze = sub.add_parser(
        "analyze",
        help="Run agent hypothesis generation on a profile",
        description=(
            "Run agent hypothesis generation on a profile. "
            "Profile data (kernel names, NVTX annotations, timing metrics) is sent to the "
            "configured LLM API provider. "
            "Only analyze profiles from trusted sources — profile data is inserted verbatim "
            "into the LLM prompt and could contain instruction-like text. "
            "See README § Risks for details."
        ),
    )
    p_analyze.add_argument(
        "profile",
        nargs="+",
        help=(
            "Path(s) to .sqlite profile(s). Pass multiple files (or a shell glob) "
            "to enable multi-rank analysis."
        ),
    )
    p_analyze.add_argument("--json", action="store_true", help="Output raw JSON")
    p_analyze.add_argument(
        "--verbose", action="store_true", help="Print the ProfileSummary sent to the AI"
    )
    p_analyze.add_argument("--quiet", action="store_true", help="Suppress agent turn logging")
    p_analyze.add_argument(
        "--max-phases",
        type=int,
        default=10,
        help="Maximum number of execution phases to detect (default: 10; use 1 to disable)",
    )
    p_analyze.add_argument(
        "--primary-rank",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Multi-rank mode: use rank N as the primary rank for detailed analysis "
            "(N is the parsed rank ID from the filename, not the file index). "
            "By default the outlier rank (highest GPU idle time) is selected automatically."
        ),
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
            "Defaults: claude-opus-4-6 (anthropic), gpt-4o (openai), gemini-2.5-flash (gemini)."
        ),
    )
    p_analyze.add_argument(
        "--allow-app-knowledge",
        action="store_true",
        help=(
            "Allow the model to draw on application-specific knowledge from training data "
            "(e.g. suggest named environment variables or library options"
            " inferred from kernel names). "
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
    p_compare = sub.add_parser(
        "compare",
        help="Compare two profiles and summarize differences",
        description=(
            "Compare two profiles and summarize differences. "
            "Profile data (kernel names, NVTX annotations, timing metrics) is sent to the "
            "configured LLM API provider. "
            "Only analyze profiles from trusted sources — profile data is inserted verbatim "
            "into the LLM prompt and could contain instruction-like text. "
            "See README § Risks for details."
        ),
    )
    p_compare.add_argument("profile_a", help="Path to first .sqlite profile")
    p_compare.add_argument("profile_b", help="Path to second .sqlite profile")
    p_compare.add_argument("--json", action="store_true", help="Output raw JSON")
    p_compare.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    p_compare.add_argument(
        "--max-phases",
        type=int,
        default=10,
        help=(
            "Maximum number of execution phases to detect per profile"
            " (default: 10; use 1 to disable)"
        ),
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
        "--allow-app-knowledge",
        action="store_true",
        help=(
            "Allow the model to draw on application-specific knowledge from training data "
            "when interpreting differences. "
            "By default observations are grounded strictly in the profile data."
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

    p_evaluate = sub.add_parser(
        "evaluate",
        help="Score PerfAdvisor against the synthetic benchmark ground truth",
        description=(
            "Run PerfAdvisor on all benchmark profiles found under profiles_dir, "
            "then score each run's hypotheses against ground_truth_meta.json using "
            "a two-tier rubric: (1) bottleneck-type detection and (2) suggestion "
            "coverage judged by an LLM. Pass --cached to rescore from saved results "
            "without re-running PerfAdvisor."
        ),
    )
    p_evaluate.add_argument(
        "profiles_dir",
        nargs="?",
        help=(
            "Directory containing {1gpu,4gpu,8gpu}/ subdirs with benchmark profiles. "
            "Required unless --cached is given."
        ),
    )
    p_evaluate.add_argument(
        "--ground-truth",
        metavar="BENCH_DIR",
        default=None,
        help=(
            "Directory containing ground_truth_meta.json (usually bench/). "
            "Defaults to the parent of profiles_dir."
        ),
    )
    p_evaluate.add_argument(
        "--model",
        default=None,
        help=(
            "Model for hypothesis generation. Same format as 'analyze --model'. "
            "Defaults: claude-opus-4-6 (anthropic), gpt-4o (openai)."
        ),
    )
    p_evaluate.add_argument(
        "--judge-model",
        default=None,
        metavar="MODEL",
        help=(
            "Model for LLM-judge suggestion scoring. Same provider:model format. "
            "Default: claude-haiku-4-5-20251001."
        ),
    )
    p_evaluate.add_argument(
        "--max-turns",
        type=int,
        default=MAX_TURNS,
        help=f"Max PerfAdvisor tool-call turns per run (default: {MAX_TURNS}).",
    )
    p_evaluate.add_argument(
        "--max-phases",
        type=int,
        default=10,
        help="Max execution phases to detect per profile (default: 10).",
    )
    p_evaluate.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help=(
            "Write full results (hypotheses + scores) to this JSON file. "
            "Written incrementally after each run so a partial eval is preserved."
        ),
    )
    p_evaluate.add_argument(
        "--cached",
        metavar="PATH",
        default=None,
        help=(
            "Load hypotheses from a previously-saved --output file and re-run "
            "scoring only (no PerfAdvisor calls). Useful for iterating on the "
            "scoring rubric without paying for LLM hypothesis generation again."
        ),
    )
    p_evaluate.add_argument(
        "--skip-judge",
        action="store_true",
        help=(
            "Skip LLM-judge suggestion scoring; only evaluate bottleneck detection. "
            "Faster and cheaper for quick checks."
        ),
    )
    p_evaluate.add_argument(
        "--allow-app-knowledge",
        action="store_true",
        help="Same as 'analyze --allow-app-knowledge': lift the grounding constraint.",
    )
    p_evaluate.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-suggestion judge scores for every run.",
    )
    p_evaluate.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt.",
    )
    p_evaluate.add_argument(
        "--log",
        action="store_true",
        help=(
            "Save a full LLM interaction log for each run. "
            "By default, logs are written next to each run's primary SQLite file "
            "as {run_id}_{timestamp}_log.txt. See also --log-file."
        ),
    )
    p_evaluate.add_argument(
        "--log-file",
        metavar="DIR",
        help=(
            "Directory to write per-run LLM interaction logs (implies --log). "
            "Each run produces {run_id}_{timestamp}_log.txt in this directory. "
            "The directory is created if it does not exist."
        ),
    )
    p_evaluate.add_argument(
        "--transcript",
        action="store_true",
        help="Save a transcript of the entire evaluation session to a file.",
    )
    p_evaluate.add_argument(
        "--transcript-file",
        metavar="PATH",
        help="Save the session transcript to PATH (implies --transcript).",
    )

    p_summary = sub.add_parser("summary", help="Print structured metrics summary")
    p_summary.add_argument("profile", help="Path to .sqlite profile")
    p_summary.add_argument("--json", action="store_true", help="Output raw JSON")
    p_summary.add_argument(
        "--verbose", action="store_true", help="Print phase detection debug output"
    )
    p_summary.add_argument(
        "--max-phases",
        type=int,
        default=10,
        help="Maximum number of execution phases to detect (default: 10; use 1 to disable)",
    )

    args = parser.parse_args()
    try:
        if args.command == "analyze":
            cmd_analyze(args)
        elif args.command == "compare":
            cmd_compare(args)
        elif args.command == "summary":
            cmd_summary(args)
        elif args.command == "evaluate":
            cmd_evaluate(args)
        else:
            parser.print_help()
            sys.exit(1)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
