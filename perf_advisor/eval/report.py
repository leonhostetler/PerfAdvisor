"""Rich-table output and JSON serialization for evaluation results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from perf_advisor.eval.scorer import RunScore


def print_summary_table(results: list[RunScore], console: Console) -> None:
    """Print the aggregate results table and summary statistics."""
    t = Table(title="PerfAdvisor Evaluation Results", show_lines=False, show_header=True)
    t.add_column("Run", style="dim", no_wrap=True)
    t.add_column("Scenario")
    t.add_column("Detection", justify="center")
    t.add_column("Match", style="dim", justify="center")
    t.add_column("Suggestion Coverage", justify="right")
    t.add_column("False Pos", justify="right")
    t.add_column("Time (s)", justify="right", style="dim")

    for r in results:
        if r.error:
            t.add_row(
                r.run_id,
                r.scenario,
                "[red]ERROR[/red]",
                "—",
                "—",
                "—",
                f"{r.elapsed_s:.1f}",
            )
            continue

        det_str = "[green]✓[/green]" if r.bottleneck_detected else "[red]✗[/red]"
        match_str = r.match_type or "—"

        if r.suggestion_scores:
            valid = [s for s in r.suggestion_scores if s.score >= 0]
            raw = sum(s.score for s in valid)
            mx = 2 * len(valid)
            skipped = len(r.suggestion_scores) - len(valid)
            cov_str = f"{raw} / {mx}  ({r.coverage_pct:.0f}%)"
            if skipped:
                cov_str += f" [dim][{skipped} skipped][/dim]"
        else:
            cov_str = "[dim]—[/dim]"

        fp_color = "red" if r.false_positive_count > 0 else "green"
        fp_str = f"[{fp_color}]{r.false_positive_count}[/{fp_color}]"

        t.add_row(
            r.run_id,
            r.scenario,
            det_str,
            match_str,
            cov_str,
            fp_str,
            f"{r.elapsed_s:.1f}",
        )

    console.print(t)

    # Aggregate stats (errors excluded)
    completed = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    if not completed:
        return

    n_detected = sum(1 for r in completed if r.bottleneck_detected)
    all_sugg = [s for r in completed for s in r.suggestion_scores if s.score >= 0]
    total_fp = sum(r.false_positive_count for r in completed)

    console.print(
        f"\n  Detection accuracy:  "
        f"[bold]{n_detected} / {len(completed)}[/bold]"
        f"  ({100 * n_detected / len(completed):.0f}%)"
    )
    if all_sugg:
        raw_total = sum(s.score for s in all_sugg)
        mx_total = 2 * len(all_sugg)
        console.print(
            f"  Suggestion coverage: "
            f"[bold]{raw_total} / {mx_total}[/bold]"
            f"  ({100 * raw_total / mx_total:.0f}%)"
        )
    else:
        console.print("  Suggestion coverage: [dim]—  (judge skipped)[/dim]")
    console.print(f"  False positives:     [bold]{total_fp}[/bold] total")
    if errors:
        console.print(f"  [red]Errors:              {len(errors)} run(s) failed[/red]")


def print_run_details(r: RunScore, console: Console) -> None:
    """Print per-suggestion judge scores for a single run (--verbose mode)."""
    if r.error:
        console.print(f"\n[bold]{r.run_id}[/bold]  [red]ERROR: {r.error}[/red]")
        return

    det = "[green]✓[/green]" if r.bottleneck_detected else "[red]✗[/red]"
    console.print(
        f"\n[bold]{r.run_id}[/bold]  scenario={r.scenario}"
        f"  expected={r.expected_bottleneck}  detection={det}"
        f"  ({r.match_type or '—'})"
    )

    if not r.suggestion_scores:
        console.print("  [dim]No suggestion scores (judge skipped)[/dim]")
        return

    score_label = {2: "full", 1: "partial", 0: "absent", -1: "skipped"}
    score_color = {2: "green", 1: "yellow", 0: "red", -1: "dim"}

    for ss in r.suggestion_scores:
        color = score_color.get(ss.score, "white")
        label = score_label.get(ss.score, "?")
        action_preview = ss.action[:72] + ("…" if len(ss.action) > 72 else "")
        console.print(f"  [{color}]{ss.score} ({label})[/{color}]  {action_preview}")
        if ss.explanation:
            console.print(f"    [dim]{ss.explanation}[/dim]")


def save_results(
    results: list[RunScore],
    output_path: Path,
    metadata: dict[str, Any],
) -> None:
    """Write evaluation results to a JSON file."""
    data = {
        "metadata": metadata,
        "runs": [r.to_dict() for r in results],
    }
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_results(path: Path) -> tuple[dict[str, Any], list[RunScore]]:
    """Load a previously-saved evaluation results JSON.

    Returns (metadata, list[RunScore]).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    runs = [RunScore.from_dict(r) for r in data.get("runs", [])]
    return metadata, runs
