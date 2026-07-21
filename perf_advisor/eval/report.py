"""Rich-table output and JSON serialization for evaluation results."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from perf_advisor.eval.scorer import (
    RunScore,
    baseline_detection_at_k,
    detection_at_k,
    detection_by_repeat,
    detection_stability,
    mean_reciprocal_rank,
    mrr_by_repeat,
    repeat_indices,
    scorable_runs,
)


def _mean_sd(values: list[float]) -> tuple[float, float | None]:
    """Mean and sample SD; SD is None with fewer than two observations."""
    if not values:
        return 0.0, None
    sd = statistics.stdev(values) if len(values) > 1 else None
    return statistics.mean(values), sd


def print_summary_table(results: list[RunScore], console: Console) -> None:
    """Print the aggregate results table and summary statistics."""
    t = Table(title="PerfAdvisor Evaluation Results", show_lines=False, show_header=True)
    t.add_column("Run", style="dim", no_wrap=True)
    t.add_column("Scenario")
    t.add_column("Detection", justify="center")
    t.add_column("Rank", style="dim", justify="center")
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

        if r.is_optimal_path:
            det_str = "[dim]—[/dim]"   # no bottleneck expected; excluded from accuracy
            rank_str = "[dim]opt[/dim]"
        else:
            det_str = "[green]✓[/green]" if r.bottleneck_detected else "[red]✗[/red]"
            if r.bottleneck_detected and r.matched_hypothesis_idx is not None:
                # 1-based for display; rank 1 means the advisor led with it.
                rank = r.matched_hypothesis_idx + 1
                rank_str = f"#{rank}" if rank == 1 else f"[yellow]#{rank}[/yellow]"
            else:
                rank_str = "—"

        valid = [s for s in r.suggestion_scores if s.score >= 0]
        if valid and r.coverage_pct is not None:
            raw = sum(s.score for s in valid)
            mx = 2 * len(valid)
            skipped = len(r.suggestion_scores) - len(valid)
            cov_str = f"{raw} / {mx}  ({r.coverage_pct:.0f}%)"
            if skipped:
                cov_str += f" [dim][{skipped} skipped][/dim]"
        elif r.suggestion_scores:
            # Every judge call failed — report "no data", never 0%.
            cov_str = f"[dim]n/a  [{len(r.suggestion_scores)} skipped][/dim]"
        else:
            cov_str = "[dim]—[/dim]"

        fp_color = "red" if r.false_positive_count > 0 else "green"
        rate = r.false_positive_rate
        # The rate, not the count, is what compares across models — a model emitting
        # 4 hypotheses and one emitting 10 are not on the same scale.
        denom = f"/{len(r.hypotheses)}" if r.hypotheses else ""
        pct = f" ({100 * rate:.0f}%)" if rate is not None else ""
        fp_str = f"[{fp_color}]{r.false_positive_count}{denom}[/{fp_color}]{pct}"
        if r.secondary_true_count:
            # Show what was excused, so the exclusion is auditable rather than
            # silently shrinking the count.
            fp_str += f" [dim](+{r.secondary_true_count} also-true)[/dim]"

        t.add_row(
            r.run_id,
            r.scenario,
            det_str,
            rank_str,
            cov_str,
            fp_str,
            f"{r.elapsed_s:.1f}",
        )

    console.print(t)

    # Aggregate stats (errors and optimal-path runs excluded from accuracy tallies)
    completed = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    if not completed:
        return

    scorable = scorable_runs(results)
    all_sugg = [s for r in scorable for s in r.suggestion_scores if s.score >= 0]
    total_fp = sum(r.false_positive_count for r in scorable)
    n_optimal = len(completed) - len(scorable)

    opt_note = f"  [dim](+{n_optimal} optimal-path run(s) excluded)[/dim]" if n_optimal else ""

    # Rank-aware detection. detection@1 is the headline: the advisor emits a
    # *ranked* list, so a correct diagnosis buried under four wrong ones is a
    # materially worse result than one the advisor led with.
    n_reps = len(repeat_indices(scorable))
    console.print()

    for k in (1, 3):
        note = opt_note if k == 1 else ""
        if n_reps > 1:
            # One observation per repetition. Repeats of the same scenario are
            # correlated and scenarios differ in difficulty, so pooling them into a
            # single proportion would understate the true spread.
            per_rep = detection_by_repeat(results, k)
            counts = [h for h, _ in per_rep]
            denom = per_rep[0][1] if per_rep else 0
            mean, sd = _mean_sd([float(c) for c in counts])
            pct_mean = 100 * mean / denom if denom else 0.0
            pct_sd = (100 * sd / denom) if (sd is not None and denom) else None
            sd_str = f" ± {pct_sd:.0f}" if pct_sd is not None else ""
            runs_str = ", ".join(str(c) for c in counts)
            console.print(
                f"  Detection@{k}:        [bold]{mean:.1f} / {denom}[/bold]"
                f"  ({pct_mean:.0f}%{sd_str}){note}"
                f"  [dim][{n_reps} repeats: {runs_str}][/dim]"
            )
        else:
            hits, denom = detection_at_k(results, k)
            pct = f"({100 * hits / denom:.0f}%)" if denom else "(n/a)"
            console.print(f"  Detection@{k}:        [bold]{hits} / {denom}[/bold]  {pct}{note}")

    if n_reps > 1:
        vals = mrr_by_repeat(results)
        mean, sd = _mean_sd(vals)
        sd_str = f" ± {sd:.2f}" if sd is not None else ""
        console.print(
            f"  Mean recip. rank:    [bold]{mean:.2f}{sd_str}[/bold]"
            f"  [dim](1.00 = always ranked first)[/dim]"
        )
    else:
        mrr = mean_reciprocal_rank(results)
        mrr_str = f"{mrr:.2f}" if mrr is not None else "n/a"
        console.print(
            f"  Mean recip. rank:    [bold]{mrr_str}[/bold]"
            f"  [dim](1.00 = always ranked first)[/dim]"
        )

    # The floor: what a plausible-sounding advisor that never opens a profile scores
    # on this same set of scenarios. If a real model is not clearly above it, the
    # numbers are not measuring analysis.
    seen_runs: set[str] = set()
    expected_labels = []
    for r in scorable:
        if r.run_id not in seen_runs:
            seen_runs.add(r.run_id)
            expected_labels.append(r.expected_bottleneck)
    if expected_labels:
        b_hits, b_total = baseline_detection_at_k(expected_labels, 1)
        console.print(
            f"  [dim]Profile-blind floor: {b_hits} / {b_total}"
            f"  ({100 * b_hits / b_total:.0f}%) — a constant advisor that never"
            f" reads a profile[/dim]"
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
        console.print("  Suggestion coverage: [dim]n/a  (nothing judged)[/dim]")
    total_secondary = sum(r.secondary_true_count for r in scorable)
    sec_note = (
        f"  [dim](+{total_secondary} known-true secondary observation(s) not counted)[/dim]"
        if total_secondary
        else ""
    )
    console.print(
        f"  False positives:     [bold]{total_fp}[/bold] total (scorable runs only){sec_note}"
    )
    tok_in = sum(r.input_tokens or 0 for r in completed)
    tok_out = sum(r.output_tokens or 0 for r in completed)
    tok_cache = sum(r.cache_read_tokens or 0 for r in completed)
    if tok_in or tok_out:
        console.print(
            f"  Tokens:              [bold]{tok_in:,}[/bold] in"
            f"  ([dim]{tok_cache:,} cache-read[/dim]),"
            f"  [bold]{tok_out:,}[/bold] out"
        )
    if errors:
        console.print(f"  [red]Errors:              {len(errors)} run(s) failed[/red]")

    if n_reps > 1:
        print_stability_table(results, console)


def print_stability_table(results: list[RunScore], console: Console) -> None:
    """Per-scenario detection counts across repetitions.

    The aggregate cannot distinguish "found it every time" from "found it
    sometimes" from "never found it" — three different capability claims that
    average to the same number. With repeats this is usually the most useful
    output of the whole run.
    """
    rows = detection_stability(results)
    if not rows:
        return
    t = Table(
        title="Per-scenario stability across repetitions",
        show_lines=False,
        show_header=True,
    )
    t.add_column("Run", style="dim", no_wrap=True)
    t.add_column("Scenario")
    t.add_column("Detected", justify="center")

    for run_id, scenario, hits, obs in rows:
        if obs == 0:
            continue
        frac = hits / obs
        if frac == 1.0:
            cell = f"[green]{hits}/{obs}[/green]"
        elif frac == 0.0:
            cell = f"[red]{hits}/{obs}[/red]"
        else:
            # Neither reliably found nor reliably missed — the case the aggregate
            # hides, and the one worth acting on.
            cell = f"[yellow]{hits}/{obs}  unstable[/yellow]"
        t.add_row(run_id, scenario, cell)

    console.print()
    console.print(t)


def print_run_details(r: RunScore, console: Console) -> None:
    """Print per-suggestion judge scores for a single run (--verbose mode)."""
    if r.error:
        console.print(f"\n[bold]{r.run_id}[/bold]  [red]ERROR: {r.error}[/red]")
        return

    if r.matched_hypothesis_idx is not None and r.hypotheses:
        detected_type = r.hypotheses[r.matched_hypothesis_idx].get("bottleneck_type", "—")
    elif r.hypotheses:
        detected_type = r.hypotheses[0].get("bottleneck_type", "—")
    else:
        detected_type = "none"
    det_mark = "[green]✓[/green]" if r.bottleneck_detected else "[red]✗[/red]"
    if r.bottleneck_detected and r.matched_hypothesis_idx is not None:
        rank_note = f"rank #{r.matched_hypothesis_idx + 1} of {len(r.hypotheses)}"
    else:
        rank_note = "—"
    console.print(
        f"\n[bold]{r.run_id}[/bold]  scenario={r.scenario}"
        f"  expected_bottleneck={r.expected_bottleneck}"
        f"  detected_bottleneck={detected_type} {det_mark}"
        f"  ({rank_note})"
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
