"""Discover benchmark runs and load ground-truth metadata.

Walks profiles_dir/{1gpu,4gpu,8gpu}/ looking for run_NN.json files.
For each, resolves the corresponding SQLite profile(s) and looks up the
scenario's evaluation rubric in ground_truth_meta.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_RUN_JSON_RE = re.compile(r"^(run_\d+)\.json$")
_RANK_SQLITE_RE = re.compile(r"^run_\d+\.(\d+)\.sqlite$")


def _rank_from_path(p: Path) -> int:
    m = re.search(r"\.(\d+)\.sqlite$", p.name)
    return int(m.group(1)) if m else 0


@dataclass
class RunConfig:
    run_id: str
    sqlite_paths: list[Path]  # sorted by rank (rank 0 first); length 1 for single-GPU runs
    gt_runtime: dict  # parsed from run_NN.json (scenario, expected_bottleneck, params)
    gt_meta: dict | None  # entry from ground_truth_meta.json; None if scenario unknown
    subdir: str  # "1gpu" | "4gpu" | "8gpu"

    @property
    def scenario(self) -> str:
        return self.gt_runtime.get("scenario", "unknown")

    @property
    def expected_bottleneck(self) -> str:
        return self.gt_runtime.get("expected_bottleneck", "unknown")

    @property
    def is_multi_rank(self) -> bool:
        return len(self.sqlite_paths) > 1


def load_ground_truth_meta(bench_dir: Path) -> dict:
    """Load ground_truth_meta.json from bench_dir."""
    meta_path = bench_dir / "ground_truth_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"ground_truth_meta.json not found in {bench_dir}.\n"
            f"Pass --ground-truth pointing to the bench/ directory."
        )
    with meta_path.open(encoding="utf-8") as f:
        return json.load(f)


def discover_runs(profiles_dir: Path, bench_dir: Path) -> list[RunConfig]:
    """Walk profiles_dir/{1gpu,4gpu,8gpu}/ and return one RunConfig per found run.

    Runs whose ground-truth JSON is missing, or whose SQLite file(s) are absent,
    are silently skipped (the caller logs a warning if desired).
    """
    meta = load_ground_truth_meta(bench_dir)
    runs: list[RunConfig] = []

    for subdir_name in ("1gpu", "4gpu", "8gpu"):
        subdir_path = profiles_dir / subdir_name
        if not subdir_path.is_dir():
            continue

        for json_path in sorted(subdir_path.glob("*.json")):
            m = _RUN_JSON_RE.match(json_path.name)
            if not m:
                continue
            run_id = m.group(1)

            try:
                with json_path.open(encoding="utf-8") as f:
                    gt_runtime = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue

            scenario = gt_runtime.get("scenario", "")
            gt_meta = meta.get(scenario)  # None if scenario not in meta

            # Resolve SQLite file(s).
            # Multi-rank: run_NN.0.sqlite, run_NN.1.sqlite, …
            # Single-GPU: run_NN.sqlite
            multi_rank = sorted(
                subdir_path.glob(f"{run_id}.[0-9]*.sqlite"),
                key=_rank_from_path,
            )
            single = subdir_path / f"{run_id}.sqlite"

            if multi_rank:
                sqlite_paths = multi_rank
            elif single.exists():
                sqlite_paths = [single]
            else:
                continue  # no SQLite found — skip

            runs.append(
                RunConfig(
                    run_id=run_id,
                    sqlite_paths=sqlite_paths,
                    gt_runtime=gt_runtime,
                    gt_meta=gt_meta,
                    subdir=subdir_name,
                )
            )

    return runs
