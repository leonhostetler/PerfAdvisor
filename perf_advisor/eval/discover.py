"""Discover benchmark runs and load ground-truth metadata.

Walks profiles_dir/{1gpu,4gpu,8gpu}/ looking for test_NN.json files.
For each, resolves the corresponding SQLite profile(s) and looks up the
scenario's evaluation rubric in ground_truth_meta.json.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_RUN_JSON_RE = re.compile(r"^(test_\d+)\.json$")
_RANK_SQLITE_RE = re.compile(r"^test_\d+\.(\d+)\.sqlite$")


def _rank_from_path(p: Path) -> int:
    """Rank index from a flat profile name (``test_05.3.sqlite`` / ``.3.db`` -> 3)."""
    m = re.search(r"\.(\d+)\.(?:sqlite|db)$", p.name)
    return int(m.group(1)) if m else 0


def _rocpd_rank(p: Path) -> tuple[int, str]:
    """Rank index from a rocpd filename (``rank_rocpd-3.db`` -> 3).

    Falls back to a large sentinel plus the name so unparseable files sort last
    deterministically rather than silently claiming rank 0.
    """
    m = re.search(r"(\d+)\.db$", p.name)
    return (int(m.group(1)), p.name) if m else (1 << 30, p.name)


@dataclass
class RunConfig:
    run_id: str
    sqlite_paths: list[Path]  # sorted by rank (rank 0 first); length 1 for single-GPU runs
    gt_runtime: dict  # parsed from test_NN.json (scenario, expected_bottleneck, params)
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

            # Resolve profile file(s). Both benches now use the same flat layout —
            # the HIP submit scripts flatten rocprof-sys's per-run subdirectory into
            # it — so only the extension differs:
            #   1. flat multi-rank: test_NN.0.sqlite (CUDA) / test_NN.0.db (HIP)
            #   2. flat single-GPU: test_NN.sqlite    (CUDA) / test_NN.db    (HIP)
            #   3. legacy rocprof-sys per-run subdirectory, kept so captures taken
            #      before the layouts were unified still load:
            #      profiles/{Ngpu}/test_NN/{prefix}rocpd-{N}.db
            multi_rank = sorted(
                [
                    *subdir_path.glob(f"{run_id}.[0-9]*.sqlite"),
                    *subdir_path.glob(f"{run_id}.[0-9]*.db"),
                ],
                key=_rank_from_path,
            )
            single_candidates = [
                subdir_path / f"{run_id}.sqlite",
                subdir_path / f"{run_id}.db",
            ]
            run_subdir = subdir_path / run_id

            if multi_rank:
                sqlite_paths = multi_rank
            elif any(c.exists() for c in single_candidates):
                sqlite_paths = [next(c for c in single_candidates if c.exists())]
            elif run_subdir.is_dir():
                # rocprof-sys output: rank_rocpd-<N>.db inside the per-run directory.
                # Sort by the trailing rank index, not lexically — rank order is NOT
                # cosmetic: sqlite_paths[0] is passed to the cross-rank analysis as the
                # reference rank. Lexical order would put rocpd-10 before rocpd-2 once a
                # run exceeds 9 ranks (the current suite tops out at 8, so this is latent).
                rocpd_files = sorted(run_subdir.glob("*.db"), key=_rocpd_rank)
                if not rocpd_files:
                    continue
                sqlite_paths = rocpd_files
            else:
                continue  # no profiles found — skip

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
