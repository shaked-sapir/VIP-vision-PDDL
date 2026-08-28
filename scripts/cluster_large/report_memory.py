"""Attribute a cell's sampled memory to each L, per fold and accumulated.

The sbatch writes logs/mem-*.tsv (cgroup total + per-worker RSS, every 20s) and
the run log marks each L with a "NUMBER OF TRAJECTORIES = <L>" banner. Joining
them on elapsed time is what turns a flat sample stream into a per-L answer.

    python scripts/cluster_large/report_memory.py logs/
    python scripts/cluster_large/report_memory.py logs/ --csv memory_by_L.csv

Two views, because they answer different questions:
  * cgroup peak  -- what the OOM killer acts on, i.e. what --mem must cover.
  * max worker   -- whether ONE fold is ballooning while its siblings stay flat,
                    which is how the uncapped rosame_milp_* arm was found.
"""

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

L_BANNER = re.compile(r"NUMBER OF TRAJECTORIES = (\d+)")
CELL = re.compile(r"task=\d+: domain=(\S+) mask=(\S+) noise=(\S+)")


def read_samples(path: Path) -> List[dict]:
    """Rows of the memory TSV, newest schema only."""
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def l_boundaries(run_log: Path) -> List[Tuple[int, float]]:
    """``(L, mtime-relative second)`` for each L banner, in order.

    The run log has no timestamps, so the banners are located by line number and
    converted to a fraction of the run -- good enough to bucket 20s samples, and
    exact whenever a fold_result lands between two banners.
    """
    if not run_log.exists():
        return []
    lines = run_log.read_text(errors="replace").splitlines()
    marks = [(int(m.group(1)), i) for i, line in enumerate(lines)
             if (m := L_BANNER.search(line))]
    if not marks:
        return []
    total = max(len(lines), 1)
    return [(L, idx / total) for L, idx in marks]


def summarise(samples: List[dict]) -> Dict[str, int]:
    """Peak and mean of the cgroup total, and the largest single worker."""
    if not samples:
        return {}
    peak = max(int(s["cgroup_peak_mb"]) for s in samples)
    mean = sum(int(s["cgroup_current_mb"]) for s in samples) // len(samples)
    worst_worker = 0
    for s in samples:
        rss = [int(v) for v in (s.get("worker_rss_mb_csv") or "").split(",") if v]
        worst_worker = max(worst_worker, max(rss, default=0))
    return {
        "samples": len(samples),
        "cgroup_peak_mb": peak,
        "cgroup_mean_mb": mean,
        "max_single_worker_mb": worst_worker,
    }


def bucket_by_l(samples: List[dict], marks: List[Tuple[int, float]]) -> Dict[int, dict]:
    """Samples split at the L banners, so each L gets its own peak."""
    if not marks or not samples:
        return {}
    span = max(float(s["elapsed_s"]) for s in samples) or 1.0
    out: Dict[int, List[dict]] = {}
    for sample in samples:
        frac = float(sample["elapsed_s"]) / span
        current = marks[0][0]
        for L, at in marks:
            if frac >= at:
                current = L
        out.setdefault(current, []).append(sample)
    return {L: summarise(rows) for L, rows in sorted(out.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs_dir", type=Path)
    parser.add_argument("--csv", type=Path, help="Also write the per-L table here")
    args = parser.parse_args()

    rows: List[dict] = []
    for mem_log in sorted(args.logs_dir.glob("mem-*.tsv")):
        samples = read_samples(mem_log)
        if not samples:
            continue
        suffix = mem_log.name.replace("mem-", "").replace(".tsv", "")
        run_log = next(args.logs_dir.glob(f"{suffix.split('-')[0]}*{suffix.split('_')[-1]}.out"), None)
        cell = "?"
        marks: List[Tuple[int, float]] = []
        if run_log and run_log.exists():
            text = run_log.read_text(errors="replace")
            if (m := CELL.search(text)):
                cell = f"{m.group(1)} mask={m.group(2)} noise={m.group(3)}"
            marks = l_boundaries(run_log)

        overall = summarise(samples)
        print(f"\n{mem_log.name}  [{cell}]")
        print(f"  whole cell: peak={overall['cgroup_peak_mb']}MB "
              f"mean={overall['cgroup_mean_mb']}MB "
              f"largest_single_worker={overall['max_single_worker_mb']}MB "
              f"({overall['samples']} samples)")
        per_l = bucket_by_l(samples, marks)
        if per_l:
            print(f"  {'L':>6} {'peak MB':>9} {'mean MB':>9} {'max worker MB':>14}")
            for L, stats in per_l.items():
                print(f"  {L:>6} {stats['cgroup_peak_mb']:>9} "
                      f"{stats['cgroup_mean_mb']:>9} {stats['max_single_worker_mb']:>14}")
                rows.append({"log": mem_log.name, "cell": cell, "L": L, **stats})
        else:
            print("  (no L banners found in the run log; whole-cell figures only)")
            rows.append({"log": mem_log.name, "cell": cell, "L": "all", **overall})

    if args.csv and rows:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
