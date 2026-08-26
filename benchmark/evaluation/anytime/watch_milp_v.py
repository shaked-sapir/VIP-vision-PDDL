"""Read the PI-SAM MILP loop's live V stream while a run is still going.

Each ``pisam_milp_loop`` fold appends one JSON row per round to
``milp_loop_rounds.jsonl`` as that round ends, so convergence is visible without
waiting for the end-of-run ``milp_loop_rounds.json``.

    python -m benchmark.evaluation.anytime.watch_milp_v <results-or-fold-dir>
    python -m benchmark.evaluation.anytime.watch_milp_v <dir> --plot v.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

STREAM_NAME = "milp_loop_rounds.jsonl"


def read_stream(path: Path) -> List[dict]:
    """Rows from one stream. A torn final line is dropped: it is being written."""
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def find_streams(root: Path) -> Dict[str, List[dict]]:
    """Every stream under ``root``, keyed by ``<fold>/<arm>``."""
    out: Dict[str, List[dict]] = {}
    for path in sorted(root.rglob(STREAM_NAME)):
        rows = read_stream(path)
        if not rows:
            continue
        arm = path.parent.name
        fold = next(
            (p.name for p in path.parents if p.name.startswith("fold")), arm
        )
        out[f"{fold}/{arm}"] = rows
    return out


def summarise(label: str, rows: List[dict]) -> str:
    """One line: rounds so far, best V, when it was found, and the last V."""
    scored = [r for r in rows if r.get("v_raw") is not None]
    if not scored:
        return f"  {label}: {len(rows)} round(s), none scored yet"
    best = min(scored, key=lambda r: r["v_raw"])
    since_best = rows[-1]["round"] - best["round"]
    return (
        f"  {label}: {len(rows)} round(s) | best V={best['v_raw']:.4g} "
        f"@r{best['round']} | last V={scored[-1]['v_raw']:.4g} | "
        f"{since_best} round(s) since best | "
        f"{rows[-1]['elapsed_seconds']:.0f}s elapsed"
    )


def plot(streams: Dict[str, List[dict]], out_path: Path) -> Path:
    """V against round, one line per fold, with the running best dashed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, rows in sorted(streams.items()):
        scored = [r for r in rows if r.get("v_raw") is not None]
        if not scored:
            continue
        rounds = [r["round"] for r in scored]
        values = [r["v_raw"] for r in scored]
        line, = ax.plot(rounds, values, marker="o", markersize=2.5,
                        linewidth=0.9, alpha=0.55, label=label)
        running = []
        best = float("inf")
        for value in values:
            best = min(best, value)
            running.append(best)
        ax.plot(rounds, running, linestyle="--", linewidth=1.6,
                color=line.get_color())

    ax.set_xlabel("round")
    ax.set_ylabel("V (lower is better)")
    ax.set_title("PI-SAM+MILP loop: V per round (dashed = running best)")
    ax.grid(alpha=0.3)
    if len(streams) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help=f"Directory to scan for {STREAM_NAME}")
    parser.add_argument("--plot", type=Path, help="Also write a V-per-round PNG here")
    args = parser.parse_args()

    streams = find_streams(args.root)
    if not streams:
        raise SystemExit(f"No {STREAM_NAME} under {args.root}.")
    for label, rows in sorted(streams.items()):
        print(summarise(label, rows))
    if args.plot:
        print(f"\nWrote {plot(streams, args.plot)}")


if __name__ == "__main__":
    main()
