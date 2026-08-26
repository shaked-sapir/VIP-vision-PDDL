"""Plot ROSAME training-loss curves from ``snapshots.json`` index files.

Reads the indices a :class:`~benchmark.algorithm_adapters.anytime_snapshots.SnapshotWriter`
wrote during training and plots loss against epoch, one line per fold.

    python -m benchmark.evaluation.anytime.plot_loss_curves \
        benchmark/running_results/blocksworld/<experiment> -o loss.png

Records with no ``loss`` are skipped: the field is optional, and a caller that
tracks no loss writes ``None``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

INDEX_NAME = "snapshots.json"


def load_curve(index_path: Path) -> List[Tuple[int, float]]:
    """``(epoch, loss)`` pairs from one index, in file order, losses only."""
    payload = json.loads(index_path.read_text())
    records = payload if isinstance(payload, list) else payload.get(
        "snapshots", payload.get("records", [])
    )
    return [
        (r["epoch"], r["loss"]) for r in records if r.get("loss") is not None
    ]


def find_curves(root: Path) -> Dict[str, List[Tuple[int, float]]]:
    """Every snapshot index under ``root``, keyed by ``<fold>/<arm>``."""
    curves: Dict[str, List[Tuple[int, float]]] = {}
    for index_path in sorted(root.rglob(INDEX_NAME)):
        curve = load_curve(index_path)
        if not curve:
            continue
        arm = index_path.parent.name
        fold = next(
            (p.name for p in index_path.parents if p.name.startswith("fold")),
            index_path.parent.parent.name,
        )
        curves[f"{fold}/{arm}"] = curve
    return curves


def plot(curves: Dict[str, List[Tuple[int, float]]], out_path: Path) -> Path:
    """Render every curve onto one axis and write it to ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, curve in sorted(curves.items()):
        epochs = [e for e, _ in curve]
        losses = [v for _, v in curve]
        ax.plot(epochs, losses, marker="", linewidth=1.4, label=label)

    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("ROSAME training loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    if len(curves) <= 12:
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory to scan for snapshots.json")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("loss_curves.png"),
        help="Output PNG (default: loss_curves.png)",
    )
    args = parser.parse_args()

    curves = find_curves(args.root)
    if not curves:
        raise SystemExit(
            f"No {INDEX_NAME} with loss values under {args.root}. "
            "Was the run configured with snapshot_interval?"
        )
    for label, curve in sorted(curves.items()):
        first, last = curve[0][1], curve[-1][1]
        print(f"  {label}: {len(curve)} points, loss {first:.4f} -> {last:.4f}")
    print(f"\nWrote {plot(curves, args.output)}")


if __name__ == "__main__":
    main()
