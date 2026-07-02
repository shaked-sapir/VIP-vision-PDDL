"""Compare benchmark .trajectory files against GT and report per-state fluent confusion."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser

from src.utils.pddl_state import state_positive_set


@dataclass
class StateConfusion:
    state_index: int
    tp: int
    fp: int
    fn: int
    tn: int
    false_positives: Tuple[str, ...]
    false_negatives: Tuple[str, ...]

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 1.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 1.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else 1.0


@dataclass
class TrajectoryReport:
    problem: str
    n_states_gt: int
    n_states_benchmark: int
    state_confusions: List[StateConfusion]
    length_mismatch: bool

    @property
    def aggregate(self) -> StateConfusion:
        tp = sum(s.tp for s in self.state_confusions)
        fp = sum(s.fp for s in self.state_confusions)
        fn = sum(s.fn for s in self.state_confusions)
        tn = sum(s.tn for s in self.state_confusions)
        fps = tuple(sorted({f for s in self.state_confusions for f in s.false_positives}))
        fns = tuple(sorted({f for s in self.state_confusions for f in s.false_negatives}))
        return StateConfusion(-1, tp, fp, fn, tn, fps, fns)


def _load_positive_states(trajectory_path: Path, domain_path: Path) -> List[Set[str]]:
    domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
    obs = TrajectoryParser(partial_domain=domain).parse_trajectory(trajectory_path)
    states = [state_positive_set(comp.previous_state) for comp in obs.components]
    states.append(state_positive_set(obs.components[-1].next_state))
    return states


def _confusion_for_state(gt: Set[str], pred: Set[str], universe: Set[str]) -> StateConfusion:
    tp = len(gt & pred)
    fp = len(pred - gt)
    fn = len(gt - pred)
    tn = len(universe - gt - pred)
    return StateConfusion(
        state_index=-1,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        false_positives=tuple(sorted(pred - gt)),
        false_negatives=tuple(sorted(gt - pred)),
    )


def compare_trajectory(
    problem: str,
    benchmark_path: Path,
    gt_path: Path,
    domain_path: Path,
) -> TrajectoryReport:
    gt_states = _load_positive_states(gt_path, domain_path)
    pred_states = _load_positive_states(benchmark_path, domain_path)
    universe = set().union(*gt_states, *pred_states)

    n = min(len(gt_states), len(pred_states))
    confusions: List[StateConfusion] = []
    for i in range(n):
        c = _confusion_for_state(gt_states[i], pred_states[i], universe)
        confusions.append(StateConfusion(i, c.tp, c.fp, c.fn, c.tn, c.false_positives, c.false_negatives))

    return TrajectoryReport(
        problem=problem,
        n_states_gt=len(gt_states),
        n_states_benchmark=len(pred_states),
        state_confusions=confusions,
        length_mismatch=len(gt_states) != len(pred_states),
    )


def _plot_confusion_matrix(report: TrajectoryReport, output_path: Path) -> None:
    n = len(report.state_confusions)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(max(3 * n, 6), 3.5), squeeze=False)
    for i, sc in enumerate(report.state_confusions):
        ax = axes[0, i]
        mat = np.array([[sc.tn, sc.fp], [sc.fn, sc.tp]])
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred −", "Pred +"])
        ax.set_yticklabels(["GT −", "GT +"])
        ax.set_title(f"S{i}")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(mat[r, c]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{report.problem} — fluent confusion per state (CWA)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _plot_aggregate_summary(reports: List[TrajectoryReport], output_path: Path) -> None:
    problems = [r.problem for r in reports]
    tp = [r.aggregate.tp for r in reports]
    fp = [r.aggregate.fp for r in reports]
    fn = [r.aggregate.fn for r in reports]

    x = np.arange(len(problems))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(8, len(problems) * 0.8), 4))
    ax.bar(x - width, tp, width, label="TP")
    ax.bar(x, fp, width, label="FP")
    ax.bar(x + width, fn, width, label="FN")
    ax.set_xticks(x)
    ax.set_xticklabels(problems, rotation=45, ha="right")
    ax.set_ylabel("Fluent count")
    ax.set_title("Depot benchmark vs GT — aggregate fluent errors per trajectory")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark vs GT trajectory fluent confusion")
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("gt_root", type=Path, help="e.g. src/domains/depot/problems")
    parser.add_argument("--domain", type=Path, default=Path("src/domains/depot/depot.pddl"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    benchmark_trajs = sorted(args.benchmark_root.rglob("*.trajectory"))
    if not benchmark_trajs:
        raise FileNotFoundError(f"No .trajectory files under {args.benchmark_root}")

    output_dir = args.output or (args.benchmark_root / "evaluation_results" / "gt_confusion")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: List[TrajectoryReport] = []
    for bench_path in benchmark_trajs:
        problem = bench_path.stem
        gt_path = args.gt_root / problem / f"{problem}.trajectory"
        if not gt_path.exists():
            print(f"SKIP {problem}: GT not found at {gt_path}")
            continue

        report = compare_trajectory(problem, bench_path, gt_path, args.domain)
        reports.append(report)
        _plot_confusion_matrix(report, output_dir / f"{problem}_state_confusion.png")

        print(f"\n=== {problem} ===")
        if report.length_mismatch:
            print(f"  LENGTH MISMATCH: GT={report.n_states_gt} states, benchmark={report.n_states_benchmark}")
        agg = report.aggregate
        print(f"  Aggregate: TP={agg.tp} FP={agg.fp} FN={agg.fn} TN={agg.tn}  "
              f"precision={agg.precision:.3f} recall={agg.recall:.3f} accuracy={agg.accuracy:.3f}")
        for sc in report.state_confusions:
            if sc.fp or sc.fn:
                print(f"  state {sc.state_index}: FP={sc.fp} FN={sc.fn}  "
                      f"extra={list(sc.false_positives)[:5]}{'...' if len(sc.false_positives)>5 else ''}  "
                      f"missing={list(sc.false_negatives)[:5]}{'...' if len(sc.false_negatives)>5 else ''}")

    _plot_aggregate_summary(reports, output_dir / "aggregate_summary.png")

    serializable = []
    for r in reports:
        d = asdict(r)
        d["aggregate"] = asdict(r.aggregate)
        serializable.append(d)
    (output_dir / "confusion_report.json").write_text(json.dumps(serializable, indent=2))

    total = TrajectoryReport(
        "ALL",
        sum(r.n_states_gt for r in reports),
        sum(r.n_states_benchmark for r in reports),
        [sc for r in reports for sc in r.state_confusions],
        any(r.length_mismatch for r in reports),
    )
    agg = total.aggregate
    print(f"\n=== OVERALL ({len(reports)} trajectories, {len(total.state_confusions)} states) ===")
    print(f"TP={agg.tp} FP={agg.fp} FN={agg.fn} TN={agg.tn}")
    print(f"precision={agg.precision:.3f} recall={agg.recall:.3f} accuracy={agg.accuracy:.3f}")
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
