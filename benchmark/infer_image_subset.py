"""Classify a chosen subset of an existing image pool, resumably.

The pool is ``<data-dir>/training/image_trajectories/``: one folder per problem
holding ``state_*.png``, the problem ``.pddl`` and its GT ``_trajectory.json``.
The inferred ``.trajectory`` and ``.masking_info`` land in the same folder, and
a folder that already holds its ``.masking_info`` is skipped, so a second call
continues the first.

The subset is one of::

    # the problems the image run's cells will train on: the first N of every
    # fold's training order, as benchmark/experiment_running_helpers/run_fold.py
    # draws them
    python -m benchmark.infer_image_subset --domain blocksworld \\
        --data-dir benchmark/data/blocksworld/large_corpus_L500_n2500 \\
        --num-trajectories 100 --n-folds 5 --cv-scheme kfold

    # an explicit list, one problem folder name per line
    python -m benchmark.infer_image_subset --domain hanoi --data-dir <corpus> \\
        --problems-file picks.txt

    # every problem of the pool
    python -m benchmark.infer_image_subset --domain depot --data-dir <corpus> --all

Every classified problem appends one line to ``<data-dir>/inference_usage.jsonl``
(frames, wall seconds, calls, prompt / cached / completion tokens, summed
request seconds, realised mask and flip rates against ``gt_trajectories/``).
``--summary`` prints the totals of that file and, given prices, the cost.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from benchmark.algorithm_adapters.lamanna.realised_rates import (
    gt_trajectory_lookup,
    realised_rates,
)
from benchmark.data_generator import _DOMAIN_REGISTRY, _build_handler
from benchmark.experiment_running_helpers.run_fold import cv_split, select_training_pool
from src.utils.config import load_config
from src.utils.corpus_layout import FRAME_GLOB, IMAGE_POOL_SUBPATH, pool_problem_dirs

USAGE_LOG_NAME = "inference_usage.jsonl"
_TOKEN_FIELDS = ("calls", "prompt_tokens", "cached_tokens", "completion_tokens", "request_seconds")


def fold_training_subset(
    problem_dirs: Sequence[Path], n_folds: int, cv_scheme: str, num_trajectories: int
) -> List[Path]:
    """The union of every fold's first ``num_trajectories`` training problems.

    Ordered by fold, then by the fold's training order, without repeats.

    Raises:
        ValueError: if a fold's training pool is smaller than ``num_trajectories``.
    """
    subset: Dict[Path, None] = {}
    for fold in range(n_folds):
        train_idx, _ = cv_split(len(problem_dirs), fold, n_folds=n_folds, scheme=cv_scheme)
        train_dirs = [problem_dirs[i] for i in train_idx]
        if num_trajectories > len(train_dirs):
            raise ValueError(
                f"num_trajectories={num_trajectories} exceeds fold {fold}'s "
                f"training pool of {len(train_dirs)} problems"
            )
        for problem_dir in select_training_pool(train_dirs, fold, num_trajectories)[:num_trajectories]:
            subset.setdefault(problem_dir, None)
    return list(subset)


def listed_problems(names: Iterable[str], pool: Path) -> List[Path]:
    """The pool folders named in ``names``, in the given order.

    Raises:
        FileNotFoundError: if a named folder is not in the pool.
    """
    dirs = [pool / name.strip() for name in names if name.strip()]
    missing = [d.name for d in dirs if not d.is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} listed problems are not in {pool}: {missing[:5]}")
    return dirs


def is_inferred(problem_dir: Path) -> bool:
    """Whether the folder already holds an inferred trajectory and its masking."""
    name = problem_dir.name
    return (problem_dir / f"{name}.masking_info").exists() and (problem_dir / f"{name}.trajectory").exists()


def backend_usage(handler) -> Dict[str, float]:
    """Token, call and request-time totals over the handler's LLM backends."""
    totals: Dict[str, float] = {field: 0 for field in _TOKEN_FIELDS}
    for component in (handler.object_detector, handler.fluent_classifier):
        usage = getattr(getattr(component, "backend", None), "usage", None)
        if usage is None:
            continue
        for field, value in usage().items():
            totals[field] = totals.get(field, 0) + value
    return totals


def measured_rates(problem_dir: Path, domain_file: Path, data_dir: Path) -> Dict[str, float]:
    """Realised mask and flip rates of one inferred problem against its GT."""
    name = problem_dir.name
    prepared = [(problem_dir / f"{name}.trajectory", problem_dir / f"{name}.masking_info",
                 problem_dir / f"{name}.pddl")]
    rates = realised_rates(prepared, domain_file, gt_trajectory_lookup(data_dir))
    return {"mask_rate": round(rates.mask_rate, 4), "flip_rate": round(rates.noise_rate, 4)}


def infer_problem(
    domain: str, vendor: str, inference_workers: int, problem_dir: Path,
    domain_file: Path, data_dir: Path,
) -> Dict[str, object]:
    """Classify one problem folder in place; return its usage-log row."""
    row: Dict[str, object] = {
        "problem": problem_dir.name,
        "frames": len(list(problem_dir.glob(FRAME_GLOB))),
        "inference_workers": inference_workers,
    }
    started = time.perf_counter()
    try:
        handler = _build_handler(domain, vendor, inference_workers)
        handler.run_pipeline(problem_name=problem_dir.name, images_path=problem_dir)
        row.update(backend_usage(handler))
        row.update(measured_rates(problem_dir, domain_file, data_dir))
    except Exception as error:  # noqa: BLE001 - recorded in the row, the run goes on
        row["error"] = f"{type(error).__name__}: {error}"
    row["seconds"] = round(time.perf_counter() - started, 2)
    return row


def append_usage(log_path: Path, row: Dict[str, object]) -> None:
    """Append one JSON line to the usage log."""
    with log_path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def read_usage(log_path: Path) -> List[Dict[str, object]]:
    """The usage log's rows, oldest first; empty when the log does not exist."""
    if not log_path.exists():
        return []
    return [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]


def summarise(
    rows: Sequence[Dict[str, object]],
    price_input: Optional[float] = None,
    price_cached: Optional[float] = None,
    price_output: Optional[float] = None,
) -> Dict[str, object]:
    """Totals of the successful rows, with the cost when all three prices are given.

    Prices are dollars per million tokens: uncached input, cached input, output.
    """
    ok = [row for row in rows if "error" not in row]
    total = {field: sum(row.get(field, 0) for row in ok) for field in _TOKEN_FIELDS}
    frames = sum(row["frames"] for row in ok)
    summary: Dict[str, object] = {
        "problems": len(ok),
        "failed_rows": len(rows) - len(ok),
        "frames": frames,
        "wall_seconds": round(sum(row["seconds"] for row in ok), 1),
        **total,
        "request_seconds_per_call": round(total["request_seconds"] / total["calls"], 2) if total["calls"] else None,
        "cache_hit_ratio": round(total["cached_tokens"] / total["prompt_tokens"], 3) if total["prompt_tokens"] else None,
    }
    if None not in (price_input, price_cached, price_output):
        uncached = total["prompt_tokens"] - total["cached_tokens"]
        cost = (uncached * price_input + total["cached_tokens"] * price_cached
                + total["completion_tokens"] * price_output) / 1e6
        summary["cost_usd"] = round(cost, 2)
        summary["cost_usd_per_frame"] = round(cost / frames, 5) if frames else None
    return summary


def run(
    domain: str, data_dir: Path, problems: Sequence[Path], vendor: str,
    inference_workers: int, problem_workers: int,
) -> List[Dict[str, object]]:
    """Classify every not-yet-inferred problem of ``problems``; return the new rows."""
    registry = _DOMAIN_REGISTRY[domain]
    domain_file = Path(load_config()["domains"][registry["config_key"]]["domain_file"])
    log_path = data_dir / USAGE_LOG_NAME
    todo = [p for p in problems if not is_inferred(p)]
    print(f"[{domain}] {len(problems)} selected, {len(problems) - len(todo)} already inferred, "
          f"{len(todo)} to classify", flush=True)

    rows: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, problem_workers)) as pool:
        futures = [
            pool.submit(infer_problem, domain, vendor, inference_workers, p, domain_file, data_dir)
            for p in todo
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            append_usage(log_path, row)
            rows.append(row)
            print(f"  [{done}/{len(todo)}] {json.dumps(row)}", flush=True)
    return rows


def _select(args: argparse.Namespace, pool: Path) -> List[Path]:
    """The problem folders the arguments ask for."""
    if args.all:
        selected = pool_problem_dirs(pool)
    elif args.problems_file:
        selected = listed_problems(args.problems_file.read_text().splitlines(), pool)
    else:
        selected = fold_training_subset(
            pool_problem_dirs(pool), args.n_folds, args.cv_scheme, args.num_trajectories)
    return selected[:args.limit] if args.limit else selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--domain", required=True, choices=sorted(_DOMAIN_REGISTRY))
    parser.add_argument("--data-dir", type=Path, required=True, help="the corpus root")
    parser.add_argument("--vendor", default="openai", help="a vendor block of config.yaml")
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--all", action="store_true", help="every problem of the pool")
    choice.add_argument("--problems-file", type=Path, help="one problem folder name per line")
    parser.add_argument("--num-trajectories", type=int, default=100)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--cv-scheme", default="kfold", choices=["kfold", "montecarlo"])
    parser.add_argument("--limit", type=int, default=None, help="first N of the selection only")
    parser.add_argument("--inference-workers", type=int, default=12,
                        help="frames of one problem classified together")
    parser.add_argument("--problem-workers", type=int, default=1,
                        help="problems classified together")
    parser.add_argument("--list-only", action="store_true",
                        help="print the selection and its frame count, classify nothing")
    parser.add_argument("--summary", action="store_true",
                        help="print the usage log's totals, classify nothing")
    parser.add_argument("--price-input", type=float, help="$ per 1M uncached input tokens")
    parser.add_argument("--price-cached", type=float, help="$ per 1M cached input tokens")
    parser.add_argument("--price-output", type=float, help="$ per 1M output tokens")
    args = parser.parse_args()

    pool = args.data_dir / IMAGE_POOL_SUBPATH
    prices = (args.price_input, args.price_cached, args.price_output)
    if args.summary:
        print(json.dumps(summarise(read_usage(args.data_dir / USAGE_LOG_NAME), *prices), indent=2))
        return 0

    selected = _select(args, pool)
    if args.list_only:
        frames = sum(len(list(p.glob(FRAME_GLOB))) for p in selected)
        pending = sum(1 for p in selected if not is_inferred(p))
        print("\n".join(p.name for p in selected))
        print(f"# {len(selected)} problems, {frames} frames, {pending} not yet inferred", file=sys.stderr)
        return 0

    rows = run(args.domain, args.data_dir, selected, args.vendor,
               args.inference_workers, args.problem_workers)
    print(json.dumps(summarise(rows, *prices), indent=2))
    remaining = [p.name for p in selected if not is_inferred(p)]
    if remaining:
        print(f"{len(remaining)} problems still not inferred: {remaining[:10]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
