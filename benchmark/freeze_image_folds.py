"""Setup pass of an image run config: prepare each pool once, freeze every cell's inputs.

For each domain of a ``source: image`` run config this runs, in one process,
the steps on the shared pool that every fold job of that domain repeats
(normalisation when the config asks for it, then the pool's
``*_gtrate<R>_frame_axioms`` files), and writes each cell's inputs to a source
tree the pixel arms read (``benchmark/backfill_baseline.py --experiment-dir``)::

    <out-root>/<domain>/<run_name>/
        evaluation_results/run_params.json
        testing/fold<F>_numtrajs<N>_gtrate<R>/
            fold_info.json  domain_reference.pddl  original_observations/
        testing/fold<F>_gtrate<R>_shared/predictive_power_test_states/   (--test-states)

The split, the training order and the file contents are the ones
``benchmark/experiment_runner.py`` produces for the same config.

    python -m benchmark.freeze_image_folds --config benchmark/run_config_large_image.yaml
    python -m benchmark.freeze_image_folds --config benchmark/run_config_small_image.yaml \\
        --domains hanoi --test-states
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml
from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark import experiment_runner
from benchmark.evaluation.test_states_generator import generate_predictive_power_test_states
from benchmark.experiment_running_helpers.cleaned_trajectories import save_fold_observations
from benchmark.experiment_running_helpers.data_source import ImageDataSource, PreparedTrajectory
from benchmark.experiment_running_helpers.fold_corruption import record_fold_corruption
from benchmark.experiment_running_helpers.normalize import normalize_experiment_data
from benchmark.experiment_running_helpers.resume import fold_instance_dir, fold_shared_dir
from benchmark.experiment_running_helpers.run_fold import cv_split, select_training_pool
from benchmark.experiment_running_helpers.trajectory_utils import save_fold_metadata
from src.utils.atomic_json import write_json_atomic
from src.utils.corpus_layout import IMAGE_POOL_SUBPATH, pool_problem_dirs
from src.utils.masking import load_masked_observation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = Path("benchmark/running_results_image_src")
OBSERVATIONS_DIRNAME = "original_observations"
OBSERVATION_PREFIX = "original_observation"
TEST_STATES_DIRNAME = "predictive_power_test_states"


def _runner_default(name: str) -> Any:
    """The default ``experiment_runner.main`` gives the parameter ``name``."""
    return inspect.signature(experiment_runner.main).parameters[name].default


@dataclass(frozen=True)
class FoldPlan:
    """The parts of a run config that decide which cells exist and what they hold."""

    n_folds: int
    cv_scheme: str
    num_trajectories: Tuple[int, ...]
    gt_rates: Tuple[int, ...]
    frame_axiom_mode: str
    normalize: bool
    force_normalize: bool

    @classmethod
    def from_shared(cls, shared: Dict[str, Any]) -> "FoldPlan":
        """The plan of a config's ``shared`` block; an absent key takes the runner's default."""
        return cls(
            n_folds=int(shared.get("n_folds", _runner_default("n_folds"))),
            cv_scheme=shared.get("cv_scheme", _runner_default("cv_scheme")),
            num_trajectories=tuple(shared.get("num_trajectories") or [3, 4, 5, 6, 7, 8]),
            gt_rates=tuple(shared.get("gt_rates") or [0]),
            frame_axiom_mode=shared.get("frame_axiom_mode", _runner_default("frame_axiom_mode")),
            normalize=bool(shared.get("normalize", _runner_default("normalize"))),
            force_normalize=bool(shared.get("force_normalize", _runner_default("force_normalize"))),
        )


def load_image_config(path: Path) -> Dict[str, Any]:
    """A parsed run config whose source is ``image``.

    Raises:
        ValueError: if the config is not an image config or names no run.
    """
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get("source") != "image":
        raise ValueError(f"{path}: source must be 'image'")
    if not config.get("run_name") or not config.get("domains"):
        raise ValueError(f"{path}: run_name and domains are required")
    return config


def resolve_data_dir(data_dir: str) -> Path:
    """A config ``data_dir`` as an absolute path; a relative one is under the project root."""
    path = Path(data_dir).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def prepare_pool(data_dir: Path, domain_file: Path, plan: FoldPlan) -> Tuple[Path, Path, bool]:
    """The pool the cells read and their reference domain: ``(pool, domain, normalized)``.

    Raises:
        ValueError: if the config asks for normalisation and it fails.
    """
    if not plan.normalize:
        return data_dir / IMAGE_POOL_SUBPATH, domain_file, False
    pool, domain = normalize_experiment_data(data_dir, domain_file, force=plan.force_normalize)
    return pool, domain, True


def fold_problem_split(
    problem_dirs: Sequence[Path], fold: int, plan: FoldPlan
) -> Tuple[List[Path], List[Path]]:
    """``(training problems in the order a cell takes its prefix from, test problems)``."""
    train_idx, test_idx = cv_split(len(problem_dirs), fold, n_folds=plan.n_folds, scheme=plan.cv_scheme)
    train_dirs = [problem_dirs[i] for i in train_idx]
    largest = max(plan.num_trajectories)
    if largest > len(train_dirs):
        raise ValueError(
            f"num_trajectories={largest} exceeds fold {fold}'s training pool of "
            f"{len(train_dirs)} problems"
        )
    return select_training_pool(train_dirs, fold, largest), [problem_dirs[i] for i in test_idx]


def is_frozen(cell_dir: Path, num_trajectories: int) -> bool:
    """Whether ``cell_dir`` already holds its metadata, domain and every observation."""
    observations = cell_dir / OBSERVATIONS_DIRNAME
    return (
        (cell_dir / "fold_info.json").is_file()
        and (cell_dir / "domain_reference.pddl").is_file()
        and len(list(observations.glob("*.trajectory"))) == num_trajectories
        and len(list(observations.glob("*.masking_info"))) == num_trajectories
    )


def freeze_cell(
    cell_dir: Path,
    prepared: List[PreparedTrajectory],
    domain_ref: Path,
    test_problems: Sequence[Path],
    data_source: ImageDataSource,
    fold: int,
    num_trajectories: int,
    gt_rate: int,
) -> None:
    """Write one cell's ``fold_info.json``, ``domain_reference.pddl`` and observations.

    Raises:
        ValueError: if the cell would hold fewer trajectories than its name says.
    """
    if len(prepared) != num_trajectories:
        raise ValueError(
            f"{cell_dir.name}: {len(prepared)} prepared trajectories for a cell of {num_trajectories}"
        )
    cell_dir.mkdir(parents=True, exist_ok=True)
    local_domain = cell_dir / "domain_reference.pddl"
    shutil.copy2(domain_ref, local_domain)
    save_fold_metadata(
        cell_dir, prepared, fold, num_trajectories, gt_rate, [str(p) for p in test_problems]
    )
    if data_source.gt_lookup is not None:
        record_fold_corruption(
            cell_dir, prepared, local_domain, data_source.gt_lookup,
            gt_root=data_source.gt_root, tag=cell_dir.name,
        )
    partial_domain = DomainParser(local_domain, partial_parsing=True).parse_domain()
    observations = [
        load_masked_observation(Path(trajectory), Path(masking), partial_domain, problem_path=Path(problem))
        for trajectory, masking, problem, *_ in prepared
    ]
    save_fold_observations(
        observations, prepared, cell_dir / OBSERVATIONS_DIRNAME, OBSERVATION_PREFIX
    )


def freeze_domain(
    domain_key: str,
    data_dir: Path,
    experiment_name: str,
    plan: FoldPlan,
    out_root: Path,
    folds: Optional[Sequence[int]] = None,
    test_states: bool = False,
    force: bool = False,
) -> Tuple[int, int]:
    """Prepare one domain's pool and freeze its cells; return ``(written, already frozen)``.

    Raises:
        FileNotFoundError: if the pool is missing or a picked problem has no
            prepared trajectory.
        ValueError: if a training size exceeds a fold's training pool.
    """
    domain_info = experiment_runner._resolve_domain_config(domain_key)
    pool, domain_ref, normalized = prepare_pool(data_dir, domain_info["domain_file"], plan)
    if not pool.is_dir():
        raise FileNotFoundError(f"image pool not found: {pool}")
    problem_dirs = pool_problem_dirs(pool)

    data_source = ImageDataSource()
    data_source.setup(problem_dirs, domain_ref, list(plan.gt_rates), plan.frame_axiom_mode)

    experiment_dir = out_root / domain_info["display_name"] / experiment_name
    testing_dir = experiment_dir / "testing"
    write_json_atomic(experiment_dir / "evaluation_results" / "run_params.json", {
        "domain_key": domain_key,
        "display_domain_name": domain_info["display_name"],
        "data_dir": str(data_dir),
        "experiment_name": experiment_name,
        "n_folds": plan.n_folds,
        "cv_scheme": plan.cv_scheme,
        "num_trajectories_list": list(plan.num_trajectories),
        "gt_rate_percentages": list(plan.gt_rates),
        "frame_axiom_mode": plan.frame_axiom_mode,
        "normalized": normalized,
        "data_source_type": type(data_source).__name__,
        "frozen_by": "benchmark.freeze_image_folds",
    })

    written = skipped = 0
    for fold in (range(plan.n_folds) if folds is None else sorted(set(folds))):
        selected_pool, test_dirs = fold_problem_split(problem_dirs, fold, plan)
        test_problems = [d / f"{d.name}.pddl" for d in test_dirs]
        for gt_rate in plan.gt_rates:
            for num_trajectories in plan.num_trajectories:
                cell_dir = fold_instance_dir(testing_dir, fold, num_trajectories, gt_rate)
                if not force and is_frozen(cell_dir, num_trajectories):
                    skipped += 1
                    continue
                prepared, _, _ = data_source.prepare(
                    selected_pool, num_trajectories, gt_rate, fold, cell_dir
                )
                freeze_cell(cell_dir, prepared, domain_ref, test_problems, data_source,
                            fold, num_trajectories, gt_rate)
                written += 1
            if test_states:
                generate_predictive_power_test_states(
                    domain_ref_path=domain_ref,
                    test_problem_paths=[str(p) for p in test_problems],
                    output_dir=fold_shared_dir(testing_dir, fold, gt_rate) / TEST_STATES_DIRNAME,
                    seed=42 + fold,
                )
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="An image run config.")
    parser.add_argument("--domains", nargs="*", default=None, metavar="DOMAIN_KEY",
                        help="Subset of the config's domain keys (default: all).")
    parser.add_argument("--only-fold", nargs="*", type=int, default=None, metavar="N",
                        help="Freeze only these folds (default: all).")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help=f"Root of the frozen source tree (default: {DEFAULT_OUT_ROOT}).")
    parser.add_argument("--test-states", action="store_true",
                        help="Also generate each fold's predictive-power test states.")
    parser.add_argument("--force", action="store_true",
                        help="Rewrite cells that are already frozen.")
    args = parser.parse_args()

    config = load_image_config(args.config)
    plan = FoldPlan.from_shared(config.get("shared") or {})
    out_root = args.out_root if args.out_root.is_absolute() else PROJECT_ROOT / args.out_root
    wanted = set(args.domains) if args.domains else None
    unknown = (wanted or set()) - {d["domain_key"] for d in config["domains"]}
    if unknown:
        raise SystemExit(f"--domains {sorted(unknown)} not in {args.config}")

    for domain in config["domains"]:
        if wanted is not None and domain["domain_key"] not in wanted:
            continue
        written, skipped = freeze_domain(
            domain["domain_key"], resolve_data_dir(domain["data_dir"]), config["run_name"],
            plan, out_root, folds=args.only_fold, test_states=args.test_states, force=args.force,
        )
        print(f"[{domain['domain_key']}] froze {written} cells of {config['run_name']}, "
              f"{skipped} already frozen, under {out_root}")


if __name__ == "__main__":
    main()
