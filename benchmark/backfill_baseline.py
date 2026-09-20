"""Backfill baseline algorithm results into existing experiment cells.

Runs one or more baseline runners (e.g. ROSAME) on the *frozen* degraded
trajectories saved inside each ``testing/fold*_numtrajs*_gtrate*`` cell of an
already-executed experiment, evaluates them with the same harness as a live
run, and merges the resulting rows into each cell's ``fold_result.json``.

Because the baselines consume the exact ``original_observations/`` files that
CDPS/PISAM already learned from, the resulting rows are perfectly paired with
the existing ones — no data regeneration, no re-running of CDPS.

Behavior per cell:
  - Reads identity from the directory name and ``fold_info.json``.
  - Rebuilds ``prepared_trajectories`` from ``original_observations/``
    (trajectory + optional ``.masking_info``) and the problem PDDLs in the
    experiment's ``data_dir`` (read from its ``run_params.json``, or overridden
    by ``--data-dir``).
  - Runs each requested baseline, saves its model to
    ``baseline_models/<name>/model.pddl``, evaluates it against the cell's
    saved test problems and predictive-power test states.
  - Merges the row into ``fold_result.json``: existing rows are preserved,
    a previous row of the same algorithm is replaced (idempotent). A one-time
    ``fold_result.json.bak`` backup is written before the first modification.
    Cells with no ``fold_result.json`` (old-format experiments) get a fresh
    file containing only the baseline rows.

Usage:
    # data_dir is read automatically from the experiment's run_params.json:
    python -m benchmark.backfill_baseline \
        --experiment-dir benchmark/running_results/hanoi/TO=600__hanoi_generated_problem1__final-version \
        --baselines rosame_i_24

    # Whole grid at once (shell glob) — each experiment uses its own data_dir:
    python -m benchmark.backfill_baseline \
        --experiment-dir benchmark/running_results/hanoi/TO=600__* \
        --baselines rosame_i_24

Options:
    --data-dir  Override the per-experiment data_dir (only needed for old
                experiments without a run_params.json).
    --dry-run   List what would run without learning/writing anything.
    --force     Re-run and replace rows even if the algorithm already has one.
    --cells     Restrict to cell dirs whose name contains this substring.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from benchmark.backfill_common import (
    NULL_METRIC_KEYS,
    existing_algorithms,
    find_problem_pddl,
    is_cell_dir,
    merge_row,
    parse_cell_name,
    read_run_params,
    resolve_data_dir,
    resolve_problem_dir,
    worker_init,
)
from benchmark.baselines import RESIZE_FROM_TABLE, ResizeSpec, resolve_baselines
from benchmark.baselines.regime import DegradationRegime, gate_baselines
from benchmark.experiment_running_helpers.result_builders import evaluate_and_build_result
from benchmark.experiment_running_helpers.resume import FOLD_RESULT_FILENAME
from benchmark.experiment_running_helpers.statistics import count_total_transitions_and_gt


def _build_prepared_trajectories(
    cell: Path, problem_dir: Path, fold_info: dict,
) -> List[Tuple[Path, Optional[Path], Path, set]]:
    """Rebuild run_fold-style prepared trajectory tuples from a saved cell."""
    obs_dir = cell / "original_observations"
    prepared: List[Tuple[Path, Optional[Path], Path, set]] = []

    for entry in fold_info.get("trajectories", []):
        problem = entry["problem"]
        traj = obs_dir / f"original_observation_{problem}.trajectory"
        if not traj.exists():  # older naming fallback
            traj = obs_dir / f"{problem}.trajectory"
        if not traj.exists():
            print(f"    Warning: no saved trajectory for {problem}, skipping")
            continue

        masking = traj.with_suffix(".masking_info")
        masking_path = masking if masking.exists() else None

        problem_pddl = find_problem_pddl(problem_dir, problem)
        if problem_pddl is None:
            print(f"    Warning: no problem PDDL for {problem} under {problem_dir}, skipping")
            continue

        # gt_indices unknown for saved cells; baselines don't consume it.
        prepared.append((traj, masking_path, problem_pddl, set()))

    return prepared


def _experiment_regime(exp_dir: Path, data_dir: Path) -> Optional[DegradationRegime]:
    """The regime an experiment's ``run_params.json`` records, or ``None`` if it can't say.

    Without a regime no gating happens, so a runner that only belongs in one
    regime would run everywhere; the warning is what tells the operator.
    """
    params = read_run_params(exp_dir)
    if params is None:
        print(f"[WARN] {exp_dir.name}: no run_params.json; baselines are not regime-gated")
        return None
    try:
        return DegradationRegime.from_run_params(params, data_dir)
    except KeyError as err:
        print(f"[WARN] {exp_dir.name}: run_params.json lacks {err}; baselines are not regime-gated")
        return None


def _copy_shared_fields(fold_result_path: Path) -> dict:
    """Copy shared data-context fields from an existing row (if any)."""
    if not fold_result_path.exists():
        return {}
    try:
        rows = json.loads(fold_result_path.read_text())
    except json.JSONDecodeError:
        return {}
    for r in rows:
        if r.get("total_transitions") is not None:
            return {
                "total_transitions": r.get("total_transitions"),
                "total_gt_transitions": r.get("total_gt_transitions"),
            }
    return {}


def backfill_cell(
    cell: Path,
    problem_dir: Path,
    bench_name: str,
    baselines: list,
    planning_timeout: int,
    learn_timeout: int,
    force: bool,
    dry_run: bool,
    regime: Optional[DegradationRegime] = None,
) -> str:
    """Backfill one cell. Returns a status string: done | dry | skip | gated | invalid.

    ``regime`` is the cell's degradation regime; when given, runners that do
    not support it are dropped (``gated`` if none remain) and the rest are
    bound to it, exactly as a live run does per cell.
    """
    parsed = parse_cell_name(cell.name)
    if parsed is None:
        return "invalid"
    fold, num_trajs, gt_rate = parsed

    fold_info_path = cell / "fold_info.json"
    domain_ref = cell / "domain_reference.pddl"
    if not fold_info_path.exists() or not domain_ref.exists():
        print(f"  [SKIP] {cell.name}: missing fold_info.json or domain_reference.pddl")
        return "skip"
    fold_info = json.loads(fold_info_path.read_text())

    if regime is not None:
        baselines, gated = gate_baselines(baselines, regime)
        for name, why in gated.items():
            print(f"  [GATE] {cell.name}: {name} skipped in {regime.describe()}: {why}")
        if not baselines:
            return "gated"

    fold_result_path = cell / FOLD_RESULT_FILENAME
    existing = existing_algorithms(fold_result_path)
    todo = [r for r in baselines if force or r.row_name(domain_ref) not in existing]
    if not todo:
        print(f"  [SKIP] {cell.name}: all requested baselines already present")
        return "skip"

    # Inputs shared by all runners in this cell
    prepared = _build_prepared_trajectories(cell, problem_dir, fold_info)
    if not prepared:
        print(f"  [SKIP] {cell.name}: no usable trajectories")
        return "skip"

    test_problem_paths: List[str] = []
    for problem in fold_info.get("test_problems", []):
        p = find_problem_pddl(problem_dir, problem)
        if p is not None:
            test_problem_paths.append(str(p))
        else:
            print(f"    Warning: test problem PDDL not found for {problem}")
    if not test_problem_paths:
        print(f"  [SKIP] {cell.name}: no test problem PDDLs found")
        return "skip"

    test_states = cell / "predictive_power_test_states" / "test_states.json"
    test_states_str = str(test_states) if test_states.exists() else None

    shared = _copy_shared_fields(fold_result_path)
    if not shared:
        total_transitions, total_gt = count_total_transitions_and_gt(prepared)
        shared = {"total_transitions": total_transitions, "total_gt_transitions": total_gt}

    if dry_run:
        names = ", ".join(r.row_name(domain_ref) for r in todo)
        print(f"  [DRY] {cell.name}: would run [{names}] on "
              f"{len(prepared)} trajectories, {len(test_problem_paths)} test problems"
              f"{'' if test_states_str else ' (no test states!)'}")
        return "dry"

    null_metrics = {k: None for k in NULL_METRIC_KEYS}

    # AMLGym's problem_solving writes ./tmp to the cwd — work inside the cell
    # (same protection as run_fold).
    original_cwd = os.getcwd()
    os.chdir(cell)
    try:
        for runner in todo:
            algo_name = runner.row_name(domain_ref)
            print(f"  [{algo_name}] {cell.name}: learning...")
            learn_start = time.perf_counter()
            model, extra_info = runner.learn(
                domain_path=domain_ref,
                prepared_trajectories=prepared,
                work_dir=cell,
                timeout_seconds=learn_timeout,
            )
            learn_time = time.perf_counter() - learn_start

            if model:
                model_dir = cell / "baseline_models" / algo_name
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model.pddl").write_text(model)

            print(f"  [{algo_name}] {cell.name}: evaluating...")
            row = evaluate_and_build_result(
                model, algo_name, bench_name, fold, num_trajs, gt_rate,
                test_problem_paths, domain_ref, cell.parent,
                null_metrics, cell,
                total_transitions=shared.get("total_transitions"),
                total_gt_transitions=shared.get("total_gt_transitions"),
                learning_time_seconds=learn_time,
                algorithm_specific=extra_info or {},
                planning_timeout=planning_timeout,
                test_states_path=test_states_str,
            )
            merge_row(fold_result_path, row)
            print(f"  [{algo_name}] {cell.name}: row merged into {FOLD_RESULT_FILENAME}")
    finally:
        os.chdir(original_cwd)
    return "done"


# ── Parallel execution (one process per cell) ──────────────────────────────

def _parse_resize(value: str) -> ResizeSpec:
    """``--resize`` argument -> ``int`` | ``[h, w]`` | ``None`` (native)."""
    text = value.strip().lower()
    if text in ("native", "none", "null"):
        return None
    if "," in text:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--resize expects N, H,W or 'native'; got {value!r}"
            )
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"--resize H,W must be integers; got {value!r}"
            ) from None
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--resize expects N, H,W or 'native'; got {value!r}"
        ) from None


def _backfill_cell_worker(
    cell_str: str,
    problem_dir_str: str,
    bench_name: str,
    baseline_keys: List[str],
    planning_timeout: int,
    learn_timeout: int,
    force: bool,
    train_per_trajectory: bool = True,
    resize: ResizeSpec = RESIZE_FROM_TABLE,
    runner_kwargs: Optional[dict] = None,
    regime: Optional[DegradationRegime] = None,
) -> Tuple[str, str]:
    """Process-pool entry point: resolve runners locally (no pickling of torch
    objects across processes) and backfill one cell.

    Returns:
        (cell_name, status) where status is backfill_cell's status or
        ``error: <message>``.
    """
    try:
        try:
            import torch
            torch.set_num_threads(1)
        except ImportError:
            pass
        baselines = resolve_baselines(
            baseline_keys,
            train_per_trajectory=train_per_trajectory,
            resize=resize,
            **(runner_kwargs or {}),
        )
        status = backfill_cell(
            Path(cell_str), Path(problem_dir_str), bench_name, baselines,
            planning_timeout=planning_timeout, learn_timeout=learn_timeout,
            force=force, dry_run=False, regime=regime,
        )
        return cell_str, status
    except Exception as e:  # keep one failing cell from killing the whole run
        return cell_str, f"error: {e}"


def _runner_kwargs(args: argparse.Namespace) -> dict:
    """Runner options that only some baselines accept.

    ``resolve_baselines`` forwards each key only to the runners whose
    ``__init__`` takes it, so an option meant for one arm cannot reach another.
    Omitted keys leave that runner's own default in place, which is why only
    values the operator actually passed are included.
    """
    kwargs: dict = {}
    if args.epochs is not None:
        kwargs["epochs"] = args.epochs
    if args.n_seeds is not None:
        kwargs["n_seeds"] = args.n_seeds
    if getattr(args, "batch_size", None) is not None:
        kwargs["batch_size"] = args.batch_size
    if getattr(args, "normalize_base_loss", None) is not None:
        kwargs["normalize_base_loss"] = args.normalize_base_loss
    if getattr(args, "rosame_seed", None) is not None:
        kwargs["rosame_seed"] = args.rosame_seed
    if getattr(args, "rosame_convergence", None) is not None:
        kwargs["rosame_convergence"] = json.loads(args.rosame_convergence)
    if getattr(args, "agreement_stop", None) is not None:
        text = str(args.agreement_stop).strip().lower()
        kwargs["agreement_stop"] = None if text in ("none", "null", "off") else float(text)
    if getattr(args, "nolam_noise", None) is not None:
        kwargs["nolam_noise"] = args.nolam_noise
    if getattr(args, "nolam_allow_neg_precs", None) is not None:
        kwargs["nolam_allow_neg_precs"] = args.nolam_allow_neg_precs
    if getattr(args, "nolam_seed", None) is not None:
        kwargs["nolam_seed"] = args.nolam_seed
    if args.budget_mode is not None:
        kwargs["budget_mode"] = args.budget_mode
    elif args.ignore_budget:
        # Retained spelling of --budget-mode fixed.
        kwargs["budget_mode"] = "fixed"
    return kwargs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Backfill baseline results into existing experiment cells.")
    ap.add_argument("--experiment-dir", type=Path, nargs="+", required=True,
                    help="Experiment director(y/ies) containing testing/ "
                         "(shell globs expand to multiple dirs).")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="Override the data_dir problem PDDLs/images are "
                         "resolved against. By default each experiment's "
                         "data_dir is read from its evaluation_results/"
                         "run_params.json; pass this only to override that "
                         "(e.g. for old experiments without run_params.json).")
    ap.add_argument("--baselines", nargs="+", default=["rosame_24"],
                    help="Baseline registry keys to backfill (default: rosame_24).")
    ap.add_argument("--domain", default=None,
                    help="Domain/bench name for the result rows "
                         "(default: inferred from the experiment path).")
    ap.add_argument("--planning-timeout", type=int, default=60)
    ap.add_argument("--learn-timeout", type=int, default=300,
                    help="Timeout passed to the baseline's learn() (seconds).")
    ap.add_argument("--cells", default=None,
                    help="Only process cell dirs whose name contains this substring.")
    ap.add_argument("--force", action="store_true",
                    help="Re-run baselines even when their row already exists.")
    ap.add_argument("--train-per-trajectory", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Symbolic ROSAME training schedule: per-trajectory "
                         "(default, matches the vendored learn_rosame) vs pooled "
                         "(--no-train-per-trajectory). Ignored by baselines that "
                         "don't accept it, ROSAME-I included — it trains pooled "
                         "unconditionally, as ICAPS-24 train.py does.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Symbolic ROSAME arms: transitions per pooled optimizer "
                         "step; 0 = one step per trace. Default: the runner's own "
                         "(128, as ICAPS-24's DataLoader).")
    ap.add_argument("--normalize-base-loss", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="ROSAME+MILP arms: divide the base loss terms by the "
                         "step's transition count before adding the pseudo-label "
                         "CE. Default: the runner's own (on).")
    ap.add_argument("--rosame-seed", type=int, default=None,
                    help="Seed for the symbolic ROSAME arms' RNGs. Default: the "
                         "runner's own (8800, upstream ICAPS-24's default).")
    ap.add_argument("--rosame-convergence", default=None, metavar="JSON",
                    help="Symbolic ROSAME arms: the training-loss plateau rule as a "
                         "JSON mapping, e.g. '{\"window\": 10, \"min_improvement\": "
                         "0.002, \"patience\": 3, \"min_epochs\": 50}'. Omitted = "
                         "off (fixed epochs). Pair it with --no-train-per-trajectory "
                         "for rosame_24: the per-trajectory schedule has no single "
                         "loss curve, so the rule does not apply to it.")
    ap.add_argument("--agreement-stop", default=None, metavar="FLOAT|none",
                    help="ROSAME+MILP arms: stop once network/MILP agreement "
                         "reaches this level; 'none' records agreement without "
                         "stopping on it. Default: the runner's own (1.0).")
    ap.add_argument("--nolam-noise", default=None,
                    help="NOLAM: the flip probability it is given. 'oracle' "
                         "(the runner's default) measures the fold's realised "
                         "flip rate against the GT trajectories; a float pins "
                         "one e for every cell and suffixes the row name "
                         "(NOLAM__e=0.1) so it is never averaged with oracle rows.")
    ap.add_argument("--nolam-allow-neg-precs", action=argparse.BooleanOptionalAction,
                    default=None,
                    help="NOLAM: learn negative preconditions (the paper's MAP "
                         "variant, row name NOLAM__negprecs). Default: the "
                         "runner's own (off, MAP_pre+).")
    ap.add_argument("--nolam-seed", type=int, default=None,
                    help="NOLAM: NumPy seed for MAP tie-breaking. Default: the "
                         "runner's own (0).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override the per-domain epoch budget of the ICAPS-26 "
                         "arm (rosame_i_26). The configured value is a ceiling "
                         "the pre-flight budget check lowers to fit the cell "
                         "timeout; whenever the effective value differs from "
                         "the default the row name carries an __ep= suffix, so "
                         "two budgets can never be averaged under one label. "
                         "Ignored by baselines that don't accept it.")
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="Override the number of independent models trained "
                         "per fold (the lowest-final-training-loss one is "
                         "kept). It multiplies the pre-flight projection. "
                         "Ignored by baselines that don't accept it.")
    ap.add_argument("--budget-mode", choices=["preflight", "fixed", "converge"],
                    default=None,
                    help="How the ICAPS-26 arm decides when to stop. "
                         "'preflight' (default) lets the budget check lower "
                         "--epochs to fit the cell timeout, which is what the "
                         "grid runs. 'fixed' runs --epochs whatever the "
                         "projection says (gate 7's control; run it outside the "
                         "timeout). 'converge' early-stops on a training-loss "
                         "plateau with --epochs as a ceiling, and is the only "
                         "mode that emits the best-loss model rather than the "
                         "final epoch's. Non-default modes suffix the row name "
                         "so two modes are never averaged together.")
    ap.add_argument("--ignore-budget", action="store_true",
                    help="Let the ICAPS-26 arm run its configured epoch count "
                         "whatever the pre-flight projects. This is gate 7's "
                         "budget-control setting: one cell per domain at "
                         "--epochs 5000 outside the timeout, so 'underperforms "
                         "at the grid budget' and 'undertrained at it' stay "
                         "distinguishable. Not for grid cells. Superseded by "
                         "--budget-mode fixed, which it is now an alias for.")
    ap.add_argument("--resize", type=_parse_resize, default=RESIZE_FROM_TABLE,
                    metavar="N|H,W|native",
                    help="Override the per-domain image resize for the pixel "
                         "arms (ROSAME-I, ROSAME-I+MILP). 'N' resizes the "
                         "shorter edge to N and preserves aspect (upstream's "
                         "form); 'H,W' forces that size and distorts (note the "
                         "order is height,width); 'native' skips the resize. "
                         "Omit to use the per-domain table. Whenever the "
                         "effective value differs from the default, the row "
                         "name carries a suffix (e.g. ROSAME-I_24__res=64x64) so "
                         "two resolutions can never be averaged under one "
                         "label. Ignored by baselines that don't accept it.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of cells to backfill in parallel (one process "
                         "per cell, like the experiment runner's per-fold "
                         "processes). Default 1 = sequential. Dry runs are "
                         "always sequential.")
    args = ap.parse_args()

    override_data_dir = args.data_dir.resolve() if args.data_dir else None
    if override_data_dir and not override_data_dir.is_dir():
        raise SystemExit(f"--data-dir does not exist: {override_data_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    # Gather (bench_name, cell, problem_dir) work items across all experiment
    # dirs. Each experiment's data_dir is read from its own run_params.json so the
    # operator can't accidentally point at a regenerated/sibling dir (e.g.
    # *_fixed); --data-dir, when given, overrides this for every experiment. The
    # problem_dir is then resolved from it so the problem PDDLs match the cells'
    # predicate dialect.
    tasks: List[Tuple[str, Path, Path, Optional[DegradationRegime]]] = []
    for exp_dir in args.experiment_dir:
        exp_dir = exp_dir.resolve()
        testing = exp_dir / "testing"
        if not testing.is_dir():
            print(f"[SKIP] {exp_dir}: no testing/ directory")
            continue

        data_dir, src = resolve_data_dir(exp_dir, override_data_dir)
        if data_dir is None:
            print(f"[SKIP] {exp_dir}: no data_dir in run_params.json; "
                  f"pass --data-dir explicitly")
            continue
        if not data_dir.is_dir():
            print(f"[SKIP] {exp_dir}: resolved data_dir does not exist: {data_dir}")
            continue

        # running_results/<domain>/<experiment_name> → domain
        bench_name = args.domain or exp_dir.parent.name
        cells = sorted(d for d in testing.iterdir() if is_cell_dir(d))
        if args.cells:
            cells = [c for c in cells if args.cells in c.name]

        problem_dir = resolve_problem_dir(exp_dir, data_dir)
        regime = _experiment_regime(exp_dir, data_dir)
        dialect_note = "" if problem_dir == data_dir else f", problems from {problem_dir.name}/"
        regime_note = f", {regime.describe()}" if regime is not None else ""
        print(f"[{bench_name}] {exp_dir.name}: {len(cells)} cells "
              f"(data_dir from {src}: {data_dir}{dialect_note}{regime_note})")
        tasks.extend((bench_name, cell, problem_dir, regime) for cell in cells)

    if not tasks:
        print("Nothing to do.")
        return

    workers = 1 if args.dry_run else min(args.workers, len(tasks))

    if workers == 1:
        # Sequential — identical to the original behavior.
        baselines = resolve_baselines(
            args.baselines, train_per_trajectory=args.train_per_trajectory,
            resize=args.resize, **_runner_kwargs(args),
        )
        for bench_name, cell, problem_dir, regime in tasks:
            backfill_cell(
                cell, problem_dir, bench_name, baselines,
                planning_timeout=args.planning_timeout,
                learn_timeout=args.learn_timeout,
                force=args.force, dry_run=args.dry_run, regime=regime,
            )
        return

    # Parallel — one process per cell; runners are resolved inside each worker
    # (nothing torch-adjacent is pickled across processes).
    print(f"\nRunning {len(tasks)} cells with {workers} workers...")
    statuses: dict = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as executor:
        futures = {
            executor.submit(
                _backfill_cell_worker,
                str(cell), str(problem_dir), bench_name, args.baselines,
                args.planning_timeout, args.learn_timeout, args.force,
                args.train_per_trajectory, args.resize, _runner_kwargs(args),
                regime,
            ): cell
            for bench_name, cell, problem_dir, regime in tasks
        }
        for i, future in enumerate(as_completed(futures), start=1):
            try:
                cell_str, status = future.result()
            except Exception as e:  # worker process died before/outside the task
                cell_str, status = str(futures[future]), f"error: {e}"
            statuses[cell_str] = status
            print(f"[{i}/{len(futures)}] {Path(cell_str).name}: {status}")

    done = sum(1 for s in statuses.values() if s == "done")
    skipped = sum(1 for s in statuses.values() if s in ("skip", "gated", "invalid"))
    errors = {c: s for c, s in statuses.items() if s.startswith("error")}
    print(f"\nSummary: {done} done, {skipped} skipped, {len(errors)} errors")
    for cell_str, err in errors.items():
        print(f"  ERROR {cell_str}: {err}")


if __name__ == "__main__":
    main()
