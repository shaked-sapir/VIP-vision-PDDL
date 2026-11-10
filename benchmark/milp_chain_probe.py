"""Chains of MILP rounds at several subset sizes, with the prior soft or pinned.

For each selected cell (a finished experiment's fold, read from its frozen
``original_observations``), each subset size ``m`` and each seed, the probe runs
one cold round and then ``--rounds - 1`` warm rounds per mode:

- ``soft``: the previous best PI-SAM model enters the solve as the loop hands it
  over today, a solution hint plus a tie-break objective term;
- ``hard_declared``: additionally, every add/delete bit that model declares is
  pinned to 1;
- ``hard_full``: additionally, every add/delete bit of a schema that model has
  effects for is pinned to the model's value (declared 1, undeclared 0).

Round 1 is shared by the modes, so a chain costs ``1 + modes * (rounds - 1)``
solves. Every solve appends one JSON row to ``results.jsonl`` with CP-SAT's
exit status, wall time, first-solution and last-improvement times, objective
and bound, plus the round's repair cost, PI-SAM conflicts and V.

Usage:
    python -m benchmark.milp_chain_probe --domains gripper depot --noise 0.2 0.4 \\
        --m 4 8 16 --rounds 4 --seeds 42 43 --solve-time-limit 60
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Set

from src.milp import encoder as _encoder_module  # noqa: F401  (vendor path + factory registration)
from constraint_opt.factory import resolve as resolve_encoder  # noqa: E402
from planning_structs.traces import Traces  # noqa: E402

from benchmark.backfill_anchored_solve_cap import (
    frozen_fold_trajectories,
    read_anchored_config,
)
from benchmark.backfill_cdps import resolve_experiment
from benchmark.backfill_common import read_data_dir_from_run_params
from benchmark.backfill_nogt_s0 import frozen_observations_dir
from benchmark.experiment_running_helpers.learning_helpers import (
    _prepare_milp_driver_inputs,
)
from src.milp.converter import build_ps_domain
from src.plan_denoising.evaluator import EvaluationWeights
from src.plan_denoising.milp_denoiser.config import PisamMilpConfig
from src.plan_denoising.milp_denoiser.loop import (
    _OBJECTIVES_NO_PRIOR,
    _OBJECTIVES_WITH_PRIOR,
    _TraceCache,
    _evaluate,
    _subset_gt,
    model_hash,
)
from src.plan_denoising.milp_denoiser.model_prior import (
    learner_domain_to_observation_m,
)
from src.plan_denoising.milp_denoiser.single_round import _learn_with_pisam
from src.plan_denoising.milp_denoiser.trajectory_extraction import (
    extract_repaired_observations,
)
from src.utils.time import create_experiment_timestamp

MODES = ("soft", "hard_declared", "hard_full")
DEFAULT_OUT_ROOT = Path("benchmark/running_results_milp_chain_probe")
DEFAULT_MILP_CONFIG = Path("benchmark/milp_configs/npisam_schema_nonempty_large.yaml")


@dataclass
class LoadedCell:
    """One fold's frozen observations, encoded once for every chain."""

    domain: str
    noise: float
    cell_name: str
    partial_domain: Any
    observations: List[Any]
    gt_states_by_obs: Optional[Dict[int, Set[int]]]
    negative_preconditions_policy: Any
    ps_domain: Any
    cache: _TraceCache
    pool: List[int]


@dataclass
class ChainState:
    """The incumbent a chain carries from round to round."""

    best_domain: Any = None
    best_v: Optional[float] = None
    rounds: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


def _cell_dir(results_root: Path, domain: str, prefix: str, mask: float, noise: float,
              fold: int, num_trajs: int) -> Path:
    exp_dir = results_root / domain / f"{prefix}__mask={mask}__noise={noise}"
    return exp_dir / "testing" / f"fold{fold}_numtrajs{num_trajs}_gtrate0"


def _local_data_dir(exp_dir: Path, domain: str, data_root: Path) -> Optional[Path]:
    """The recorded data_dir, or its same-named local copy when it was written on another host."""
    recorded = read_data_dir_from_run_params(exp_dir)
    if recorded is None or recorded.is_dir():
        return None
    local = data_root / domain / recorded.name
    return local.resolve() if local.is_dir() else None


def load_cell(cell: Path, domain: str, noise: float, config: PisamMilpConfig,
              data_root: Path) -> LoadedCell:
    """Read a cell's frozen observations and encode every trace once."""
    exp_dir = cell.parent.parent
    settings = resolve_experiment(
        exp_dir, _local_data_dir(exp_dir, domain, data_root),
        SimpleNamespace(domain=domain, learn_timeout=None, planning_timeout=None,
                        frame_axiom_mode=None),
    )
    if settings is None:
        raise FileNotFoundError(f"{exp_dir}: run_params.json unusable")
    fold_info = json.loads((cell / "fold_info.json").read_text())
    with frozen_observations_dir(cell) as frozen_dir:
        if frozen_dir is None:
            raise FileNotFoundError(f"{cell}: no original_observations")
        trajectories = frozen_fold_trajectories(
            frozen_dir.resolve(), settings.problem_dir, fold_info, cell.name,
        )
        inputs = _prepare_milp_driver_inputs(
            cell / "domain_reference.pddl", trajectories, None, None, None, tag="PROBE",
        )
    partial_domain = inputs["partial_domain"]
    observations = list(inputs["observations"])
    gt_states = config.resolve_gt_states(inputs["gt_states_by_obs"])
    ps_domain = build_ps_domain(partial_domain)
    cache = _TraceCache(ps_domain, partial_domain, config.gt_anchoring,
                        inputs["object_types_by_obs"])
    for index, observation in enumerate(observations):
        cache.trace(index, observation, (gt_states or {}).get(index))
    return LoadedCell(
        domain=domain, noise=noise, cell_name=cell.name,
        partial_domain=partial_domain, observations=observations,
        gt_states_by_obs=gt_states,
        negative_preconditions_policy=inputs["negative_preconditions_policy"],
        ps_domain=ps_domain, cache=cache, pool=list(range(len(observations))),
    )


def _declared_schemas(obs_m, threshold: float = 0.5) -> set:
    """Schemas for which the reference model declares at least one effect."""
    return {key[0] for channel in (obs_m.add, obs_m.dele)
            for key, prob in channel.items() if prob > threshold}


def _build_encoder(cell: LoadedCell, subset: Sequence[int], prior, mode: str,
                   config: PisamMilpConfig):
    traces = [cell.cache.trace(i, cell.observations[i], (cell.gt_states_by_obs or {}).get(i))
              for i in subset]
    has_prior = prior is not None
    encoder = resolve_encoder(config.solver_encoder_key())(
        cell.ps_domain,
        Traces(instance=None, obs_m=prior, obs_t=traces),
        _OBJECTIVES_WITH_PRIOR if has_prior else _OBJECTIVES_NO_PRIOR,
        config=config.encoding_config(has_prior=has_prior),
    )
    if has_prior and mode == "hard_declared":
        encoder.fix_model_bits(prior, ("add", "dele"), declared_only=True)
    elif has_prior and mode == "hard_full":
        encoder.fix_model_bits(prior, ("add", "dele"), declared_only=False,
                               schemas=_declared_schemas(prior))
    return encoder


def run_round(cell: LoadedCell, state: ChainState, subset: List[int], mode: str,
              config: PisamMilpConfig, solve_limit: int, weights: EvaluationWeights,
              seed: int) -> Dict[str, Any]:
    """Solve one subset, learn from the repair, score V, update the incumbent."""
    prior = None
    if state.best_domain is not None:
        prior = learner_domain_to_observation_m(state.best_domain, cell.ps_domain).observation_m
    state.rounds += 1
    row: Dict[str, Any] = {
        "round": state.rounds, "mode": mode if prior is not None else "cold",
        "subset": list(subset), "prior_present": prior is not None,
        "prior_hash": model_hash(state.best_domain),
    }
    encoder = _build_encoder(cell, subset, prior, mode, config)
    solved = encoder.solve(time_limit=solve_limit)
    stats = encoder.solve_stats
    value, bound, scale = stats.get("objective_value"), stats.get("objective_bound"), stats.get("obj_scale")
    row.update({
        "solved": bool(solved),
        "exit_status": stats.get("exit_status"),
        "solve_seconds": stats.get("solve_time_seconds"),
        "first_solution_seconds": stats.get("first_solution_seconds"),
        "last_improvement_seconds": stats.get("last_improvement_seconds"),
        "n_incumbents": stats.get("n_incumbents"),
        "objective_value": value, "objective_bound": bound,
        "gap_flips": (round((bound - value) / scale, 3)
                      if solved and value is not None and bound is not None and scale else None),
        "n_model_vars": stats.get("n_model_vars"), "n_trace_vars": stats.get("n_trace_vars"),
        "pinned_model_bits": stats.get("pinned_model_bits", 0),
    })
    if not solved:
        return row

    extraction = extract_repaired_observations(encoder, [cell.observations[i] for i in subset])
    row["repair_cost"] = extraction.repair_cost
    row["repair_cost_per_trace"] = round(extraction.repair_cost / len(subset), 2)

    t0 = time.perf_counter()
    domain, conflicts, _report = _learn_with_pisam(
        cell.partial_domain, list(extraction.observations),
        _subset_gt(cell.gt_states_by_obs, subset),
        cell.negative_preconditions_policy, seed,
    )
    row["learn_seconds"] = round(time.perf_counter() - t0, 3)
    row["pisam_conflicts"] = len(conflicts)
    row["model_hash"] = model_hash(domain)

    t0 = time.perf_counter()
    evaluation = _evaluate(domain, cell.observations, weights)
    row["eval_seconds"] = round(time.perf_counter() - t0, 3)
    if evaluation is None:
        row["note"] = "PI-SAM returned no model"
        return row
    row.update({
        "v_per_transition": evaluation.v_per_transition,
        "effect_mismatches": evaluation.effect_mismatches,
        "inapplicability_events": evaluation.inapplicability_events,
    })
    improved = state.best_v is None or evaluation.v_raw < state.best_v
    row["improved"] = improved
    if improved:
        state.best_v, state.best_domain = evaluation.v_raw, domain
    return row


def _next_subset(rng: random.Random, pool: Sequence[int], m: int,
                 previous: Optional[Sequence[int]]) -> List[int]:
    eligible = [i for i in pool if not previous or i not in set(previous)]
    if len(eligible) < m:
        eligible = list(pool)
    return sorted(rng.sample(eligible, m))


def run_chain(cell: LoadedCell, m: int, seed: int, rounds: int, modes: Sequence[str],
              config: PisamMilpConfig, solve_limit: int, weights: EvaluationWeights,
              emit) -> None:
    """One cold round, then ``rounds - 1`` warm rounds per mode on shared subsets."""
    rng = random.Random(seed)
    subsets = []
    previous = None
    for _ in range(rounds):
        previous = _next_subset(rng, cell.pool, m, previous)
        subsets.append(previous)

    base = {"domain": cell.domain, "noise": cell.noise, "cell": cell.cell_name,
            "pool_size": len(cell.pool), "m": m, "seed": seed}
    cold = ChainState()
    emit({**base, **run_round(cell, cold, subsets[0], "soft", config, solve_limit, weights, seed),
          "chain_mode": "shared"})
    for mode in modes:
        state = ChainState(best_domain=cold.best_domain, best_v=cold.best_v, rounds=1)
        for subset in subsets[1:]:
            emit({**base, **run_round(cell, state, subset, mode, config, solve_limit, weights, seed),
                  "chain_mode": mode})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=Path("benchmark/running_results"))
    parser.add_argument("--data-root", type=Path, default=Path("benchmark/data"),
                        help="where a cell's recorded data_dir is looked up by name when it is not on this host")
    parser.add_argument("--experiment-prefix", default="large-corpora__L-sweep__v4")
    parser.add_argument("--domains", nargs="+",
                        default=["blocksworld", "depot", "gripper", "hanoi", "npuzzle"])
    parser.add_argument("--mask", type=float, default=0.1)
    parser.add_argument("--noise", nargs="+", type=float, default=[0.2, 0.4])
    parser.add_argument("--num-trajs", type=int, default=50, help="the cell's L")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--m", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--rounds", type=int, default=4, help="rounds per chain, cold one included")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43])
    parser.add_argument("--solve-time-limit", type=int, default=60)
    parser.add_argument("--milp-config", type=Path, default=DEFAULT_MILP_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help=f"default {DEFAULT_OUT_ROOT}/<timestamp>/")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = read_anchored_config(args.milp_config)
    weights = EvaluationWeights(effect_mismatch=config.eval.effect_mismatch_weight,
                                inapplicability=config.eval.inapplicability_weight)
    out_dir = args.out_dir or DEFAULT_OUT_ROOT / create_experiment_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_params.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, indent=2))
    results = out_dir / "results.jsonl"
    print(f"[probe] writing {results}", flush=True)

    def emit(row: Dict[str, Any]) -> None:
        with results.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(f"  {row['domain']:11} noise={row['noise']} m={row['m']:2} seed={row['seed']} "
              f"r{row['round']} {row['mode']:13} {row['exit_status']:22} "
              f"{row['solve_seconds']:7.2f}s first={row['first_solution_seconds']} "
              f"gap={row.get('gap_flips')} cost={row.get('repair_cost')} "
              f"V={row.get('v_per_transition')}", flush=True)

    cells: Dict[tuple, LoadedCell] = {}
    for domain in args.domains:
        for noise in args.noise:
            cell = _cell_dir(args.results_root, domain, args.experiment_prefix, args.mask,
                             noise, args.fold, args.num_trajs)
            t0 = time.perf_counter()
            cells[(domain, noise)] = load_cell(cell, domain, noise, config, args.data_root)
            print(f"[probe] loaded {domain} noise={noise}: {len(cells[(domain, noise)].pool)} "
                  f"traces in {time.perf_counter() - t0:.1f}s", flush=True)

    for m in args.m:
        for (domain, noise), cell in cells.items():
            if m > len(cell.pool):
                print(f"[probe] skip {domain} noise={noise} m={m}: pool has {len(cell.pool)}")
                continue
            for seed in args.seeds:
                run_chain(cell, m, seed, args.rounds, args.modes, config,
                          args.solve_time_limit, weights, emit)
    print("[probe] done", flush=True)


if __name__ == "__main__":
    main()
