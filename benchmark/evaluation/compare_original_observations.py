"""Compare fold original_observations against source GT100 trajectories.

For each ``original_observation_{problem}.trajectory`` in a fold, loads the
matching ``{problem}_gtrate100.trajectory`` and compares grounded positive
fluents state-by-state (symmetric difference, excluding state 0).

Usage:
    python -m benchmark.evaluation.compare_original_observations \\
        <experiment_dir> --fold fold0_numtrajs3_gtrate0

    python -m benchmark.evaluation.compare_original_observations \\
        /path/to/testing/fold0_numtrajs3_gtrate0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser

from benchmark.experiment_running_helpers.simulated_data_utils import build_gt_trajectory_lookup
from src.utils.pddl_state import state_positive_set

ORIGINAL_OBS_PREFIX = "original_observation_"
ORIGINAL_OBS_DIR = "original_observations"


@dataclass(frozen=True)
class StateDiff:
    problem: str
    state_index: int
    n_diff: int
    only_in_observation: Tuple[str, ...]
    only_in_gt100: Tuple[str, ...]

    @property
    def differing_fluents(self) -> Tuple[str, ...]:
        return tuple(sorted(set(self.only_in_observation) | set(self.only_in_gt100)))


def _load_run_params(experiment_dir: Path) -> dict:
    for candidate in (
        experiment_dir / "evaluation_results" / "run_params.json",
        experiment_dir / "run_params.json",
    ):
        if candidate.exists():
            return json.loads(candidate.read_text())
    raise FileNotFoundError(
        f"No run_params.json under {experiment_dir} or {experiment_dir / 'evaluation_results'}"
    )


def _infer_domain_key(experiment_dir: Path, run_params: dict) -> str:
    domains = run_params.get("domains_to_run") or []
    if len(domains) == 1:
        return domains[0]

    parent_name = experiment_dir.parent.name
    if parent_name in domains:
        return parent_name
    if parent_name == "npuzzle":
        return "n_puzzle_typed"
    raise ValueError(
        f"Could not infer domain key for {experiment_dir}. "
        f"domains_to_run={domains}"
    )


def _benchmark_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_gt100_lookup(experiment_dir: Path, run_params: dict) -> Dict[str, Path]:
    simulated_paths = run_params.get("simulated_gt_trajectories") or []
    if simulated_paths:
        return build_gt_trajectory_lookup([Path(p) for p in simulated_paths])

    domain_key = _infer_domain_key(experiment_dir, run_params)
    data_dirs = run_params.get("experiment_data_dirs", {}).get(domain_key) or []
    if not data_dirs:
        raise FileNotFoundError(
            f"No experiment_data_dirs for domain '{domain_key}' in run_params.json"
        )

    data_dir_name = data_dirs[0]
    training_root = (
        _benchmark_root() / "data" / domain_key / data_dir_name / "training" / "trajectories"
    )
    if not training_root.exists():
        raise FileNotFoundError(f"Training trajectories directory not found: {training_root}")

    lookup: Dict[str, Path] = {}
    for problem_dir in sorted(training_root.iterdir()):
        if not problem_dir.is_dir():
            continue
        gt100 = problem_dir / f"{problem_dir.name}_gtrate100.trajectory"
        if gt100.exists():
            lookup[problem_dir.name] = gt100
    if not lookup:
        raise FileNotFoundError(f"No *_gtrate100.trajectory files under {training_root}")
    return lookup


def _resolve_fold_dir(experiment_dir: Path, fold: Optional[str]) -> Path:
    if fold is None:
        raise ValueError("--fold is required when passing an experiment directory")

    fold_path = Path(fold)
    if fold_path.is_dir() and (fold_path / ORIGINAL_OBS_DIR).is_dir():
        return fold_path.resolve()

    candidate = experiment_dir / "testing" / fold
    if candidate.is_dir() and (candidate / ORIGINAL_OBS_DIR).is_dir():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Fold directory with {ORIGINAL_OBS_DIR}/ not found: {fold} "
        f"(under {experiment_dir / 'testing'})"
    )


def _resolve_experiment_dir(fold_or_experiment: Path) -> Tuple[Path, Optional[str]]:
    """Allow passing a fold directory directly as the only positional argument."""
    fold_or_experiment = fold_or_experiment.resolve()
    if (fold_or_experiment / ORIGINAL_OBS_DIR).is_dir():
        if fold_or_experiment.parent.name == "testing":
            return fold_or_experiment.parent.parent, fold_or_experiment.name
        return fold_or_experiment.parent, fold_or_experiment.name
    return fold_or_experiment, None


def _problem_from_original_obs(path: Path) -> str:
    stem = path.stem
    if not stem.startswith(ORIGINAL_OBS_PREFIX):
        raise ValueError(f"Unexpected original observation filename: {path.name}")
    return stem[len(ORIGINAL_OBS_PREFIX):]


def _load_state_fluent_sets(trajectory_path: Path, domain_path: Path) -> List[Set[str]]:
    domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
    observation = TrajectoryParser(partial_domain=domain).parse_trajectory(trajectory_path)
    states = [observation.components[0].previous_state] + [
        component.next_state for component in observation.components
    ]
    return [state_positive_set(state, unmasked_only=False) for state in states]


def _compare_trajectory_pair(
    problem: str,
    observation_path: Path,
    gt100_path: Path,
    domain_path: Path,
) -> Tuple[List[StateDiff], Optional[str]]:
    obs_states = _load_state_fluent_sets(observation_path, domain_path)
    gt_states = _load_state_fluent_sets(gt100_path, domain_path)

    if len(obs_states) != len(gt_states):
        length_note = (
            f"state count mismatch for {problem}: "
            f"observation={len(obs_states)}, gt100={len(gt_states)}"
        )
    else:
        length_note = None

    diffs: List[StateDiff] = []
    max_index = min(len(obs_states), len(gt_states))
    for state_index in range(1, max_index):
        obs_fluents = obs_states[state_index]
        gt_fluents = gt_states[state_index]
        only_in_obs = tuple(sorted(obs_fluents - gt_fluents))
        only_in_gt = tuple(sorted(gt_fluents - obs_fluents))
        n_diff = len(only_in_obs) + len(only_in_gt)
        diffs.append(
            StateDiff(
                problem=problem,
                state_index=state_index,
                n_diff=n_diff,
                only_in_observation=only_in_obs,
                only_in_gt100=only_in_gt,
            )
        )
    return diffs, length_note


def compare_fold_original_observations(fold_dir: Path, experiment_dir: Path) -> dict:
    original_dir = fold_dir / ORIGINAL_OBS_DIR
    if not original_dir.is_dir():
        raise FileNotFoundError(f"Missing {ORIGINAL_OBS_DIR}/ under {fold_dir}")

    domain_path = fold_dir / "domain_reference.pddl"
    if not domain_path.exists():
        raise FileNotFoundError(f"Missing domain reference PDDL: {domain_path}")

    run_params = _load_run_params(experiment_dir)
    gt100_lookup = _build_gt100_lookup(experiment_dir, run_params)

    problems: List[dict] = []
    all_diffs: List[StateDiff] = []
    warnings: List[str] = []
    problems_compared: List[str] = []

    for obs_path in sorted(original_dir.glob("*.trajectory")):
        problem = _problem_from_original_obs(obs_path)
        gt100_path = gt100_lookup.get(problem)
        if gt100_path is None or not gt100_path.exists():
            warnings.append(f"No GT100 trajectory found for {problem}")
            continue

        diffs, length_note = _compare_trajectory_pair(
            problem, obs_path, gt100_path, domain_path
        )
        if length_note:
            warnings.append(length_note)
        all_diffs.extend(diffs)
        problems_compared.append(problem)
        problems.append({
            "problem": problem,
            "observation_path": str(obs_path),
            "gt100_path": str(gt100_path),
            "state_diffs": [
                {
                    "state_index": diff.state_index,
                    "n_diff": diff.n_diff,
                    "only_in_observation": list(diff.only_in_observation),
                    "only_in_gt100": list(diff.only_in_gt100),
                    "differing_fluents": list(diff.differing_fluents),
                }
                for diff in diffs
            ],
        })

    return {
        "experiment_dir": str(experiment_dir),
        "fold_dir": str(fold_dir),
        "problems_compared": problems_compared,
        "warnings": warnings,
        "problems": problems,
        "state_diffs": [
            {
                "problem": diff.problem,
                "state_index": diff.state_index,
                "n_diff": diff.n_diff,
                "only_in_observation": list(diff.only_in_observation),
                "only_in_gt100": list(diff.only_in_gt100),
                "differing_fluents": list(diff.differing_fluents),
            }
            for diff in all_diffs
        ],
    }


def _format_problem_table(problem: str, state_diffs: List[dict]) -> str:
    headers = ["state", "n_diff", "only_in_observation", "only_in_gt100"]
    rows = [
        [
            str(diff["state_index"]),
            str(diff["n_diff"]),
            ", ".join(diff["only_in_observation"]) or "-",
            ", ".join(diff["only_in_gt100"]) or "-",
        ]
        for diff in state_diffs
    ]

    if not rows:
        return f"=== {problem} (no states to compare) ===\n"

    col_widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(cell))

    def _fmt_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(col_widths[idx]) for idx, cell in enumerate(cells))

    separator = "-+-".join("-" * width for width in col_widths)
    n_matching = sum(1 for diff in state_diffs if diff["n_diff"] == 0)
    n_differing = len(state_diffs) - n_matching
    title = (
        f"=== {problem} ({len(state_diffs)} states, state 0 excluded; "
        f"{n_differing} differ, {n_matching} match) ==="
    )
    lines = [title, _fmt_row(headers), separator]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _format_all_problems(problems: List[dict]) -> str:
    if not problems:
        return "(no problems compared)"
    return "\n\n".join(
        _format_problem_table(problem_entry["problem"], problem_entry["state_diffs"])
        for problem_entry in problems
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare original_observations in a fold against source GT100 trajectories, "
            "state by state (symmetric difference, excluding state 0)."
        )
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Experiment directory, or a fold directory containing original_observations/",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default=None,
        help="Fold directory name or path under <experiment>/testing/ (required if path is experiment dir).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full comparison result as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    experiment_dir, fold_from_path = _resolve_experiment_dir(args.path)
    fold_name = args.fold or fold_from_path
    fold_dir = _resolve_fold_dir(experiment_dir, fold_name)

    result = compare_fold_original_observations(fold_dir, experiment_dir)
    print(f"Fold: {fold_dir.name}")
    print(f"Problems compared: {', '.join(result['problems_compared']) or '(none)'}")
    if result["warnings"]:
        print("\nWarnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    print("\nState diffs vs GT100 (symmetric difference, state 0 excluded):\n")
    print(_format_all_problems(result["problems"]))

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2))
        print(f"\nJSON written to: {args.json_out}")


if __name__ == "__main__":
    main()
