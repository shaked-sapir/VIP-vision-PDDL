"""ROSAME-I baseline runner (imaged mode only).

Trains the joint CV + ROSAME model (:class:`RosameI_Runner`) directly from each
trajectory's *state images*, using the observed action sequence and the GT final
state as supervision — never the (VLM-inferred / degraded) trajectory states.
This is the ICAPS-24 ROSAME-I baseline; see
``docs/rosame-i-implementation-plan.md`` for the full design.

Two training schedules are selectable via the ``train_per_trajectory`` flag
(default per-trajectory). To mitigate small-data variance, ``n_seeds`` independent
models are trained and the one with the lowest final training loss is kept (a
selection rule that never touches test data).

On simulation-mode cells (no images on disk) the runner emits a clear skip
message and returns ``(None, {})`` so the harness records a null row.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, State

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.experiment_running_helpers.normalize import _normalize_hyphens

# Paper per-domain defaults (epochs, lambda_, gamma), keyed by our bench names.
_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "blocksworld": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
    "hanoi": {"epochs": 70, "lambda_": 0.2, "gamma": 10},
    "npuzzle": {"epochs": 300, "lambda_": 0.4, "gamma": 10},
    "gripper": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
    "depot": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
}
_DEFAULT_HYPERPARAMS = {"epochs": 100, "lambda_": 0.2, "gamma": 10}

# PDDL domain-name / path-part → bench key (domain names differ from bench keys).
_DOMAIN_ALIASES: Dict[str, str] = {
    "blocks": "blocksworld",
    "blocksworld": "blocksworld",
    "hanoi": "hanoi",
    "n_puzzle": "npuzzle",
    "n_puzzle_typed": "npuzzle",
    "npuzzle": "npuzzle",
    "gripper": "gripper",
    "depot": "depot",
}

# Domains whose renderings are horizontal-flip invariant (paper augments these).
_AUGMENT_DOMAINS = {"blocksworld"}

# Staging directory names that sit one level above a problem's trajectory dir.
_TRAJECTORY_DIR_NAMES = {"trajectories", "trajectories_normalized"}


class RosameIBaselineRunner(BaselineRunner):
    """ROSAME-I (imaged-mode CV + ROSAME) baseline."""

    def __init__(
        self,
        train_per_trajectory: bool = True,
        n_seeds: int = 3,
        device: Optional[str] = None,
        base_seed: int = 8800,
    ) -> None:
        self.train_per_trajectory = train_per_trajectory
        self.n_seeds = n_seeds
        self.device = device
        self.base_seed = base_seed

    @property
    def name(self) -> str:
        return "ROSAME-I"

    @property
    def display_name(self) -> str:
        return "ROSAME-I"

    @property
    def color(self) -> str:
        return "#d55181"

    # ------------------------------------------------------------------ learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
        bench = self._infer_domain_name(domain_path, partial_domain)
        hp = _HYPERPARAMS.get(bench, _DEFAULT_HYPERPARAMS)
        augment = bench in _AUGMENT_DOMAINS

        prepared_problems, _gt_paths = self._resolve_inputs(
            partial_domain, prepared_trajectories, bench
        )
        if not prepared_problems:
            print("  [ROSAME-I] skipping: no images (simulation-mode cell?)")
            return None, {}

        # Imported lazily so a torch-less environment can still import the registry.
        from benchmark.algorithm_adapters.rosame_i_runner import RosameI_Runner

        start = time.perf_counter()

        def timeout_check() -> bool:
            return (time.perf_counter() - start) > timeout_seconds

        seed_losses: Dict[int, float] = {}
        seed_models: Dict[int, str] = {}
        skipped_seeds: List[int] = []

        for i in range(self.n_seeds):
            seed = self.base_seed + i
            if seed_models and timeout_check():
                skipped_seeds.append(seed)
                continue
            try:
                runner = RosameI_Runner(str(domain_path), device=self.device, seed=seed)
                final_loss = runner.learn_full(
                    prepared_problems,
                    train_per_trajectory=self.train_per_trajectory,
                    epochs=int(hp["epochs"]),
                    gamma=float(hp["gamma"]),
                    lambda_=float(hp["lambda_"]),
                    augment=augment,
                    timeout_check=timeout_check,
                )
            except Exception as e:  # keep one bad seed from killing the cell
                print(f"  [ROSAME-I] seed {seed} failed: {e}")
                continue

            if final_loss is None:
                continue
            model = runner.to_pddl()
            if not model or ":action" not in model:
                print(f"  [ROSAME-I] seed {seed}: invalid model, skipping")
                continue

            seed_losses[seed] = final_loss
            seed_models[seed] = model
            seed_dir = Path(work_dir) / "baseline_models" / self.name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "model.pddl").write_text(model)

        if not seed_models:
            print("  [ROSAME-I] no seed produced a valid model")
            return None, {}

        chosen = min(seed_losses, key=seed_losses.get)
        extra_info: Dict = {
            "seeds": seed_losses,
            "chosen_seed": chosen,
            "chosen_final_loss": seed_losses[chosen],
            "train_per_trajectory": self.train_per_trajectory,
        }
        if skipped_seeds:
            extra_info["skipped_seeds_timeout"] = skipped_seeds
        return seed_models[chosen], extra_info

    # ------------------------------------------------------------------ inputs

    def _resolve_inputs(
        self,
        partial_domain: Domain,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        bench: str,
    ) -> Tuple[List[Tuple[object, List[Path], List[str], List[str]]], List[Optional[Path]]]:
        """Per trajectory, resolve (problem, image paths, action strings, GT final).

        Trajectories with no images on disk are dropped; an empty result means a
        simulation-mode cell (handled by the caller).

        Returns:
            ``(prepared_problems, gt_trajectory_paths)``, the latter positionally
            aligned with the former and holding the trajectory each final-state
            anchor was read from (``None`` where none was found).
        """
        prepared_problems: List[Tuple[object, List[Path], List[str], List[str]]] = []
        gt_trajectory_paths: List[Optional[Path]] = []

        for traj_path, _masking_path, problem_pddl_path, *_ in prepared_trajectories:
            problem_dir = problem_pddl_path.parent
            problem_name = problem_pddl_path.stem

            image_paths = self._resolve_images(problem_dir, bench, problem_name)
            if not image_paths:
                continue

            problem = self._parse_problem_normalized(partial_domain, problem_pddl_path)

            # Actions only (states here are degraded and must NOT be used).
            observation = self._parse_trajectory_normalized(partial_domain, problem, traj_path)
            action_strings = [
                str(component.grounded_action_call)[1:-1]
                for component in observation.components
            ]

            gt_path = self.resolve_final_state_path(problem_dir, problem_name)
            final_preds = self._resolve_final_state(
                partial_domain, problem, gt_path, problem_name
            )
            prepared_problems.append((problem, image_paths, action_strings, final_preds))
            gt_trajectory_paths.append(gt_path)

        return prepared_problems, gt_trajectory_paths

    @staticmethod
    def _resolve_images(problem_dir: Path, bench: str, problem_name: str) -> List[Path]:
        """Numerically-sorted ``state_*.png`` frames for one trajectory.

        Looks next to the problem PDDL first (PDDLGym-generated domains keep
        frames in the data dir), then falls back to the external-image domain
        layout ``src/domains/<bench>/problems/<problem>/`` (e.g. depot, gripper),
        whose frames never get copied into ``benchmark/data``.
        """
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            problem_dir,
            repo_root / "src" / "domains" / bench / "problems" / problem_name,
        ]
        for cand in candidates:
            images = list(cand.glob("state_*.png"))
            if images:
                return sorted(images, key=lambda p: int(re.search(r"\d+", p.stem).group()))
        return []

    @staticmethod
    def resolve_final_state_path(problem_dir: Path, problem_name: str) -> Optional[Path]:
        """Path of the trajectory the final-state anchor is read from, if any.

        Prefers ``<data_dir>/gt_trajectories/<problem>/<problem>.trajectory`` over
        the (degraded) trajectory sitting next to the problem PDDL.
        """
        candidates: List[Path] = []
        # Standard layout: <data_dir>/training/<staging_dir>/<problem>/ → data_dir.
        if problem_dir.parent.name in _TRAJECTORY_DIR_NAMES:
            data_dir = problem_dir.parents[2]
            candidates.append(
                data_dir / "gt_trajectories" / problem_name / f"{problem_name}.trajectory"
            )
        candidates.append(problem_dir / f"{problem_name}.trajectory")

        for cand in candidates:
            if cand.exists():
                return cand
        return None

    def _resolve_final_state(
        self,
        partial_domain: Domain,
        problem,
        final_state_path: Optional[Path],
        problem_name: str,
    ) -> List[str]:
        """GT final-state positive predicate strings from an already-resolved path."""
        if final_state_path is not None:
            observation = self._parse_trajectory_normalized(
                partial_domain, problem, final_state_path
            )
            if observation.components:
                return self._state_positive_predicates(observation.components[-1].next_state)

        print(f"    [ROSAME-I] warning: no GT trajectory for {problem_name}; "
              f"final-state anchor will be empty (all-false)")
        return []

    # In image mode the reference domain is hyphen-normalized (underscores), but
    # the problem PDDLs and gt_trajectories on disk keep their original (often
    # hyphenated) identifiers. Parsing those raw against the normalized domain
    # raises an ``illegal state component`` error, so every input is normalized to
    # underscores first — a no-op for domains that never used hyphens.

    @staticmethod
    def _normalized_tempfile(source: Path, suffix: str) -> Path:
        """Write a hyphen→underscore-normalized copy of ``source`` to a temp file."""
        normalized_text = _normalize_hyphens(source.read_text())
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
            tmp.write(normalized_text)
            return Path(tmp.name)

    @classmethod
    def _parse_problem_normalized(cls, partial_domain: Domain, problem_pddl_path: Path) -> Problem:
        """Parse a problem PDDL whose identifiers may be hyphenated."""
        tmp_path = cls._normalized_tempfile(problem_pddl_path, ".pddl")
        try:
            return ProblemParser(tmp_path, partial_domain).parse_problem()
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def _parse_trajectory_normalized(
        cls, partial_domain: Domain, problem: Problem, trajectory_path: Path
    ):
        """Parse a trajectory whose identifiers may be hyphenated."""
        tmp_path = cls._normalized_tempfile(trajectory_path, ".trajectory")
        try:
            return TrajectoryParser(partial_domain, problem).parse_trajectory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _state_positive_predicates(state: State) -> List[str]:
        """Untyped predicate strings (no parens) of a state's positive fluents."""
        predicates: List[str] = []
        for grounded_set in state.state_predicates.values():
            for pred in grounded_set:
                predicates.append(pred.untyped_representation[1:-1])
        return predicates

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _infer_domain_name(domain_path: Path, partial_domain: Domain) -> str:
        """Map a PDDL domain name / experiment path to a bench key."""
        domain_name = (partial_domain.name or "").lower()
        if domain_name in _DOMAIN_ALIASES:
            return _DOMAIN_ALIASES[domain_name]
        for part in Path(domain_path).resolve().parts:
            key = part.lower()
            if key in _HYPERPARAMS:
                return key
            if key in _DOMAIN_ALIASES:
                return _DOMAIN_ALIASES[key]
        return domain_name
