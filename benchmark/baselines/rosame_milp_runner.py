"""ROSAME+MILP baseline runners (simulation mode).

Two registered variants (ROSAME-side glue in
``benchmark/algorithm_adapters/rosame_milp/``; the MILP encoder itself lives in
``src/milp/``, shared with the ``pisam_milp_*`` learners):

- ``rosame_milp_24`` (:class:`RosameMilpRunner`) — the ICAPS-26 iterative loop:
  ``pre_mip_epochs`` warmup, then a MILP solve every ``mip_interval`` epochs
  whose 4-way model pseudo-labels supervise further training (undecayed, per
  upstream code); output = decode of the final MILP solution.
- ``rosame_milp_24_tag`` (:class:`RosameMilpTagRunner`) — the same loop with the
  ``tag`` encoding rules (``MilpEncodingConfig.tag``): >=1 add effect per
  schema (no precondition requirement) and no redundant-add ban. The ``tag``
  comparison is scoped to this arm alone; no other arm has a ``tag`` variant.

:class:`RosameMilpBaseRunner` holds the MILP plumbing both share and the
one-shot ``learn`` they inherit. It is not a registered arm: ``name`` is
abstract there, so it cannot be instantiated.

The MILP constraint rule-set is bundled in
:class:`~src.milp.encoding_config.MilpEncodingConfig`
and passed to the runners via ``encoding_config`` (default ``upstream()``).

Both fall back to the plain ROSAME model (with ``milp_failed=True`` in the
report) when no feasible MILP solution is found.
"""

from __future__ import annotations

import random
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser

import benchmark.algorithm_adapters.rosame_milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp import encoder as _encoder_module  # noqa: F401  (factory registration)
from src.milp.converter import (
    build_ps_domain,
    build_ps_instance,
    find_gt_trajectory,
    gt_final_state_fluents,
    observation_to_trace,
)
from src.milp.encoding_config import MilpEncodingConfig
from benchmark.algorithm_adapters.rosame_milp.milp_loop import MilpPORosame
from benchmark.algorithm_adapters.rosame_milp.model_bridge import (
    extract_model_labels,
    model_agreement,
    rosame_to_observation_m,
)
from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner
from benchmark.baselines.rosame_runner import RosameBaselineRunner, _setup_rosame_workspace
from benchmark.experiment_running_helpers.normalize import _normalize_hyphens

from constraint_opt.factory import resolve as resolve_encoder
from planning_structs.traces import Traces

_OBJECTIVES = {"state", "model"}


def goal_fluents_from_trajectory(
    gt_trajectory_path: Optional[Path],
    normalize_identifiers: bool = False,
) -> Optional[set]:
    """Final-state fluents of a GT trajectory, or None (-> soft final state).

    Args:
        gt_trajectory_path: an already-resolved GT ``.trajectory``.
        normalize_identifiers: hyphen→underscore the fluent names and arguments,
            matching the normalization image-mode staging applies to the domain
            and to trajectories but not to ``gt_trajectories/``.
    """
    if gt_trajectory_path is None:
        return None
    fluents = gt_final_state_fluents(gt_trajectory_path)
    if fluents is None or not normalize_identifiers:
        return fluents
    return {
        (_normalize_hyphens(name), tuple(_normalize_hyphens(arg) for arg in args))
        for name, args in fluents
    }


def goal_fluents_for(
    problem_pddl_path: Optional[Path],
    goal_mode: str = "gt",
    normalize_identifiers: bool = False,
) -> Optional[set]:
    """GT final-state fluents for one problem, or None (-> soft final state)."""
    if goal_mode != "gt" or problem_pddl_path is None:
        return None
    gt_path = find_gt_trajectory(problem_pddl_path)
    if gt_path is None:
        print(f"  [MILP] Warning: no GT trajectory for "
              f"{problem_pddl_path.stem} — final state left soft")
        return None
    return goal_fluents_from_trajectory(gt_path, normalize_identifiers)


class RosameMilpBaseRunner(RosameBaselineRunner):
    """Shared MILP plumbing plus a one-shot ``learn``: train, then project once.

    Abstract: ``name`` is re-declared abstract here, so only its subclasses are
    instantiable. Without that it would inherit ``RosameBaselineRunner``'s and
    file its rows under that arm's label.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __init__(
        self,
        train_per_trajectory: bool = True,
        epochs: int = 100,
        mip_time_limit: int = 60,
        encoding_config: Optional[MilpEncodingConfig] = None,
        goal_mode: str = "gt",
        milp_solver: str = "cp-sat-observed",
    ) -> None:
        super().__init__(train_per_trajectory=train_per_trajectory)
        self.epochs = epochs
        self.mip_time_limit = mip_time_limit
        self.encoding_config = encoding_config or MilpEncodingConfig.upstream()
        self.goal_mode = goal_mode
        self.milp_solver = milp_solver

    # ------------------------------------------------------------ MILP plumbing

    def _goal_fluents_for(self, problem_pddl_path: Optional[Path]):
        """GT final-state fluents for one problem, or None (-> soft final state)."""
        return goal_fluents_for(problem_pddl_path, self.goal_mode)

    def _build_milp_traces(
        self,
        partial_domain,
        prepared: List[Tuple[object, object]],
        original_problem_paths: Dict[str, Path],
    ):
        """(ps_domain, obs_t list) from the trained-side prepared pairs."""
        ps_domain = build_ps_domain(partial_domain)
        obs_t = []
        n_gt_goals = 0
        for problem, observation in prepared:
            instance = build_ps_instance(ps_domain, partial_domain, problem)
            goal = self._goal_fluents_for(original_problem_paths.get(problem.name))
            trace = observation_to_trace(instance, observation, goal)
            if trace is None:
                continue
            if goal is not None:
                n_gt_goals += 1
            obs_t.append(trace)
        return ps_domain, obs_t, n_gt_goals

    def _solve(self, ps_domain, obs_t, obs_m):
        """One MILP solve; returns (encoder, ok)."""
        traces = Traces(instance=None, obs_m=obs_m, obs_t=obs_t)
        encoder = resolve_encoder(self.milp_solver)(
            ps_domain,
            traces,
            _OBJECTIVES,
            config=self.encoding_config,
        )
        ok = encoder.solve(time_limit=self.mip_time_limit)
        return encoder, ok

    @staticmethod
    def _original_problem_paths(prepared_trajectories) -> Dict[str, Path]:
        """problem name -> original problem PDDL path (for GT trajectory lookup)."""
        return {
            problem_pddl_path.stem: problem_pddl_path
            for _traj, _mask, problem_pddl_path, *_ in prepared_trajectories
        }

    # ------------------------------------------------------------ learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        traj_paths = _setup_rosame_workspace(prepared_trajectories, work_dir)
        if not traj_paths:
            print(f"  [{self.name}] No valid trajectories, skipping")
            return None, {}

        extra: Dict = {
            "train_per_trajectory": self.train_per_trajectory,
            "goal_mode": self.goal_mode,
            "milp_solver": self.milp_solver,
            "encoding_config": self.encoding_config.as_stats(),
        }
        try:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            rosame = PORosame_Runner(str(domain_path))
            prepared = self._build_prepared(traj_paths, partial_domain)

            rosame.learn_full(
                prepared, train_per_trajectory=self.train_per_trajectory, epochs=self.epochs
            )

            ps_domain, obs_t, n_gt_goals = self._build_milp_traces(
                partial_domain, prepared, self._original_problem_paths(prepared_trajectories)
            )
            extra["n_traces"] = len(obs_t)
            extra["n_gt_goals"] = n_gt_goals
            if not obs_t:
                raise ValueError("No usable traces for the MILP")

            obs_m = rosame_to_observation_m(rosame, ps_domain)
            encoder, ok = self._solve(ps_domain, obs_t, obs_m)
            extra["milp"] = encoder.solve_stats

            # The learned model is what this arm reports; the MILP supervises it
            # through pseudo-labels rather than replacing it. See
            # ``RosameIMilpRunner._model_from`` for why, and what changed.
            model = rosame.rosame_to_pddl()
            extra["milp_failed"] = not ok
            if not ok:
                print(f"  [{self.name}] MILP failed — the model is DL-only for this cell")

            if model and ":action" in model:
                return model, extra
            raise ValueError("Invalid ROSAME+MILP model")

        except Exception as e:
            print(f"  Warning: {self.name} learning failed: {e}")
            return None, extra


class RosameMilpRunner(RosameMilpBaseRunner):
    """Iterative variant: the paper's train/solve/pseudo-label loop.

    Runs on the pooled schedule only (the paper trains pooled batches);
    ``train_per_trajectory`` is fixed to False.
    """

    def __init__(
        self,
        epochs: int = 100,
        pre_mip_epochs: int = 50,
        mip_interval: int = 1,
        mip_traces: Optional[int] = None,
        agreement_stop: float = 1.0,
        mip_time_limit: int = 60,
        encoding_config: Optional[MilpEncodingConfig] = None,
        goal_mode: str = "gt",
        milp_solver: str = "cp-sat-observed",
    ) -> None:
        super().__init__(
            train_per_trajectory=False,
            epochs=epochs,
            mip_time_limit=mip_time_limit,
            encoding_config=encoding_config,
            goal_mode=goal_mode,
            milp_solver=milp_solver,
        )
        self.pre_mip_epochs = pre_mip_epochs
        self.mip_interval = mip_interval
        self.mip_traces = mip_traces
        self.agreement_stop = agreement_stop

    @property
    def name(self) -> str:
        return "ROSAME_MILP_24"

    @property
    def display_name(self) -> str:
        return "ROSAME+MILP (24)"

    @property
    def color(self) -> str:
        return "#9085e9"

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        traj_paths = _setup_rosame_workspace(prepared_trajectories, work_dir)
        if not traj_paths:
            print(f"  [{self.name}] No valid trajectories, skipping")
            return None, {}

        extra: Dict = {
            "goal_mode": self.goal_mode,
            "milp_solver": self.milp_solver,
            "encoding_config": self.encoding_config.as_stats(),
            "pre_mip_epochs": self.pre_mip_epochs,
            "mip_interval": self.mip_interval,
            "mip_traces": self.mip_traces,
        }
        try:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            rosame = MilpPORosame(str(domain_path))
            prepared = self._build_prepared(traj_paths, partial_domain)

            ps_domain, obs_t, n_gt_goals = self._build_milp_traces(
                partial_domain, prepared, self._original_problem_paths(prepared_trajectories)
            )
            extra["n_traces"] = len(obs_t)
            extra["n_gt_goals"] = n_gt_goals
            if not obs_t:
                raise ValueError("No usable traces for the MILP")

            def milp_round():
                round_obs_t = obs_t
                if self.mip_traces is not None and self.mip_traces < len(obs_t):
                    round_obs_t = random.sample(obs_t, self.mip_traces)
                obs_m = rosame_to_observation_m(rosame, ps_domain)
                encoder, ok = self._solve(ps_domain, round_obs_t, obs_m)
                if not ok:
                    return {}, 0.0, encoder.solve_stats, None
                solution = encoder.action_model_sol()
                labels = extract_model_labels(rosame, ps_domain, solution)
                agreement = model_agreement(rosame, labels)
                return labels, agreement, encoder.solve_stats, solution

            start = time.perf_counter()
            report = rosame.learn_pooled_with_milp(
                prepared,
                milp_round,
                epochs=self.epochs,
                pre_mip_epochs=self.pre_mip_epochs,
                mip_interval=self.mip_interval,
                agreement_stop=self.agreement_stop,
            )
            extra["loop_seconds"] = round(time.perf_counter() - start, 2)
            extra["milp_rounds"] = report["rounds"]
            extra["stop_reason"] = report["stop_reason"]
            extra["final_agreement"] = report["final_agreement"]

            solution = report["final_solution"]
            if solution is None:
                # last-chance whole-fold solve before giving up on the MILP
                obs_m = rosame_to_observation_m(rosame, ps_domain)
                encoder, ok = self._solve(ps_domain, obs_t, obs_m)
                extra.setdefault("milp_rounds", []).append(
                    {"epoch": "final_fallback", **encoder.solve_stats})
                solution = encoder.action_model_sol() if ok else None

            model = rosame.rosame_to_pddl()
            extra["milp_failed"] = solution is None
            if solution is None:
                print(f"  [{self.name}] MILP failed — the model is DL-only for this cell")

            if model and ":action" in model:
                return model, extra
            raise ValueError("Invalid ROSAME+MILP model")

        except Exception as e:
            print(f"  Warning: {self.name} learning failed: {e}")
            return None, extra


class RosameMilpTagRunner(RosameMilpRunner):
    """Iterative loop with the ``tag`` encoding rules.

    Same train/solve/pseudo-label loop as :class:`RosameMilpRunner`, but the
    MILP uses :meth:`MilpEncodingConfig.tag`: every schema must have >=1 add
    effect (no precondition requirement) and the redundant-add ban is dropped.
    """

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("encoding_config", MilpEncodingConfig.tag())
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "ROSAME_MILP_24_TAG"

    @property
    def display_name(self) -> str:
        return "ROSAME+MILP (24, tag)"

    @property
    def color(self) -> str:
        return "#e98545"
