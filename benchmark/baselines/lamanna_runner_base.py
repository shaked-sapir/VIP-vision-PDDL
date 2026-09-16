"""The runner half every Lamanna-learner arm shares.

A fresh per-arm workspace, the fold's observations rewritten in the learners'
trace grammar, the fold's realised corruption rates, and the child-process
call. Concrete arms add their gate, their knobs and the in-child function.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.algorithm_adapters.lamanna import (
    RealisedRates,
    gt_trajectory_lookup,
    realised_rates,
    run_learner,
    write_partial_trace,
)
from benchmark.algorithm_adapters.lamanna.domain_dialect import write_typed_domain
from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.regime import DegradationRegime
from benchmark.experiment_running_helpers.simulated_data_utils import load_gt_observation
from src.utils.masking import load_masked_observation


class LamannaBaselineRunner(BaselineRunner):
    """Common machinery for arms over the ``offlam`` / ``nolam`` packages.

    Subclasses set :attr:`workspace_dirname` and implement :attr:`name`,
    :attr:`paper`, :meth:`supports` and :meth:`learn`; the latter composes
    :meth:`_prepare`, :meth:`_base_report` and :meth:`_run`.
    """

    workspace_dirname: str = "lamanna_workspace"

    def __init__(self, regime: Optional[DegradationRegime] = None) -> None:
        self.regime = regime

    @property
    def input_kind(self) -> str:
        return "symbolic"

    @property
    def uses_milp(self) -> bool:
        return False

    def for_regime(self, regime: DegradationRegime) -> "LamannaBaselineRunner":
        bound = copy.copy(self)
        bound.regime = regime
        return bound

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #

    def _fresh_workspace(self, work_dir: Path) -> Path:
        workspace = Path(work_dir) / self.workspace_dirname
        if workspace.exists():
            shutil.rmtree(workspace)
        workspace.mkdir(parents=True)
        return workspace

    def _write_traces(
        self, prepared_trajectories: List[Tuple], domain, traces_dir: Path
    ) -> List[Path]:
        """One trace file per prepared trajectory, masked atoms omitted."""
        paths: List[Path] = []
        for index, (trajectory, masking, problem, *_) in enumerate(prepared_trajectories):
            problem = Path(problem)
            if masking is not None and Path(masking).exists():
                observation = load_masked_observation(Path(trajectory), Path(masking), domain, problem)
            else:
                observation = load_gt_observation(Path(trajectory), domain, problem)
            paths.append(write_partial_trace(
                observation, traces_dir / f"{index}_{problem.stem}.trajectory"
            ))
        return paths

    def _measure(
        self, prepared_trajectories: List[Tuple], domain_path: Path
    ) -> Optional[RealisedRates]:
        """The fold's realised rates, or ``None`` when no GT is reachable."""
        tag = self.name
        if self.regime is None or self.regime.data_dir is None:
            print(f"  [{tag}] no corpus root bound; realised rates not measured")
            return None
        lookup = gt_trajectory_lookup(self.regime.data_dir)
        if not lookup:
            print(f"  [{tag}] no gt_trajectories/ under {self.regime.data_dir}; realised rates not measured")
            return None
        try:
            return realised_rates(prepared_trajectories, domain_path, lookup)
        except (KeyError, ValueError) as err:
            print(f"  [{tag}] realised rates not measured: {err}")
            return None

    def _prepare(
        self, domain_path: Path, prepared_trajectories: List[Tuple], work_dir: Path
    ) -> Tuple[Path, Path, List[Path], Optional[RealisedRates]]:
        """``(workspace, typed domain copy, trace paths, realised rates)`` for one fold.

        The domain copy types every untyped parameter as ``object`` (see
        :mod:`domain_dialect`); the reference domain itself is not modified.
        """
        domain_path = Path(domain_path)
        workspace = self._fresh_workspace(work_dir)
        domain_copy = write_typed_domain(domain_path, workspace / "domain.pddl")
        domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
        trace_paths = self._write_traces(prepared_trajectories, domain, workspace / "traces")
        rates = self._measure(prepared_trajectories, domain_path)
        return workspace, domain_copy, trace_paths, rates

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #

    def _base_report(self, rates: Optional[RealisedRates], n_traces: int) -> Dict[str, object]:
        """The row fields every Lamanna arm records."""
        return {
            **self.run_params(),
            "nominal_mask_rate": None if self.regime is None else self.regime.masking_p,
            "nominal_noise_rate": None if self.regime is None else self.regime.noising_p,
            "realised_mask_rate": None if rates is None else rates.mask_rate,
            "realised_noise_rate": None if rates is None else rates.noise_rate,
            "n_traces": n_traces,
        }

    def _run(
        self,
        fn: Callable[..., str],
        kwargs: Mapping[str, object],
        workspace: Path,
        timeout_seconds: int,
        report: Dict[str, object],
    ) -> Tuple[Optional[str], Dict[str, object]]:
        """Call ``fn`` in the child, fold the outcome into ``report``."""
        outcome = run_learner(fn, kwargs, cwd=workspace, timeout_seconds=timeout_seconds)
        report["terminated_by"] = outcome.terminated_by
        report["wall_seconds"] = round(outcome.wall_seconds, 3)
        report["learner_log"] = str(outcome.log_path)
        if not outcome.completed:
            first_line = outcome.detail.strip().splitlines()[0] if outcome.detail else ""
            print(f"  Warning: {self.name} {outcome.terminated_by}: {first_line}")
        return outcome.model_text, report
