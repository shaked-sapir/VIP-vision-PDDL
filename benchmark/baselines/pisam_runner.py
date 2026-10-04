"""PI-SAM baseline runner (Le, Juba, Stern, AAAI 2024), with no repair step.

Learns one model from all of the fold's frozen observations as they are: masked
fluents stay unknown and every unmasked fluent is taken as true. It is the
learner the ``pisam_milp_*`` arms call after their MILP, run without the MILP.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from utilities import NegativePreconditionPolicy

from benchmark.baselines.base_runner import BaselineRunner
from src.plan_denoising.noisy_pisam_learning import NoisyPisamLearner
from src.utils.masking import load_masked_observation

INITIAL_STATE_INDEX = 0


class PisamRawRunner(BaselineRunner):
    """PI-SAM over the fold's frozen observations, with nothing repaired."""

    def __init__(
        self,
        negative_preconditions_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        pisam_seed: int = 42,
    ) -> None:
        self.negative_preconditions_policy = negative_preconditions_policy
        self.pisam_seed = pisam_seed

    @property
    def name(self) -> str:
        return "PISAM"

    @property
    def display_name(self) -> str:
        return "PI-SAM (AAAI-24)"

    @property
    def color(self) -> str:
        return "#008300"

    @property
    def input_kind(self) -> str:
        return "symbolic"

    @property
    def paper(self) -> str:
        return "aaai24"

    @property
    def uses_milp(self) -> bool:
        return False

    def run_params(self) -> Dict[str, object]:
        return {
            "negative_preconditions_policy": self.negative_preconditions_policy.name,
            "pisam_seed": self.pisam_seed,
        }

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        report: Dict[str, object] = dict(self.run_params())
        partial_domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        observations = [
            load_masked_observation(Path(trajectory), Path(masking), partial_domain, problem)
            for trajectory, masking, problem, *_ in prepared_trajectories
            if masking is not None
        ]
        report["n_traces"] = len(observations)
        if not observations:
            print("  [PISAM] No usable trajectories, skipping")
            return None, report

        learner = NoisyPisamLearner(
            partial_domain=deepcopy(partial_domain),
            negative_preconditions_policy=self.negative_preconditions_policy,
            seed=self.pisam_seed,
        )
        gt_states = {index: {INITIAL_STATE_INDEX} for index in range(len(observations))}
        learned_domain, conflicts, _ = learner.learn_action_model_with_conflicts(
            observations=observations,
            fluent_patches=set(),
            model_patches=set(),
            gt_states_by_obs=gt_states,
        )
        report["pisam_conflicts"] = len(conflicts)
        report["n_operators_emitted"] = len(learned_domain.actions)
        return learned_domain.to_pddl(), report
