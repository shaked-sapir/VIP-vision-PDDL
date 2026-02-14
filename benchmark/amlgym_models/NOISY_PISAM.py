import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from amlgym.algorithms.AlgorithmAdapter import AlgorithmAdapter
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Observation
from utilities import NegativePreconditionPolicy

from src.pi_sam import PISAMLearner
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.utils.masking import load_masked_observation


@dataclass
class NOISY_PISAM(AlgorithmAdapter):
    """
    Adapter class for running the PISAM algorithm: "Safe Learning of Lifted Action Models",
    B. Juba and H. S. Le, and R. Stern, Proceedings of the 18th International Conference
    on Principles of Knowledge Representation and Reasoning, 2021.
    https://proceedings.kr.org/2021/36/

    Example:
        .. code-block:: python

            from amlgym.algorithms import get_algorithm
            sam = get_algorithm('SAM')
            model = sam.learn('path/to/domain.pddl', ['path/to/trace0', 'path/to/trace1'])
            print(model)

    """
    negative_precondition_policy = NegativePreconditionPolicy.hard

    # Conflict search parameters
    fluent_patch_cost = 1
    model_patch_cost = 1
    max_search_nodes = None
    timeout_seconds = 60
    seed = 42

    def learn(self,
              domain_path: str,
              trajectory_paths: List[str],
              use_problems: bool = False,
              with_new_traj: bool = False) -> Tuple[str, list[Observation], dict]:
        """
        Learns a PDDL action model from:
         (i)    a (possibly empty) input model which is required to specify the predicates and operators signature;
         (ii)   a list of trajectory file paths.

        :parameter domain_path: input PDDL domain file path
        :parameter trajectory_paths: list of trajectory file paths
        :parameter use_problems: boolean flag indicating whether to provide the set of objects
            specified in the problem from which the trajectories have been generated

        :return: tuple of (learned_pddl_model, patched_observations, learning_report)
        """

        # Instantiate SAM algorithm
        partial_domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        conflict_search = ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(partial_domain),
            negative_preconditions_policy=self.negative_precondition_policy,
            seed=self.seed,
            logger=None
        )
        # Parse input trajectories

        masked_observations = []
        if use_problems:
            raise NotImplementedError("use_problems=True is not implemented yet for PISAM.")
        else: # we use our own trajectories and masks, not amlgym's
            for traj_path in trajectory_paths:
                traj_path = Path(traj_path)

                # Look for masking_info file with the same stem as the trajectory file
                masking_info_path = traj_path.parent / f"{traj_path.stem}.masking_info"

                if not masking_info_path.exists():
                    self.logger.warning(f"Masking info file not found for {traj_path.stem}, skipping")
                    continue

                masked_obs = load_masked_observation(traj_path, masking_info_path, partial_domain)
                masked_observations.append(masked_obs)

        # Learn action model
        learned_model, conflicts, model_constraints, fluent_patches, cost, report, patched_observations = conflict_search.run(
            observations=masked_observations,
            max_nodes=self.max_search_nodes,
            timeout_seconds=self.timeout_seconds
        )

        # TODO: show conflicts and patches at the end of the learning?
        return learned_model.to_pddl(), patched_observations, report
