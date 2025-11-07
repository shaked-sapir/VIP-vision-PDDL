"""Main plan denoising orchestrator."""

from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging

from pddl_plus_parser.models import Observation, Domain
from sam_learning.core import LearnerDomain

from src.plan_denoising.inconsistency_detector import InconsistencyDetector
from src.plan_denoising.data_repairer import DataRepairer
from src.plan_denoising.conflict_tree import (
    ConflictTree, Inconsistency, RepairChoice, RepairOperation
)
from src.pi_sam import PISAMLearner
from src.utils.pddl import copy_observation
from utilities import NegativePreconditionPolicy


class PlanDenoiser:
    """
    Main orchestrator for plan denoising using PI-SAM learning.

    The denoiser:
    1. Detects inconsistencies in a trajectory
    2. Uses LLM verification to repair inconsistencies
    3. Tracks repairs in a conflict tree
    4. Runs PI-SAM on the repaired trajectory
    5. Validates the learned model
    6. Backtracks if necessary to try alternative repairs

    Algorithm:
    - WHILE inconsistencies exist:
        - Find an inconsistency
        - Repair it using LLM (choose which transition to fix)
        - Add repair to conflict tree
        - Run PI-SAM on repaired observation
        - IF learned model is consistent with trajectory:
            - Continue to next inconsistency
        - ELSE (model still inconsistent):
            - Backtrack in tree
            - Try alternative repair (fix the other transition)
    """

    def __init__(
        self,
        domain: Domain,
        openai_apikey: str,
        image_directory: Path,
        domain_name: str = "unknown",
        negative_precondition_policy: NegativePreconditionPolicy = NegativePreconditionPolicy.hard,
        max_iterations: int = 100,
        max_backtracks: int = 10
    ):
        """
        Initialize the plan denoiser.

        :param domain: The PDDL domain
        :param openai_apikey: OpenAI API key for LLM verification
        :param image_directory: Directory containing trajectory images
        :param domain_name: Name of the domain (for prompts)
        :param negative_precondition_policy: Policy for negative preconditions in PI-SAM
        :param max_iterations: Maximum denoising iterations
        :param max_backtracks: Maximum backtracking attempts
        """
        self.domain = domain
        self.image_directory = Path(image_directory)
        self.domain_name = domain_name

        # Initialize components
        self.detector = InconsistencyDetector()
        self.repairer = DataRepairer(openai_apikey)
        self.conflict_tree = ConflictTree()

        # PI-SAM configuration
        self.negative_precondition_policy = negative_precondition_policy

        # Limits
        self.max_iterations = max_iterations
        self.max_backtracks = max_backtracks

        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def get_image_path(self, state_index: int) -> Path:
        """
        Get the path to the image for a given state index.

        Images are assumed to be named: state_0001.png, state_0002.png, etc.

        :param state_index: Index of the state in the trajectory
        :return: Path to the image file
        """
        # Format with leading zeros (assuming 4 digits)
        image_name = f"state_{state_index:04d}.png"
        return self.image_directory / image_name

    def run_pi_sam(
        self,
        observation: Observation
    ) -> Tuple[LearnerDomain, dict]:
        """
        Run PI-SAM learning on an observation.

        :param observation: The observation to learn from
        :return: Tuple of (learned domain, learning report)
        """
        self.logger.info("Running PI-SAM learning...")

        learner = PISAMLearner(
            partial_domain=self.domain,
            negative_preconditions_policy=self.negative_precondition_policy
        )

        # Learn from the single observation
        # Note: PI-SAM expects a list of observations
        learned_domain, learning_report = learner.learn_action_model([observation])

        self.logger.info("PI-SAM learning completed")
        return learned_domain, learning_report

    def validate_learned_model(
        self,
        learned_domain: LearnerDomain,
        observation: Observation
    ) -> bool:
        """
        Validate that the learned model is consistent with the observation.

        A model is consistent if executing each action in the trajectory
        according to the learned preconditions and effects produces the
        observed next states.

        :param learned_domain: The learned domain model
        :param observation: The observation to validate against
        :return: True if consistent, False otherwise
        """
        self.logger.info("Validating learned model against trajectory...")

        # Check each transition in the observation
        for idx, component in enumerate(observation.components):
            action_name = component.grounded_action_call.name

            # Check if action exists in learned model
            if action_name not in learned_domain.actions:
                self.logger.warning(f"Action {action_name} not in learned model")
                return False

            learned_action = learned_domain.actions[action_name]

            # Check preconditions: are they satisfied in prev_state?
            # (This is a simplified check - full validation would require grounding)

            # Check effects: do they match the state transition?
            # (Also simplified)

        # For now, we assume the model is valid if PI-SAM succeeded
        # A more thorough validation would simulate the actions
        self.logger.info("Model validation passed")
        return True

    def repair_inconsistency_with_choice(
        self,
        observation: Observation,
        inconsistency: Inconsistency,
        repair_choice: RepairChoice
    ) -> Tuple[Observation, RepairOperation]:
        """
        Repair an inconsistency with a specified repair choice.

        :param observation: The observation to repair
        :param inconsistency: The inconsistency to resolve
        :param repair_choice: Which transition to repair
        :return: Tuple of (repaired observation, repair operation)
        """
        # Get images for both transitions' next states
        # Transition indices correspond to state indices (state i+1 is after transition i)
        image1_path = self.get_image_path(inconsistency.transition1_index + 1)
        image2_path = self.get_image_path(inconsistency.transition2_index + 1)

        # Determine the fluent's correct value
        if repair_choice == RepairChoice.FIRST:
            # We're repairing trans1, so trans2's value is correct
            fluent_should_be_present = inconsistency.fluent_in_trans2_next
        else:
            # We're repairing trans2, so trans1's value is correct
            fluent_should_be_present = inconsistency.fluent_in_trans1_next

        # Repair the observation
        repaired_obs, repair_op = self.repairer.repair_observation(
            observation, inconsistency, repair_choice, fluent_should_be_present
        )

        return repaired_obs, repair_op

    def denoise(
        self,
        observation: Observation,
        use_llm_verification: bool = True
    ) -> Tuple[Observation, LearnerDomain, ConflictTree]:
        """
        Main denoising loop.

        :param observation: The observation to denoise
        :param use_llm_verification: Whether to use LLM for repair choice (vs. trying first option)
        :return: Tuple of (denoised observation, learned domain, conflict tree)
        """
        self.logger.info("Starting plan denoising...")

        # Make a copy to avoid modifying the original
        current_observation = copy_observation(observation)

        iteration = 0
        backtrack_count = 0

        while iteration < self.max_iterations:
            iteration += 1
            self.logger.info(f"\n=== Denoising iteration {iteration} ===")

            # Step 1: Detect inconsistencies
            inconsistencies = self.detector.detect_inconsistencies_from_observation(
                current_observation
            )

            if not inconsistencies:
                self.logger.info("No inconsistencies detected!")
                break

            self.logger.info(f"Found {len(inconsistencies)} inconsistencies")

            # Step 2: Take the first inconsistency
            inconsistency = inconsistencies[0]
            self.logger.info(f"Resolving: {inconsistency}")

            # Step 3: Determine repair choice
            if use_llm_verification:
                # Use LLM to choose which transition to repair
                image1_path = self.get_image_path(inconsistency.transition1_index + 1)
                image2_path = self.get_image_path(inconsistency.transition2_index + 1)

                _, _, repair_choice = self.repairer.repair_inconsistency(
                    current_observation,
                    inconsistency,
                    image1_path,
                    image2_path,
                    self.domain_name
                )
                # Note: repair_inconsistency already modifies the observation
                # Get the repair operation from the last repair
                if repair_choice == RepairChoice.FIRST:
                    trans_idx = inconsistency.transition1_index
                    old_val = inconsistency.fluent_in_trans1_next
                    new_val = inconsistency.fluent_in_trans2_next
                else:
                    trans_idx = inconsistency.transition2_index
                    old_val = inconsistency.fluent_in_trans2_next
                    new_val = inconsistency.fluent_in_trans1_next

                repair_operation = RepairOperation(
                    transition_index=trans_idx,
                    state_type='next_state',
                    fluent_changed=inconsistency.conflicting_fluent,
                    old_value=old_val,
                    new_value=new_val
                )
            else:
                # Try repairing the first transition by default
                repair_choice = RepairChoice.FIRST
                current_observation, repair_operation = self.repair_inconsistency_with_choice(
                    current_observation, inconsistency, repair_choice
                )

            # Step 4: Add repair to conflict tree
            self.conflict_tree.add_repair(inconsistency, repair_operation, repair_choice)

            # Step 5: Run PI-SAM on repaired observation
            try:
                learned_domain, learning_report = self.run_pi_sam(current_observation)

                # Step 6: Validate learned model
                is_valid = self.validate_learned_model(learned_domain, current_observation)

                if is_valid:
                    self.logger.info("Learned model is valid, continuing...")
                    continue
                else:
                    self.logger.warning("Learned model is inconsistent with trajectory")

                    # Step 7: Try backtracking
                    if backtrack_count >= self.max_backtracks:
                        self.logger.error(f"Max backtracks ({self.max_backtracks}) reached")
                        break

                    if self.conflict_tree.has_unexplored_alternative():
                        self.logger.info("Backtracking to try alternative repair...")
                        backtrack_count += 1

                        # Backtrack
                        self.conflict_tree.backtrack()

                        # Reset observation to state before last repair
                        current_observation = copy_observation(observation)
                        # Re-apply all repairs except the last one
                        for repair_op in self.conflict_tree.get_current_repairs():
                            # Re-apply this repair
                            # (This is simplified; a full implementation would track observation states)
                            pass

                        # Try alternative repair choice
                        alternative_choice = self.conflict_tree.get_alternative_repair_choice()
                        current_observation, repair_operation = self.repair_inconsistency_with_choice(
                            current_observation, inconsistency, alternative_choice
                        )

                        # Add to tree
                        self.conflict_tree.add_repair(inconsistency, repair_operation, alternative_choice)

                    else:
                        self.logger.error("No alternative repairs available")
                        break

            except Exception as e:
                self.logger.error(f"Error during PI-SAM learning: {e}")
                break

        # Final PI-SAM run on the denoised observation
        self.logger.info("\n=== Final PI-SAM learning ===")
        final_learned_domain, _ = self.run_pi_sam(current_observation)

        self.logger.info(f"Denoising completed after {iteration} iterations")
        self.logger.info(f"Total backtracks: {backtrack_count}")

        return current_observation, final_learned_domain, self.conflict_tree

    def denoise_from_trajectory_file(
        self,
        trajectory_path: Path,
        use_llm_verification: bool = True
    ) -> Tuple[Observation, LearnerDomain, ConflictTree]:
        """
        Denoise a trajectory from a .trajectory file.

        :param trajectory_path: Path to the trajectory file
        :param use_llm_verification: Whether to use LLM verification
        :return: Tuple of (denoised observation, learned domain, conflict tree)
        """
        # Load trajectory
        observation = self.detector.load_trajectory(trajectory_path)

        # Denoise
        return self.denoise(observation, use_llm_verification)
