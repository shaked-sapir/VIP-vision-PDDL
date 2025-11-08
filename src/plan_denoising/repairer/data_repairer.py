"""
Data repairer for fixing trajectory inconsistencies using LLM verification.

This module provides functionality to repair effects violations (determinism violations)
in PDDL trajectories by using vision-language models to verify which observed state
is correct when two transitions with the same preconditions lead to different outcomes.
"""

from pathlib import Path
from typing import Tuple, Union, List

from openai import OpenAI
from pddl_plus_parser.models import Observation, GroundedPredicate, State

from src.plan_denoising.detectors.effects_detector import EffectsViolation
from src.plan_denoising.conflict_tree import RepairOperation, RepairChoice
from src.plan_denoising.repairer.prompts import (
    create_binary_verification_prompt,
    create_ternary_verification_prompt
)
from src.llms.utils import encode_image


class DataRepairer:
    """
    Repairs trajectory inconsistencies using LLM-based visual verification.

    When an effects violation is detected (two transitions with the same action and
    prev_state have different next_states), this class an LLM to:
    1. Examine images of both conflicting next_states
    2. Determine which image correctly represents the fluent in question
    3. Repair the incorrectly classified state in the observation

    The approach assumes that visual observations are generally reliable, but
    classification errors may occur. The LLM acts as a tie-breaker to determine
    which of the two conflicting observations is correct.
    """

    def __init__(self, openai_apikey: str, model: str, use_uncertain: bool = False):
        """
        Initialize the data repairer with LLM access.

        Args:
            openai_apikey: OpenAI API key for GPT-4 Vision access
            model: Vision model identifier (default: gpt-4o)
            use_uncertain: If True, uses ternary mode (IMAGE1/IMAGE2/UNCERTAIN) and masks
                         uncertain fluents. If False, uses binary mode (IMAGE1/IMAGE2 only)
                         and repairs to the correct state.
        """
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model
        self.use_uncertain = use_uncertain

    # ==================== LLM Prompting ====================

    def _get_system_prompt(self, fluent: str, domain_name: str) -> str: # TODO: why fluent ?
        """
        Get the system prompt for verifying fluent presence based on use_uncertain mode.

        Args:
            fluent: PDDL fluent to verify (e.g., "(on red blue)")
            domain_name: Name of the planning domain (for context)

        Returns:
            System prompt string for the LLM
        """
        if self.use_uncertain:
            return create_ternary_verification_prompt(fluent, domain_name)
        else:
            return create_binary_verification_prompt(fluent, domain_name)

    def verify_fluent_with_llm(
        self,
        image1_path: Path,
        image2_path: Path,
        fluent: str,
        domain_name: str,
        temperature: float = 0.2
    ) -> str:
        """
        Use VisionModel to determine which image(s) contain the fluent.

        This is the core verification step: the LLM examines both images and
        determines the ground truth about the fluent's presence.

        Args:
            image1_path: Path to first transition's next_state image
            image2_path: Path to second transition's next_state image
            fluent: PDDL fluent to verify (e.g., "(on red blue)")
            domain_name: Planning domain name
            temperature: LLM sampling temperature (lower = more deterministic)

        Returns:
            - Binary mode: One of "IMAGE1", "IMAGE2"
            - Ternary mode: One of "IMAGE1", "IMAGE2", "UNCERTAIN"
        """
        # Step 1: Encode both images to base64
        base64_image1 = encode_image(image1_path)
        base64_image2 = encode_image(image2_path)

        # Step 2: Get the system prompt based on use_uncertain mode
        system_prompt = self._get_system_prompt(fluent, domain_name)

        # Step 3: Construct the user message with both images
        user_prompt = [
            {"type": "text", "text": "IMAGE 1:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image1}"}
            },
            {"type": "text", "text": "IMAGE 2:"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image2}"}
            },
            {"type": "text", "text": f"Which image(s) satisfy the fluent: {fluent}?"}
        ]

        # Step 4: Call Vision API
        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        )

        # Step 5: Parse the response
        response_text = response.choices[0].message.content.strip().upper()

        # Step 6: Extract the structured answer based on mode
        if self.use_uncertain:
            # Ternary mode: IMAGE1, IMAGE2, or UNCERTAIN
            if "UNCERTAIN" in response_text:
                return "UNCERTAIN"
            elif "IMAGE1" in response_text and "IMAGE2" not in response_text:
                return "IMAGE1"
            elif "IMAGE2" in response_text and "IMAGE1" not in response_text:
                return "IMAGE2"
            else:
                # Fallback: if response is unclear, default to UNCERTAIN
                print(f"Warning: Unclear LLM response: {response_text}, defaulting to UNCERTAIN")
                return "UNCERTAIN"
        else:
            # Binary mode: IMAGE1 or IMAGE2
            if "IMAGE1" in response_text and "IMAGE2" not in response_text:
                return "IMAGE1"
            elif "IMAGE2" in response_text and "IMAGE1" not in response_text:
                return "IMAGE2"
            else:
                # Fallback: if response is unclear, default to IMAGE1
                print(f"Warning: Unclear LLM response: {response_text}, defaulting to IMAGE1")
                return "IMAGE1"

    # ==================== Repair Decision Logic ====================

    def determine_repair_choice(
        self,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str
    ) -> Tuple[RepairChoice, Union[bool, None]]:
        """
        Determine which transition to repair based on LLM verification.

        Given an effects violation (two transitions with same action+prev_state but
        different next_states), this method:
        1. Asks the LLM which image(s) contain the conflicting fluent
        2. Determines which transition's classification was incorrect
        3. Decides how to repair it (add or remove the fluent, or mask it)

        Args:
            violation: The effects violation to resolve
            image1_path: Path to image of transition1's next_state
            image2_path: Path to image of transition2's next_state
            domain_name: Planning domain name

        Returns:
            Tuple of:
            - RepairChoice: Which transition to repair (FIRST, SECOND, or BOTH)
            - bool or None: Whether the fluent should be present (True/False) or masked (None)
        """
        # Step 1: Query the LLM about fluent presence
        llm_result = self.verify_fluent_with_llm(
            image1_path,
            image2_path,
            violation.conflicting_fluent,
            domain_name
        )

        # Step 2: Interpret LLM result and determine repair

        if llm_result == "UNCERTAIN":
            # LLM is uncertain: mask the fluent in both states
            return RepairChoice.BOTH, None

        elif llm_result == "IMAGE1":
            # Ground truth: fluent is present in IMAGE1
            # Therefore, transition2 should also have the fluent
            if violation.fluent_in_trans1_next and not violation.fluent_in_trans2_next:
                # Trans1 correctly has it, trans2 is missing it
                return RepairChoice.SECOND, True
            else:
                # Trans1 doesn't have it but should - repair trans1
                return RepairChoice.FIRST, True

        elif llm_result == "IMAGE2":
            # Ground truth: fluent is present in IMAGE2
            # Therefore, transition1 should also have the fluent
            if violation.fluent_in_trans2_next and not violation.fluent_in_trans1_next:
                # Trans2 correctly has it, trans1 is missing it
                return RepairChoice.FIRST, True
            else:
                # Trans2 doesn't have it but should - repair trans2
                return RepairChoice.SECOND, True

        else:
            # In binary mode, we should never get here
            # Default to repairing first transition
            print(f"Warning: Unexpected LLM result: {llm_result}")
            return RepairChoice.FIRST, True

    # ==================== State Repair ====================

    @staticmethod
    def _repair_state(
        state: State,
        fluent_str: str,
        should_be_present: Union[bool, None]
    ) -> None:
        """
        Repair a state by modifying a fluent's truth value or masking it.

        This modifies the State object in-place by:
        1. Finding the GroundedPredicate matching fluent_str
        2. Removing the old predicate
        3. Adding a corrected predicate with the right is_positive value or masked

        Args:
            state: The State object to repair (modified in-place)
            fluent_str: String representation of the fluent (e.g., "(on red blue)")
            should_be_present: True for positive, False for negative, None to mask

        Note:
            The state's internal structure organizes predicates by name in
            state.state_predicates[predicate_name], so we must:
            - Extract the predicate name from fluent_str
            - Find the matching GroundedPredicate in that set
            - Replace it with a corrected version
        """
        # Step 1: Extract predicate name from fluent string
        # Format: "predicate_name(arg1, arg2, ...)" or just "predicate_name"
        if '(' in fluent_str:
            pred_name = fluent_str[:fluent_str.index('(')]
        else:
            pred_name = fluent_str

        # Step 2: Find the predicate set for this predicate name
        if pred_name not in state.state_predicates:
            # Predicate not in state - nothing to repair
            return

        predicates_set = state.state_predicates[pred_name]

        # Step 3: Find the specific grounded predicate matching fluent_str
        target_pred = None
        for pred in predicates_set:
            if pred.untyped_representation == fluent_str:
                target_pred = pred
                break

        if not target_pred:
            # Specific grounding not found - nothing to repair
            return

        # Step 4: Remove the old predicate
        predicates_set.discard(target_pred)

        # Step 5: Create and add the corrected predicate
        if should_be_present is None:
            # Mask the fluent (uncertain case)
            corrected_pred = GroundedPredicate(
                name=target_pred.name,
                signature=target_pred.signature,
                object_mapping=target_pred.object_mapping,
                is_positive=target_pred.is_positive,  # Keep original truth value
                is_masked=True  # Mark as masked
            )
        else:
            # Set specific truth value (binary mode)
            corrected_pred = GroundedPredicate(
                name=target_pred.name,
                signature=target_pred.signature,
                object_mapping=target_pred.object_mapping,
                is_positive=should_be_present,  # The corrected truth value
                is_masked=target_pred.is_masked
            )
        predicates_set.add(corrected_pred)

    def repair_observation(
        self,
        observation: Observation,
        violation: EffectsViolation,
        repair_choice: RepairChoice,
        fluent_should_be_present: Union[bool, None]
    ) -> Tuple[Observation, List[RepairOperation]]:
        """
        Repair an observation by fixing the next_state of a chosen transition.

        This is the core repair operation that modifies the observation to
        resolve the effects violation.

        Args:
            observation: The trajectory observation (modified in-place)
            violation: The effects violation being resolved
            repair_choice: Which transition to repair (FIRST, SECOND, or BOTH)
            fluent_should_be_present: Correct truth value for the fluent, or None to mask

        Returns:
            Tuple of:
            - Observation: The repaired observation (same object, modified)
            - List[RepairOperation]: Records of the repair(s) for tracking/backtracking
        """
        repair_operations = []

        if repair_choice == RepairChoice.BOTH:
            # Mask the fluent in both transitions' next_states
            # Repair transition 1
            component1 = observation.components[violation.transition1_index]
            self._repair_state(
                component1.next_state,
                violation.conflicting_fluent,
                None  # Mask the fluent
            )
            repair_op1 = RepairOperation(
                transition_index=violation.transition1_index,
                state_type='next_state',
                fluent_changed=violation.conflicting_fluent,
                old_value=violation.fluent_in_trans1_next,
                new_value=violation.fluent_in_trans1_next  # Truth value unchanged, but masked
            )
            repair_operations.append(repair_op1)

            # Repair transition 2
            component2 = observation.components[violation.transition2_index]
            self._repair_state(
                component2.next_state,
                violation.conflicting_fluent,
                None  # Mask the fluent
            )
            repair_op2 = RepairOperation(
                transition_index=violation.transition2_index,
                state_type='next_state',
                fluent_changed=violation.conflicting_fluent,
                old_value=violation.fluent_in_trans2_next,
                new_value=violation.fluent_in_trans2_next  # Truth value unchanged, but masked
            )
            repair_operations.append(repair_op2)

        else:
            # Single transition repair (FIRST or SECOND)
            if repair_choice == RepairChoice.FIRST:
                trans_index = violation.transition1_index
                old_value = violation.fluent_in_trans1_next
            else:  # RepairChoice.SECOND
                trans_index = violation.transition2_index
                old_value = violation.fluent_in_trans2_next

            # Get the observation component for this transition
            component = observation.components[trans_index]

            # Repair the next_state by modifying the fluent
            self._repair_state(
                component.next_state,
                violation.conflicting_fluent,
                fluent_should_be_present
            )

            # Create a repair operation record for tracking
            repair_op = RepairOperation(
                transition_index=trans_index,
                state_type='next_state',
                fluent_changed=violation.conflicting_fluent,
                old_value=old_value,
                new_value=fluent_should_be_present
            )
            repair_operations.append(repair_op)

        return observation, repair_operations

    # ==================== High-Level Interface ====================

    def repair_violation(
        self,
        observation: Observation,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str = "unknown"
    ) -> Tuple[Observation, List[RepairOperation], RepairChoice]:
        """
        Complete pipeline: verify and repair an effects violation.

        This is the main public interface that combines all steps:
        1. LLM verification of which image is correct
        2. Determination of which transition to repair
        3. Application of the repair to the observation

        Args:
            observation: The trajectory observation to repair
            violation: The effects violation to resolve
            image1_path: Path to transition1's next_state image
            image2_path: Path to transition2's next_state image
            domain_name: Planning domain name

        Returns:
            Tuple of:
            - Observation: The repaired observation
            - List[RepairOperation]: Record(s) of the repair(s) performed
            - RepairChoice: Which transition(s) were repaired
        """
        # Step 1: Determine repair strategy using LLM
        repair_choice, fluent_should_be_present = self.determine_repair_choice(
            violation, image1_path, image2_path, domain_name
        )

        # Step 2: Apply the repair to the observation
        repaired_obs, repair_ops = self.repair_observation(
            observation, violation, repair_choice, fluent_should_be_present
        )

        # Step 3: Log the repair for debugging/tracking
        print(f"Repaired violation: {violation}")
        print(f"  Repair choice: {repair_choice.value}")
        for repair_op in repair_ops:
            print(f"  Repair operation: {repair_op}")

        return repaired_obs, repair_ops, repair_choice
