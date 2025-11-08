"""
Data repairer for fixing trajectory inconsistencies using LLM verification.

This module provides functionality to repair effects violations (determinism violations)
in PDDL trajectories by using vision-language models to verify which observed state
is correct when two transitions with the same preconditions lead to different outcomes.
"""

import base64
from pathlib import Path
from typing import Tuple, Union

from openai import OpenAI
from pddl_plus_parser.models import Observation, GroundedPredicate, State

from src.plan_denoising.detectors.effects_detector import EffectsViolation
from src.plan_denoising.conflict_tree import RepairOperation, RepairChoice


class DataRepairer:
    """
    Repairs trajectory inconsistencies using LLM-based visual verification.

    When an effects violation is detected (two transitions with the same action and
    prev_state have different next_states), this class uses GPT-4 Vision to:
    1. Examine images of both conflicting next_states
    2. Determine which image correctly represents the fluent in question
    3. Repair the incorrectly classified state in the observation

    The approach assumes that visual observations are generally reliable, but
    classification errors may occur. The LLM acts as a tie-breaker to determine
    which of the two conflicting observations is correct.
    """

    def __init__(self, openai_apikey: str, model: str = "gpt-4o"):
        """
        Initialize the data repairer with LLM access.

        Args:
            openai_apikey: OpenAI API key for GPT-4 Vision access
            model: GPT-4 Vision model identifier (default: gpt-4o)
        """
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model

    # ==================== Image Processing ====================

    @staticmethod
    def _encode_image(image_path: Union[Path, str]) -> str:
        """
        Encode an image file to base64 for LLM API consumption.

        Args:
            image_path: Path to the image file

        Returns:
            Base64-encoded string representation of the image
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ==================== LLM Prompting ====================

    def _create_verification_prompt(self, fluent: str, domain_name: str) -> str:
        """
        Create a system prompt instructing the LLM to verify fluent presence.

        The prompt is designed to elicit a clear, unambiguous response about
        which image(s) satisfy the given fluent.

        Args:
            fluent: PDDL fluent to verify (e.g., "(on red blue)")
            domain_name: Name of the planning domain (for context)

        Returns:
            System prompt string for the LLM
        """
        return f"""You are a visual reasoning expert for PDDL planning domains.

You will be shown TWO images from the {domain_name} domain.

Your task: Determine which image (if any) satisfies the following fluent (predicate):

**Fluent to verify**: {fluent}

Analyze each image carefully and determine:
1. Does IMAGE 1 satisfy this fluent?
2. Does IMAGE 2 satisfy this fluent?

Respond with EXACTLY one of the following:
- "IMAGE1" if only the first image satisfies the fluent
- "IMAGE2" if only the second image satisfies the fluent
- "BOTH" if both images satisfy the fluent
- "NEITHER" if neither image satisfies the fluent

Be precise and base your answer only on clear visual evidence."""

    def verify_fluent_with_llm(
        self,
        image1_path: Path,
        image2_path: Path,
        fluent: str,
        domain_name: str = "unknown",
        temperature: float = 0.2
    ) -> str:
        """
        Use GPT-4 Vision to determine which image(s) contain the fluent.

        This is the core verification step: the LLM examines both images and
        determines the ground truth about the fluent's presence.

        Args:
            image1_path: Path to first transition's next_state image
            image2_path: Path to second transition's next_state image
            fluent: PDDL fluent to verify (e.g., "(on red blue)")
            domain_name: Planning domain name
            temperature: LLM sampling temperature (lower = more deterministic)

        Returns:
            One of: "IMAGE1", "IMAGE2", "BOTH", "NEITHER"
        """
        # Step 1: Encode both images to base64
        base64_image1 = self._encode_image(image1_path)
        base64_image2 = self._encode_image(image2_path)

        # Step 2: Create the system prompt
        system_prompt = self._create_verification_prompt(fluent, domain_name)

        # Step 3: Construct the user message with both images
        user_message = [
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

        # Step 4: Call GPT-4 Vision API
        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100
        )

        # Step 5: Parse the response
        response_text = response.choices[0].message.content.strip().upper()

        # Step 6: Extract the structured answer
        if "IMAGE1" in response_text and "IMAGE2" not in response_text:
            return "IMAGE1"
        elif "IMAGE2" in response_text and "IMAGE1" not in response_text:
            return "IMAGE2"
        elif "BOTH" in response_text:
            return "BOTH"
        elif "NEITHER" in response_text:
            return "NEITHER"
        else:
            # Fallback: if response is unclear, default to NEITHER
            print(f"Warning: Unclear LLM response: {response_text}")
            return "NEITHER"

    # ==================== Repair Decision Logic ====================

    def determine_repair_choice(
        self,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str = "unknown"
    ) -> Tuple[RepairChoice, bool]:
        """
        Determine which transition to repair based on LLM verification.

        Given an effects violation (two transitions with same action+prev_state but
        different next_states), this method:
        1. Asks the LLM which image(s) contain the conflicting fluent
        2. Determines which transition's classification was incorrect
        3. Decides how to repair it (add or remove the fluent)

        Args:
            violation: The effects violation to resolve
            image1_path: Path to image of transition1's next_state
            image2_path: Path to image of transition2's next_state
            domain_name: Planning domain name

        Returns:
            Tuple of:
            - RepairChoice: Which transition to repair (FIRST or SECOND)
            - bool: Whether the fluent should be present in the repaired state
        """
        # Step 1: Query the LLM about fluent presence
        llm_result = self.verify_fluent_with_llm(
            image1_path,
            image2_path,
            violation.conflicting_fluent,
            domain_name
        )

        # Step 2: Interpret LLM result and determine repair
        # The logic: if LLM says fluent is in IMAGE1, then transition1's
        # classification is correct, so we repair transition2 to match it.

        if llm_result == "IMAGE1":
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

        elif llm_result == "NEITHER":
            # Ground truth: fluent is absent in both images
            # Remove it from whichever state incorrectly has it
            if violation.fluent_in_trans1_next:
                # Trans1 incorrectly has the fluent
                return RepairChoice.FIRST, False
            else:
                # Trans2 incorrectly has the fluent
                return RepairChoice.SECOND, False

        else:  # llm_result == "BOTH"
            # Ground truth: fluent is present in both images
            # Add it to whichever state is missing it
            if not violation.fluent_in_trans1_next:
                # Trans1 is missing the fluent
                return RepairChoice.FIRST, True
            else:
                # Trans2 is missing the fluent
                return RepairChoice.SECOND, True

    # ==================== State Repair ====================

    @staticmethod
    def _repair_state(
        state: State,
        fluent_str: str,
        should_be_present: bool
    ) -> None:
        """
        Repair a state by modifying a fluent's truth value.

        This modifies the State object in-place by:
        1. Finding the GroundedPredicate matching fluent_str
        2. Removing the old predicate
        3. Adding a corrected predicate with the right is_positive value

        Args:
            state: The State object to repair (modified in-place)
            fluent_str: String representation of the fluent (e.g., "(on red blue)")
            should_be_present: True to make fluent positive, False for negative

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
        fluent_should_be_present: bool
    ) -> Tuple[Observation, RepairOperation]:
        """
        Repair an observation by fixing the next_state of a chosen transition.

        This is the core repair operation that modifies the observation to
        resolve the effects violation.

        Args:
            observation: The trajectory observation (modified in-place)
            violation: The effects violation being resolved
            repair_choice: Which transition to repair (FIRST or SECOND)
            fluent_should_be_present: Correct truth value for the fluent

        Returns:
            Tuple of:
            - Observation: The repaired observation (same object, modified)
            - RepairOperation: Record of the repair for tracking/backtracking
        """
        # Step 1: Determine which transition to repair
        if repair_choice == RepairChoice.FIRST:
            trans_index = violation.transition1_index
            old_value = violation.fluent_in_trans1_next
        else:  # RepairChoice.SECOND
            trans_index = violation.transition2_index
            old_value = violation.fluent_in_trans2_next

        # Step 2: Get the observation component for this transition
        component = observation.components[trans_index]

        # Step 3: Repair the next_state by modifying the fluent
        self._repair_state(
            component.next_state,
            violation.conflicting_fluent,
            fluent_should_be_present
        )

        # Step 4: Create a repair operation record for tracking
        repair_op = RepairOperation(
            transition_index=trans_index,
            state_type='next_state',
            fluent_changed=violation.conflicting_fluent,
            old_value=old_value,
            new_value=fluent_should_be_present
        )

        return observation, repair_op

    # ==================== High-Level Interface ====================

    def repair_violation(
        self,
        observation: Observation,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str = "unknown"
    ) -> Tuple[Observation, RepairOperation, RepairChoice]:
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
            - RepairOperation: Record of the repair performed
            - RepairChoice: Which transition was repaired
        """
        # Step 1: Determine repair strategy using LLM
        repair_choice, fluent_should_be_present = self.determine_repair_choice(
            violation, image1_path, image2_path, domain_name
        )

        # Step 2: Apply the repair to the observation
        repaired_obs, repair_op = self.repair_observation(
            observation, violation, repair_choice, fluent_should_be_present
        )

        # Step 3: Log the repair for debugging/tracking
        print(f"Repaired violation: {violation}")
        print(f"  Repair choice: {repair_choice.value}")
        print(f"  Repair operation: {repair_op}")

        return repaired_obs, repair_op, repair_choice
