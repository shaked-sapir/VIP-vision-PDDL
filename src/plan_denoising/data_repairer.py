"""Data repairer for fixing inconsistencies using LLM verification."""

import base64
from pathlib import Path
from typing import Tuple, Union

from openai import OpenAI
from pddl_plus_parser.models import Observation, GroundedPredicate, State

from src.plan_denoising.detectors.effects_detector import EffectsViolation
from src.plan_denoising.conflict_tree import RepairOperation, RepairChoice


class DataRepairer:
    """
    Uses LLM to determine which state has the correct fluent value and repairs the trajectory.

    Given an inconsistency between two transitions, the repairer:
    1. Takes images of both next_states
    2. Asks an LLM which image contains the conflicting fluent
    3. Fixes the wrongly classified state in the observation
    """

    def __init__(self, openai_apikey: str, model: str = "gpt-4o"):
        """
        Initialize the data repairer.

        :param openai_apikey: OpenAI API key for GPT-4 Vision
        :param model: GPT-4 Vision model to use
        """
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model

    @staticmethod
    def _encode_image(image_path: Union[Path, str]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _create_verification_prompt(self, fluent: str, domain_name: str) -> str:
        """
        Create a prompt for the LLM to verify which image contains the fluent.

        :param fluent: The fluent to verify (e.g., "on(block1, block2)")
        :param domain_name: Name of the domain (for context)
        :return: System prompt for the LLM
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
        Use LLM to determine which image contains the fluent.

        :param image1_path: Path to first image
        :param image2_path: Path to second image
        :param fluent: Fluent to verify
        :param domain_name: Name of the domain
        :param temperature: Temperature for LLM generation (lower = more deterministic)
        :return: One of "IMAGE1", "IMAGE2", "BOTH", "NEITHER"
        """
        # Encode both images
        base64_image1 = self._encode_image(image1_path)
        base64_image2 = self._encode_image(image2_path)

        # Create system prompt
        system_prompt = self._create_verification_prompt(fluent, domain_name)

        # Create user message with both images
        user_message = [
            {
                "type": "text",
                "text": "IMAGE 1:"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image1}"}
            },
            {
                "type": "text",
                "text": "IMAGE 2:"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image2}"}
            },
            {
                "type": "text",
                "text": f"Which image(s) satisfy the fluent: {fluent}?"
            }
        ]

        # Call GPT-4 Vision
        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=100
        )

        response_text = response.choices[0].message.content.strip().upper()

        # Extract the answer
        if "IMAGE1" in response_text and "IMAGE2" not in response_text:
            return "IMAGE1"
        elif "IMAGE2" in response_text and "IMAGE1" not in response_text:
            return "IMAGE2"
        elif "BOTH" in response_text:
            return "BOTH"
        elif "NEITHER" in response_text:
            return "NEITHER"
        else:
            # Default to NEITHER if unclear
            print(f"Warning: Unclear LLM response: {response_text}")
            return "NEITHER"

    def determine_repair_choice(
        self,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str = "unknown"
    ) -> Tuple[RepairChoice, bool]:
        """
        Determine which transition should be repaired based on LLM verification.

        :param violation: The inconsistency to resolve
        :param image1_path: Path to image of transition1's next_state
        :param image2_path: Path to image of transition2's next_state
        :param domain_name: Name of the domain
        :return: Tuple of (RepairChoice, fluent_should_be_present)
                 - RepairChoice: which transition to repair (FIRST or SECOND)
                 - fluent_should_be_present: whether fluent should be present
        """
        # Ask LLM which image contains the fluent
        llm_result = self.verify_fluent_with_llm(
            image1_path, image2_path,
            violation.conflicting_fluent,
            domain_name
        )

        # Determine which transition is wrong based on LLM result
        if llm_result == "IMAGE1":
            # IMAGE1 has the fluent, IMAGE2 should also have it (repair trans2)
            if violation.fluent_in_trans1_next and not violation.fluent_in_trans2_next:
                return RepairChoice.SECOND, True
            else:
                # Unexpected: trans1 doesn't have it but LLM says it should
                return RepairChoice.FIRST, True

        elif llm_result == "IMAGE2":
            # IMAGE2 has the fluent, IMAGE1 should also have it (repair trans1)
            if violation.fluent_in_trans2_next and not violation.fluent_in_trans1_next:
                return RepairChoice.FIRST, True
            else:
                # Unexpected: trans2 doesn't have it but LLM says it should
                return RepairChoice.SECOND, True

        elif llm_result == "NEITHER":
            # Neither has the fluent - repair whichever claims to have it
            if violation.fluent_in_trans1_next:
                return RepairChoice.FIRST, False
            else:
                return RepairChoice.SECOND, False

        else:  # "BOTH"
            # Both should have the fluent - repair whichever doesn't
            if not violation.fluent_in_trans1_next:
                return RepairChoice.FIRST, True
            else:
                return RepairChoice.SECOND, True

    def repair_observation(
        self,
        observation: Observation,
        violation: EffectsViolation,
        repair_choice: RepairChoice,
        fluent_should_be_present: bool
    ) -> Tuple[Observation, RepairOperation]:
        """
        Repair the observation by fixing the next_state of the chosen transition.

        :param observation: The observation to repair (will be modified in-place)
        :param violation: The inconsistency being resolved
        :param repair_choice: Which transition to repair
        :param fluent_should_be_present: Whether the fluent should be present
        :return: Tuple of (repaired observation, repair operation)
        """
        # Determine which transition to repair
        if repair_choice == RepairChoice.FIRST:
            trans_index = violation.transition1_index
            old_value = violation.fluent_in_trans1_next
        else:
            trans_index = violation.transition2_index
            old_value = violation.fluent_in_trans2_next

        # Get the component to repair
        component = observation.components[trans_index]

        # Repair the next_state
        self._repair_state(
            component.next_state,
            violation.conflicting_fluent,
            fluent_should_be_present
        )

        # Create repair operation record
        repair_op = RepairOperation(
            transition_index=trans_index,
            state_type='next_state',
            fluent_changed=violation.conflicting_fluent,
            old_value=old_value,
            new_value=fluent_should_be_present
        )

        return observation, repair_op

    @staticmethod
    def _repair_state(
        state: State,
        fluent_str: str,
        should_be_present: bool
    ) -> None:
        """
        Repair a state by adding or removing a fluent.

        :param state: The state to repair (modified in-place)
        :param fluent_str: String representation of the fluent
        :param should_be_present: Whether fluent should be present (True) or absent (False)
        """
        # Find the predicate in the state
        # Parse fluent_str to extract predicate name
        # Format: "predicate_name(arg1, arg2, ...)"
        if '(' in fluent_str:
            pred_name = fluent_str[:fluent_str.index('(')]
        else:
            pred_name = fluent_str

        # Find matching grounded predicate in state
        if pred_name in state.state_predicates:
            predicates_set = state.state_predicates[pred_name]

            # Find the specific grounded predicate matching fluent_str
            target_pred = None
            for pred in predicates_set:
                if pred.untyped_representation == fluent_str:
                    target_pred = pred
                    break

            if target_pred:
                # Remove old predicate
                predicates_set.discard(target_pred)

                # Add corrected predicate
                corrected_pred = GroundedPredicate(
                    name=target_pred.name,
                    signature=target_pred.signature,
                    object_mapping=target_pred.object_mapping,
                    is_positive=should_be_present,
                    is_masked=target_pred.is_masked
                )
                predicates_set.add(corrected_pred)

    def repair_inconsistency(
        self,
        observation: Observation,
        violation: EffectsViolation,
        image1_path: Path,
        image2_path: Path,
        domain_name: str = "unknown"
    ) -> Tuple[Observation, RepairOperation, RepairChoice]:
        """
        Main method: Repair an inconsistency using LLM verification.

        :param observation: The observation to repair
        :param violation: The inconsistency to resolve
        :param image1_path: Path to image of transition1's next_state
        :param image2_path: Path to image of transition2's next_state
        :param domain_name: Name of the domain
        :return: Tuple of (repaired observation, repair operation, repair choice)
        """
        # Determine which transition to repair
        repair_choice, fluent_should_be_present = self.determine_repair_choice(
            inconsistency, image1_path, image2_path, domain_name
        )

        # Repair the observation
        repaired_obs, repair_op = self.repair_observation(
            observation, inconsistency, repair_choice, fluent_should_be_present
        )

        print(f"Repaired inconsistency: {inconsistency}")
        print(f"  Choice: {repair_choice.value}")
        print(f"  Operation: {repair_op}")

        return repaired_obs, repair_op, repair_choice
