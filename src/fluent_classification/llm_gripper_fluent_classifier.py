"""LLM-based fluent classifier for the Gripper domain."""

import itertools
import re
from pathlib import Path

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.gripper.prompts import confidence_system_prompt


class LLMGripperFluentClassifier(LLMFluentClassifier):
    """LLM-based fluent classifier for the Gripper domain.

    Uses a VLM to extract predicates from images of gripper scenarios.

    Name mappings from image labels to PDDL names:
    - Room labels: "ROOMA" → "rooma" (lowercase)
    - Gripper labels: "R" → "right", "L" → "left"
    - Ball numbers: already "ball1", "ball2" in the prompt output
    """

    # Map image gripper labels to PDDL names.
    GRIPPER_NAME_MAP = {"R": "right", "L": "left"}

    def __init__(
        self,
        llm_backend: ImageLLMBackend,
        init_state_image_path: Path,
        type_to_objects: dict[str, list[str]] = None,
        temperature: float = None,
        use_uncertain: bool = True,
    ):
        self.use_uncertain = use_uncertain

        super().__init__(
            llm_backend=llm_backend,
            type_to_objects=type_to_objects,
            temperature=temperature,
            init_state_image_path=init_state_image_path,
        )

        # Identity mapping — gripper prompt instructs the LLM to use PDDL names.
        self.imaged_obj_to_gym_obj_name = {}

        preds = self.extract_predicates_from_gt_state()
        self.fewshot_examples = [(init_state_image_path, preds)]

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def set_use_uncertain(self, use_uncertain: bool) -> None:
        """Set whether to allow uncertain predictions."""
        self.use_uncertain = use_uncertain
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Gripper domain."""
        assert self.type_to_objects is not None, (
            "type_to_objects must be set before getting system prompt."
        )

        return confidence_system_prompt(
            ball_ids=sorted(self.type_to_objects.get("ball", [])),
            rooms=sorted(self.type_to_objects.get("room", [])),
            grippers=sorted(self.type_to_objects.get("gripper", [])),
        )

    @staticmethod
    def _get_result_regex() -> str:
        """Returns regex for gripper fluent classification output.

        Matches predicates like: at-robby(rooma:room): 2, carry(ball1:ball,left:gripper): 1
        """
        return r'([a-zA-Z_-]+\([a-zA-Z0-9_:,]+\))\s*:\s*([0-2])'

    @staticmethod
    def _alter_predicate_from_llm_to_problem(predicate: str) -> str:
        """Normalize names from image labels to PDDL names.

        - Room names: ROOMA → rooma, ROOMB → roomb (uppercase to lowercase)
        - Gripper names: R → right, L → left (if the LLM outputs short names)

        The prompt instructs the LLM to use full PDDL names, but this handles
        edge cases where the LLM falls back to image labels.
        """
        # Lowercase room names (e.g., ROOMA:room → rooma:room)
        predicate = re.sub(
            r'\b(ROOM[A-Z]):room\b',
            lambda m: f'{m.group(1).lower()}:room',
            predicate,
        )

        return predicate

    def _generate_all_possible_predicates(self) -> set[str]:
        """Generate all possible grounded predicates for the Gripper domain.

        Predicate forms:
            at-robby(r:room)
            at(b:ball, r:room)
            free(g:gripper)
            carry(b:ball, g:gripper)
        """
        assert self.type_to_objects is not None, (
            "type_to_objects must be set before generating predicates."
        )

        rooms = self.type_to_objects.get("room", [])
        balls = self.type_to_objects.get("ball", [])
        grippers = self.type_to_objects.get("gripper", [])

        predicates: set[str] = set()

        # at-robby(r:room)
        for r in rooms:
            predicates.add(f"at-robby({r}:room)")

        # at(b:ball, r:room)
        for b in balls:
            for r in rooms:
                predicates.add(f"at({b}:ball,{r}:room)")

        # free(g:gripper)
        for g in grippers:
            predicates.add(f"free({g}:gripper)")

        # carry(b:ball, g:gripper)
        for b in balls:
            for g in grippers:
                predicates.add(f"carry({b}:ball,{g}:gripper)")

        return predicates
