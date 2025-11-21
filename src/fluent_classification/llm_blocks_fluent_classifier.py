import itertools

from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.blocks.prompts import with_uncertain_confidence_system_prompt, no_uncertain_confidence_system_prompt


class LLMBlocksFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Blocks domain.
    Uses VisionModel to extract predicates from images of blocks world scenarios.
    """

    def __init__(self, openai_apikey: str, type_to_objects: dict[str, list[str]] = None,
                 model: str = "gpt-4o", use_uncertain: bool = True):
        """
        Initialize the blocks fluent classifier.

        :param openai_apikey: OpenAI API key
        :param type_to_objects: Mapping of object types to object names
        :param model: model to use
        :param use_uncertain: If True, allows score 1 (uncertain); if False, only 0 or 2
        """
        self.use_uncertain = use_uncertain

        super().__init__(
            openai_apikey=openai_apikey,
            type_to_objects=type_to_objects,
            model=model
        )

        self.imaged_obj_to_gym_obj_name = {
            "red": "a",
            "cyan": "b",
            "blue": "c",
            "green": "d",
            "yellow": "e",
            "pink": "f",
            "gripper": "robot"
        }

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def set_use_uncertain(self, use_uncertain: bool) -> None:
        """
        Set whether to allow uncertain predictions.

        :param use_uncertain: If True, allows score 1 (uncertain); if False, only 0 or 2
        """
        self.use_uncertain = use_uncertain
        # Update the system prompt
        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Blocks domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        if self.use_uncertain:
            return with_uncertain_confidence_system_prompt(
                self.type_to_objects['block'])
        else:
            return no_uncertain_confidence_system_prompt(
                self.type_to_objects['block'])

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the blocks domain.

        Returns:
            Set of all possible predicate strings for the blocks domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type with defaults
        blocks = self.type_to_objects.get('block', ['red', 'cyan', 'blue', 'green'])  # Default blocks from problem1.pddl
        # Use the first robot if multiple are provided
        gripper_name = 'gripper'

        predicates = set()

        # on(block1, block2) predicates
        for block1, block2 in itertools.permutations(blocks, 2):
            predicates.add(f"on({block1}:block,{block2}:block)")

        # ontable(block) predicates
        for block in blocks:
            predicates.add(f"ontable({block}:block)")

        # clear(block) predicates
        for block in blocks:
            predicates.add(f"clear({block}:block)")

        # handempty(robot) predicate
        predicates.add(f"handempty({gripper_name}:gripper)")

        # handfull(robot) predicate
        predicates.add(f"handfull({gripper_name}:gripper)")

        # holding(block) predicates
        for block in blocks:
            predicates.add(f"holding({block}:block)")

        return predicates
