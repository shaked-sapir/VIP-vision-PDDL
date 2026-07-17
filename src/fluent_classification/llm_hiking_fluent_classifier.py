import itertools
from pathlib import Path

from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.llms.domains.hiking.prompts import confidence_system_prompt


class LLMHikingFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the N-puzzle domain.
    """

    def __init__(
            self,
            llm_backend: ImageLLMBackend,
            init_state_image_path: Path,
            type_to_objects: dict[str, list[str]] = None,
            temperature: float = None,
            use_uncertain: bool = True
    ):
        self.use_uncertain = use_uncertain

        super().__init__(
            llm_backend=llm_backend,
            type_to_objects=type_to_objects,
            temperature=temperature,
            init_state_image_path=init_state_image_path
        )

        # Mapping from LLM-detected object names to gym object names
        # For N-puzzle, we use tiles (t_X) and positions (p_X_Y)
        self.imaged_obj_to_gym_obj_name = {
            f"c{i}_r{j}": f"c{i}_r{j}" for i in range(0,30) for j in range(0,30) # positions up to 5x5
        }

        self.fewshot_examples = [(init_state_image_path, self.extract_predicates_from_gt_state())]

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        locations = sorted(self.type_to_objects['loc'])

        return confidence_system_prompt(locations)

    def _alter_predicate_from_llm_to_problem(self, predicate: str) -> str:
        """Alters the predicate from LLM format to the problem format."""
        return predicate.replace("isbrown", "ishill")

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the npuzzle domain.

        Returns:
            Set of all possible predicate strings for the Hanoi domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type
        locations = sorted(self.type_to_objects['loc'])

        predicates = set()

        # at(location) predicates - person is at location in the grid
        for location in locations:
            predicates.add(f"at({location}:loc)")

        # iswater(location) predicates - location is water
        for location in locations:
            predicates.add(f"iswater({location}:loc)")

        # ishill(location) predicates - location is hill
        for location in locations:
            predicates.add(f"ishill({location}:loc)")

        # isgoal(location) predicates - location is goal
        for location in locations:
            predicates.add(f"isgoal({location}:loc)")

        # adjacent(location1, location2) predicates - location1 and location2 are adjacent
        for loc1, loc2 in itertools.combinations(locations, 2):
            predicates.add(f"adjacent({loc1}:loc,{loc2}:loc)")
            predicates.add(f"adjacent({loc2}:loc,{loc1}:loc)")

        return predicates
    print("done")