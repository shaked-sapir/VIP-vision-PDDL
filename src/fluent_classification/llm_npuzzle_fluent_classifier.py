import itertools

from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.n_puzzle.prompts import confidence_system_prompt


class LLMNpuzzleFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the N-puzzle domain.
    """

    def __init__(self, openai_apikey: str, type_to_objects: dict[str, list[str]] = None, model: str = "gpt-4o",
                 temperature: float = 1.0, use_uncertain: bool = True):
        self.use_uncertain = use_uncertain

        super().__init__(
            openai_apikey=openai_apikey,
            type_to_objects=type_to_objects,
            model=model,
            temperature=temperature
        )

        # Mapping from LLM-detected object names to gym object names
        # For N-puzzle, we use tiles (t_X) and positions (p_X_Y)
        self.imaged_obj_to_gym_obj_name = {
            **{f"t_{i}": f"t_{i}" for i in range(1, 25)},
            **{f"p_{i}_{j}": f"p_{i}_{j}" for i in range(1,6) for j in range(1,6)} # positions up to 5x5
        }

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        tiles = sorted(self.type_to_objects['tile'])
        positions = sorted(self.type_to_objects['position'])

        return confidence_system_prompt(tiles, positions)

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the npuzzle domain.

        Returns:
            Set of all possible predicate strings for the Hanoi domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type
        tiles = sorted(self.type_to_objects['tile'])
        positions = sorted(self.type_to_objects['position'])

        predicates = set()

        # at(tile, position) predicates - tile is at position in the grid
        for tile in tiles:
            for position in positions:
                predicates.add(f"at({tile}:tile,{position}:position)")

        # empty(position) predicates - position is the blank position
        for position in positions:
            predicates.add(f"empty({position}:position)")

        # neighbor(position1, position2) predicates - position1 and position2 share an edge
        for pos1, pos2 in itertools.combinations(positions, 2):
            predicates.add(f"neighbor({pos1}:position,{pos2}:position)")
            predicates.add(f"neighbor({pos2}:position,{pos1}:position)")

        return predicates
