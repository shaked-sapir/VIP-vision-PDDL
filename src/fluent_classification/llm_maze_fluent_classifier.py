from src.fluent_classification.base_fluent_classifier import PredicateTruthValue
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.maze.prompts import confidence_system_prompt


class LLMMazeFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Maze domain.
    Supports constant predicates that are always true without LLM inference.
    """

    def __init__(self, openai_apikey: str, type_to_objects: dict[str, list[str]] = None, model: str = "gpt-4o",
                 temperature: float = 1.0, use_uncertain: bool = True, const_predicates: set = None):
        self.use_uncertain = use_uncertain
        self.const_predicates = const_predicates if const_predicates is not None else set()

        super().__init__(
            openai_apikey=openai_apikey,
            type_to_objects=type_to_objects,
            model=model,
            temperature=temperature
        )

        # Mapping from LLM-detected object names to gym object names
        # For N-puzzle, we use tiles (t_X) and positions (p_X_Y)
        self.imaged_obj_to_gym_obj_name = {
            **{f"loc_{i}_{j}": f"loc-{i}-{j}" for i in range(1,20) for j in range(1,20)},
            "robot": "player-1",
            "doll": "doll"
        }

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        locations = sorted(self.type_to_objects['location'])

        return confidence_system_prompt(locations)

    def _alter_predicate_from_llm_to_problem(self, predicate: str) -> str:
        """Alters the predicate from LLM format to the problem format.
        the original domain predicates are with hypens but llm doesnt like it"""
        return predicate.replace("_", "-")

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the maze domain.

        Returns:
            Set of all possible predicate strings for the Maze domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type
        locations = sorted(self.type_to_objects['location'])

        predicates = set()

        # at(player, location) predicates - player is at location in the grid
        for location in locations:
            predicates.add(f"at(player-1:player,{location}:location)")

        # clear(location) predicates - location is clear (not occupied)
        for location in locations:
            predicates.add(f"clear({location}:location)")

        # is-goal(location) predicates - location is the goal
        for location in locations:
            predicates.add(f"is-goal({location}:location)")

        # oriented_{direction}(player) predicates - player is oriented in a direction
        directions = ['right', 'left', 'up', 'down']
        for direction in directions:
            predicates.add(f"oriented-{direction}(player-1:player)")

        # move-dir predicates are added from const_predicates (not generated here)

        return predicates

    def classify(self, image_path):
        """
        Override classify to include constant predicates as TRUE.

        Args:
            image_path: Path to the image to classify

        Returns:
            Dict mapping predicates to PredicateTruthValue
        """
        # Call parent classify to get LLM-inferred predicates
        result = super().classify(image_path)

        # Add all constant predicates as TRUE (they don't change throughout the trajectory)
        for const_pred in self.const_predicates:
            result[const_pred] = PredicateTruthValue.TRUE

        return result
