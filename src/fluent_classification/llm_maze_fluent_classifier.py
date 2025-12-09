from pathlib import Path

from src.fluent_classification.base_fluent_classifier import PredicateTruthValue
from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.llms.domains.maze.prompts import confidence_system_prompt


class LLMMazeFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Maze domain.
    Supports constant predicates that are always true without LLM inference.
    """

    def __init__(
            self,
            llm_backend: ImageLLMBackend,
            init_state_image_path: Path,
            type_to_objects: dict[str, list[str]] = None,
            temperature: float = 1.0,
            use_uncertain: bool = True
    ):
        self.use_uncertain = use_uncertain

        super().__init__(
            llm_backend=llm_backend,
            type_to_objects=type_to_objects,
            temperature=temperature,
            init_state_image_path=init_state_image_path
        )
        self.use_uncertain = use_uncertain

        # Mapping from LLM-detected object names to gym object names
        # For N-puzzle, we use tiles (t_X) and positions (p_X_Y)
        self.imaged_obj_to_gym_obj_name = {
            **{f"loc_{i}_{j}": f"loc-{i}-{j}" for i in range(1,20) for j in range(1,20)},
            "robot": "player-1",
            "doll": "doll"
        }

        gt_preds = self.extract_predicates_from_gt_state()
        fewshot_preds = sorted([f"{pred}: 2".replace("player", "robot") for pred in gt_preds])

        self.fewshot_examples = [(init_state_image_path, fewshot_preds)]

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        locations = sorted(self.type_to_objects['location'])

        return confidence_system_prompt(locations)

    @classmethod
    def _get_user_instruction(cls) -> str:
        base_instruction = super(LLMMazeFluentClassifier, cls)._get_user_instruction()
        return f"{base_instruction}\nPay carefull attention to the blue robot location and to the bear location."

    def _alter_predicate_from_llm_to_problem(self, predicate: str) -> str:
        """Alters the predicate from LLM format to the problem format.
        the original domain predicates are with hypens but llm doesnt like it"""
        return predicate.replace("_", "-").replace("robot:robot", "player-1:player")

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract predicates from LLM response."""
        # Pattern to match predicate with confidence score
        # Examples: ('move-dir-up(loc-1-3,loc-3-2)', '2')
        return r'([a-zA-Z0-9_-]+\([^)]*\)):\s*([0-2])'  # relevance is int 0,1,2

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

        for location1, location2 in [(l1, l2) for l1 in locations for l2 in locations if l1 != l2]:
            for dir in directions:
                predicates.add(f"move-dir-{dir}({location1}:location,{location2}:location)")

        # move-dir predicates are added from const_predicates (not generated here)

        return predicates


if __name__ == "__main__":
    type_to_objects = {
            "location":[f"loc_{i}_{j}" for i in range(1,10) for j in range(1,10)],
            "robot": ["robot"],
            "doll": ["doll"]
        }
    init_image = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/maze/experiment_07-12-2025T16:16:54__model=gpt-5.1__steps=100__planner/training/trajectories/problem1/state_0000.png")

    openai_backend = OpenAIImageLLMBackend(
        api_key="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",  # Replace with your actual OpenAI API key,
        model="gpt-5.1",
    )
    hanoi_openai = LLMMazeFluentClassifier(
        llm_backend=openai_backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
        temperature=0.0,
    )

    # Gemini version
    gemini_backend = GeminiImageLLMBackend(
        api_key="AIzaSyANQZLrjfLEQqp5gC-Ip7-sLwbP9OR46xs",
        model="gemini-2.5-flash",
    )
    hanoi_gemini = LLMMazeFluentClassifier(
        llm_backend=gemini_backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
        temperature=0.0,
    )

    # Both expose the same API:
    preds_gemini = hanoi_gemini.classify(init_image.parent / "state_0010.png")
    preds_openai = hanoi_openai.classify(init_image.parent / "state_0010.png")
    print("done")