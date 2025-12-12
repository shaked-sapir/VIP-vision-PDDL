import itertools
import re
from pathlib import Path

from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.llms.domains.n_puzzle.prompts import confidence_system_prompt


class LLMNpuzzleFluentClassifier(LLMFluentClassifier):
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
            **{f"t_{i}": f"t_{i}" for i in range(1, 25)},
            **{f"p_{i}_{j}": f"p_{i}_{j}" for i in range(1,6) for j in range(1,6)} # positions up to 5x5
        }

        new_preds, max_idx = [], 0

        for p in self.extract_predicates_from_gt_state():
            if p.startswith("position("):
                for n in re.findall(r"(\d+):default", p):
                    max_idx = max(max_idx, int(n))
            elif p.startswith("at("):
                t, x, y = re.findall(r"(\d+):default", p)
                new_preds.append(f"at(t_{t}:tile,p_{x}_{y}:position)")
            elif p.startswith("blank("):
                x, y = re.findall(r"(\d+):default", p)
                new_preds.append(f"empty(p_{x}_{y})")

        neighbors = set()
        for i in range(1, max_idx + 1):
            for j in range(1, max_idx + 1):
                for di, dj in ((1, 0), (0, 1)):  # right & down generate all undirected pairs
                    ni, nj = i + di, j + dj
                    if 1 <= ni <= max_idx and 1 <= nj <= max_idx:
                        neighbors.add(f"neighbor(p_{i}_{j}:position,p_{ni}_{nj}:position)")
                        neighbors.add(f"neighbor(p_{ni}_{nj}:position,p_{i}_{j}:position)")

        new_preds.extend(sorted(neighbors))

        self.fewshot_examples = [(init_state_image_path, new_preds)]

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

if __name__ == "__main__":
    type_to_objects = {
        "tile": [f"t_{i}" for i in range(1, 25)],
        "position": [f"p_{i}_{j}" for i in range(1, 6) for j in range(1, 6)],
    }
    init_image = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/n_puzzle_typed/multi_problem_06-12-2025T13:32:59__model=gpt-5.1__steps=100__planner/training/trajectories/problem8/state_0000.png")

    openai_backend = OpenAIImageLLMBackend(
        api_key="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",  # Replace with your actual OpenAI API key,
        model="gpt-5.1",
    )
    hanoi_openai = LLMNpuzzleFluentClassifier(
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
    hanoi_gemini = LLMNpuzzleFluentClassifier(
        llm_backend=gemini_backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
        temperature=0.0,
    )

    # Both expose the same API:
    preds_gemini = hanoi_gemini.classify(init_image.parent / "state_0001.png")
    preds_openai = hanoi_openai.classify(init_image.parent / "state_0001.png")
    print("done")