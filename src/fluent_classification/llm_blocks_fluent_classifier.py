import itertools
from pathlib import Path

from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.llms.domains.blocks.prompts import confidence_system_prompt


class LLMBlocksFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Blocks domain.
    Uses VisionModel to extract predicates from images of blocksworld world scenarios.
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

        self.imaged_obj_to_gym_obj_name = {
            "red": "a",
            "cyan": "b",
            "blue": "c",
            "green": "d",
            "yellow": "e",
            "pink": "f",
            "gripper": "robot"
        }

        preds = self.extract_predicates_from_gt_state()
        preds = [
            ("handempty()" if "handempty" in p else p)
            for p in preds
            if "handfull" not in p
        ]
        self.fewshot_examples = [(init_state_image_path, preds)]

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

        return confidence_system_prompt(self.type_to_objects['block'])

    @staticmethod
    def _alter_predicate_from_llm_to_problem(predicate: str) -> str:
        """
        Alters a predicate string from LLM format to problem format.

        Args:
            predicate: Predicate string in LLM format.

        Returns:
            Predicate string in problem format.
        """
        # In this case, no alteration is needed; return as is
        return predicate if "handempty" not in predicate else "handempty()"

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the blocksworld domain.

        Returns:
            Set of all possible predicate strings for the blocksworld domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type with defaults
        blocks = self.type_to_objects.get('block', ['red', 'cyan', 'blue', 'green'])
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
        predicates.add(f"handempty()")

        # holding(block) predicates
        for block in blocks:
            predicates.add(f"holding({block}:block)")

        return predicates

if __name__ == "__main__":
    type_to_objects = {
        "block": ["red", "cyan", "blue", "green", "yellow", "pink"],
    }
    init_image = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/blocksworld/multi_problem_04-12-2025T12:00:44__model=gpt-5.1__steps=50__planner/training/trajectories/problem7/state_0005.png")

    openai_backend = OpenAIImageLLMBackend(
        api_key="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",  # Replace with your actual OpenAI API key,
        model="gpt-5.1",
    )
    hanoi_openai = LLMBlocksFluentClassifier(
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
    hanoi_gemini = LLMBlocksFluentClassifier(
        llm_backend=gemini_backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
        temperature=0.0,
    )

    # Both expose the same API:
    preds_gemini = hanoi_gemini.classify(init_image.parent / "state_0006.png")
    preds_openai = hanoi_openai.classify(init_image.parent / "state_0006.png")
    print("done")