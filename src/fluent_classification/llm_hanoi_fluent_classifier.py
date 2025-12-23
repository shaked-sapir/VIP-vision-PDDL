"""
LLM-based fluent classifier for Hanoi domain.

Uses GPT-4 Vision to extract and classify predicates from Hanoi puzzle images.
"""

import itertools
from pathlib import Path

from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.llms.domains.hanoi.prompts import confidence_system_prompt


class LLMHanoiFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Hanoi domain.
    Uses GPT-4 Vision to extract predicates from images of Hanoi puzzle scenarios.
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
        # For Hanoi, we keep names simple: d1, d2, d3, peg1, peg2, peg3
        self.imaged_obj_to_gym_obj_name = {
            "d1": "d1",
            "d2": "d2",
            "d3": "d3",
            "d4": "d4",
            "d5": "d5",
            "d6": "d6",
            "d7": "d7",
            "d8": "d8",
            "d9": "d9",
            "d10": "d10",
            "peg1": "peg1",
            "peg2": "peg2",
            "peg3": "peg3"
        }

        preds = self.extract_predicates_from_gt_state()
        updated_preds = []
        for pred in preds:
            if "clear(peg" in pred:
                pred = pred.replace("clear", "clear-peg").replace("default", "peg")
            elif "clear(d" in pred:
                pred = pred.replace("clear", "clear-disc").replace("default", "disc")
            elif "smaller(peg" in pred:
                pred = pred.replace("smaller", "smaller-peg").replace("default", "peg",1).replace("default", "disc")
            elif "smaller(d" in pred:
                pred = pred.replace("smaller", "smaller-disc").replace("default", "disc")
            elif "on(" in pred and "peg" in pred:
                pred = pred.replace("on", "on-peg").replace("default", "disc", 1).replace("default", "peg")
            elif "on(" in pred and "peg" not in pred:
                pred = pred.replace("on", "on-disc").replace("default", "disc")
            updated_preds.append(f"{pred}: 2")

        self.fewshot_examples = [(init_state_image_path, updated_preds)]

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    @staticmethod
    def _get_result_regex() -> str:
        return r'\b((?:on[_-]disc|on[_-]peg|smaller[_-]disc|smaller[_-]peg|clear[_-]disc|clear[_-]peg)\([^)]*\))\s*:\s*([0-2])'

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        discs = sorted(self.type_to_objects.get('disc', ['d1', 'd2', 'd3']))
        pegs = sorted(self.type_to_objects.get('peg', ['peg1', 'peg2', 'peg3']))

        return confidence_system_prompt(discs, pegs)

    @staticmethod
    def _alter_predicate_from_llm_to_problem(predicate: str) -> str:
        return predicate.replace('_', '-')

    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Generates all possible predicates for the Hanoi domain.

        Returns:
            Set of all possible predicate strings for the Hanoi domain.
        """
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        # Extract objects by type
        discs = self.type_to_objects.get('disc', ['d1', 'd2', 'd3'])
        pegs = self.type_to_objects.get('peg', ['peg1', 'peg2', 'peg3'])

        predicates = set()

        # on(disc, disc) predicates - disc x is on disc y
        for disc1, disc2 in itertools.permutations(discs, 2):
            predicates.add(f"on-disc({disc1}:disc,{disc2}:disc)")

        # on(disc, peg) predicates - disc is on peg (at the bottom)
        for disc in discs:
            for peg in pegs:
                predicates.add(f"on-peg({disc}:disc,{peg}:peg)")

        # clear(disc) predicates - no disc on top
        for disc in discs:
            predicates.add(f"clear-disc({disc}:disc)")

        # clear(peg) predicates - peg has no discs
        for peg in pegs:
            predicates.add(f"clear-peg({peg}:peg)")

        # smaller(disc, disc) predicates - static size relationships
        for disc1, disc2 in itertools.permutations(discs, 2):
            predicates.add(f"smaller-disc({disc1}:disc,{disc2}:disc)")

        # smaller(peg, disc) predicates - pegs are larger than all discs (always true)
        for peg in pegs:
            for disc in discs:
                predicates.add(f"smaller-peg({peg}:peg,{disc}:disc)")

        return predicates


if __name__ == "__main__":
    type_to_objects = {
        "disc": ["d1", "d2", "d3", "d4", "d5"],
        "peg": ["peg1", "peg2", "peg3"],
    }
    init_image = Path("/Users/shakedsapir/Documents/BGU/thesis/VIP-vision-PDDL/benchmark/data/hanoi/multi_problem_06-12-2025T13:58:24__model=gpt-5.1__steps=100__planner/training/trajectories/problem5/state_0000.png")

    openai_backend = OpenAIImageLLMBackend(
        api_key="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",  # Replace with your actual OpenAI API key,
        model="gpt-5.1",
    )
    hanoi_openai = LLMHanoiFluentClassifier(
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
    hanoi_gemini = LLMHanoiFluentClassifier(
        llm_backend=gemini_backend,
        init_state_image_path=init_image,
        type_to_objects=type_to_objects,
        temperature=0.0,
    )

    # Both expose the same API:
    preds_gemini = hanoi_gemini.classify(init_image.parent / "state_0001.png")
    # preds_openai = hanoi_openai.classify(init_image.parent / "state_0001.png")
    print("done")
