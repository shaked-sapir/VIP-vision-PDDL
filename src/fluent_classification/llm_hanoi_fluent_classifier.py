"""
LLM-based fluent classifier for Hanoi domain.

Uses GPT-4 Vision to extract and classify predicates from Hanoi puzzle images.
"""

import itertools

from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.hanoi.prompts import confidence_system_prompt


class LLMHanoiFluentClassifier(LLMFluentClassifier):
    """
    LLM-based fluent classifier for the Hanoi domain.
    Uses GPT-4 Vision to extract predicates from images of Hanoi puzzle scenarios.
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
        # For Hanoi, we keep names simple: d1, d2, d3, peg1, peg2, peg3
        self.imaged_obj_to_gym_obj_name = {
            "d1": "d1",
            "d2": "d2",
            "d3": "d3",
            "d4": "d4",
            "d5": "d5",
            "d6": "d6",
            "peg1": "peg1",
            "peg2": "peg2",
            "peg3": "peg3"
        }

    def set_type_to_objects(self, type_to_objects: dict[str, list[str]]) -> None:
        """Sets the type_to_objects mapping and regenerates possible predicates."""
        self.type_to_objects = type_to_objects

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for the Hanoi domain."""
        assert self.type_to_objects is not None, "type_to_objects must be set before getting system prompt."

        discs = self.type_to_objects.get('disc', ['d1', 'd2', 'd3'])
        pegs = self.type_to_objects.get('peg', ['peg1', 'peg2', 'peg3'])

        return confidence_system_prompt(discs, pegs)

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
            predicates.add(f"on({disc1}:disc,{disc2}:disc)")

        # on(disc, peg) predicates - disc is on peg (at the bottom)
        for disc in discs:
            for peg in pegs:
                predicates.add(f"on({disc}:disc,{peg}:peg)")

        # clear(disc) predicates - no disc on top
        for disc in discs:
            predicates.add(f"clear({disc}:disc)")

        # clear(peg) predicates - peg has no discs
        for peg in pegs:
            predicates.add(f"clear({peg}:peg)")

        # smaller(disc, disc) predicates - static size relationships
        for disc1, disc2 in itertools.permutations(discs, 2):
            predicates.add(f"smaller({disc1}:disc,{disc2}:disc)")

        # smaller(peg, disc) predicates - pegs are larger than all discs (always true)
        for peg in pegs:
            for disc in discs:
                predicates.add(f"smaller({peg}:peg,{disc}:disc)")

        return predicates
