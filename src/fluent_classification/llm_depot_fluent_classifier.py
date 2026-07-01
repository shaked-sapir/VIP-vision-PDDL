"""LLM-based fluent classifier for the Depot domain."""

import itertools
from pathlib import Path

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier
from src.llms.domains.depot.prompts import confidence_system_prompt


class LLMDepotFluentClassifier(LLMFluentClassifier):
    """LLM-based fluent classifier for the Depot domain.

    Uses a VLM to extract predicates from images of depot scenarios.
    Depot uses identity name mapping — image labels match PDDL names directly.
    """

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

        # Identity mapping — depot image labels match PDDL object names.
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
        """Returns the system prompt for the Depot domain."""
        assert self.type_to_objects is not None, (
            "type_to_objects must be set before getting system prompt."
        )

        return confidence_system_prompt(
            depots=sorted(self.type_to_objects.get("depot", [])),
            trucks=sorted(self.type_to_objects.get("truck", [])),
            cranes=sorted(self.type_to_objects.get("crane", [])),
            piles=sorted(self.type_to_objects.get("pile", [])),
            packages=sorted(self.type_to_objects.get("package", [])),
        )

    @staticmethod
    def _get_result_regex() -> str:
        """Returns regex for depot fluent classification output.

        Matches predicates like: at-truck(t1:truck,D1:depot): 2
        Depot labels include uppercase letters and digits.
        """
        return r'([a-zA-Z_-]+\([a-zA-Z0-9_:,]+\))\s*:\s*([0-2])'

    @staticmethod
    def _alter_predicate_from_llm_to_problem(predicate: str) -> str:
        """Normalize depot names from uppercase (D1) to lowercase (d1).

        The LLM sees image labels like "D1:depot" but PDDL uses "d1:depot".
        """
        import re
        return re.sub(r'\bD(\d+)', lambda m: f'd{m.group(1)}', predicate)

    def _generate_all_possible_predicates(self) -> set[str]:
        """Generate all possible grounded predicates for the Depot domain.

        Predicate forms:
            at-truck(t:truck,d:depot)
            at-crane(c:crane,d:depot)
            at-pile(pl:pile,d:depot)
            at(p:package,d:depot)
            on(p:package,q:package)          — excluding self-relations
            on-pile(p:package,pl:pile)
            clear(x:package) / clear(x:pile)
            holding(c:crane,p:package)
            empty-crane(c:crane)
            in-truck(p:package,t:truck)
        """
        assert self.type_to_objects is not None, (
            "type_to_objects must be set before generating predicates."
        )

        depots = self.type_to_objects.get("depot", [])
        trucks = self.type_to_objects.get("truck", [])
        cranes = self.type_to_objects.get("crane", [])
        piles = self.type_to_objects.get("pile", [])
        packages = self.type_to_objects.get("package", [])

        predicates: set[str] = set()

        # at-truck(t:truck, d:depot)
        for t in trucks:
            for d in depots:
                predicates.add(f"at-truck({t}:truck,{d}:depot)")

        # at-crane(c:crane, d:depot)
        for c in cranes:
            for d in depots:
                predicates.add(f"at-crane({c}:crane,{d}:depot)")

        # at-pile(pl:pile, d:depot)
        for pl in piles:
            for d in depots:
                predicates.add(f"at-pile({pl}:pile,{d}:depot)")

        # at(p:package, d:depot)
        for p in packages:
            for d in depots:
                predicates.add(f"at({p}:package,{d}:depot)")

        # on(p:package, q:package) — excluding self-relations
        for p, q in itertools.permutations(packages, 2):
            predicates.add(f"on({p}:package,{q}:package)")

        # on-pile(p:package, pl:pile)
        for p in packages:
            for pl in piles:
                predicates.add(f"on-pile({p}:package,{pl}:pile)")

        # clear(x) — packages and piles can be clear
        for p in packages:
            predicates.add(f"clear({p}:package)")
        for pl in piles:
            predicates.add(f"clear({pl}:pile)")

        # holding(c:crane, p:package)
        for c in cranes:
            for p in packages:
                predicates.add(f"holding({c}:crane,{p}:package)")

        # empty-crane(c:crane)
        for c in cranes:
            predicates.add(f"empty-crane({c}:crane)")

        # in-truck(p:package, t:truck)
        for p in packages:
            for t in trucks:
                predicates.add(f"in-truck({p}:package,{t}:truck)")

        return predicates
