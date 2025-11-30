import base64
import re
from abc import abstractmethod, ABC
from collections import Counter
from pathlib import Path
from typing import Dict, Union

from openai import OpenAI

from src.fluent_classification.base_fluent_classifier import FluentClassifier, PredicateTruthValue
from src.utils.pddl import multi_replace_predicate
from src.utils.visualize import encode_image_to_base64


class LLMFluentClassifier(FluentClassifier, ABC):
    system_prompt: str

    def __init__(self, openai_apikey: str, type_to_objects: dict[str, list[str]], model: str, temperature: float = 1.0):
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model
        self.type_to_objects = type_to_objects  # dict of type -> list of object names
        self.temperature = temperature

        # Mapping from objects detected by LLM to their gym instances for predicate back-translation. define in subclass
        self.imaged_obj_to_gym_obj_name = {}
        self.uncertainty_label = 1

        # initiate LLM related attributes
        self.system_prompt = self._get_system_prompt()
        self.result_regex = self._get_result_regex()
        self.llm_result_parse_func = self._parse_llm_predicate_relevance

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for fluent classification, depending on domain."""
        raise NotImplementedError

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract predicates from LLM response."""
        # Pattern to match predicate with confidence score
        # Examples: "on(red,blue) 0.9", "ontable(green) 0.8", "clear(cyan) 0.7"
        return r'([a-zA-Z_]+\([a-zA-Z0-9_:,]+\))\s*:\s*([0-9]*\.?[0-9]+)'  # relevance is int 0,1,2

    @staticmethod
    def _parse_llm_predicate_relevance(predicate_fact: tuple[str, int]) -> tuple[str, int]:
        return predicate_fact[0].replace(" ", ""), int(predicate_fact[1])

    def extract_facts_once(self, image_path: Path | str, temperature) -> set[tuple[str, int]]:
        base64_image: str = encode_image_to_base64(image_path)
        user_prompt = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            },
            {
                "type": "text",
                "text": "Extract all predicates as described above. Return one predicate per line."
            }
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.system_prompt, "cache_control": {"type": "ephemeral"}}
                    ]
                },
                {"role": "user", "content": user_prompt}
            ],
        )
        response_text: str = response.choices[0].message.content.strip()
        facts: list[tuple[str, int]] = re.findall(self.result_regex, response_text)
        return set([self.llm_result_parse_func(fact) for fact in facts])

    def simulate_predicate_probabilities(self, image_path: Path, temperature: float, trials: int = 10
                                         ) -> dict[str, float]:
        predicate_counts = Counter()
        for _ in range(trials):
            predicates = self.extract_facts_once(image_path, temperature)
            predicate_counts.update(predicates)
        return {p: predicate_counts[p] / trials for p in predicate_counts}

    def simulate_relevance_judgement(self, image_path: Path | str, temperature: float) -> Dict[str, int]:
        predicates = self.extract_facts_once(image_path, temperature)
        return {p: rel for p, rel in predicates}

    @staticmethod
    def fill_missing_predicates_with_uncertainty(relevance_dict: dict[str, int], all_possible_preds: set[str],
                                                 uncertainty_label: int = 1) -> dict[str, int]:
        """
        Fills in any missing predicates with a default relevance score of 1 (uncertain).
        """
        return {pred: relevance_dict.get(pred, uncertainty_label) for pred in all_possible_preds}

    @abstractmethod
    def _generate_all_possible_predicates(self) -> set[str]:
        """
        Abstract method to generate all possible predicates for a specific domain.
        Must be implemented by subclasses with domain-specific logic.
        It uses the self.grounded_objects to generate predicates.

        Returns:
            Set of all possible predicate strings for the domain
        """
        raise NotImplementedError

    @staticmethod
    def _alter_predicate_from_llm_to_problem(predicate: str) -> str:
        return predicate  # by default, no alteration

    def classify(self, image_path: Path | str) -> Dict[str, PredicateTruthValue]:
        print(f"Classifying image: {image_path.split('/')[-1]} with temperature = {self.temperature}")
        predicates_with_rel_judgement = (
            {
                self._alter_predicate_from_llm_to_problem(pred): rel
                for pred, rel
                in self.simulate_relevance_judgement(image_path=image_path, temperature=self.temperature).items()
            }
        )

        all_possible_predicates = self._generate_all_possible_predicates()
        predicates_with_rel_judgement = self.fill_missing_predicates_with_uncertainty(
            predicates_with_rel_judgement, all_possible_predicates, self.uncertainty_label)

        for p in predicates_with_rel_judgement:
            if p not in all_possible_predicates:
                print(f"⚠️ Warning: Predicate {p} not in all possible predicates.")

        # turn all predicates into their gym object names
        predicates_with_rel_judgement = {
            **{ # predicates with arguments
                multi_replace_predicate(p, self.imaged_obj_to_gym_obj_name): rel
                for p, rel in predicates_with_rel_judgement.items()
                for obj in self.imaged_obj_to_gym_obj_name.keys() if obj in p
            },
            **{ # argless predicates
                p: rel
                for p, rel in predicates_with_rel_judgement.items()
                if all(obj not in p for obj in self.imaged_obj_to_gym_obj_name.keys())
            }
        }

        # Convert relevance scores to PredicateTruthValue
        result = {}
        for predicate, relevance_score in predicates_with_rel_judgement.items():
            if relevance_score == 2:
                result[predicate] = PredicateTruthValue.TRUE
            elif relevance_score == 0:
                result[predicate] = PredicateTruthValue.FALSE
            else:
                result[predicate] = PredicateTruthValue.UNCERTAIN
                
        return result
