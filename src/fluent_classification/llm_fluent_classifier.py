import base64
import re
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Union

from openai import OpenAI

from src.fluent_classification.base_fluent_classifier import FluentClassifier, PredicateTruthValue


class LLMFluentClassifier(FluentClassifier):
    system_prompt: str

    def __init__(self, openai_apikey: str, system_prompt_text: str, result_regex: str, result_parse_func: callable):
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.system_prompt = system_prompt_text
        self.result_regex = result_regex
        self.result_parse_func = result_parse_func


    @staticmethod
    def encode_image(image_path: Union[Path, str]):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract_facts_once(self, image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                           result_parse_func: callable, temperature=1.3):
        base64_image: str = self.encode_image(image_path)
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
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt_text},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3000
        )
        response_text: str = response.choices[0].message.content.strip()
        facts: list[str] = re.findall(result_regex, response_text)
        return set([result_parse_func(fact) for fact in facts])  # remove spaces guardedly added by the LLM

    def simulate_predicate_probabilities(self, image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                                         result_parse_func: callable, temperature: float, trials: int = 10):
        predicate_counts = Counter()
        for _ in range(trials):
            predicates = self.extract_facts_once(image_path, model, system_prompt_text, result_regex, result_parse_func,
                                            temperature)
            predicate_counts.update(predicates)
        return {p: predicate_counts[p] / trials for p in predicate_counts}

    def simulate_relevance_judgement(self, image_path: Path, model: str, system_prompt_text: str, result_regex: str,
                                     result_parse_func: callable, temperature: float):
        predicates = self.extract_facts_once(image_path, model, system_prompt_text, result_regex, result_parse_func,
                                        temperature)
        return {p: rel for p, rel in predicates}

    @staticmethod
    def fill_missing_predicates_with_uncertainty(relevance_dict: dict, all_possible_preds: set[str],
                                                 uncertainty_label: int = 1) -> dict:
        """
        Fills in any missing predicates with a default relevance score of 1 (uncertain).
        """
        return {pred: relevance_dict.get(pred, uncertainty_label) for pred in all_possible_preds}

    @abstractmethod
    def _generate_all_possible_predicates(self, *args, **kwargs) -> set[str]:
        """
        Abstract method to generate all possible predicates for a specific domain.
        Must be implemented by subclasses with domain-specific logic.

        Returns:
            Set of all possible predicate strings for the domain
        """
        pass

    def classify(self, image) -> Dict[str, PredicateTruthValue]:
        predicates_with_rel_judgement = self.simulate_relevance_judgement(
            image_path=image,
            model="gpt-4-vision-preview",
            system_prompt_text=self.system_prompt,
            result_regex=self.result_regex,
            result_parse_func=self.result_parse_func,
            temperature=1.3
        )

        all_possible_predicates = self._generate_all_possible_predicates(*args, **kwargs)
        predicates_with_rel_judgement = self.fill_missing_predicates_with_uncertainty(predicates_with_rel_judgement)
        
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
