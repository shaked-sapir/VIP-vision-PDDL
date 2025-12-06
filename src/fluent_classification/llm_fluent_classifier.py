import base64
import json
import re
from abc import abstractmethod, ABC
from collections import Counter
from pathlib import Path
from typing import Dict, Union

from openai import OpenAI

from src.fluent_classification.base_fluent_classifier import FluentClassifier, PredicateTruthValue
from src.utils.pddl import multi_replace_predicate, translate_pddlgym_state_to_image_predicates
from src.utils.visualize import encode_image_to_base64


class LLMFluentClassifier(FluentClassifier, ABC):
    system_prompt: str

    def __init__(self, openai_apikey: str, type_to_objects: dict[str, list[str]], model: str, temperature: float,
                 init_state_image_path: Path):
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

        # Find trajectory JSON in the same directory as the image
        trajectory_files = list(init_state_image_path.parent.glob("*_trajectory.json"))
        self.gt_json_trajectory_path: Path = trajectory_files[0]
        self.gt_json_trajectory = json.loads(self.gt_json_trajectory_path.read_text())
        self.fewshot_examples = []

        self.user_instruction = self._get_user_instruction()

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for fluent classification, depending on domain."""
        raise NotImplementedError

    @classmethod
    def _get_user_instruction(cls) -> str:
        """Returns the user instruction for fluent classification, depending on domain."""
        return ("Extract facts on this image, based on the example image with its extracted facts, "
                "One predicate per line.")

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract predicates from LLM response."""
        # Pattern to match predicate with confidence score
        # Examples: "on(red,blue) 0.9", "ontable(green) 0.8", "clear(cyan) 0.7"
        return r'([a-zA-Z_]+\([a-zA-Z0-9_:,]+\))\s*:\s*([0-9]*\.?[0-9]+)'  # relevance is int 0,1,2

    @staticmethod
    def _parse_llm_predicate_relevance(predicate_fact: tuple[str, int]) -> tuple[str, int]:
        return predicate_fact[0].replace(" ", ""), int(predicate_fact[1])

    def create_examples_for_few_shot(
            self,
            image_paths: list[Path | str],
            gt_trajectory_indices: list[int],
            gt_trajectory_path: Path
    ) -> list[tuple[Path | str, list[str]]]:
        """
        Creates few-shot examples from the ground-truth trajectory.
        :param image_paths: images to use as examples
        :param gt_trajectory_indices: the corresponding indices in the gt trajectory for each image
        :param gt_trajectory_path: path to the ground-truth trajectory file
        :return: paired list of (image_path, list of predicates) for each example
        """
        # Load steps
        steps = json.loads(Path(gt_trajectory_path).read_text())

        # Build step → literals map
        step_literals = {
            s["step"]: s["current_state"]["literals"]
            for s in steps
        }

        # Build examples compactly
        return [
            (
                img_path,
                self.extract_predicates_from_gt_state(
                    step_literals[idx + 1]
                )
            )
            for img_path, idx in zip(image_paths, gt_trajectory_indices)
        ]

    def extract_facts_once(
            self,
            image_path: Path | str,
            temperature: float,
            examples: list[tuple[Path | str, list[str]]]
    ) -> set[tuple[str, int]]:
        """
        This method performs a single extraction of predicates with relevance scores from the given image.
        - Zero-shot:
            Call without examples (default).
        - One-shot:
            Call with a single (example_image_path, example_facts_text) pair.
        - Few-shot:
            Call with multiple (example_image_path, example_facts_text) pairs.

        Parameters
        ----------
        image_path : Path | str
            The target image we want to extract predicates from.
        temperature : float
            Sampling temperature for the model.
        examples : Optional[Iterable[Tuple[Path | str, str]]]
            Iterable of (example_image_path, example_facts_text) pairs.
            Each `example_facts_text` should be a multi-line string with
            one predicate per line, in the same format you expect from the model.

        Returns
        -------
        set[tuple[str, int]]
            Parsed facts as returned by `self.llm_result_parse_func`.
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
            }
        ]

        # Add all provided examples directly (one-shot or few-shot)
        if examples:
            for example_img, example_facts in examples:
                example_b64 = encode_image_to_base64(example_img)
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{example_b64}"}
                        },
                        {
                            "type": "text",
                            "text": (
                                "Example image. According to the predicate definitions in the system "
                                "prompt, these are the correct grounded predicates for this image."
                            )
                        }
                    ]
                })
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": '\n'.join(example_facts)}]
                })

        # Target example
        target_img_b64 = encode_image_to_base64(image_path)
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{target_img_b64}"}
                },
                {
                    "type": "text",
                    "text": self.user_instruction
                }
            ]
        })

        # Run model
        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )

        text = response.choices[0].message.content.strip()
        facts = re.findall(self.result_regex, text)
        return {self.llm_result_parse_func(f) for f in facts}

    def simulate_predicate_probabilities(self, image_path: Path, temperature: float, trials: int = 10
                                         ) -> dict[str, float]:
        predicate_counts = Counter()
        for _ in range(trials):
            predicates = self.extract_facts_once(image_path, temperature)
            predicate_counts.update(predicates)
        return {p: predicate_counts[p] / trials for p in predicate_counts}

    def simulate_relevance_judgement(self, image_path: Path | str, temperature: float,
                                     examples: list[tuple[Path | str, list[str]]]) -> Dict[str, int]:
        predicates = self.extract_facts_once(image_path, temperature, examples)
        return {p: rel for p, rel in predicates}

    def extract_predicates_from_gt_state(self, state_index: int = 0) -> list[str]:
        state = self.gt_json_trajectory[state_index]["current_state"]
        fluents = state["literals"]
        return translate_pddlgym_state_to_image_predicates(fluents, self.imaged_obj_to_gym_obj_name)

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

    def classify(self, image_path: Path | str,
                 examples: list[tuple[Path | str, list[str]]] = None) -> Dict[str, PredicateTruthValue]:
        print(f"Classifying image: {image_path.split('/')[-1]} with temperature = {self.temperature}")
        examples = examples if examples is not None else self.fewshot_examples
        predicates_with_rel_judgement = (
            {
                self._alter_predicate_from_llm_to_problem(pred): rel
                for pred, rel
                in self.simulate_relevance_judgement(image_path, self.temperature, examples).items()
            }
        )

        all_possible_predicates = self._generate_all_possible_predicates()
        predicates_with_rel_judgement_with_unk = self.fill_missing_predicates_with_uncertainty(
            predicates_with_rel_judgement, all_possible_predicates, self.uncertainty_label)

        for p in predicates_with_rel_judgement_with_unk:
            if p not in all_possible_predicates:
                pass
                # print(f"⚠️ Warning: Predicate {p} not in all possible predicates.")

        # turn all predicates into their gym object names
        predicates_with_rel_judgement_with_unk = {
            **{ # predicates with arguments
                multi_replace_predicate(p, self.imaged_obj_to_gym_obj_name): rel
                for p, rel in predicates_with_rel_judgement_with_unk.items()
                for obj in self.imaged_obj_to_gym_obj_name.keys() if obj in p
            },
            **{ # argless predicates
                p: rel
                for p, rel in predicates_with_rel_judgement_with_unk.items()
                if all(obj not in p for obj in self.imaged_obj_to_gym_obj_name.keys())
            }
        }

        # Convert relevance scores to PredicateTruthValue
        result = {}
        for predicate, relevance_score in predicates_with_rel_judgement_with_unk.items():
            if relevance_score == 2:
                result[predicate] = PredicateTruthValue.TRUE
            elif relevance_score == 0:
                result[predicate] = PredicateTruthValue.FALSE
            else:
                result[predicate] = PredicateTruthValue.UNCERTAIN
                
        return result
