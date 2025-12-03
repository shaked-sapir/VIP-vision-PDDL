import base64
import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Set

from openai import OpenAI

from src.object_detection.base_object_detector import ObjectDetector
from src.utils.pddl import extract_objects_from_pddlgym_state
from src.utils.visualize import encode_image_to_base64


class LLMObjectDetector(ObjectDetector, ABC):
    """
    Abstract LLM-based object detector.

    This class encapsulates an LLM prompt/parse cycle to extract grounded objects present in an image.
    Subclasses must provide a domain-specific system prompt, regex, and parsing function that
    converts model output into a mapping of object typings to grounded object names.
    """

    def __init__(self, openai_apikey: str, model: str, temperature: float, init_state_image_path: Path):
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model
        self.temperature = temperature
        self.system_prompt = self._get_system_prompt()
        self.result_regex = self._get_result_regex()
        self.llm_result_parse_func = self._parse_llm_object_detection
        self.initial_state_image_path = init_state_image_path
        self.imaged_obj_to_gym_obj_name = {}

        # Find trajectory JSON in the same directory as the image
        trajectory_files = list(init_state_image_path.parent.glob("*_trajectory.json"))
        self.gt_json_trajectory_path: Path = trajectory_files[0]
        self.gt_json_trajectory = json.loads(self.gt_json_trajectory_path.read_text())
        self.fewshot_examples = []

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for fluent classification, depending on domain."""
        raise NotImplementedError

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract predicates from LLM response."""
        # Pattern to match object detection statement in "<name>:<type>" format
        # Examples: "red:block", "table:table", "robot:robot"
        return r"\b[a-z]+:[a-z]+\b"

    @staticmethod
    def _parse_llm_object_detection(obj_detect_fact: str) -> str:
        return obj_detect_fact.replace(" ", "")  # remove spaces guardedly added by the LLM

    def _detect_once(
            self,
            image_path: Path | str,
            temperature: float,
            examples: list[tuple[Path | str, list[str]]] | None = None,
    ) -> set[str]:
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
        examples = examples if examples is not None else self.fewshot_examples
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
                            "Example image. According to the types definitions in the system "
                            "prompt, these are the correct object and their types for this image."
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
                    "text": "Extract all object with their types for this image. One line per object."
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

    def extract_objects_from_gt_state(
            self,
            state_index: int = 0
    ) -> List[str]:
        state = self.gt_json_trajectory[state_index]["current_state"]
        objects = state["objects"]

        return list(extract_objects_from_pddlgym_state(objects, self.imaged_obj_to_gym_obj_name))

    def detect(self, image: Union[Path, str], examples: list[tuple[Path | str, list[str]]] = None,
               *args, **kwargs) -> dict[str, List[str]]:
        print(f"Detecting objects with temperature = {self.temperature}")
        detected_objects = self._detect_once(image, self.temperature, examples)
        type_to_objects: Dict[str, List[str]] = defaultdict(list)
        for obj in detected_objects:
            name, type_ = obj.split(":")
            type_to_objects[type_].append(name)

        return type_to_objects
