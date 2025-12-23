import json
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union, Set

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.object_detection.base_object_detector import ObjectDetector
from src.utils.pddl import extract_objects_from_pddlgym_state


class LLMObjectDetector(ObjectDetector, ABC):
    """
    Abstract LLM-based object detector.

    This class encapsulates an LLM prompt/parse cycle to extract grounded objects present in an image.
    Subclasses must provide a domain-specific system prompt, regex, and parsing function that
    converts model output into a mapping of object typings to grounded object names.
    """

    def __init__(
        self,
        llm_backend: ImageLLMBackend,
        init_state_image_path: Path,
        temperature: float = None,
        inference_mode: bool = False
    ):
        self.backend = llm_backend
        self.temperature = temperature if temperature is not None else llm_backend.temperature

        self.system_prompt = self._get_system_prompt()
        self.user_instruction = self._get_user_instruction()
        self.result_regex = self._get_result_regex()
        self.llm_result_parse_func = self._parse_llm_object_detection
        self.initial_state_image_path = init_state_image_path
        self.imaged_obj_to_gym_obj_name = {}

        # Find trajectory JSON in the same directory as the image
        trajectory_files = list(init_state_image_path.parent.glob("*_trajectory.json"))
        self.gt_json_trajectory_path: Path = trajectory_files[0]
        self.gt_json_trajectory = json.loads(self.gt_json_trajectory_path.read_text())
        self.inference_mode = inference_mode
        self.fewshot_examples = []

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Returns the system prompt for fluent classification, depending on domain."""
        raise NotImplementedError

    @classmethod
    def _get_user_instruction(cls) -> str:
        """User instruction sent with the target image."""
        return (
            "Extract all objects that appear in this image, in the form 'name:type'.\n"
            "Output one object per line."
        )

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
        examples: List[tuple[Path | str, List[str]]] | None = None,
    ) -> Set[str]:
        """
        Perform a single LLM-based object detection call.

        Parameters
        ----------
        image_path : Path | str
            Target image.
        temperature : float
            Sampling temperature.
        examples : list[(example_image_path, ["obj:type", ...])]
            Optional few-shot examples.

        Returns
        -------
        set[str]
            Parsed object strings in the 'name:type' format.
        """
        examples = examples if examples is not None else self.fewshot_examples
        if not self.inference_mode:
            return set(self.fewshot_examples[0][1])
        else:
            text = self.backend.generate_text(
                system_prompt=self.system_prompt,
                user_instruction=self.user_instruction,
                image_path=image_path,
                temperature=temperature,
                examples=examples,
            )

            matches = re.findall(self.result_regex, text)
            return {self.llm_result_parse_func(m) for m in matches}

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
