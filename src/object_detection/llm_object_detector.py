import base64
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Union

from openai import OpenAI

from src.object_detection.base_object_detector import ObjectDetector


class LLMObjectDetector(ObjectDetector, ABC):
    """
    Abstract LLM-based object detector.

    This class encapsulates an LLM prompt/parse cycle to extract grounded objects present in an image.
    Subclasses must provide a domain-specific system prompt, regex, and parsing function that
    converts model output into a mapping of object typings to grounded object names.
    """

    def __init__(self, openai_apikey: str, model: str):
        self.openai_client = OpenAI(api_key=openai_apikey)
        self.model = model
        self.system_prompt = self._get_system_prompt()
        self.result_regex = self._get_result_regex()
        self.llm_result_parse_func = self._parse_llm_object_detection

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

    @staticmethod
    def _encode_image(image_path: Union[Path, str]) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _detect_once(self, image_path: Union[Path, str], temperature: float = 1.0) -> set[str]:
        base64_image: str = self._encode_image(image_path)
        user_prompt = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            },
            {
                "type": "text",
                "text": "Extract all objects as described above. Return one object per line."
            }
        ]

        response = self.openai_client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # max_tokens=3000  # TODO: make configurable
        )
        response_text: str = response.choices[0].message.content.strip()
        facts: list[str] = re.findall(self.result_regex, response_text)  # assumes that a fact is like "<obj_name>:<obj_type>"
        return set([self.llm_result_parse_func(fact) for fact in facts])

    def detect(self, image: Union[Path, str], *args, **kwargs) -> dict[str, List[str]]:
        detected_objects = self._detect_once(image, *args, **kwargs)
        type_to_objects: Dict[str, List[str]] = defaultdict(list)
        for obj in detected_objects:
            name, type_ = obj.split(":")
            type_to_objects[type_].append(name)

        return type_to_objects
