"""
LLM-based object detector for Hanoi domain.

Uses GPT-4 Vision to detect and identify discs and pegs in Hanoi puzzle images.
"""

from typing import Dict, List, Union
from pathlib import Path
from collections import defaultdict

from src.llms.domains.hanoi.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMHanoiObjectDetector(LLMObjectDetector):
    """
    LLM-based object detector for the Hanoi domain.
    Uses GPT-4 Vision API to detect discs and pegs from images.

    The LLM returns objects as "d1:disc", "p1:peg", etc.
    This class maps them to the format expected by the fluent classifier:
    - d1, d2, d3, ... (discs stay the same)
    - p1 → peg1, p2 → peg2, p3 → peg3 (pegs are renamed)
    """

    def __init__(self, openai_apikey: str, model: str = "gpt-4o", temperature: float = 1.0):
        super().__init__(
            openai_apikey=openai_apikey,
            model=model,
            temperature=temperature
        )

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Hanoi object detection."""
        return object_detection_system_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract object detections from LLM response."""
        # Pattern to match object detection in "<name>:<type>" format
        # where name can include digits (e.g., "d1:disc", "p1:peg")
        return r"\b[a-z]\d+:[a-z]+\b"

    def detect(self, image: Union[Path, str], *args, **kwargs) -> Dict[str, List[str]]:
        """
        Detect objects in image and map peg names from p1, p2, p3 to peg1, peg2, peg3.

        :param image: Path to the image
        :return: Dictionary mapping type to list of object names
        """
        # Call parent's detect method
        detected_objects = super().detect(image, *args, **kwargs)

        # Map peg names: p1 → peg1, p2 → peg2, etc.
        if 'peg' in detected_objects:
            mapped_pegs = []
            for peg_name in detected_objects['peg']:
                if peg_name.startswith('p') and peg_name[1:].isdigit():
                    # Convert p1 → peg1
                    mapped_name = f"peg{peg_name[1:]}"
                    mapped_pegs.append(mapped_name)
                else:
                    # Keep as-is if already in correct format
                    mapped_pegs.append(peg_name)
            detected_objects['peg'] = mapped_pegs

        return detected_objects
