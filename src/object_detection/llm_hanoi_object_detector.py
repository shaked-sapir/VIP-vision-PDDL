"""
LLM-based object detector for Hanoi domain.

Uses GPT-4 Vision to detect and identify discs and pegs in Hanoi puzzle images.
"""

from src.llms.domains.hanoi.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMHanoiObjectDetector(LLMObjectDetector):
    """
    LLM-based object detector for the Hanoi domain.
    Uses GPT-4 Vision API to detect discs and pegs from images.
    """

    def __init__(self, openai_apikey: str, model: str = "gpt-4o"):
        super().__init__(
            openai_apikey=openai_apikey,
            model=model
        )

    def _get_system_prompt(self) -> str:
        """Returns the system prompt for Hanoi object detection."""
        return object_detection_system_prompt
