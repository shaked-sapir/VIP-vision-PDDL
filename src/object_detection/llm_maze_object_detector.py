from src.llms.domains.maze.prompts import maze_object_detection_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMMazeObjectDetector(LLMObjectDetector):
    """
    LLM-based object detector for the Slide (8-puzzle) domain.

    The LLM returns objects as "loc-1-3:location".
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
        return maze_object_detection_prompt

    @staticmethod
    def _get_result_regex() -> str:
        """Returns the regex pattern to extract object detections from LLM response."""
        # Pattern to match object detection in "<name>:<type>" format
        # where name can include digits (e.g., "d1:disc", "p1:peg")
        return r'(?:loc-\d+-\d+:location|robot:player|doll:doll)'
