from src.llms.domains.blocks.prompts import object_detection_system_prompt
from src.object_detection.llm_object_detector import LLMObjectDetector


class LLMBlocksObjectDetector(LLMObjectDetector):
    def __init__(self, openai_apikey: str, model: str = "gpt-4o", temperature: float = 1.0):
        super().__init__(
            openai_apikey=openai_apikey,
            model=model,
            temperature=temperature
        )

    def _get_system_prompt(self) -> str:
        return object_detection_system_prompt  # TODO: find a proper way to handle the objects, not hardcoding from the .prompts file
