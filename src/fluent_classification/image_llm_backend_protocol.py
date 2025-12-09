from pathlib import Path
from typing import Protocol, List, Set


class ImageLLMBackend(Protocol):
    system_prompt: str

    """
        Generic interface for an LLM that extracts predicates from a single image.

        - domain-agnostic: it just knows about system_prompt / user_instruction / examples.
        - vendor-specific implementations (OpenAI, Gemini, etc.) will implement this.
        """

    def generate_text(
        self,
        system_prompt: str,
        user_instruction: str,
        image_path: Path | str,
        temperature: float,
        examples: List[tuple[Path | str, List[str]]] | None = None,
    ) -> str:
        ...
