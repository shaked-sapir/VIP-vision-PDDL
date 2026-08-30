from pathlib import Path
from typing import Protocol, List, Set


class ImageLLMBackend(Protocol):
    system_prompt: str
    temperature: float
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
        cache_key: str | None = None,
    ) -> str:
        """Args:
            cache_key: Routing hint so requests sharing a prompt prefix land on
                the same backend and hit its cache. Vendors that do not support
                one ignore it.
        """
        ...
