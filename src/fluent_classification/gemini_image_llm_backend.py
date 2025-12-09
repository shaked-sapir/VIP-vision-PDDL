from pathlib import Path
from typing import List

from google import genai
from google.genai import types

from .image_llm_backend_protocol import ImageLLMBackend


class GeminiImageLLMBackend(ImageLLMBackend):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.0,
        default_mime_type: str = "image/png",
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.default_mime_type = default_mime_type

    def _image_part(self, image_path: Path | str) -> types.Part:
        path = Path(image_path)
        with path.open("rb") as f:
            image_bytes = f.read()
        return types.Part.from_bytes(
            data=image_bytes,
            mime_type=self.default_mime_type,
        )

    def generate_text(
        self,
        system_prompt: str,
        user_instruction: str,
        image_path: Path | str,
        temperature: float = None,
        examples: List[tuple[Path | str, List[str]]] | None = None,
    ) -> str:
        """
        Build a single multimodal prompt:

        [ex1_image, "Example image ...", "correct predicates for ex1",
         ex2_image, "Example image ...", "correct predicates for ex2",
         target_image, user_instruction]
        """

        contents: List[object] = []
        temperature = temperature if temperature is not None else self.temperature

        # -------- Few-shot examples --------
        if examples:
            for example_img, example_facts in examples:
                img_part = self._image_part(example_img)

                # Example image + explanation + gold predicates
                contents.append(img_part)
                contents.append(
                    "Example image. According to the predicate definitions "
                    "in the system prompt, these are the correct grounded "
                    "predicates for this image."
                )
                contents.append("\n".join(example_facts))

        # -------- Target image + instruction --------
        target_img_part = self._image_part(image_path)
        contents.append(target_img_part)
        contents.append(user_instruction)

        # -------- Gemini call --------
        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={
                "system_instruction": system_prompt,
                "temperature": temperature,
            },
        )

        return (resp.text or "").strip()