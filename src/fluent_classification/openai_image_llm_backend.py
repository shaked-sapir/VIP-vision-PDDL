from abc import ABC
from pathlib import Path

from openai import OpenAI
from typing import List

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.utils.visualize import encode_image_to_base64


class OpenAIImageLLMBackend(ImageLLMBackend, ABC):
    system_prompt: str

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate_text(
        self,
        system_prompt: str,
        user_instruction: str,
        image_path: Path | str,
        temperature: float = None,
        examples: List[tuple[Path | str, List[str]]] | None = None,
    ) -> str:
        temperature = temperature if temperature is not None else self.temperature

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        # Few-shot examples
        if examples:
            for example_img, example_facts in examples:
                example_b64 = encode_image_to_base64(example_img)
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{example_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Example image. According to the predicate "
                                    "definitions in the system prompt, these are "
                                    "the correct grounded predicates for this image."
                                ),
                            },
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "\n".join(example_facts)}
                        ],
                    }
                )

        # Target image
        target_b64 = encode_image_to_base64(image_path)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{target_b64}"},
                    },
                    {"type": "text", "text": user_instruction},
                ],
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )

        return response.choices[0].message.content.strip()
