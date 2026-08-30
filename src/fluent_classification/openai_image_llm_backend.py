import threading
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
        self.prompt_tokens_total = 0
        self.prompt_tokens_cached = 0
        # One backend serves every worker of a window under concurrent
        # inference, and `+=` is not atomic: without this the ratio reads low,
        # which is the one number that says whether the caching works.
        self._usage_lock = threading.Lock()

    def generate_text(
        self,
        system_prompt: str,
        user_instruction: str,
        image_path: Path | str,
        temperature: float = None,
        examples: List[tuple[Path | str, List[str]]] | None = None,
        cache_key: str | None = None,
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

        # The prompt prefix is identical for every image of a corpus (one walk
        # over one problem -> one object universe), so it caches -- but only if
        # the requests reach the backend that holds it. `prompt_cache_key` is
        # the routing hint; without it a long run scatters across machines.
        request = {
            "model": self.model,
            "temperature": temperature,
            "messages": messages,
        }
        if cache_key is not None:
            request["prompt_cache_key"] = cache_key

        response = self.client.chat.completions.create(**request)
        self._record_cache_usage(response)
        return response.choices[0].message.content.strip()

    def _record_cache_usage(self, response) -> None:
        """Accumulate cached-vs-total prompt tokens, so a miss is visible.

        Silent misses are the failure mode worth catching: the run completes
        either way, and only the bill differs.
        """
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) or 0
        total = getattr(usage, "prompt_tokens", 0) or 0
        with self._usage_lock:
            self.prompt_tokens_total += total
            self.prompt_tokens_cached += cached

    @property
    def cache_hit_ratio(self) -> float:
        """Share of prompt tokens served from cache, or 0.0 before any call."""
        if not self.prompt_tokens_total:
            return 0.0
        return self.prompt_tokens_cached / self.prompt_tokens_total
