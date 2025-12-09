"""
Factory for creating ImageLLMBackend instances based on vendor.

This centralizes backend creation logic to avoid code duplication across trajectory handlers.
"""

from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend


class ImageLLMBackendFactory:
    """
    Factory for creating ImageLLMBackend instances based on vendor.

    Example:
        >>> backend = ImageLLMBackendFactory.create(
        ...     vendor="openai",
        ...     api_key="sk-...",
        ...     model="gpt-4o",
        ...     temperature=0.0
        ... )
    """

    @staticmethod
    def create(
        vendor: str,
        api_key: str,
        model: str,
        temperature: float
    ) -> ImageLLMBackend:
        """
        Create an ImageLLMBackend instance based on the vendor.

        Args:
            vendor: LLM vendor ("openai" or "google")
            api_key: API key for the vendor
            model: Model name (e.g., "gpt-4o", "gemini-2.5-pro")
            temperature: Temperature for model inference

        Returns:
            ImageLLMBackend instance (OpenAIImageLLMBackend or GeminiImageLLMBackend)

        Raises:
            ValueError: If vendor is not supported
        """
        if vendor == "google":
            return GeminiImageLLMBackend(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        elif vendor == "openai":
            return OpenAIImageLLMBackend(
                api_key=api_key,
                model=model,
                temperature=temperature
            )
        else:
            raise ValueError(f"Unsupported vendor: {vendor}. Must be 'openai' or 'google'.")
