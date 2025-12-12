"""
Factory for creating ImageLLMBackend instances based on vendor.

This centralizes backend creation logic to avoid code duplication across trajectory handlers.
Loads configuration from config.yaml to determine API keys, models, and temperatures.
"""

from src.fluent_classification.gemini_image_llm_backend import GeminiImageLLMBackend
from src.fluent_classification.image_llm_backend_protocol import ImageLLMBackend
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend
from src.utils.config import load_config


class ImageLLMBackendFactory:
    """
    Factory for creating ImageLLMBackend instances based on vendor and model type.

    Automatically loads configuration from config.yaml to get:
    - API keys for the vendor
    - Model names for the task type (object_detection or fluent_classification)
    - Temperature settings

    Example:
        >>> backend = ImageLLMBackendFactory.create(
        ...     vendor="openai",
        ...     model_type="object_detection"
        ... )
        >>> # Returns OpenAIImageLLMBackend with gpt-5.1 and temp from config

        >>> backend = ImageLLMBackendFactory.create(
        ...     vendor="google",
        ...     model_type="fluent_classification"
        ... )
        >>> # Returns GeminiImageLLMBackend with gemini-2.5-pro and temp from config
    """

    @staticmethod
    def create(
        vendor: str,
        model_type: str,
    ) -> ImageLLMBackend:
        """
        Create an ImageLLMBackend instance based on the vendor and model type.

        Args:
            vendor: LLM vendor ("openai" or "google")
            model_type: Task type for the model ("object_detection" or "fluent_classification")

        Returns:
            ImageLLMBackend instance (OpenAIImageLLMBackend or GeminiImageLLMBackend)
            configured with API key, model, and temperature from config.yaml

        Raises:
            ValueError: If vendor is not supported or model_type is invalid
            KeyError: If config.yaml is missing required keys
        """
        # Validate model_type
        valid_model_types = ["object_detection", "fluent_classification"]
        if model_type not in valid_model_types:
            raise ValueError(
                f"Invalid model_type: {model_type}. Must be one of {valid_model_types}"
            )

        # Load configuration
        config = load_config()

        # Extract configuration for the vendor and model type
        try:
            api_key = config[vendor]['api_key']
            model_config_key = f"{model_type}_model"
            model = config[vendor][model_config_key]['model_name']
            temperature = config[vendor][model_config_key]['temperature']
        except KeyError as e:
            raise KeyError(
                f"Missing configuration key for vendor '{vendor}' and model_type '{model_type}': {e}"
            )

        # Create the appropriate backend
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
