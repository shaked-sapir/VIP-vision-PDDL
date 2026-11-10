"""A vendor block with ``api: openai`` and a ``base_url`` builds the OpenAI
backend pointed at that server.

    python -m pytest src/fluent_classification/test_backend_factory.py
"""

import pytest

import src.object_detection  # noqa: F401  (breaks a circular import in visualize)
from src.fluent_classification import image_llm_backend_factory as factory_module
from src.fluent_classification.image_llm_backend_factory import ImageLLMBackendFactory
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend

_CONFIG = {
    "openai": {
        "api_key": "sk-test",
        "fluent_classification_model": {"model_name": "gpt-5.2", "temperature": 0.0},
    },
    "vllm": {
        "api": "openai",
        "api_key": "EMPTY",
        "base_url": "http://localhost:8000/v1",
        "request_extras": {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
        "fluent_classification_model": {"model_name": "qwen3-vl-8b", "temperature": 0.0},
    },
    "odd": {
        "api": "carrier-pigeon",
        "api_key": "x",
        "fluent_classification_model": {"model_name": "m", "temperature": 0.0},
    },
}


@pytest.fixture(autouse=True)
def _config(monkeypatch):
    monkeypatch.setattr(factory_module, "load_config", lambda: _CONFIG)


def test_a_base_url_vendor_is_the_openai_backend_on_that_server():
    backend = ImageLLMBackendFactory.create("vllm", "fluent_classification")
    assert isinstance(backend, OpenAIImageLLMBackend)
    assert str(backend.client.base_url).rstrip("/") == "http://localhost:8000/v1"
    assert backend.model == "qwen3-vl-8b"


def test_the_openai_vendor_keeps_the_default_server():
    backend = ImageLLMBackendFactory.create("openai", "fluent_classification")
    assert "api.openai.com" in str(backend.client.base_url)


def test_an_unknown_api_is_refused():
    with pytest.raises(ValueError, match="carrier-pigeon"):
        ImageLLMBackendFactory.create("odd", "fluent_classification")


def test_the_environment_can_redirect_the_server(monkeypatch):
    monkeypatch.setenv("VIP_LLM_BASE_URL", "http://ise-4090-14:8000/v1")
    backend = ImageLLMBackendFactory.create("vllm", "fluent_classification")
    assert str(backend.client.base_url).rstrip("/") == "http://ise-4090-14:8000/v1"


def test_request_extras_reach_the_backend():
    backend = ImageLLMBackendFactory.create("vllm", "fluent_classification")
    assert backend.request_extras == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    assert ImageLLMBackendFactory.create("openai", "fluent_classification").request_extras == {}
