"""The cache-hit counters must survive concurrent inference.

    python -m pytest src/fluent_classification/test_usage_accounting.py

One backend serves every worker of a window under ``ConcurrentFrameClassifier``,
and ``+=`` is not atomic. A lost update only skews the reported ratio, never a
trajectory -- but that ratio is the one number that says whether the prompt
caching is working, so it has to be right.
"""

import threading
from types import SimpleNamespace

import src.object_detection  # noqa: F401  (breaks a circular import in visualize)
from src.fluent_classification.openai_image_llm_backend import OpenAIImageLLMBackend

PROMPT_TOKENS = 1201
CACHED_TOKENS = 1152


def _backend():
    """A backend with its counters initialised, without touching the network."""
    backend = OpenAIImageLLMBackend.__new__(OpenAIImageLLMBackend)
    backend.prompt_tokens_total = 0
    backend.prompt_tokens_cached = 0
    backend._usage_lock = threading.Lock()
    return backend


def _response(prompt_tokens=PROMPT_TOKENS, cached=CACHED_TOKENS):
    return SimpleNamespace(usage=SimpleNamespace(
        prompt_tokens=prompt_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
    ))


class TestConcurrentAccounting:
    def test_no_updates_are_lost(self):
        backend, response, per_thread, threads = _backend(), _response(), 500, 10
        workers = [
            threading.Thread(
                target=lambda: [backend._record_cache_usage(response)
                                for _ in range(per_thread)]
            )
            for _ in range(threads)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()

        assert backend.prompt_tokens_total == threads * per_thread * PROMPT_TOKENS
        assert backend.prompt_tokens_cached == threads * per_thread * CACHED_TOKENS

    def test_ratio_is_exact_under_concurrency(self):
        backend, response = _backend(), _response()
        workers = [
            threading.Thread(
                target=lambda: [backend._record_cache_usage(response)
                                for _ in range(200)]
            )
            for _ in range(8)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join()
        assert backend.cache_hit_ratio == CACHED_TOKENS / PROMPT_TOKENS


class TestDegenerateResponses:
    def test_no_usage_is_ignored(self):
        backend = _backend()
        backend._record_cache_usage(SimpleNamespace(usage=None))
        assert backend.prompt_tokens_total == 0

    def test_missing_cache_details_count_as_a_miss(self):
        backend = _backend()
        backend._record_cache_usage(
            SimpleNamespace(usage=SimpleNamespace(prompt_tokens=900,
                                                  prompt_tokens_details=None)))
        assert backend.prompt_tokens_total == 900
        assert backend.prompt_tokens_cached == 0
        assert backend.cache_hit_ratio == 0.0

    def test_ratio_before_any_call(self):
        assert _backend().cache_hit_ratio == 0.0
