"""Cache keys for LLM prompt-prefix routing."""

import hashlib


def prompt_cache_key(task: str, system_prompt: str) -> str:
    """A routing key identifying one stable prompt prefix.

    Derived from the prompt rather than passed in, so it cannot drift out of
    sync with what is actually sent: change the prompt and the key changes with
    it, which is what should happen — the old prefix is no longer cacheable.

    The whole corpus shares one key because a corpus is one walk over one
    problem, so every window has the same object universe and therefore the same
    system prompt (``uniform_object_universe`` in generation_info.json).

    Args:
        task: What the prompt is for, e.g. ``"fluent_classification"``. Keeps
            detection and classification on separate keys; their prompts differ,
            so sharing one would route two prefixes to the same backend.
        system_prompt: The prefix itself.

    Returns:
        ``"<task>_<12 hex chars>"``.
    """
    digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]
    return f"{task}_{digest}"
