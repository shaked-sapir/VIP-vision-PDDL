"""The few-shot example the model sees scores every grounded predicate.

    python -m pytest src/fluent_classification/test_scored_examples.py

The example is stored as its frame's true facts; the system prompt asks for a
0/1/2 score on every predicate, so the turn sent to the model must show that
shape, or a model that imitates the example lists bare facts the parser cannot
read and omits the false ones, which then count as uncertain.
"""

import src.object_detection  # noqa: F401  (breaks a circular import in visualize)
from src.fluent_classification.llm_fluent_classifier import LLMFluentClassifier


class _Classifier(LLMFluentClassifier):
    def _get_system_prompt(self) -> str:
        return ""

    def _generate_all_possible_predicates(self) -> set[str]:
        return {"clear(a:block)", "holding(a:block)", "ontable(a:block)", "handempty()"}


def _classifier() -> _Classifier:
    return _Classifier.__new__(_Classifier)


def test_true_facts_score_2_and_the_rest_0():
    rendered = _classifier()._scored_examples([("img.png", ["ontable(a:block)", "handempty()"])])
    assert rendered == [("img.png", [
        "clear(a:block): 0",
        "handempty(): 2",
        "holding(a:block): 0",
        "ontable(a:block): 2",
    ])]


def test_a_fact_outside_the_universe_is_kept_as_true():
    _, lines = _classifier()._scored_examples([("img.png", ["odd(a:block)"])])[0]
    assert "odd(a:block): 2" in lines
    assert all(line.endswith(": 0") for line in lines if not line.startswith("odd"))


def test_the_stored_example_is_untouched():
    """``construct_trajectory_from_images`` still reads the bare facts as state 0."""
    classifier = _classifier()
    classifier.fewshot_examples = [("img.png", ["handempty()"])]
    classifier._scored_examples(classifier.fewshot_examples)
    assert classifier.fewshot_examples == [("img.png", ["handempty()"])]
