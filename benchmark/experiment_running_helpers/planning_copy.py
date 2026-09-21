"""The model a planner is handed, and the solving metrics read off it.

An operator with no effects maps every state to itself, so removing it changes
neither which problems are solvable nor which plans are valid. It does change
whether the planner runs at all: unified-planning writes such an action without
an ``:effect`` field and Fast Downward's translator rejects the whole domain
(exit code 31), an outcome AMLGym's ``problem_solving`` counts in no bucket.
Planning therefore runs on a copy without those operators; every other metric
keeps reading the learned model as it is.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from amlgym.metrics import problem_solving

ACTION_OPEN = "(:action"
PLANNING_COPY_SUFFIX = ".planning.pddl"

#: Row fields this module produces, in schema order.
SOLVING_FIELDS: Tuple[str, ...] = (
    "solving_ratio",
    "false_plans_ratio",
    "unsolvable_ratio",
    "planning_timed_out_ratio",
    "planning_syntax_error_ratio",
    "planning_error_ratio",
    "planning_dropped_operators",
)

_ACTION_NAME = re.compile(r"\(:action\s+([\w-]+)")
_NOT_AN_ATOM = re.compile(r"[()\s]|\band\b")
_NEXT_FIELD = re.compile(r":(?:precondition|parameters)\b")


def action_spans(text: str) -> List[Tuple[int, int]]:
    """``(start, end)`` of every balanced ``(:action ...)`` block in ``text``."""
    spans: List[Tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(ACTION_OPEN, cursor)
        if start < 0:
            return spans
        depth = 0
        for index in range(start, len(text)):
            if text[index] == "(":
                depth += 1
            elif text[index] == ")":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1))
                    break
        else:
            raise ValueError("unbalanced parentheses in an (:action block")
        cursor = spans[-1][1]


def effect_is_empty(action_block: str) -> bool:
    """True when the block's ``:effect`` holds no atom, or is absent."""
    _, found, effect = action_block.partition(":effect")
    if not found:
        return True
    effect = _NEXT_FIELD.split(effect, maxsplit=1)[0]
    return _NOT_AN_ATOM.sub("", effect) == ""


def without_inert_operators(text: str) -> Tuple[str, List[str]]:
    """``text`` minus its no-effect operators, and their names in file order."""
    dropped: List[str] = []
    for start, end in reversed(action_spans(text)):
        block = text[start:end]
        if effect_is_empty(block):
            dropped.append(_ACTION_NAME.search(block).group(1))
            text = text[:start] + text[end:]
    dropped.reverse()
    return text, dropped


@contextmanager
def planning_copy(model_path: Path) -> Iterator[Tuple[Path, List[str]]]:
    """Yield ``(path to plan with, dropped operator names)``.

    The path is ``model_path`` itself when nothing was dropped; otherwise a
    sibling file that is removed on exit.
    """
    model_path = Path(model_path)
    text, dropped = without_inert_operators(model_path.read_text())
    if not dropped:
        yield model_path, dropped
        return
    copy_path = model_path.with_name(model_path.name + PLANNING_COPY_SUFFIX)
    copy_path.write_text(text)
    try:
        yield copy_path, dropped
    finally:
        copy_path.unlink(missing_ok=True)


def _error_ratio(result: Dict[str, float]) -> float:
    """The share of problems that ended in none of ``problem_solving``'s buckets."""
    classified = sum(
        result.get(key) or 0.0
        for key in ("solving_ratio", "false_plans_ratio", "unsolvable_ratio",
                    "timed_out", "syntax_errors")
    )
    return round(max(0.0, 1.0 - classified), 6)


def solving_metrics(
    model_path: Path,
    ref_domain_path: Path,
    test_problems: Sequence[str],
    timeout: int = 60,
) -> Dict[str, Optional[object]]:
    """Plan the test problems with the model's planning copy.

    Args:
        model_path: The learned model.
        ref_domain_path: The reference domain plans are validated against.
        test_problems: Problem file paths.
        timeout: Planner timeout per problem, in seconds.

    Returns:
        The :data:`SOLVING_FIELDS`. ``planning_error_ratio`` is the share of
        problems the planner neither solved, failed, timed out on nor refused
        to parse; ``planning_dropped_operators`` is the comma-joined names of
        the no-effect operators left out of the planning copy.
    """
    with planning_copy(Path(model_path)) as (path, dropped):
        result = problem_solving(
            str(path), str(ref_domain_path), list(test_problems), timeout=timeout
        )
    return {
        "solving_ratio": result.get("solving_ratio"),
        "false_plans_ratio": result.get("false_plans_ratio"),
        "unsolvable_ratio": result.get("unsolvable_ratio"),
        "planning_timed_out_ratio": result.get("timed_out"),
        "planning_syntax_error_ratio": result.get("syntax_errors"),
        "planning_error_ratio": _error_ratio(result),
        "planning_dropped_operators": ",".join(dropped),
    }
