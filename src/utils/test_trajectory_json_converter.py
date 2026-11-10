"""A state with no positive literal is still a state.

    python -m pytest src/utils/test_trajectory_json_converter.py

A frame whose every fluent came back UNCERTAIN (or false) is written as
``(:state )``; the parser must keep it, or the action count no longer matches
the state count and the whole problem is rejected.
"""

from pathlib import Path

from src.utils.trajectory_json_converter import _parse_trajectory_file


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "p.trajectory"
    path.write_text(text)
    return path


def test_an_empty_state_is_kept(tmp_path: Path) -> None:
    path = _write(tmp_path, """(
(:init (clear a) (ontable a) (handempty))
(operator: (pick-up a))
(:state )
(operator: (put-down a))
(:state (clear a) (ontable a) (handempty))
)""")
    init, actions, states = _parse_trajectory_file(path)
    assert init == ["clear a", "ontable a", "handempty"]
    assert actions == ["pick-up a", "put-down a"]
    assert states == [[], ["clear a", "ontable a", "handempty"]]


def test_a_full_trajectory_parses_as_before(tmp_path: Path) -> None:
    path = _write(tmp_path, """(
(:init (clear a) (handempty))
(operator: (pick-up a))
(:state (holding a))
)""")
    init, actions, states = _parse_trajectory_file(path)
    assert (init, actions, states) == (["clear a", "handempty"], ["pick-up a"], [["holding a"]])
