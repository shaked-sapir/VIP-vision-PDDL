"""Unit tests for :class:`RenderedSource` with a fake renderer.

    ./venv11/bin/python -m pytest src/trace_generation/test_frame_renderer.py -q
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Hashable, Iterator, List, Optional, Tuple

from PIL import Image

from src.trace_generation.cutter import CutMode, cut
from src.trace_generation.emitter import emit_window
from src.trace_generation.frame_renderer import RenderedSource, state_key
from src.trace_generation.sources import TraceSource
from src.trace_generation.trace_step import TraceState, TraceStep

OBJECTS = ("a:block", "b:block")
S0 = TraceState(("ontable(a:block)", "ontable(b:block)", "clear(a:block)",
                 "clear(b:block)", "handempty()"), OBJECTS)
S1 = TraceState(("holding(a:block)", "ontable(b:block)", "clear(b:block)"), OBJECTS)


class FakeRenderer:
    """Paints a solid colour per distinct state and counts its calls."""

    def __init__(self) -> None:
        self.calls: List[Tuple[TraceState, int, Optional[str]]] = []

    def render(self, state: TraceState, *, step: int, action: Optional[str]) -> Image.Image:
        self.calls.append((state, step, action))
        shade = (len(state.literals) * 40) % 256
        return Image.new("RGB", (4, 4), (shade, shade, shade))

    def frame_key(self, state: TraceState, *, step: int, action: Optional[str]) -> Hashable:
        return state_key(state)

    def describe(self) -> Dict[str, Any]:
        return {"name": "fake"}


class ListSource(TraceSource):
    """A source over a fixed list of symbolic steps."""

    def __init__(self, steps: List[TraceStep]) -> None:
        self._steps = steps

    @property
    def domain_name(self) -> str:
        return "blocks"

    @property
    def objects(self) -> Tuple[str, ...]:
        return OBJECTS

    def steps(self) -> Iterator[TraceStep]:
        yield from self._steps

    def describe(self) -> Dict[str, Any]:
        return {"source_kind": "list", "source_file": "memory"}


def _back_and_forth(rounds: int = 3) -> List[TraceStep]:
    """pick_up a, put_down a, repeated: only two distinct states."""
    steps = []
    for _ in range(rounds):
        steps.append(TraceStep(S0, "pick_up(a:block)", S1))
        steps.append(TraceStep(S1, "put_down(a:block)", S0))
    return steps


def test_every_step_carries_both_frames(tmp_path):
    source = RenderedSource(ListSource(_back_and_forth()), FakeRenderer(),
                            frames_dir=tmp_path)
    steps = list(source.steps())
    assert len(steps) == 6
    assert all(step.has_frames for step in steps)
    assert all(step.frame_before.is_file() and step.frame_after.is_file()
               for step in steps)


def test_symbolic_content_is_passed_through_unchanged(tmp_path):
    inner = _back_and_forth()
    steps = list(RenderedSource(ListSource(inner), FakeRenderer(),
                                frames_dir=tmp_path).steps())
    assert [(s.prev_state, s.action, s.next_state) for s in steps] == \
        [(s.prev_state, s.action, s.next_state) for s in inner]


def test_a_repeated_state_is_rendered_once(tmp_path):
    renderer = FakeRenderer()
    source = RenderedSource(ListSource(_back_and_forth(rounds=5)), renderer,
                            frames_dir=tmp_path)
    steps = list(source.steps())
    assert len(renderer.calls) == 2
    assert source.frames_rendered == 2
    assert {step.frame_before for step in steps} | {step.frame_after for step in steps} \
        == {steps[0].frame_before, steps[0].frame_after}


def test_consecutive_steps_share_the_frame_of_their_common_state(tmp_path):
    steps = list(RenderedSource(ListSource(_back_and_forth()), FakeRenderer(),
                                frames_dir=tmp_path).steps())
    for earlier, later in zip(steps, steps[1:]):
        assert later.frame_before == earlier.frame_after


def test_the_renderer_sees_trace_step_index_and_leading_action(tmp_path):
    renderer = FakeRenderer()
    list(RenderedSource(ListSource(_back_and_forth(rounds=1)), renderer,
                        frames_dir=tmp_path).steps())
    assert renderer.calls[0][1:] == (0, None)
    assert renderer.calls[1][1:] == (1, "pick_up(a:block)")


def test_a_key_that_includes_the_step_renders_every_state(tmp_path):
    class PerStepRenderer(FakeRenderer):
        def frame_key(self, state, *, step, action):
            return state_key(state), step

    renderer = PerStepRenderer()
    source = RenderedSource(ListSource(_back_and_forth(rounds=2)), renderer,
                            frames_dir=tmp_path)
    list(source.steps())
    assert source.frames_rendered == 5


def test_frames_dir_defaults_to_a_fresh_temporary_directory():
    source = RenderedSource(ListSource(_back_and_forth(rounds=1)), FakeRenderer())
    assert source.frames_dir.is_dir()
    assert not any(source.frames_dir.iterdir())


def test_describe_records_the_inner_source_and_the_renderer(tmp_path):
    described = RenderedSource(ListSource([]), FakeRenderer(), frames_dir=tmp_path).describe()
    assert described["source_kind"] == "list"
    assert described["renderer"] == {"name": "fake"}


def test_domain_name_and_objects_come_from_the_inner_source(tmp_path):
    source = RenderedSource(ListSource([]), FakeRenderer(), frames_dir=tmp_path)
    assert source.domain_name == "blocks"
    assert source.objects == OBJECTS


S2 = TraceState(("ontable(a:block)", "ontable(b:block)", "clear(a:block)",
                 "clear(b:block)", "handempty()", "on(b:block,a:block)"), OBJECTS)


def _three_states() -> List[TraceStep]:
    """A window whose first and last states differ, so the cutter keeps it."""
    return [TraceStep(S0, "pick_up(a:block)", S1), TraceStep(S1, "stack(b:block,a:block)", S2)]


def test_the_emitter_writes_the_rendered_frames(tmp_path):
    source = RenderedSource(ListSource(_three_states()), FakeRenderer(),
                            frames_dir=tmp_path / "frames")
    window = cut(source.steps(), mode=CutMode.NONE)[0]
    emitted = emit_window(window, problem_name="problem0", index=0,
                          domain_name="blocks", corpus_root=tmp_path / "corpus",
                          render=True)
    assert emitted.rendered
    frames = sorted(emitted.problem_dir.glob("state_*.png"))
    assert [f.name for f in frames] == ["state_000000.png", "state_000001.png",
                                        "state_000002.png"]
    assert Image.open(frames[0]).tobytes() != Image.open(frames[1]).tobytes()
    assert Image.open(frames[1]).tobytes() == Image.open(
        window.steps[0].frame_after).tobytes()


def test_without_render_the_emitter_writes_no_frames(tmp_path):
    source = RenderedSource(ListSource(_three_states()), FakeRenderer(),
                            frames_dir=tmp_path / "frames")
    window = cut(source.steps(), mode=CutMode.NONE)[0]
    emitted = emit_window(window, problem_name="problem0", index=0,
                          domain_name="blocks", corpus_root=tmp_path / "corpus",
                          render=False)
    assert not emitted.rendered
    assert not list(emitted.problem_dir.glob("state_*.png"))
