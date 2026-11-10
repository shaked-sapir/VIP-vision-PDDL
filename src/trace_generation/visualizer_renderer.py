"""Render eval-schema states with the ``planning-visualizer`` canvas renderers.

The visualizer (a sibling checkout) draws a frame in two stages, and its
``backend/api/generate_images.ts`` is the reference pipeline the depot and
gripper corpora were produced with:

    1. ``backend/planner`` ``RendererFactory`` lays a state out as a
       ``RenderedState`` (positioned visual objects and relations), in Python;
    2. the frontend's TypeScript canvas renderers draw it, in Node, through the
       ``canvas`` package.

:class:`VisualizerRenderer` runs stage 1 in-process and streams stage 2 to a
long-lived ``backend/api/render_states.ts`` worker. The step/action overlay
``generate_images.ts`` draws by default is off here, as it was for the corpora.
With the overlay on the image depends on the step index and action too, so the
cache key then includes both.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Hashable, Optional, Sequence, Set

from PIL import Image

from src.trace_generation.frame_renderer import state_key
from src.trace_generation.pddlgym_renderer import parse_eval_literal
from src.trace_generation.trace_step import TraceState
from src.utils.pddl_gym import parse_gym_to_pddl_ground_action

PLANNER_SUBDIR = Path("backend") / "planner"
API_SUBDIR = Path("backend") / "api"
WORKER_SCRIPT = "render_states.ts"
DOMAIN_FILES = {"depot": "depot", "gripper": "gripper"}
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 600


def planner_dir(visualizer_root: Path) -> Path:
    """The visualizer's Python planner package directory."""
    return Path(visualizer_root) / PLANNER_SUBDIR


def import_visualizer(visualizer_root: Path) -> None:
    """Put the visualizer's planner package on ``sys.path`` once."""
    planner = str(planner_dir(visualizer_root).resolve())
    if not Path(planner).is_dir():
        raise FileNotFoundError(
            f"No planning-visualizer planner package at {planner}. Point "
            "rendering.planning_visualizer_dir in config.yaml at the checkout.")
    if planner not in sys.path:
        sys.path.insert(0, planner)


def visualizer_domain_file(visualizer_root: Path, domain_key: str) -> Path:
    """The visualizer's own PDDL domain file for ``domain_key``."""
    return planner_dir(visualizer_root) / "domains" / DOMAIN_FILES[domain_key] / "domain.pddl"


class CanvasWorker:
    """One ``render_states.ts`` process, fed one JSON request per line."""

    def __init__(self, visualizer_root: Path) -> None:
        api_dir = Path(visualizer_root) / API_SUBDIR
        script = api_dir / WORKER_SCRIPT
        if not script.is_file():
            raise FileNotFoundError(f"No canvas worker at {script}.")
        self._process = subprocess.Popen(
            ["npx", "tsx", WORKER_SCRIPT],
            cwd=api_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1,
        )

    def render(self, request: Dict[str, Any]) -> Path:
        """Send one request and wait for its PNG to be written."""
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            stderr = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"The canvas worker exited:\n{stderr}")
        reply = json.loads(line)
        if "error" in reply:
            raise RuntimeError(f"The canvas worker failed: {reply['error']}")
        return Path(reply["ok"])

    def close(self) -> None:
        """Stop the worker."""
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=30)

    def __del__(self) -> None:
        if self._process.poll() is None:
            self._process.kill()


class VisualizerRenderer:
    """A :class:`FrameRenderer` over one visualizer domain renderer."""

    def __init__(
        self,
        visualizer_root: Path,
        domain_key: str,
        *,
        overlay: bool = False,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
    ) -> None:
        """Bind the visualizer's renderer for ``domain_key``.

        Args:
            visualizer_root: The ``planning-visualizer`` checkout.
            domain_key: ``"depot"`` or ``"gripper"``, the visualizer's name.
            overlay: Stamp ``Step N`` / ``Action: ...`` on each frame.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        import_visualizer(visualizer_root)
        from state_renderer import RendererFactory

        self.visualizer_root = Path(visualizer_root)
        self.domain_key = domain_key
        self.overlay = overlay
        self.width = width
        self.height = height
        self.object_order: Optional[Sequence[str]] = None
        self._layout = RendererFactory.get_renderer(domain_key)
        self._worker: Optional[CanvasWorker] = None

    def use_problem(self, problem_file: Path) -> None:
        """Lay objects out in ``problem_file``'s declared order, as the visualizer does.

        The canvas renderers place same-typed objects left to right in the
        order the visualizer's own PDDL parser lists them, so frames of an
        existing problem are reproduced only with that order. Without a problem
        the state's sorted object universe is used.
        """
        from state_generator.pddl_parser import PDDLParser

        parsed = PDDLParser(str(visualizer_domain_file(self.visualizer_root, self.domain_key)),
                            str(problem_file))
        self.object_order = list(parsed.objects)

    def render(self, state: TraceState, *, step: int, action: Optional[str]) -> Image.Image:
        """Lay the state out, draw it in the worker, and load the PNG back."""
        metadata: Dict[str, Any] = {"step": step}
        if action is not None:
            metadata["action"] = parse_gym_to_pddl_ground_action(action)
        rendered = self._layout.render(
            _facts(state), _object_types(state, self.object_order), metadata)
        path = self._worker_process().render({
            "out": str(self._scratch_frame()),
            "state": _rendered_state_dict(rendered),
            "metadata": self.overlay,
            "width": self.width,
            "height": self.height,
        })
        with Image.open(path) as image:
            return image.copy()

    def frame_key(self, state: TraceState, *, step: int, action: Optional[str]) -> Hashable:
        """The state and object order, plus the overlay's inputs when it is drawn."""
        order = tuple(self.object_order) if self.object_order is not None else None
        if self.overlay:
            return state_key(state), order, step, action
        return state_key(state), order

    def describe(self) -> Dict[str, Any]:
        """Renderer name, checkout, overlay and size."""
        return {"name": "planning-visualizer", "domain": self.domain_key,
                "visualizer_root": str(self.visualizer_root.resolve()),
                "overlay": self.overlay, "width": self.width, "height": self.height}

    def close(self) -> None:
        """Stop the canvas worker, if one was started."""
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    def _worker_process(self) -> CanvasWorker:
        if self._worker is None:
            self._worker = CanvasWorker(self.visualizer_root)
        return self._worker

    def _scratch_frame(self) -> Path:
        import tempfile

        if not hasattr(self, "_scratch"):
            self._scratch = Path(tempfile.mkdtemp(prefix="visualizer_frame_")) / "frame.png"
        return self._scratch


def _facts(state: TraceState) -> Set[Any]:
    """The state's literals as visualizer ``Predicate`` objects."""
    from state_generator.pddl_parser import Predicate

    facts = set()
    for literal in state.literals:
        name, args = parse_eval_literal(literal)
        facts.add(Predicate(name, [obj for obj, _ in args]))
    return facts


def _object_types(state: TraceState,
                  order: Optional[Sequence[str]] = None) -> Dict[str, str]:
    """``{object_name: type_name}`` from the state's universe, in ``order`` if given."""
    types: Dict[str, str] = {}
    for obj in state.objects:
        name, _, typ = obj.partition(":")
        types[name] = typ
    if order is None:
        return types
    if set(order) != set(types):
        raise ValueError(
            f"The problem declares objects {sorted(order)} but the state's universe "
            f"is {sorted(types)}.")
    return {name: types[name] for name in order}


def _rendered_state_dict(rendered: Any) -> Dict[str, Any]:
    """A ``RenderedState`` as the JSON ``generate_images.ts`` hands the canvas."""
    return {
        "domain": rendered.domain,
        "objects": [
            {"id": o.id, "type": o.type, "label": o.label,
             "position": list(o.position) if o.position is not None else None,
             "properties": o.properties or {}}
            for o in rendered.objects
        ],
        "relations": [
            {"type": r.type, "source": r.source, "target": r.target}
            for r in rendered.relations
        ],
        "metadata": rendered.metadata or {},
    }
