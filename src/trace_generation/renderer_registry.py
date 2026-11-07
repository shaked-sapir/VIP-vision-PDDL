"""Pick a :class:`FrameRenderer` by domain key.

    blocksworld, hanoi, npuzzle  ->  PDDLGymRenderer (pddlgym.rendering.*)
    depot, gripper               ->  VisualizerRenderer (../planning-visualizer)

The visualizer checkout is read from ``rendering.planning_visualizer_dir`` in
``config.yaml``, falling back to the sibling ``../planning-visualizer`` checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.trace_generation.frame_renderer import FrameRenderer
from src.trace_generation.pddlgym_renderer import (
    BLOCKS_SCHEMA,
    HANOI_SCHEMA,
    NPUZZLE_SCHEMA,
    PDDLGymRenderer,
    prime_block_colors,
)
from src.trace_generation.visualizer_renderer import VisualizerRenderer
from src.utils.config import load_config

CONFIG_SECTION = "rendering"
CONFIG_VISUALIZER_KEY = "planning_visualizer_dir"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VISUALIZER_ROOT = PROJECT_ROOT.parent / "planning-visualizer"


def _blocksworld(_: Dict[str, Any]) -> FrameRenderer:
    from pddlgym.rendering.blocks import render

    return PDDLGymRenderer(render, BLOCKS_SCHEMA, prime_colors=prime_block_colors,
                           name="pddlgym.blocks")


def _hanoi(_: Dict[str, Any]) -> FrameRenderer:
    from pddlgym.rendering.hanoi import render

    return PDDLGymRenderer(render, HANOI_SCHEMA, name="pddlgym.hanoi")


def _npuzzle(_: Dict[str, Any]) -> FrameRenderer:
    from pddlgym.rendering.slidetile import render

    return PDDLGymRenderer(render, NPUZZLE_SCHEMA, name="pddlgym.slidetile")


def _visualizer(domain_key: str) -> Callable[[Dict[str, Any]], FrameRenderer]:
    def build(options: Dict[str, Any]) -> FrameRenderer:
        return VisualizerRenderer(
            options["visualizer_root"], domain_key,
            overlay=options.get("overlay", False))
    return build


RENDERER_REGISTRY: Dict[str, Callable[[Dict[str, Any]], FrameRenderer]] = {
    "blocksworld": _blocksworld,
    "hanoi": _hanoi,
    "npuzzle": _npuzzle,
    "depot": _visualizer("depot"),
    "gripper": _visualizer("gripper"),
}

RENDERABLE_DOMAINS = sorted(RENDERER_REGISTRY)


def visualizer_root_from_config(config: Optional[Dict[str, Any]] = None) -> Path:
    """``rendering.planning_visualizer_dir`` from ``config.yaml``, or the sibling checkout."""
    config = config if config is not None else load_config()
    configured = config.get(CONFIG_SECTION, {}).get(CONFIG_VISUALIZER_KEY)
    return Path(configured) if configured else DEFAULT_VISUALIZER_ROOT


def build_renderer(
    domain: str,
    *,
    visualizer_root: Optional[Path] = None,
    overlay: bool = False,
) -> FrameRenderer:
    """The renderer registered for ``domain``.

    Args:
        domain: A key of :data:`RENDERER_REGISTRY`.
        visualizer_root: The ``planning-visualizer`` checkout, for depot and
            gripper. Read from ``config.yaml`` when omitted.
        overlay: Draw the visualizer's step/action stamp. Ignored by PDDLGym.
            Off by default, as it was for the small depot and gripper corpora.

    Raises:
        ValueError: ``domain`` has no renderer.
    """
    if domain not in RENDERER_REGISTRY:
        raise ValueError(
            f"No renderer for domain {domain!r}; renderable domains are "
            f"{RENDERABLE_DOMAINS}.")
    options: Dict[str, Any] = {"overlay": overlay}
    if domain in ("depot", "gripper"):
        options["visualizer_root"] = (visualizer_root if visualizer_root is not None
                                      else visualizer_root_from_config())
    return RENDERER_REGISTRY[domain](options)
