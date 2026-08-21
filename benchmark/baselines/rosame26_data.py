"""One prepared fold as the ICAPS-26 network's inputs.

:mod:`benchmark.baselines.image_fold_inputs` resolves a fold into
:class:`~benchmark.baselines.image_fold_inputs.ResolvedTrace` objects — parsed
problem, frames on disk, observed actions, GT endpoint states. This module turns
those into the tensors :mod:`src.milp.trace_tensors` defines, which means four
things the tensor contract deliberately does not know about:

* the **grounding**, which is one ``Instance`` over the object union of the whole
  corpus rather than the fold (``docs/rosame-i-milp-26-implementation-plan.md``
  §4.2a), plus the ``domain_model.json`` / ``objects.json`` the DL head is built
  from and the two head-column maps that bridge the two argument orders (§4.2b);
* **frame selection**, dropping both endpoints per §4.1, and reporting the traces
  that leaves with nothing;
* **resize**, per-domain configurable and defaulting to the aspect-preserving
  ``Resize(64)`` (§4.6), reusing the 24 arm's transform rather than restating it;
* **standardisation**, computed once over every frame of the corpus and cached on
  disk under a ``(domain, resize-form)`` key (§4.4).

The entry point is :func:`build_fold_batch`.

Two dialects reach this module. Predicate and action *names* come from
``src/domains/*.pddl`` via :func:`~src.milp.domain_assets.build_domain` and keep
their hyphens; object names and the experiment's own domain may have been
underscored by ``normalize_experiment_data``. Every name is therefore compared in
the underscored form :func:`canonical` produces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Sequence

import torch
from PIL import Image
from pddl_plus_parser.models import Domain

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.domain_assets import build_domain, write_grounding_assets
from src.milp.head_alignment import action_index_by_name, proposition_index_by_name
from src.milp.trace_tensors import (
    PaddedTraces,
    TraceInput,
    build_padded_traces,
    interior_frame_indices,
)

from benchmark.algorithm_adapters.rosame_i_runner import build_image_tf
from benchmark.baselines.image_fold_inputs import (
    TRAJECTORY_DIR_NAMES,
    ResolvedTrace,
    parse_problem_normalized,
    resolve_images,
)
from benchmark.baselines.rosame_i_runner import ResizeSpec, _resize_tag

from dl.util.ROSAME.rosame import get_domain_model
from planning_structs.instance import Instance as PSInstance

#: Floor applied to the per-pixel standard deviation before dividing by it.
#: Upstream adds ``1e-20``, which on our renders (52%-71% of pixels constant
#: across the corpus) produces values up to 3.6e13.
STD_FLOOR: float = 1e-6


def canonical(name: str) -> str:
    """A name in the underscored dialect with its whitespace normalised.

    Two sources spell the same symbol differently. Hyphens survive in the
    reference domains but not in an experiment that ran
    ``normalize_experiment_data``, and ``untyped_representation`` renders a
    nullary predicate as ``"(handempty )"``, so the fold walk's ``[1:-1]`` slice
    hands over a trailing space the head's key does not have.
    """
    return " ".join(name.replace("-", "_").split())


@dataclass(frozen=True)
class Grounding:
    """The one grounding a run shares, and the maps into the DL head.

    Attributes:
        bench: The bench key, e.g. ``"blocksworld"``.
        instance: The CP grounding, over ``objects``.
        domain_model: The vendored ``Domain_Model`` the DL head is sized from.
        proposition_index: ``"on a b"`` -> head column, keys canonicalised.
        action_index: ``"pick_up a"`` -> head index, keys canonicalised.
        objects: ``{type: [object]}``, the corpus union.
        assets_root: What the network takes as ``domain_assets_root``.
    """

    bench: str
    instance: PSInstance
    domain_model: object
    proposition_index: Dict[str, int]
    action_index: Dict[str, int]
    objects: Dict[str, List[str]]
    assets_root: Path


@dataclass(frozen=True)
class NormalisationStats:
    """Per-pixel statistics over every frame of a corpus.

    Attributes:
        mean: ``[C, H, W]``.
        std: ``[C, H, W]``, already floored at :data:`STD_FLOOR`.
    """

    mean: torch.Tensor
    std: torch.Tensor


@dataclass(frozen=True)
class FoldBatch:
    """One fold's batch, with the traces it could not use named.

    Attributes:
        batch: The padded tensors, in ``kept`` order.
        grounding: The run's shared grounding.
        stats: The corpus statistics the frames were standardised by.
        kept: Problem names in the batch, in row order.
        dropped: Problem name -> why it is not in the batch.
    """

    batch: PaddedTraces
    grounding: Grounding
    stats: NormalisationStats
    kept: List[str]
    dropped: Dict[str, str]


def corpus_dir(traces: Sequence[ResolvedTrace]) -> Path:
    """The single staging directory every trace of a fold was prepared into.

    Raises:
        ValueError: if ``traces`` is empty, if they span more than one staging
            directory, or if that directory is not one of
            :data:`~benchmark.baselines.image_fold_inputs.TRAJECTORY_DIR_NAMES`.
    """
    if not traces:
        raise ValueError("no resolved traces, so there is no corpus to ground over")

    dirs = sorted({trace.problem_dir.parent for trace in traces})
    if len(dirs) != 1:
        raise ValueError(
            f"a fold's traces must come from one staging directory; got "
            f"{[str(d) for d in dirs]}. Two staging directories hold the same "
            f"frames in different dialects, so their object unions differ"
        )

    staging = dirs[0]
    if staging.name not in TRAJECTORY_DIR_NAMES:
        raise ValueError(
            f"'{staging}' is not a trajectory staging directory; expected one of "
            f"{sorted(TRAJECTORY_DIR_NAMES)}"
        )
    return staging


def corpus_problem_pddls(staging_dir: Path) -> List[Path]:
    """Every ``<problem>/<problem>.pddl`` under a staging directory, sorted.

    Staging directories also hold loose files such as ``_domain.pddl``, which are
    not problems and are skipped.

    Raises:
        FileNotFoundError: if the directory holds no problem at all.
    """
    pddls = [
        entry / f"{entry.name}.pddl"
        for entry in sorted(staging_dir.iterdir())
        if entry.is_dir() and (entry / f"{entry.name}.pddl").exists()
    ]
    if not pddls:
        raise FileNotFoundError(
            f"no '<problem>/<problem>.pddl' under '{staging_dir}', so the corpus "
            f"object union would be empty"
        )
    return pddls


def corpus_object_union(
    partial_domain: Domain, staging_dir: Path, bench: str
) -> Dict[str, List[str]]:
    """``{type: [object]}`` over every problem of the corpus, not just the fold.

    Type names are returned in the spelling ``src/domains`` declares, whatever
    dialect the problem PDDLs are written in.

    Raises:
        ValueError: if one object name is declared under two types, or if a
            problem uses a type the reference domain does not declare.
    """
    declared = {canonical(t.name): t.name for t in build_domain(bench).types_set}

    by_type: Dict[str, List[str]] = {}
    type_of: Dict[str, str] = {}
    for problem_pddl in corpus_problem_pddls(staging_dir):
        problem = parse_problem_normalized(partial_domain, problem_pddl)
        for name, obj in problem.objects.items():
            key = canonical(obj.type.name)
            if key not in declared:
                raise ValueError(
                    f"'{problem_pddl.name}' declares object '{name}' of type "
                    f"'{obj.type.name}', which '{bench}' does not have; declared "
                    f"types are {sorted(declared.values())}"
                )
            type_name = declared[key]
            if name in type_of and type_of[name] != type_name:
                raise ValueError(
                    f"object '{name}' is '{type_of[name]}' in one problem of "
                    f"'{staging_dir}' and '{type_name}' in '{problem_pddl.name}'; "
                    f"one grounding cannot serve both"
                )
            type_of[name] = type_name
            names = by_type.setdefault(type_name, [])
            if name not in names:
                names.append(name)

    return {type_name: sorted(names) for type_name, names in sorted(by_type.items())}


def build_grounding(
    bench: str, objects: Mapping[str, Sequence[str]], assets_root: Path
) -> Grounding:
    """Ground ``bench`` over ``objects`` and write the head's assets.

    The CP grounding and the DL head are two independent groundings of the same
    domain; :func:`~src.milp.head_alignment.proposition_index_by_name` raises
    unless they turn out to be a bijection.
    """
    instance = PSInstance(
        build_domain(bench),
        sorted(
            (name, type_name)
            for type_name, names in objects.items()
            for name in names
        ),
    )
    write_grounding_assets(bench, objects, assets_root)
    domain_model = get_domain_model(bench, str(assets_root))
    return Grounding(
        bench=bench,
        instance=instance,
        domain_model=domain_model,
        proposition_index=_canonical_keys(
            proposition_index_by_name(instance, domain_model), "proposition"
        ),
        action_index=_canonical_keys(
            action_index_by_name(instance, domain_model), "action"
        ),
        objects={t: list(names) for t, names in objects.items()},
        assets_root=Path(assets_root),
    )


def corpus_frame_paths(staging_dir: Path, bench: str) -> List[Path]:
    """Every ``state_*.png`` of the corpus, in problem order then frame order.

    Endpoint frames are included: the statistic is a property of the corpus, not
    of which frames §4.1 happens to feed the network.

    Raises:
        FileNotFoundError: if no problem of the corpus has frames on disk, which
            means a simulation-mode cell.
    """
    frames: List[Path] = []
    for problem_pddl in corpus_problem_pddls(staging_dir):
        frames.extend(
            resolve_images(problem_pddl.parent, bench, problem_pddl.stem)
        )
    if not frames:
        raise FileNotFoundError(
            f"no 'state_*.png' under any problem of '{staging_dir}' nor under "
            f"src/domains/{bench}/problems/, so there are no images to normalise"
        )
    return frames


def load_frames(
    paths: Sequence[Path], image_tf: Callable[[Image.Image], torch.Tensor]
) -> torch.Tensor:
    """``[N, C, H, W]`` from ``paths``, transformed by ``image_tf``.

    Raises:
        ValueError: if ``paths`` is empty, or if two frames transform to
            different shapes and so cannot be stacked.
    """
    if not paths:
        raise ValueError("no frames to load")

    frames: List[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(image_tf(img.convert("RGB")))
        if frames[-1].shape != frames[0].shape:
            raise ValueError(
                f"'{path}' transforms to {tuple(frames[-1].shape)} but "
                f"'{paths[0]}' to {tuple(frames[0].shape)}; an int resize "
                f"preserves aspect ratio, so frames of different native shapes "
                f"stay different"
            )
    return torch.stack(frames)


def normalisation_stats(
    staging_dir: Path,
    bench: str,
    resize: ResizeSpec,
    image_tf: Callable[[Image.Image], torch.Tensor],
) -> NormalisationStats:
    """Corpus-level per-pixel mean and std, cached on disk (§4.4).

    The cache is keyed on ``(bench, resize-form)`` and lives beside the corpus,
    so a fold, a re-run and a backfill of the same cell standardise identically.
    A cached entry whose shape does not match the frames is stale and recomputed;
    one frame is loaded to check that, rather than the whole corpus.
    """
    cache_path = staging_dir / f".rosame26_norm__{bench}__res={_resize_tag(resize)}.pt"
    paths = corpus_frame_paths(staging_dir, bench)

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        probe = load_frames(paths[:1], image_tf)
        if tuple(cached["mean"].shape) == tuple(probe.shape[1:]):
            return NormalisationStats(mean=cached["mean"], std=cached["std"])

    frames = load_frames(paths, image_tf)
    stats = NormalisationStats(
        mean=frames.mean(dim=0),
        std=frames.std(dim=0, unbiased=False).clamp_min(STD_FLOOR),
    )
    torch.save({"mean": stats.mean, "std": stats.std}, cache_path)
    return stats


def standardise(images: torch.Tensor, stats: NormalisationStats) -> torch.Tensor:
    """``(images - mean) / std``, broadcasting ``[C, H, W]`` over ``[T, ...]``.

    Raises:
        ValueError: if the frame shape does not match the statistic's.
    """
    if tuple(images.shape[1:]) != tuple(stats.mean.shape):
        raise ValueError(
            f"frames are {tuple(images.shape[1:])} but the corpus statistic is "
            f"{tuple(stats.mean.shape)}; the two were computed under different "
            f"resize settings"
        )
    return (images - stats.mean) / stats.std


def build_fold_batch(
    traces: Sequence[ResolvedTrace],
    partial_domain: Domain,
    bench: str,
    assets_root: Path,
    resize: ResizeSpec,
) -> FoldBatch:
    """Turn one resolved fold into the network's inputs.

    Traces with fewer than three frames are dropped and named in
    :attr:`FoldBatch.dropped`; everything else that does not fit the contract is
    a data defect and raises.

    Raises:
        ValueError: if every trace is dropped, or if a trace's action count does
            not span its frames.
    """
    staging_dir = corpus_dir(traces)
    grounding = build_grounding(
        bench, corpus_object_union(partial_domain, staging_dir, bench), assets_root
    )
    image_tf = build_image_tf(resize)
    stats = normalisation_stats(staging_dir, bench, resize, image_tf)

    inputs: List[TraceInput] = []
    kept: List[str] = []
    dropped: Dict[str, str] = {}
    for trace in traces:
        try:
            indices = interior_frame_indices(len(trace.image_paths))
        except ValueError as exc:
            dropped[trace.problem_name] = str(exc)
            continue
        inputs.append(_trace_input(trace, indices, image_tf, stats))
        kept.append(trace.problem_name)

    if not inputs:
        raise ValueError(
            f"every trace of '{staging_dir}' was dropped for having no interior "
            f"frame: {sorted(dropped)}"
        )

    return FoldBatch(
        batch=build_padded_traces(
            inputs,
            proposition_index=grounding.proposition_index,
            action_index=grounding.action_index,
        ),
        grounding=grounding,
        stats=stats,
        kept=kept,
        dropped=dropped,
    )


def _trace_input(
    trace: ResolvedTrace,
    indices: Sequence[int],
    image_tf: Callable[[Image.Image], torch.Tensor],
    stats: NormalisationStats,
) -> TraceInput:
    """One trace's interior frames and canonicalised symbols."""
    if len(trace.action_strings) != len(indices) + 1:
        raise ValueError(
            f"'{trace.problem_name}' has {len(trace.image_paths)} frames, which "
            f"span {len(indices) + 1} actions, but its trajectory carries "
            f"{len(trace.action_strings)}"
        )
    frames = load_frames([trace.image_paths[i] for i in indices], image_tf)
    return TraceInput(
        images=standardise(frames, stats),
        init_predicates=[canonical(p) for p in trace.gt_init_predicates],
        goal_predicates=[canonical(p) for p in trace.gt_final_predicates],
        action_strings=[canonical(a) for a in trace.action_strings],
    )


def _canonical_keys(index: Mapping[str, int], kind: str) -> Dict[str, int]:
    """``index`` rekeyed by :func:`canonical`, raising if two keys collapse."""
    out: Dict[str, int] = {}
    for key, value in index.items():
        collapsed = canonical(key)
        if collapsed in out:
            raise ValueError(
                f"{kind}s {key!r} and another collapse to {collapsed!r} once "
                f"hyphens are normalised, so they cannot be told apart"
            )
        out[collapsed] = value
    return out
