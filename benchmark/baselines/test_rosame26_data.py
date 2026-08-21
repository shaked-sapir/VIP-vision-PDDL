"""Tests for :mod:`benchmark.baselines.rosame26_data`.

    python -m pytest benchmark/baselines/test_rosame26_data.py

The fold adapter's job is the four things the pure tensor contract cannot know:
the grounding is corpus-wide and not fold-wide (§4.2a), both endpoint frames are
dropped and the traces that leaves empty are *named* rather than silently missing
(§4.1, §8 item 11b), the corpus statistic is cached under a key that survives a
re-run (§4.4), and hyphens do not decide whether a proposition has a column.

The corpus below is written to disk rather than mocked, because three of those
four are properties of a directory layout: a staging directory holds loose files
that are not problems, its problems declare overlapping-but-unequal object sets,
and the frames of a domain like depot do not live next to the problem PDDL.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import pytest

torch = pytest.importorskip("torch")
from PIL import Image

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from benchmark.baselines.image_fold_inputs import (
    ResolvedTrace,
    parse_problem_normalized,
)
from benchmark.baselines.rosame26_data import (
    STD_FLOOR,
    build_fold_batch,
    build_grounding,
    canonical,
    corpus_dir,
    corpus_frame_paths,
    corpus_object_union,
    corpus_problem_pddls,
    load_frames,
    normalisation_stats,
    standardise,
)
from src.utils.config import load_config

from benchmark.algorithm_adapters.rosame_i_runner import build_image_tf

RESIZE = 8
FRAME_SIZE = (16, 16)


@pytest.fixture(scope="module")
def blocks_domain() -> Domain:
    return DomainParser(
        Path(load_config()["domains"]["blocksworld"]["domain_file"]),
        partial_parsing=True,
    ).parse_domain()


def _problem_pddl(name: str, blocks: Sequence[str]) -> str:
    objects = " ".join(blocks)
    init = " ".join(f"(clear {b}) (ontable {b})" for b in blocks)
    return (
        f"(define (problem {name}) (:domain blocks)\n"
        f" (:objects {objects} - block)\n"
        f" (:init {init} (handempty))\n"
        f" (:goal (and (holding {blocks[0]}))))\n"
    )


def _write_frames(problem_dir: Path, count: int, size=FRAME_SIZE) -> List[Path]:
    """``count`` distinct frames, so the corpus statistic is not degenerate."""
    paths = []
    for index in range(count):
        path = problem_dir / f"state_{index}.png"
        Image.new("RGB", size, (index * 7 % 256, 30, 200)).save(path)
        paths.append(path)
    return paths


def _write_corpus(
    root: Path, problems: Dict[str, Sequence[str]], frames: Dict[str, int]
) -> Path:
    """A ``<data>/training/trajectories/`` staging directory, frames included."""
    staging = root / "training" / "trajectories"
    staging.mkdir(parents=True)
    for name, blocks in problems.items():
        problem_dir = staging / name
        problem_dir.mkdir()
        (problem_dir / f"{name}.pddl").write_text(_problem_pddl(name, blocks))
        _write_frames(problem_dir, frames.get(name, 4))
    return staging


def _trace(
    domain: Domain, staging: Path, name: str, actions: Sequence[str] | None = None
) -> ResolvedTrace:
    problem_dir = staging / name
    image_paths = sorted(
        problem_dir.glob("state_*.png"), key=lambda p: int(p.stem.split("_")[1])
    )
    problem = parse_problem_normalized(domain, problem_dir / f"{name}.pddl")
    blocks = sorted(problem.objects)
    return ResolvedTrace(
        problem=problem,
        problem_name=name,
        problem_dir=problem_dir,
        image_paths=image_paths,
        action_strings=(
            list(actions)
            if actions is not None
            else [f"pick_up {blocks[0]}"] * max(len(image_paths) - 1, 0)
        ),
        # "handempty " keeps the trailing space `state_positive_predicates`
        # produces for a nullary; the adapter has to survive it.
        gt_init_predicates=[f"clear {b}" for b in blocks] + ["handempty "],
        gt_final_predicates=[f"holding {blocks[0]}"],
        gt_trajectory_path=problem_dir / f"{name}.trajectory",
    )


@pytest.fixture
def corpus(tmp_path, blocks_domain) -> Path:
    """Three problems; ``problem3`` alone names block ``d``, and has 2 frames."""
    return _write_corpus(
        tmp_path,
        {"problem1": ["a", "b"], "problem2": ["a", "c"], "problem3": ["a", "d"]},
        {"problem1": 4, "problem2": 5, "problem3": 2},
    )


# ── locating the corpus ─────────────────────────────────────────────────


class TestCorpusDir:
    def test_it_is_the_staging_directory_the_fold_was_prepared_into(
        self, corpus, blocks_domain
    ) -> None:
        traces = [_trace(blocks_domain, corpus, "problem1")]
        assert corpus_dir(traces) == corpus

    def test_an_empty_fold_raises(self) -> None:
        with pytest.raises(ValueError, match="no resolved traces"):
            corpus_dir([])

    def test_two_staging_directories_raise(self, tmp_path, blocks_domain) -> None:
        one = _write_corpus(tmp_path / "a", {"problem1": ["a"]}, {})
        two = _write_corpus(tmp_path / "b", {"problem2": ["a"]}, {})
        traces = [
            _trace(blocks_domain, one, "problem1"),
            _trace(blocks_domain, two, "problem2"),
        ]
        with pytest.raises(ValueError, match="one staging directory"):
            corpus_dir(traces)

    def test_a_problem_outside_the_known_layout_raises(
        self, tmp_path, blocks_domain, corpus
    ) -> None:
        stray = tmp_path / "elsewhere" / "problem1"
        stray.mkdir(parents=True)
        (stray / "problem1.pddl").write_text(_problem_pddl("problem1", ["a"]))
        trace = _trace(blocks_domain, corpus, "problem1")
        moved = ResolvedTrace(**{**vars(trace), "problem_dir": stray})
        with pytest.raises(ValueError, match="not a trajectory staging directory"):
            corpus_dir([moved])


class TestCorpusProblemPddls:
    def test_every_problem_is_found(self, corpus) -> None:
        assert [p.stem for p in corpus_problem_pddls(corpus)] == [
            "problem1",
            "problem2",
            "problem3",
        ]

    def test_a_loose_file_is_not_mistaken_for_a_problem(self, corpus) -> None:
        """Real staging directories ship a `_domain.pddl` next to the problems."""
        (corpus / "_domain.pddl").write_text("(define (domain blocks))")
        assert len(corpus_problem_pddls(corpus)) == 3

    def test_a_directory_without_its_own_pddl_is_skipped(self, corpus) -> None:
        (corpus / "leftovers").mkdir()
        assert len(corpus_problem_pddls(corpus)) == 3

    def test_an_empty_staging_directory_raises(self, tmp_path) -> None:
        empty = tmp_path / "training" / "trajectories"
        empty.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="no '<problem>/<problem>.pddl'"):
            corpus_problem_pddls(empty)


# ── the object union (§4.2a) ────────────────────────────────────────────


class TestCorpusObjectUnion:
    def test_it_spans_the_corpus_not_the_fold(self, corpus, blocks_domain) -> None:
        union = corpus_object_union(blocks_domain, corpus, "blocksworld")
        assert union == {"block": ["a", "b", "c", "d"]}

    def test_one_object_name_under_two_types_raises(self, tmp_path) -> None:
        """One `Instance` for the run cannot serve two typings of the same name."""
        domain = DomainParser(
            Path(load_config()["domains"]["gripper"]["domain_file"]),
            partial_parsing=True,
        ).parse_domain()
        staging = tmp_path / "training" / "trajectories"
        for name, objects in [
            ("g1", "x - ball rooma - room left - gripper"),
            ("g2", "x - room ball1 - ball left - gripper"),
        ]:
            problem_dir = staging / name
            problem_dir.mkdir(parents=True)
            (problem_dir / f"{name}.pddl").write_text(
                f"(define (problem {name}) (:domain gripper)\n"
                f" (:objects {objects})\n"
                f" (:init (free left))\n (:goal (and (free left))))\n"
            )
        with pytest.raises(ValueError, match="one grounding cannot serve both"):
            corpus_object_union(domain, staging, "gripper")


# ── the grounding and the head maps ─────────────────────────────────────


class TestBuildGrounding:
    def test_the_two_groundings_are_the_same_width(self, tmp_path) -> None:
        grounding = build_grounding(
            "blocksworld", {"block": ["a", "b", "c"]}, tmp_path
        )
        assert len(grounding.proposition_index) == len(
            grounding.instance.propositions
        )
        assert len(grounding.action_index) == len(grounding.instance.actions)

    def test_the_head_assets_are_written_where_the_network_looks(
        self, tmp_path
    ) -> None:
        build_grounding("blocksworld", {"block": ["a", "b"]}, tmp_path)
        assets = tmp_path / "blocksworld"
        assert (assets / "domain_model.json").exists()
        assert json.loads((assets / "objects.json").read_text()) == {
            "block": ["a", "b"]
        }

    def test_a_hyphenated_predicate_is_keyed_underscored(self, tmp_path) -> None:
        """gripper's `at-robby` must match an underscored experiment dialect."""
        grounding = build_grounding(
            "gripper",
            {"room": ["rooma"], "ball": ["ball1"], "gripper": ["left"]},
            tmp_path,
        )
        assert "at_robby rooma" in grounding.proposition_index
        assert "at-robby rooma" not in grounding.proposition_index

    def test_a_nullary_predicate_gets_a_bare_key(self, tmp_path) -> None:
        grounding = build_grounding("blocksworld", {"block": ["a"]}, tmp_path)
        assert "handempty" in grounding.proposition_index


# ── frames ──────────────────────────────────────────────────────────────


class TestCorpusFramePaths:
    def test_every_frame_of_every_problem(self, corpus) -> None:
        assert len(corpus_frame_paths(corpus, "blocksworld")) == 4 + 5 + 2

    def test_a_corpus_without_frames_raises(self, tmp_path, blocks_domain) -> None:
        staging = _write_corpus(tmp_path, {"problem1": ["a"]}, {"problem1": 0})
        with pytest.raises(FileNotFoundError, match="no 'state_..png'"):
            corpus_frame_paths(staging, "blocksworld")


class TestLoadFrames:
    def test_the_frames_are_stacked(self, corpus) -> None:
        paths = corpus_frame_paths(corpus, "blocksworld")
        frames = load_frames(paths, build_image_tf(RESIZE))
        assert tuple(frames.shape) == (len(paths), 3, RESIZE, RESIZE)

    def test_no_frames_raises(self) -> None:
        with pytest.raises(ValueError, match="no frames to load"):
            load_frames([], build_image_tf(RESIZE))

    def test_frames_of_two_native_shapes_raise(self, corpus) -> None:
        """An int resize preserves aspect, so two shapes stay two shapes."""
        odd = corpus / "problem1" / "state_9.png"
        Image.new("RGB", (32, 16), (1, 2, 3)).save(odd)
        paths = corpus_frame_paths(corpus, "blocksworld")
        with pytest.raises(ValueError, match="transforms to"):
            load_frames(paths, build_image_tf(RESIZE))


# ── the corpus statistic (§4.4) ─────────────────────────────────────────


class TestNormalisationStats:
    def test_the_shape_is_one_frame(self, corpus) -> None:
        stats = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        assert tuple(stats.mean.shape) == (3, RESIZE, RESIZE)
        assert tuple(stats.std.shape) == (3, RESIZE, RESIZE)

    def test_the_std_is_floored_so_a_constant_pixel_stays_finite(
        self, corpus
    ) -> None:
        """Our renders are 52%-71% constant; upstream's `+1e-20` gives ~1e13."""
        stats = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        assert bool((stats.std >= torch.tensor(STD_FLOOR, dtype=stats.std.dtype)).all())
        frames = load_frames(
            corpus_frame_paths(corpus, "blocksworld"), build_image_tf(RESIZE)
        )
        assert bool(torch.isfinite(standardise(frames, stats)).all())
        assert float(standardise(frames, stats).abs().max()) < 1e3

    def test_the_cache_is_written_under_the_resize_key(self, corpus) -> None:
        normalisation_stats(corpus, "blocksworld", RESIZE, build_image_tf(RESIZE))
        assert (corpus / f".rosame26_norm__blocksworld__res={RESIZE}.pt").exists()

    def test_two_resize_settings_do_not_share_a_cache(self, corpus) -> None:
        normalisation_stats(corpus, "blocksworld", RESIZE, build_image_tf(RESIZE))
        normalisation_stats(corpus, "blocksworld", None, build_image_tf(None))
        assert (corpus / f".rosame26_norm__blocksworld__res={RESIZE}.pt").exists()
        assert (corpus / ".rosame26_norm__blocksworld__res=native.pt").exists()

    def test_a_second_call_reads_the_cache_rather_than_the_corpus(
        self, corpus
    ) -> None:
        """Removing frames after the first call cannot change the answer."""
        first = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        for frame in (corpus / "problem2").glob("state_*.png"):
            frame.unlink()
        second = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        assert torch.equal(first.mean, second.mean)
        assert torch.equal(first.std, second.std)

    def test_a_stale_shape_is_recomputed(self, corpus) -> None:
        cache = corpus / f".rosame26_norm__blocksworld__res={RESIZE}.pt"
        torch.save({"mean": torch.zeros(3, 99, 99), "std": torch.ones(3, 99, 99)}, cache)
        stats = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        assert tuple(stats.mean.shape) == (3, RESIZE, RESIZE)


class TestStandardise:
    def test_it_centres_the_corpus(self, corpus) -> None:
        frames = load_frames(
            corpus_frame_paths(corpus, "blocksworld"), build_image_tf(RESIZE)
        )
        stats = normalisation_stats(
            corpus, "blocksworld", RESIZE, build_image_tf(RESIZE)
        )
        assert float(standardise(frames, stats).mean(dim=0).abs().max()) < 1e-5

    def test_a_statistic_of_the_wrong_shape_raises(self, corpus) -> None:
        frames = load_frames(
            corpus_frame_paths(corpus, "blocksworld"), build_image_tf(RESIZE)
        )
        stats = normalisation_stats(corpus, "blocksworld", None, build_image_tf(None))
        with pytest.raises(ValueError, match="different resize settings"):
            standardise(frames, stats)


# ── the whole fold ──────────────────────────────────────────────────────


class TestBuildFoldBatch:
    def _build(self, corpus, blocks_domain, tmp_path, names):
        return build_fold_batch(
            [_trace(blocks_domain, corpus, name) for name in names],
            blocks_domain,
            "blocksworld",
            tmp_path / "assets",
            RESIZE,
        )

    def test_the_shapes_follow_the_frame_arithmetic(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem1", "problem2"])
        # 4 frames -> T=2, 5 frames -> T=3.
        assert fold.batch.lengths.tolist() == [2, 3]
        assert tuple(fold.batch.images.shape) == (2, 3, 3, RESIZE, RESIZE)
        n_props = len(fold.grounding.proposition_index)
        assert tuple(fold.batch.state_traces.shape) == (2, 5, n_props)

    def test_a_two_frame_trace_is_dropped_and_named(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        """§8 item 11b: the drop is expected, being silent about it is not."""
        fold = self._build(
            corpus, blocks_domain, tmp_path, ["problem1", "problem3", "problem2"]
        )
        assert fold.kept == ["problem1", "problem2"]
        assert list(fold.dropped) == ["problem3"]
        assert "no interior frame" in fold.dropped["problem3"]

    def test_the_grounding_covers_objects_no_kept_trace_uses(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        """`d` is problem3's alone, and problem3 was dropped."""
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem1"])
        assert fold.grounding.objects == {"block": ["a", "b", "c", "d"]}
        assert "clear d" in fold.grounding.proposition_index

    def test_the_batch_rows_follow_the_kept_order(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem2", "problem1"])
        assert fold.kept == ["problem2", "problem1"]
        assert fold.batch.lengths.tolist() == [3, 2]

    def test_the_gt_endpoints_are_the_state_trace_endpoints(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem1"])
        index = fold.grounding.proposition_index
        init = fold.batch.state_traces[0, 0]
        goal = fold.batch.state_traces[0, -1]
        assert float(init[index["handempty"]]) == 1.0
        assert float(init[index["clear a"]]) == 1.0
        assert float(goal[index["holding a"]]) == 1.0
        assert float(goal[index["handempty"]]) == 0.0

    def test_the_interior_state_rows_are_filler_not_our_states(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem1"])
        assert float(fold.batch.state_traces[0, 1:-1].abs().sum()) == 0.0

    def test_a_fold_of_only_short_traces_raises(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        with pytest.raises(ValueError, match="every trace .* was dropped"):
            self._build(corpus, blocks_domain, tmp_path, ["problem3"])

    def test_an_action_count_that_does_not_span_the_frames_raises(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        trace = _trace(blocks_domain, corpus, "problem1", actions=["pick_up a"])
        with pytest.raises(ValueError, match="span 3 actions, but its trajectory"):
            build_fold_batch(
                [trace], blocks_domain, "blocksworld", tmp_path / "assets", RESIZE
            )

    def test_the_frames_handed_over_are_the_interior_ones(
        self, corpus, blocks_domain, tmp_path
    ) -> None:
        fold = self._build(corpus, blocks_domain, tmp_path, ["problem1"])
        expected = standardise(
            load_frames(
                [corpus / "problem1" / f"state_{i}.png" for i in (1, 2)],
                build_image_tf(RESIZE),
            ),
            fold.stats,
        )
        assert torch.allclose(fold.batch.images[0], expected)


class TestCanonical:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("at-robby rooma", "at_robby rooma"),
            ("on a b", "on a b"),
            # `untyped_representation[1:-1]` of a nullary keeps a trailing space.
            ("handempty ", "handempty"),
            ("empty-crane  hoist0", "empty_crane hoist0"),
            ("", ""),
        ],
    )
    def test_the_two_dialects_meet(self, name, expected) -> None:
        assert canonical(name) == expected

    def test_it_is_idempotent(self) -> None:
        assert canonical(canonical("at-robby  rooma ")) == "at_robby rooma"
