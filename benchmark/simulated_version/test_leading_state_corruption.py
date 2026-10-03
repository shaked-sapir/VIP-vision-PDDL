"""Tests for redrawing a frozen observation's leading states from ground truth.

    python -m pytest benchmark/simulated_version/test_leading_state_corruption.py
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain, Observation

from benchmark.experiment_running_helpers.cleaned_trajectories import (
    extract_masking_info_from_observation,
)
from benchmark.experiment_running_helpers.simulated_data_utils import (
    load_gt_observation,
    prepare_simulated_observations,
)
from benchmark.simulated_version.leading_state_corruption import (
    LEADING_STATE_INDICES,
    DegradationSpec,
    hide_masked_values,
    leading_state_seed,
    observation_states,
    stage_noisy_initial_observation,
)
from src.observation_degradation.masking import MaskingType
from src.observation_degradation.noising import NoisingType
from src.utils.masking import load_masked_observation, save_masking_info
from src.utils.pddl_state import get_state_grounded_predicates
from src.utils.pddl_trajectory import observation_to_trajectory_file

BLOCKS_DOMAIN = Path(__file__).resolve().parents[2] / "src" / "domains" / "blocks" / "blocks.pddl"
PROBLEM = "problem7"

_PROBLEM_PDDL = """
(define (problem problem7) (:domain blocks)
  (:objects a - block b - block c - block)
  (:init (ontable a) (ontable b) (ontable c) (clear a) (clear b) (clear c) (handempty))
  (:goal (and (on c a) (on a b)))
)
"""

_GT_TRAJECTORY = "\n".join([
    "(",
    "(:init (clear a) (clear b) (clear c) (handempty) (ontable a) (ontable b) (ontable c))",
    "(operator: (pick_up a))",
    "(:state (clear b) (clear c) (holding a) (ontable b) (ontable c))",
    "(operator: (stack a b))",
    "(:state (clear a) (clear c) (handempty) (on a b) (ontable b) (ontable c))",
    "(operator: (pick_up c))",
    "(:state (clear a) (holding c) (on a b) (ontable b))",
    "(operator: (stack c a))",
    "(:state (clear c) (handempty) (on a b) (on c a) (ontable b))",
    ")",
])


def spec(masking_p: float, noising_p: float, seed: int = 42) -> DegradationSpec:
    return DegradationSpec(
        MaskingType.PERCENTAGE, masking_p, NoisingType.PERCENTAGE, noising_p, seed,
    )


def write_blocks_fixture(root: Path, masking_p: float, noising_p: float) -> Dict[str, Path]:
    """A one-problem data dir plus that problem's frozen degraded observation.

    Returns the paths under the keys ``data_dir``, ``problem``, ``gt``,
    ``frozen_trajectory``, ``frozen_masking`` and ``frozen_dir``.
    """
    data_dir = root / "data"
    problem_dir = data_dir / "training" / "trajectories" / PROBLEM
    gt_dir = data_dir / "gt_trajectories" / PROBLEM
    frozen_dir = root / "frozen" / "original_observations"
    for directory in (problem_dir, gt_dir, frozen_dir):
        directory.mkdir(parents=True)

    problem = problem_dir / f"{PROBLEM}.pddl"
    problem.write_text(_PROBLEM_PDDL)
    gt = gt_dir / f"{PROBLEM}.trajectory"
    gt.write_text(_GT_TRAJECTORY)

    observations, _ = prepare_simulated_observations(
        domain_path=BLOCKS_DOMAIN,
        gt_trajectory_paths=[gt],
        masking_strategy=MaskingType.PERCENTAGE,
        masking_p=masking_p,
        noising_strategy=NoisingType.PERCENTAGE,
        noising_p=noising_p,
        seed=42,
        problem_paths=[problem],
    )
    stem = f"original_observation_{PROBLEM}"
    frozen_trajectory = observation_to_trajectory_file(
        observations[0], frozen_dir / f"{stem}.trajectory"
    )
    save_masking_info(
        frozen_dir, stem, extract_masking_info_from_observation(observations[0])
    )
    return {
        "data_dir": data_dir,
        "problem": problem,
        "gt": gt,
        "frozen_trajectory": frozen_trajectory,
        "frozen_masking": frozen_dir / f"{stem}.masking_info",
        "frozen_dir": frozen_dir,
    }


def blocks_domain() -> Domain:
    return DomainParser(BLOCKS_DOMAIN, partial_parsing=True).parse_domain()


def _stage(paths: Dict[str, Path], out: Path, degradation: DegradationSpec,
           fold: int = 0, hide_masked: bool = True) -> Tuple[Path, Path]:
    return stage_noisy_initial_observation(
        paths["frozen_trajectory"], paths["frozen_masking"], paths["gt"],
        paths["problem"], blocks_domain(), degradation, fold, out,
        hide_masked=hide_masked,
    )


def _lines(path: Path) -> List[str]:
    return path.read_text().split("\n")


def _masking_lines(path: Path) -> List[str]:
    return path.read_text().split("\n")[:-1]


def _fluents(state_line: str) -> set:
    """The fluents listed on a ``(:init ...)`` / ``(:state ...)`` line."""
    return {re.sub(r"\s+\)", ")", fluent) for fluent in re.findall(r"\([^()]*\)", state_line)}


def _fluent_key(predicate) -> str:
    """A grounded predicate's text without its polarity."""
    text = predicate.untyped_representation
    return text[len("(not "):-1] if text.startswith("(not ") else text


def _counts_against_gt(staged: Observation, gt: Observation, state_index: int) -> Tuple[int, int, int]:
    """(grounded fluents, masked, flipped among the unmasked) of one staged state."""
    truth = {
        _fluent_key(p): p.is_positive
        for p in get_state_grounded_predicates(observation_states(gt)[state_index])
    }
    predicates = get_state_grounded_predicates(observation_states(staged)[state_index])
    masked = sum(1 for p in predicates if p.is_masked)
    flipped = sum(
        1 for p in predicates
        if not p.is_masked and truth[_fluent_key(p)] != p.is_positive
    )
    return len(predicates), masked, flipped


class TestLaterStatesAreUntouched:
    def test_only_the_leading_state_lines_change(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        staged, _ = _stage(paths, tmp_path / "out", spec(0.2, 0.2))

        frozen_lines, staged_lines = _lines(paths["frozen_trajectory"]), _lines(staged)
        assert len(frozen_lines) == len(staged_lines)
        leading_lines = {1 + 2 * i for i in LEADING_STATE_INDICES}
        for index, (frozen, new) in enumerate(zip(frozen_lines, staged_lines)):
            if index not in leading_lines:
                assert frozen == new, f"line {index} changed"

    def test_later_masking_lines_are_kept_verbatim_when_values_are_kept(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        _, staged_masking = _stage(paths, tmp_path / "out", spec(0.2, 0.2), hide_masked=False)

        frozen, staged = _masking_lines(paths["frozen_masking"]), _masking_lines(staged_masking)
        assert len(frozen) == len(staged) == 5
        assert frozen[len(LEADING_STATE_INDICES):] == staged[len(LEADING_STATE_INDICES):]

    def test_staged_files_keep_the_frozen_names(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        staged, staged_masking = _stage(paths, tmp_path / "out", spec(0.2, 0.2))
        assert staged.name == paths["frozen_trajectory"].name
        assert staged_masking.name == paths["frozen_masking"].name


class TestLeadingStatesAreDegradedLikeTheRest:
    @pytest.mark.parametrize("masking_p, noising_p", [(0.2, 0.2), (0.0, 0.3), (0.4, 0.0)])
    def test_nominal_number_of_masked_and_flipped_fluents(self, tmp_path, masking_p, noising_p):
        paths = write_blocks_fixture(tmp_path, masking_p, noising_p)
        staged, staged_masking = _stage(paths, tmp_path / "out", spec(masking_p, noising_p))
        domain = blocks_domain()
        staged_obs = load_masked_observation(staged, staged_masking, domain, paths["problem"])
        gt_obs = load_gt_observation(paths["gt"], domain, paths["problem"])

        for state_index in LEADING_STATE_INDICES:
            total, masked, flipped = _counts_against_gt(staged_obs, gt_obs, state_index)
            expected_masked = max(1, round(total * masking_p)) if masking_p else 0
            visible = total - expected_masked
            expected_flipped = max(1, round(visible * noising_p)) if noising_p else 0
            assert masked == expected_masked
            assert flipped == expected_flipped

    def test_the_frozen_initial_state_was_clean(self, tmp_path):
        """The premise: without staging, state 0 carries no masking and no flips."""
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        domain = blocks_domain()
        frozen_obs = load_masked_observation(
            paths["frozen_trajectory"], paths["frozen_masking"], domain, paths["problem"],
        )
        gt_obs = load_gt_observation(paths["gt"], domain, paths["problem"])
        _total, masked, flipped = _counts_against_gt(frozen_obs, gt_obs, 0)
        assert (masked, flipped) == (0, 0)

    def test_zero_rates_reproduce_ground_truth(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.0, 0.0)
        staged, staged_masking = _stage(paths, tmp_path / "out", spec(0.0, 0.0))

        gt_lines, staged_lines = _lines(paths["gt"]), _lines(staged)
        frozen_lines = _lines(paths["frozen_trajectory"])
        for state_index in LEADING_STATE_INDICES:
            line = 1 + 2 * state_index
            assert _fluents(staged_lines[line]) == _fluents(gt_lines[line])
            assert _fluents(staged_lines[line]) == _fluents(frozen_lines[line])
        assert _masking_lines(staged_masking) == [""] * 5


class TestDeterminism:
    def test_same_inputs_give_identical_files(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        first = _stage(paths, tmp_path / "out1", spec(0.2, 0.2), hide_masked=False)
        second = _stage(paths, tmp_path / "out2", spec(0.2, 0.2), hide_masked=False)
        assert first[0].read_text() == second[0].read_text()
        assert first[1].read_text() == second[1].read_text()

    def test_redrawn_lines_list_their_fluents_in_sorted_order(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.4, 0.2)
        staged, staged_masking = _stage(paths, tmp_path / "out", spec(0.4, 0.2))
        for state_index in LEADING_STATE_INDICES:
            fluents = re.findall(r"\([^()]*\)", _lines(staged)[1 + 2 * state_index])
            entries = _masking_lines(staged_masking)[state_index].split(", ")
            assert fluents == sorted(fluents)
            assert entries == sorted(entries)

    def test_the_fold_changes_the_draw(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        fold0, _ = _stage(paths, tmp_path / "out0", spec(0.2, 0.2), fold=0)
        fold1, _ = _stage(paths, tmp_path / "out1", spec(0.2, 0.2), fold=1)
        assert _fluents(_lines(fold0)[1]) != _fluents(_lines(fold1)[1])

    def test_seed_is_stable_and_separates_its_inputs(self):
        assert leading_state_seed(42, 0, "problem7", 0) == leading_state_seed(42, 0, "problem7", 0)
        seeds = {
            leading_state_seed(42, 0, "problem7", 0),
            leading_state_seed(42, 0, "problem7", 1),
            leading_state_seed(42, 1, "problem7", 0),
            leading_state_seed(42, 0, "problem8", 0),
            leading_state_seed(43, 0, "problem7", 0),
        }
        assert len(seeds) == 5


class TestMaskedValuesAreHidden:
    def test_hide_masked_values_strips_the_negation(self):
        line = "(not (on a - block b - block)), (clear c - block), (not (handempty ))"
        assert hide_masked_values(line) == "(on a - block b - block), (clear c - block), (handempty )"

    def test_no_staged_masking_entry_is_negated(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.4, 0.2)
        assert "(not" in paths["frozen_masking"].read_text()
        _, staged_masking = _stage(paths, tmp_path / "out", spec(0.4, 0.2))
        assert "(not" not in staged_masking.read_text()

    def test_the_same_fluents_stay_masked(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.4, 0.2)
        _, kept = _stage(paths, tmp_path / "kept", spec(0.4, 0.2), hide_masked=False)
        _, hidden = _stage(paths, tmp_path / "hidden", spec(0.4, 0.2), hide_masked=True)
        for kept_line, hidden_line in zip(_masking_lines(kept), _masking_lines(hidden)):
            assert set(hide_masked_values(kept_line).split(", ")) == set(hidden_line.split(", "))

    def test_loaded_masked_predicates_carry_no_truth_value(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.4, 0.2)
        staged, staged_masking = _stage(paths, tmp_path / "out", spec(0.4, 0.2))
        observation = load_masked_observation(staged, staged_masking, blocks_domain(), paths["problem"])
        masked = [
            p for state in observation_states(observation)
            for p in get_state_grounded_predicates(state) if p.is_masked
        ]
        assert masked
        assert all(p.is_positive for p in masked)


class TestMismatchedInputsAreRejected:
    def test_a_different_action_sequence_raises(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        other_gt = tmp_path / "other.trajectory"
        other_gt.write_text(_GT_TRAJECTORY.replace("(pick_up c)", "(pick_up b)"))
        paths["gt"] = other_gt
        with pytest.raises(ValueError, match="different actions"):
            _stage(paths, tmp_path / "out", spec(0.2, 0.2))

    def test_a_different_length_raises(self, tmp_path):
        paths = write_blocks_fixture(tmp_path, 0.2, 0.2)
        shorter = _GT_TRAJECTORY.split("\n")
        short_gt = tmp_path / "short.trajectory"
        short_gt.write_text("\n".join(shorter[:-3] + [")"]))
        paths["gt"] = short_gt
        with pytest.raises(ValueError, match="differ in length"):
            _stage(paths, tmp_path / "out", spec(0.2, 0.2))
