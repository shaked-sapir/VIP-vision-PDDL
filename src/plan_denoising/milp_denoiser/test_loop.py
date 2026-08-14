"""Unit tests for the ``cdps_milp_loop`` driver and its model-prior projection.

    python -m unittest src.plan_denoising.milp_denoiser.test_loop

Nothing here runs CP-SAT or PI-SAM. Those are covered end-to-end elsewhere; what
this suite pins down are the loop's *decision* rules, which are cheap to test and
expensive to get wrong because they fail silently:

1. **Model identity** (:func:`model_hash`). Dedup and the fixpoint stop rule both
   key on it, so an identity that is not a function of the model makes the loop
   re-run solved rounds and never terminate on fixpoint. This actually happened:
   the first implementation hashed ``to_pddl()``, which renders an augmented
   ``:requirements`` *set* and so can return two different strings for one
   object. ``test_hash_is_not_derived_from_to_pddl`` is that regression.
2. **Sampling.** Size, no-replacement, cooldown-as-preference, and the
   deterministic tie-break that makes a round reproducible from its log.
3. **Dedup and the fixpoint.** A ``(subset, M_best)`` pair is never solved twice;
   the same subset *is* re-drawn once the incumbent changes; both are off under
   ``pool_policy: replace``, where the pair no longer identifies a round.
4. **Stop rules**, including their precedence and the per-solve budget clamp.
5. **The prior projection**, whose whole point is that what it cannot express is
   counted rather than dropped in silence.

The fixtures are hand-built stand-ins with just the attributes the code reads
(``signature`` / ``preconditions`` / ``discrete_effects``), so the tests state
their inputs literally instead of learning them.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from pddl_plus_parser.models import Predicate
from planning_structs.domain import Domain as PSDomain

from src.plan_denoising.milp_denoiser.config import (
    CdpsMilpConfig,
    StopRules,
    SubsetSize,
)
from src.milp.converter import GtAnchoring
from src.plan_denoising.milp_denoiser.loop import (
    NO_MODEL_HASH,
    ROUND_MODELS_DIR,
    LoopResult,
    RoundLog,
    _LoopState,
    _TraceCache,
    _learner_input,
    _per_trace_scores,
    _remaining_budget,
    _sample_hardest_first,
    _sample_random,
    _stop_reason,
    _subset_gt,
    model_hash,
    save_round_log,
    save_round_model,
)
from src.plan_denoising.milp_denoiser.model_prior import (
    NEUTRAL_PROBABILITY,
    PRESENT_PROBABILITY,
    learner_domain_to_observation_m,
)


# ---------------------------------------------------------------- fixtures


def _predicate(name: str, *arguments: str, positive: bool = True) -> Predicate:
    """A lifted literal over the given action-parameter names (types unused here)."""
    return Predicate(name=name, signature={a: None for a in arguments}, is_positive=positive)


class _FakeAction:
    """The three attributes ``model_hash`` and the prior projection actually read."""

    def __init__(
        self,
        name: str,
        signature: Dict[str, str],
        preconditions: Sequence[Predicate],
        effects: Sequence[Predicate],
    ) -> None:
        self.name = name
        self.signature = signature
        # The real ``CompoundPrecondition`` iterates as (operator, operand) pairs.
        self.preconditions = [("and", p) for p in preconditions]
        self.discrete_effects = list(effects)


class _FakeLearnedDomain:
    """A learned model. ``to_pddl`` raises: no consumer here may depend on it."""

    def __init__(self, actions: Sequence[_FakeAction]) -> None:
        self.actions = {a.name: a for a in actions}

    def to_pddl(self) -> str:
        raise AssertionError("model identity must not be derived from to_pddl()")


def _move_model(*, extra_effect: bool = False) -> _FakeLearnedDomain:
    """``move(?f ?t)``: pre ``at(?f)``, effects ``at(?t)`` and ``not at(?f)``."""
    effects = [_predicate("at", "?t"), _predicate("at", "?f", positive=False)]
    if extra_effect:
        effects.append(_predicate("visited", "?t"))
    return _FakeLearnedDomain([
        _FakeAction("move", {"?f": "room", "?t": "room"}, [_predicate("at", "?f")], effects)
    ])


class _RenderableDomain:
    """A model whose ``to_pddl`` works — for the one consumer allowed to call it.

    Separate from :class:`_FakeLearnedDomain` on purpose. That fixture's
    ``to_pddl`` raises to protect model *identity* from the unstable text;
    writing the artifact is the opposite case, and needs its own fixture so the
    protection cannot be relaxed by accident.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    def to_pddl(self) -> str:
        return self._text


def _micro_ps_domain() -> PSDomain:
    """``move(?from ?to)`` over ``at`` / ``visited`` — the prior's key vocabulary."""
    return PSDomain(
        [("room", None)],
        [("at", ["room"]), ("visited", ["room"])],
        [("move", ["room", "room"])],
        name="micro",
    )


class _FakeTrace:
    def __init__(self, observation_index: int, v_per_transition: float) -> None:
        self.observation_index = observation_index
        self.v_per_transition = v_per_transition


class _FakeEvaluation:
    def __init__(self, per_trace: Sequence[_FakeTrace]) -> None:
        self.per_trace = list(per_trace)


def _config(**overrides) -> CdpsMilpConfig:
    """A loop config built through the real YAML validation path."""
    return CdpsMilpConfig.from_dict({"variant": "loop", **overrides})


def _state(**overrides) -> _LoopState:
    defaults = dict(pool_observations={}, repaired={}, rng=random.Random(0))
    defaults.update(overrides)
    return _LoopState(**defaults)


# ---------------------------------------------------------------- model identity


class TestModelHash(unittest.TestCase):
    """The loop's notion of "the same model"."""

    def test_no_model_has_a_sentinel_hash(self) -> None:
        self.assertEqual(model_hash(None), NO_MODEL_HASH)

    def test_hash_is_not_derived_from_to_pddl(self) -> None:
        """Regression: ``to_pddl()`` is not a stable function of the model.

        The fixture's ``to_pddl`` raises, so this fails loudly if the textual
        identity is ever reintroduced.
        """
        self.assertNotEqual(model_hash(_move_model()), NO_MODEL_HASH)

    def test_hash_ignores_the_order_literals_were_stored_in(self) -> None:
        """Preconditions and effects live in sets upstream; iteration order varies."""
        forward = _move_model()
        action = forward.actions["move"]
        reversed_model = _FakeLearnedDomain([
            _FakeAction(
                "move",
                dict(reversed(list(action.signature.items()))),
                [operand for _operator, operand in reversed(action.preconditions)],
                list(reversed(action.discrete_effects)),
            )
        ])
        self.assertEqual(model_hash(forward), model_hash(reversed_model))

    def test_hash_changes_when_the_model_changes(self) -> None:
        self.assertNotEqual(model_hash(_move_model()), model_hash(_move_model(extra_effect=True)))

    def test_hash_is_stable_across_calls(self) -> None:
        model = _move_model()
        self.assertEqual(model_hash(model), model_hash(model))


# ---------------------------------------------------------------- samplers


class TestSamplers(unittest.TestCase):
    """Subset selection: size, no replacement, cooldown, determinism."""

    POOL = [0, 1, 2, 3, 4]

    def test_random_draw_is_a_sorted_subset_of_the_pool(self) -> None:
        subset = _sample_random(self.POOL, 3, random.Random(1), blocked=set())
        self.assertEqual(len(subset), 3)
        self.assertEqual(len(set(subset)), 3, "drawn without replacement")
        self.assertEqual(subset, sorted(subset))
        self.assertTrue(set(subset) <= set(self.POOL))

    def test_random_draw_avoids_blocked_traces_when_it_can(self) -> None:
        subset = _sample_random(self.POOL, 2, random.Random(1), blocked={0, 1, 2})
        self.assertEqual(set(subset), {3, 4})

    def test_random_draw_ignores_the_cooldown_rather_than_shrink(self) -> None:
        """The cooldown is a preference: honouring it must never change ``m``."""
        subset = _sample_random(self.POOL, 3, random.Random(1), blocked={0, 1, 2, 3})
        self.assertEqual(len(subset), 3)

    def test_hardest_first_takes_the_worst_scoring_traces(self) -> None:
        scores = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.0}
        subset = _sample_hardest_first(
            self.POOL, 2, scores, random.Random(1), blocked=set()
        )
        self.assertEqual(subset, [1, 3])

    def test_hardest_first_breaks_ties_by_index_not_randomly(self) -> None:
        """A round has to be reproducible from its log, so ties cannot be random."""
        scores = {i: 0.5 for i in self.POOL}
        first = _sample_hardest_first(self.POOL, 3, scores, random.Random(1), set())
        second = _sample_hardest_first(self.POOL, 3, scores, random.Random(999), set())
        self.assertEqual(first, second)
        self.assertEqual(first, [0, 1, 2])

    def test_hardest_first_gives_forced_traces_the_first_slots(self) -> None:
        scores = {0: 0.1, 1: 0.9, 2: 0.2, 3: 0.8, 4: 0.0}
        subset = _sample_hardest_first(
            self.POOL, 2, scores, random.Random(1), blocked=set(), forced=[4]
        )
        self.assertIn(4, subset, "the co-sampled trace must survive its low score")
        self.assertEqual(subset, [1, 4])

    def test_hardest_first_tops_up_when_scores_are_missing(self) -> None:
        subset = _sample_hardest_first(self.POOL, 3, {0: 1.0}, random.Random(1), set())
        self.assertEqual(len(subset), 3)
        self.assertIn(0, subset)

    def test_per_trace_scores_key_on_the_observation_index(self) -> None:
        """Not on position: V covers every original, the pool may be a subset."""
        evaluation = _FakeEvaluation([
            _FakeTrace(0, 0.1), _FakeTrace(1, 0.2), _FakeTrace(2, 0.3)
        ])
        self.assertEqual(_per_trace_scores(evaluation, [0, 2]), {0: 0.1, 2: 0.3})

    def test_per_trace_scores_of_no_evaluation_is_empty(self) -> None:
        self.assertEqual(_per_trace_scores(None, [0, 1]), {})


# ---------------------------------------------------------------- dedup / fixpoint


class TestDrawAndDedup(unittest.TestCase):
    """``(subset, M_best)`` as a round's identity."""

    POOL = [0, 1, 2]

    def test_a_solved_pair_is_never_drawn_again(self) -> None:
        state = _state(solved_pairs={((0, 1), "h1")})
        for _ in range(20):
            subset = state.draw(_config(), self.POOL, 2, "h1", total_subsets=3)
            self.assertIsNotNone(subset)
            self.assertNotEqual(tuple(subset), (0, 1))

    def test_the_same_subset_is_redrawn_under_a_new_incumbent(self) -> None:
        """Dedup keys on the pair, not the subset: a new prior is new work."""
        state = _state(solved_pairs={((0, 1), "h1")})
        drawn = {
            tuple(state.draw(_config(), self.POOL, 2, "h2", total_subsets=3))
            for _ in range(30)
        }
        self.assertIn((0, 1), drawn)

    def test_draw_reports_the_fixpoint_when_every_subset_is_seen(self) -> None:
        state = _state(solved_pairs={((0, 1), "h1"), ((0, 2), "h1"), ((1, 2), "h1")})
        self.assertIsNone(state.draw(_config(), self.POOL, 2, "h1", total_subsets=3))

    def test_the_fixpoint_is_per_incumbent(self) -> None:
        state = _state(solved_pairs={((0, 1), "h1"), ((0, 2), "h1"), ((1, 2), "h1")})
        self.assertIsNotNone(state.draw(_config(), self.POOL, 2, "h2", total_subsets=3))

    def test_replace_keeps_drawing_solved_pairs(self) -> None:
        """With a mutating pool the same subset denotes different data each round."""
        config = _config(pool_policy="replace")
        state = _state(solved_pairs={((0, 1), "h1"), ((0, 2), "h1"), ((1, 2), "h1")})
        self.assertIsNotNone(state.draw(config, self.POOL, 2, "h1", total_subsets=3))

    def test_hardest_first_does_not_spin_on_its_own_deterministic_pick(self) -> None:
        """It is a function of the incumbent, so a redraw must not just retry it."""
        config = _config(sampler="hardest_first")
        state = _state(
            solved_pairs={((1, 2), "h1")}, per_trace_v={0: 0.1, 1: 0.9, 2: 0.8}
        )
        subset = state.draw(config, self.POOL, 2, "h1", total_subsets=3)
        self.assertIsNotNone(subset, "must fall back rather than re-propose (1, 2)")
        self.assertNotEqual(tuple(subset), (1, 2))

    def test_subsets_seen_counts_only_the_matching_prior(self) -> None:
        state = _state(solved_pairs={((0, 1), "h1"), ((0, 2), "h1"), ((0, 1), "h2")})
        self.assertEqual(state.subsets_seen_for("h1"), 2)
        self.assertEqual(state.subsets_seen_for("h2"), 1)
        self.assertEqual(state.subsets_seen_for(NO_MODEL_HASH), 0)


# ---------------------------------------------------------------- stop rules


class TestStopRules(unittest.TestCase):
    """Termination: any-of semantics, checked in a fixed order."""

    def _rounds(self, count: int) -> List[RoundLog]:
        return [
            RoundLog(round_index=i, subset=[0], prior_hash=NO_MODEL_HASH,
                     elapsed_seconds=0.0, round_seconds=0.0)
            for i in range(1, count + 1)
        ]

    def test_keeps_going_when_no_rule_fires(self) -> None:
        rules = StopRules(no_improvement_rounds=None)
        self.assertIsNone(_stop_reason(rules, self._rounds(3), 5.0, 1.0, 0, False))

    def test_budget(self) -> None:
        rules = StopRules(budget_seconds=10, no_improvement_rounds=None)
        self.assertEqual(_stop_reason(rules, [], None, 10.0, 0, False), "budget_seconds")

    def test_max_rounds(self) -> None:
        rules = StopRules(max_rounds=3, no_improvement_rounds=None)
        self.assertEqual(
            _stop_reason(rules, self._rounds(3), 5.0, 1.0, 0, False), "max_rounds"
        )

    def test_perfect_fit(self) -> None:
        rules = StopRules(no_improvement_rounds=None)
        self.assertEqual(_stop_reason(rules, self._rounds(1), 0.0, 1.0, 0, False), "perfect_fit")

    def test_perfect_fit_can_be_switched_off(self) -> None:
        rules = StopRules(stop_on_perfect_fit=False, no_improvement_rounds=None)
        self.assertIsNone(_stop_reason(rules, self._rounds(1), 0.0, 1.0, 0, False))

    def test_no_improvement(self) -> None:
        rules = StopRules(no_improvement_rounds=2)
        self.assertEqual(
            _stop_reason(rules, self._rounds(5), 5.0, 1.0, 2, False), "no_improvement"
        )

    def test_fixpoint(self) -> None:
        rules = StopRules(no_improvement_rounds=None)
        self.assertEqual(_stop_reason(rules, self._rounds(3), 5.0, 1.0, 0, True), "fixpoint")

    def test_fixpoint_can_be_switched_off(self) -> None:
        rules = StopRules(stop_on_fixpoint=False, no_improvement_rounds=None)
        self.assertIsNone(_stop_reason(rules, self._rounds(3), 5.0, 1.0, 0, True))

    def test_budget_outranks_the_others(self) -> None:
        """Wall clock first: an overrun must stop the loop whatever else holds."""
        rules = StopRules(budget_seconds=10, max_rounds=99)
        self.assertEqual(
            _stop_reason(rules, self._rounds(3), 0.0, 12.0, 9, True), "budget_seconds"
        )

    def test_replace_disables_the_fixpoint_rule(self) -> None:
        rules = _config(pool_policy="replace").effective_stop_rules()
        self.assertFalse(rules.stop_on_fixpoint)
        self.assertTrue(_config().effective_stop_rules().stop_on_fixpoint)


class TestRemainingBudget(unittest.TestCase):
    """One solve must never be allowed to overrun the whole loop."""

    def test_no_loop_budget_leaves_the_solve_limit_alone(self) -> None:
        self.assertEqual(_remaining_budget(None, 60, elapsed=10.0), 60)
        self.assertIsNone(_remaining_budget(None, None, elapsed=10.0))

    def test_the_loop_budget_clamps_the_solve_limit(self) -> None:
        self.assertEqual(_remaining_budget(100, 60, elapsed=70.0), 30)

    def test_the_solve_limit_still_applies_when_it_is_tighter(self) -> None:
        self.assertEqual(_remaining_budget(100, 10, elapsed=5.0), 10)

    def test_an_exhausted_budget_still_yields_a_positive_limit(self) -> None:
        """A zero time limit means "unlimited" to the solver — never emit one."""
        self.assertEqual(_remaining_budget(100, None, elapsed=200.0), 1)


# ---------------------------------------------------------------- round plumbing


class TestLearnerInput(unittest.TestCase):
    """What PI-SAM is shown, and which pool indices it corresponds to."""

    def test_subset_only_shows_exactly_this_round_s_repairs(self) -> None:
        observations, indices = _learner_input(
            _config(), [2, 5], ["r2", "r5"], {0: "old0", 2: "old2"}
        )
        self.assertEqual(observations, ["r2", "r5"])
        self.assertEqual(indices, [2, 5])

    def test_accumulated_shows_every_repair_so_far_in_index_order(self) -> None:
        observations, indices = _learner_input(
            _config(learner_input="accumulated"), [2, 5], ["r2", "r5"],
            {5: "r5", 0: "r0", 2: "r2"},
        )
        self.assertEqual(indices, [0, 2, 5])
        self.assertEqual(observations, ["r0", "r2", "r5"])


class TestSubsetGt(unittest.TestCase):
    """The GT map is keyed by pool index; a solve sees local 0..m-1 positions."""

    def test_indices_are_rekeyed_to_local_positions(self) -> None:
        self.assertEqual(_subset_gt({0: {0}, 3: {0, 2}, 7: {1}}, [3, 7]), {0: {0, 2}, 1: {1}})

    def test_absent_observations_are_simply_missing(self) -> None:
        self.assertEqual(_subset_gt({3: {0}}, [1, 3]), {1: {0}})

    def test_no_gt_at_all_is_none_not_an_empty_map(self) -> None:
        self.assertIsNone(_subset_gt(None, [0, 1]))
        self.assertIsNone(_subset_gt({}, [0, 1]))
        self.assertIsNone(_subset_gt({9: {0}}, [0, 1]))


# ---------------------------------------------------------------- reporting


class TestReporting(unittest.TestCase):
    """The round log is the raw material for the anytime profile (P5)."""

    def _result(self) -> LoopResult:
        rounds = [
            RoundLog(1, [0, 1], NO_MODEL_HASH, 0.0, 1.0, solved=True, v_raw=9.0, improved=True),
            RoundLog(2, [1, 2], "abc", 1.0, 1.0, solved=True, v_raw=9.0, tied_with_best=True),
            RoundLog(3, [0, 2], "abc", 2.0, 1.0, solved=False),
            RoundLog(4, [0, 1], "abc", 3.0, 1.0, solved=True, v_raw=7.0, improved=True,
                     mixed_set_conflict=True),
        ]
        return LoopResult(
            learned_domain=object(), solved=True, best_v=7.0, best_round=4,
            best_round_repair_cost=11, best_round_subset_size=2,
            stop_reason="fixpoint", rounds=rounds, stats={"n_traces": 3},
        )

    def test_report_counts_every_round_category(self) -> None:
        report = self._result().as_report()
        self.assertEqual(report["n_rounds"], 4)
        self.assertEqual(report["n_rounds_solved"], 3)
        self.assertEqual(report["n_rounds_improved"], 2)
        self.assertEqual(report["n_rounds_tied"], 1)
        self.assertEqual(report["n_mixed_set_conflicts"], 1)
        self.assertEqual(report["best_round"], 4)
        self.assertEqual(report["stop_reason"], "fixpoint")
        self.assertEqual(report["n_traces"], 3, "stats are carried through")

    def test_a_model_with_conflicts_is_not_conflict_free(self) -> None:
        result = self._result()
        self.assertTrue(result.is_conflict_free)
        result.conflicts = ["some conflict"]
        self.assertFalse(result.is_conflict_free)
        self.assertEqual(result.as_report()["conflict_free_model_count"], 0)

    def test_an_unsolved_loop_reports_no_conflict_count(self) -> None:
        result = LoopResult(learned_domain=None, solved=False, stop_reason="budget_seconds")
        report = result.as_report()
        self.assertFalse(report["milp_solved"])
        self.assertIsNone(report["pisam_conflicts_on_feasible"])

    def test_the_report_speaks_the_single_round_vocabulary(self) -> None:
        """``run_fold._milp_specific`` reads one key set for both MILP arms."""
        report = self._result().as_report()
        for key in ("algorithm", "milp_solved", "repair_cost", "best_cost",
                    "conflict_free_model_count", "pisam_conflicts_on_feasible"):
            self.assertIn(key, report)
        self.assertEqual(report["algorithm"], "cdps_milp_loop")

    def test_the_fold_wide_cost_keys_stay_empty(self) -> None:
        """A subset cost must not sit in the column CDPS is compared against.

        ``repair_cost`` carries the design §7.1 lower bound, which needs one
        solve over every trace. The loop has none, so it reports its cost under
        a name that says how many traces it covered.
        """
        report = self._result().as_report()
        self.assertIsNone(report["repair_cost"])
        self.assertIsNone(report["best_cost"])
        self.assertEqual(report["best_round_repair_cost"], 11)
        self.assertEqual(report["best_round_subset_size"], 2)

    def test_round_rows_are_json_serialisable(self) -> None:
        rows = [r.as_dict() for r in self._result().rounds]
        self.assertEqual(json.loads(json.dumps(rows))[0]["subset"], [0, 1])

    def test_save_round_log_writes_the_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "milp"
            save_round_log(self._result(), target)
            payload = json.loads((target / "milp_loop_rounds.json").read_text())
        self.assertEqual(payload["best_v"], 7.0)
        self.assertEqual(payload["stop_reason"], "fixpoint")
        self.assertEqual([r["round"] for r in payload["rounds"]], [1, 2, 3, 4])


# ---------------------------------------------------------------- round models


class TestSaveRoundModel(unittest.TestCase):
    """Per-round models on disk — what makes a round re-scorable offline (D4)."""

    def test_a_round_model_lands_under_its_own_round_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ROUND_MODELS_DIR
            path = save_round_model(_RenderableDomain("(define (domain m))"), root, 3)
            self.assertEqual(path, root / "round_3" / "model.pddl")
            self.assertEqual(path.read_text(), "(define (domain m))")

    def test_rounds_do_not_overwrite_each_other(self) -> None:
        """The loop keeps one winner; the curve needs every candidate kept apart."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ROUND_MODELS_DIR
            save_round_model(_RenderableDomain("first"), root, 1)
            save_round_model(_RenderableDomain("second"), root, 2)
            written = sorted(p.parent.name for p in root.glob("round_*/model.pddl"))
        self.assertEqual(written, ["round_1", "round_2"])

    def test_a_round_without_a_model_writes_nothing(self) -> None:
        """An infeasible round leaves a gap, which the log explains via its hash."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / ROUND_MODELS_DIR
            self.assertIsNone(save_round_model(None, root, 1))
            self.assertFalse(root.exists())


# ---------------------------------------------------------------- subset size


class TestSubsetSize(unittest.TestCase):
    """``m``, resolved once per fold (Q9)."""

    def test_half_rounds_up_with_a_floor_of_two(self) -> None:
        half = SubsetSize.parse("half")
        self.assertEqual([half.resolve(n) for n in (1, 2, 3, 4, 5, 8)], [1, 2, 2, 2, 3, 4])

    def test_all_is_the_whole_pool(self) -> None:
        self.assertEqual(SubsetSize.parse("all").resolve(7), 7)

    def test_a_fixed_size_is_clamped_to_the_pool(self) -> None:
        self.assertEqual(SubsetSize.parse(4).resolve(3), 3)
        self.assertEqual(SubsetSize.parse("4").resolve(9), 4)

    def test_an_empty_pool_resolves_to_zero(self) -> None:
        self.assertEqual(SubsetSize.parse("half").resolve(0), 0)

    def test_invalid_sizes_are_rejected_at_parse_time(self) -> None:
        for bad in ("most", 0, -1, True, 3.5):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                SubsetSize.parse(bad)


# ---------------------------------------------------------------- trace cache


class _FakeGP:
    """The duck-typed surface ``converter`` reads off a grounded predicate."""

    def __init__(self, name: str, arg_names: Sequence[str]) -> None:
        self.name = name
        self.signature = {f"?x{i}": "room" for i in range(len(arg_names))}
        self.object_mapping = dict(zip(self.signature, arg_names))
        self.is_positive = True
        self.is_masked = False


class _FakeState:
    def __init__(self, gps: Sequence[_FakeGP]) -> None:
        self.state_predicates = {"at": list(gps)}


class _FakeComponent:
    def __init__(self, previous_state, action_name, parameters, next_state) -> None:
        self.previous_state = previous_state
        self.next_state = next_state
        self.grounded_action_call = type(
            "Call", (), {"name": action_name, "parameters": list(parameters)}
        )()


class _FakeObservation:
    def __init__(self, components, grounded_objects) -> None:
        self.components = components
        self.grounded_objects = grounded_objects


class _FakePDDLObject:
    def __init__(self, type_name: str) -> None:
        self.type = type_name


class _NoConstants:
    constants: dict = {}


def _move_observation(objects: Dict[str, _FakePDDLObject]) -> _FakeObservation:
    """``move(r1, r2)`` over whatever object map the caller supplies."""
    before = _FakeState([_FakeGP("at", ["r1"])])
    after = _FakeState([_FakeGP("at", ["r2"])])
    return _FakeObservation(
        [_FakeComponent(before, "move", ["r1", "r2"], after)], objects
    )


class TestTraceCache(unittest.TestCase):
    """Which conversion failures the loop tolerates, and which abort it."""

    def setUp(self) -> None:
        self.ps_domain = _micro_ps_domain()
        self.well_typed = {n: _FakePDDLObject("room") for n in ("r1", "r2")}
        # What TrajectoryParser infers when r2's last mention is an untyped slot.
        self.mistyped = {"r1": _FakePDDLObject("room"), "r2": _FakePDDLObject("object")}

    def _cache(self, object_types=None) -> _TraceCache:
        return _TraceCache(
            self.ps_domain, _NoConstants(), GtAnchoring.INIT_ONLY, object_types
        )

    def test_a_pool_trace_that_cannot_be_encoded_raises(self) -> None:
        """A frozen input that will not encode is a defect, not a smaller pool.

        Dropping it would change ``subset_size`` and ``n_possible_subsets``, so
        the loop would silently run a different experiment than the one its
        stats describe.
        """
        cache = self._cache()
        with self.assertRaises(ValueError) as caught:
            cache.trace(0, _move_observation(self.mistyped), None)
        self.assertIn("argument type mismatch", str(caught.exception))

    def test_a_hint_trace_that_cannot_be_encoded_is_none(self) -> None:
        """A warm start is optional, so its failure must stay tolerated."""
        cache = self._cache()
        trace = cache.trace(0, _move_observation(self.mistyped), None, kind="hint")
        self.assertIsNone(trace)

    def test_declared_object_types_rescue_the_pool_trace(self) -> None:
        """The overlay is what turns the raising case back into a usable trace."""
        cache = self._cache(object_types={0: {"r1": "room", "r2": "room"}})
        trace = cache.trace(0, _move_observation(self.mistyped), None)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.step, 1)

    def test_the_overlay_is_per_observation(self) -> None:
        """Each observation is grounded on its own problem's types — folds mix problems."""
        cache = self._cache(object_types={1: {"r2": "room"}})
        with self.assertRaises(ValueError):
            cache.trace(0, _move_observation(self.mistyped), None)
        self.assertIsNotNone(cache.trace(1, _move_observation(self.mistyped), None))

    def test_a_well_typed_trace_needs_no_overlay(self) -> None:
        """The overlay must be inert when the inferred types were already right."""
        without = self._cache().trace(0, _move_observation(self.well_typed), None)
        with_overlay = self._cache(
            object_types={0: {"r1": "room", "r2": "room"}}
        ).trace(0, _move_observation(self.well_typed), None)
        self.assertEqual(without.step, with_overlay.step)
        self.assertEqual(set(without.init), set(with_overlay.init))


# ---------------------------------------------------------------- model prior


class TestModelPrior(unittest.TestCase):
    """``M_best`` projected onto the encoder's ``(schema, predicate, binding)`` grid.

    The projection is lossy by construction; every test here is about the loss
    being *counted* rather than silent.
    """

    def setUp(self) -> None:
        self.ps_domain = _micro_ps_domain()

    def _project(self, model: _FakeLearnedDomain):
        return learner_domain_to_observation_m(model, self.ps_domain)

    def _key(self, action: str, predicate: str, binding: Tuple[int, ...]):
        return (
            self.ps_domain.get_action_schema(action),
            self.ps_domain.get_predicate(predicate),
            binding,
        )

    def test_every_slot_starts_neutral(self) -> None:
        """PI-SAM's silence is "not justified", never "known absent"."""
        empty = _FakeLearnedDomain([])
        projection = self._project(empty)
        values = set(projection.observation_m.pre.values())
        self.assertEqual(values, {NEUTRAL_PROBABILITY})
        self.assertGreater(projection.stats["slots"], 0)

    def test_declared_literals_are_marked_present(self) -> None:
        projection = self._project(_move_model())
        observation_m = projection.observation_m
        self.assertEqual(observation_m.pre[self._key("move", "at", (1,))], PRESENT_PROBABILITY)
        self.assertEqual(observation_m.add[self._key("move", "at", (2,))], PRESENT_PROBABILITY)
        self.assertEqual(observation_m.dele[self._key("move", "at", (1,))], PRESENT_PROBABILITY)

    def test_an_undeclared_slot_stays_neutral(self) -> None:
        observation_m = self._project(_move_model()).observation_m
        self.assertEqual(
            observation_m.add[self._key("move", "visited", (1,))], NEUTRAL_PROBABILITY
        )

    def test_a_faithful_projection_is_lossless(self) -> None:
        self.assertTrue(self._project(_move_model()).is_lossless)

    def test_a_negative_precondition_is_dropped_and_counted(self) -> None:
        """The encoding has one polarity-free "is a precondition" bit."""
        model = _FakeLearnedDomain([
            _FakeAction("move", {"?f": "room", "?t": "room"},
                        [_predicate("at", "?t", positive=False)], [])
        ])
        projection = self._project(model)
        self.assertEqual(projection.stats["negative_preconditions_dropped"], 1)
        self.assertFalse(projection.is_lossless)
        self.assertEqual(
            projection.observation_m.pre[self._key("move", "at", (2,))], NEUTRAL_PROBABILITY
        )

    def test_a_literal_over_a_non_parameter_is_unbindable(self) -> None:
        """Only action parameters have positions; a domain constant has none."""
        model = _FakeLearnedDomain([
            _FakeAction("move", {"?f": "room", "?t": "room"}, [], [_predicate("at", "depot1")])
        ])
        projection = self._project(model)
        self.assertEqual(projection.stats["unbindable_literals"], 1)
        self.assertFalse(projection.is_lossless)

    def test_a_repeated_parameter_literal_cannot_occur(self) -> None:
        """Documents why ``_binding``'s distinctness guard is defence only.

        ``build_predicate_arguments`` enumerates permutations of *distinct*
        slots, so ``(on ?x ?x)`` would indeed have no key. But it can never
        reach the projection: ``Predicate.signature`` is a *dict* keyed by
        argument name, so the repeat collapses into a single argument before
        the projection ever sees it.
        """
        self.assertEqual(len(_predicate("at", "?f", "?f").signature), 1)

    def test_an_unknown_action_is_counted_not_fatal(self) -> None:
        model = _FakeLearnedDomain([_FakeAction("teleport", {"?t": "room"}, [], [])])
        projection = self._project(model)
        self.assertEqual(projection.stats["unknown_actions"], 1)
        self.assertFalse(projection.is_lossless)

    def test_an_unknown_predicate_is_counted_not_fatal(self) -> None:
        model = _FakeLearnedDomain([
            _FakeAction("move", {"?f": "room", "?t": "room"}, [_predicate("humming", "?f")], [])
        ])
        self.assertEqual(self._project(model).stats["unknown_predicates"], 1)

    def test_a_non_predicate_precondition_is_counted_not_fatal(self) -> None:
        action = _FakeAction("move", {"?f": "room", "?t": "room"}, [], [])
        action.preconditions = [(">", "(some numeric thing)")]
        projection = self._project(_FakeLearnedDomain([action]))
        self.assertEqual(projection.stats["non_predicate_preconditions"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
