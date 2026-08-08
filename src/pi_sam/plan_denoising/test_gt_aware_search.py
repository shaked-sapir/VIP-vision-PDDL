"""GT-aware conflict search tests.

Covers the ground-truth handling of ConflictDrivenPatchSearch on small
synthetic blocks trajectories (no external data files needed):

  1. GT states are never fluent-patched (init is always GT; further GT states
     come from gt_states_by_obs — e.g. the anchored final state).
  2. Two-sided data fixes: a must-effect conflict whose next state is GT is
     repaired from the (non-GT) prev side instead of being dropped.
  3. REQUIRE_EFFECT_VS_CANNOT with a GT next state has no data fix, and a
     REQUIRE constraint refuted by GT next-state evidence is never generated.
  4. Frame-axiom Child B never patches a GT prev state (per-conflict gate).
  5. Nodes carrying an unresolvable conflict are pruned, not expanded.

The synthetic trajectory writes .trajectory/.masking_info files into a temp
dir and loads them exactly like the benchmark pipeline does.
"""

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Set

from pddl_plus_parser.lisp_parsers import DomainParser
from utilities import NegativePreconditionPolicy

from src.pi_sam.noisy_pisam.typings import (
    Conflict,
    ConflictType,
    ModelPart,
    ParameterBoundLiteral,
    PatchOperation,
)
from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearch
from src.pi_sam.plan_denoising.frontier import SearchNode
from src.utils.masking import load_masked_observation

DOMAIN_PDDL = """
(define (domain blocks)
    (:requirements :strips :typing)
    (:types block)
    (:predicates
        (on ?x - block ?y - block)
        (ontable ?x - block)
        (clear ?x - block)
        (handempty)
        (holding ?x - block)
    )
    (:action pick_up
        :parameters (?x - block)
        :precondition (and (clear ?x) (ontable ?x) (handempty))
        :effect (and (not (ontable ?x)) (not (clear ?x)) (not (handempty)) (holding ?x))
    )
    (:action put_down
        :parameters (?x - block)
        :precondition (and (holding ?x))
        :effect (and (not (holding ?x)) (clear ?x) (handempty) (ontable ?x))
    )
    (:action stack
        :parameters (?x - block ?y - block)
        :precondition (and (holding ?x) (clear ?y))
        :effect (and (not (holding ?x)) (not (clear ?y)) (clear ?x) (handempty) (on ?x ?y))
    )
    (:action unstack
        :parameters (?x - block ?y - block)
        :precondition (and (on ?x ?y) (clear ?x) (handempty))
        :effect (and (holding ?x) (clear ?y) (not (clear ?x)) (not (handempty)) (not (on ?x ?y)))
    )
)
"""

# Clean 4-step trajectory over blocks a, b, c (states s0..s4).
CLEAN_STATES = [
    "(on c b) (on b a) (ontable a) (clear c) (handempty)",
    "(holding c) (clear b) (on b a) (ontable a)",
    "(ontable c) (clear c) (handempty) (clear b) (on b a) (ontable a)",
    "(holding b) (clear a) (ontable a) (ontable c) (clear c)",
    "(ontable b) (clear b) (handempty) (clear a) (ontable a) (ontable c) (clear c)",
]
ACTIONS = ["(unstack c b)", "(put_down c)", "(unstack b a)", "(put_down b)"]


def _write_trajectory(dir_path: Path, states, actions, name="traj") -> Path:
    lines = ["(", f"(:init {states[0]})"]
    for action, state in zip(actions, states[1:]):
        lines.append(f"(operator: {action})")
        lines.append(f"(:state {state})")
    lines.append(")")
    traj = dir_path / f"{name}.trajectory"
    traj.write_text("\n".join(lines) + "\n")
    (dir_path / f"{name}.masking_info").write_text("\n" * len(states))
    return traj


class _GtSearchTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="gt_search_test_")
        cls.tmp_dir = Path(cls._tmp.name)
        domain_file = cls.tmp_dir / "blocks.pddl"
        domain_file.write_text(DOMAIN_PDDL)
        cls.domain = DomainParser(domain_file, partial_parsing=True).parse_domain()

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _load_observation(self, states, actions, name):
        traj = _write_trajectory(self.tmp_dir, states, actions, name=name)
        return load_masked_observation(
            traj, traj.with_suffix(".masking_info"), self.domain
        )

    def _make_search(self, **kwargs) -> ConflictDrivenPatchSearch:
        return ConflictDrivenPatchSearch(
            partial_domain_template=deepcopy(self.domain),
            negative_preconditions_policy=NegativePreconditionPolicy.hard,
            seed=42,
            **kwargs,
        )

    def _assert_no_gt_patch(self, result, gt_states: Set[int], obs_idx: int = 0):
        """No fluent patch may touch a GT state (prev@c -> state c, next@c -> c+1)."""
        for fp in result.fluent_patches:
            if fp.observation_index != obs_idx:
                continue
            touched = (
                fp.component_index
                if fp.state_type == "prev"
                else fp.component_index + 1
            )
            self.assertNotIn(
                touched, gt_states | {0},
                f"Patch {fp} flips GT state {touched}",
            )


class TestMustConflictAtGtFinal(_GtSearchTestBase):
    """A seeded FORBID contradicted by the GT final transition dead-ends safely.

    Child A patches only the next state; with the final state GT and the
    constraint key occupied, the conflict at the last component has no fix —
    the search must prune (never patching the GT state) rather than "solve" it
    by corrupting the final state.
    """

    def _run(self, gt_states: Optional[Dict[int, Set[int]]]):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_must_final")
        search = self._make_search()
        # Forbid (ontable ?x) as an effect of put_down. The trajectory shows it
        # as a must-effect of put_down at components 1 and 3 (the last one).
        pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=True)
        forbid = {("put_down", ModelPart.EFFECT, pbl): PatchOperation.FORBID}
        return search.run(
            observations=[obs],
            initial_model_constraints=forbid,
            timeout_seconds=30,
            gt_states_by_obs=gt_states,
        )

    def test_gt_final_state_is_never_patched_and_subtree_pruned(self):
        result = self._run(gt_states={0: {0, 4}})
        self.assertTrue(result.conflicts, "the seeded FORBID cannot be satisfied")
        self._assert_no_gt_patch(result, gt_states={0, 4})
        self.assertGreaterEqual(result.report["pruned_unresolvable_nodes"], 1)
        self.assertEqual(result.report["conflict_free_model_count"], 0)

    def test_without_final_gt_next_side_fix_is_allowed(self):
        result = self._run(gt_states={0: {0}})
        self.assertEqual(result.conflicts, [])
        self._assert_no_gt_patch(result, gt_states={0})


class TestRequireRefutedByGtNext(_GtSearchTestBase):
    """REQUIRE evidence lives in the next state: GT next refutes it outright."""

    def _run_with_require_holding(self, gt_states):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_require")
        search = self._make_search()
        # Require (holding ?x) as an effect of put_down — false in every
        # put_down next state, including the GT final state s4.
        pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)
        require = {("put_down", ModelPart.EFFECT, pbl): PatchOperation.REQUIRE}
        return search.run(
            observations=[obs],
            initial_model_constraints=require,
            timeout_seconds=30,
            gt_states_by_obs=gt_states,
        )

    def test_no_data_fix_against_gt_next_and_subtree_pruned(self):
        result = self._run_with_require_holding(gt_states={0: {0, 4}})
        # The REQUIRE contradicts the GT final state; its conflict at the last
        # component has no data fix and the key is occupied -> the entire tree
        # is unresolvable and must be pruned without patching the GT state.
        self.assertTrue(result.conflicts, "search must NOT report conflict-free")
        self._assert_no_gt_patch(result, gt_states={0, 4})
        self.assertGreaterEqual(result.report["pruned_unresolvable_nodes"], 1)
        self.assertEqual(result.report["conflict_free_model_count"], 0)

    def test_without_gt_final_data_fix_exists(self):
        # Same setup but s4 NOT ground truth: flipping (holding b) into the
        # next states is now a legal (if wrong) repair, so the search may
        # legitimately find conflict-free models.
        result = self._run_with_require_holding(gt_states={0: {0}})
        self.assertEqual(result.conflicts, [])

    def test_model_guard_refuses_gt_refuted_require(self):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_guard")
        search = self._make_search()
        search.run(observations=[obs], timeout_seconds=5, gt_states_by_obs={0: {0, 4}})

        # Direct unit check of the guard: REQUIRE (holding ?x) in eff of
        # put_down is refuted by the GT final state (holding b is false in s4).
        pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)
        key = ("put_down", ModelPart.EFFECT, pbl)
        self.assertTrue(
            search._gt_evidence_contradicts(key, PatchOperation.REQUIRE)
        )
        conflict = Conflict(
            action_name="put_down",
            pbl=pbl,
            conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,  # desires REQUIRE
            observation_index=0,
            component_index=1,
            grounded_fluent="(holding c)",
        )
        updated = search._build_model_patch(conflict, {})
        self.assertEqual(updated, {}, "GT-refuted REQUIRE must not be generated")

        # Sanity: a REQUIRE that GT evidence supports IS generated.
        ok_pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=True)
        ok_conflict = Conflict(
            action_name="put_down",
            pbl=ok_pbl,
            conflict_type=ConflictType.FORBID_EFFECT_VS_MUST,
            observation_index=0,
            component_index=1,
            grounded_fluent="(ontable c)",
        )
        updated = search._build_model_patch(ok_conflict, {})
        self.assertEqual(
            updated[("put_down", ModelPart.EFFECT, ok_pbl)], PatchOperation.REQUIRE
        )


class TestForbidRefutedOnlyByGtPair(_GtSearchTestBase):
    """FORBID needs two-state must-evidence: one GT endpoint never refutes it."""

    def test_single_gt_endpoint_does_not_refute_forbid(self):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_forbid")
        search = self._make_search()
        search.run(observations=[obs], timeout_seconds=5, gt_states_by_obs={0: {0, 4}})

        pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=True)
        key = ("put_down", ModelPart.EFFECT, pbl)
        # s3 (prev of the final transition) is noisy-side: no GT->GT pair for
        # put_down exists, so the FORBID stays soft (not refuted).
        self.assertFalse(search._gt_evidence_contradicts(key, PatchOperation.FORBID))

    def test_adjacent_gt_pair_refutes_forbid(self):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_forbid_pair")
        search = self._make_search()
        # s3 AND s4 both GT -> the put_down at component 3 is a certain
        # must-effect of (ontable b): the FORBID is hard-refuted.
        search.run(
            observations=[obs], timeout_seconds=5, gt_states_by_obs={0: {0, 3, 4}}
        )
        pbl = ParameterBoundLiteral("ontable", ("?x",), is_positive=True)
        key = ("put_down", ModelPart.EFFECT, pbl)
        self.assertTrue(search._gt_evidence_contradicts(key, PatchOperation.FORBID))


class TestDataPatchGtGate(_GtSearchTestBase):
    """Unit checks: Child-A patches target next state, GT next -> no patch."""

    def _search_with_gt(self, gt_states):
        search = self._make_search()
        search._gt_states_by_obs = gt_states
        return search

    def _conflict(self, ctype, comp):
        return Conflict(
            action_name="put_down",
            pbl=ParameterBoundLiteral("ontable", ("?x",), is_positive=True),
            conflict_type=ctype,
            observation_index=0,
            component_index=comp,
            grounded_fluent="(ontable b)",
        )

    def test_default_is_next_side(self):
        search = self._search_with_gt({0: {0}})
        for ctype in (ConflictType.FORBID_EFFECT_VS_MUST,
                      ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                      ConflictType.FRAME_AXIOM):
            fp = search._build_data_patch(self._conflict(ctype, comp=3))
            self.assertEqual(fp.state_type, "next")
            self.assertEqual(fp.component_index, 3)

    def test_no_patch_when_next_is_gt(self):
        search = self._search_with_gt({0: {0, 4}})
        for ctype in (ConflictType.FORBID_EFFECT_VS_MUST,
                      ConflictType.REQUIRE_EFFECT_VS_CANNOT,
                      ConflictType.FRAME_AXIOM):
            self.assertIsNone(search._build_data_patch(self._conflict(ctype, comp=3)))

    def test_interior_gt_state_blocks_its_incoming_patch(self):
        search = self._search_with_gt({0: {0, 2}})
        # state 2 is GT: comp 1's next-patch is blocked, comp 2's is fine.
        self.assertIsNone(
            search._build_data_patch(self._conflict(ConflictType.FORBID_EFFECT_VS_MUST, comp=1))
        )
        self.assertIsNotNone(
            search._build_data_patch(self._conflict(ConflictType.FORBID_EFFECT_VS_MUST, comp=2))
        )


class TestUnresolvablePruning(_GtSearchTestBase):
    """_conflict_is_resolvable + the pruning counter."""

    def _node(self, constraints=None, patches=None):
        return SearchNode(
            cost=0.0, depth=0,
            model_constraints=constraints or {},
            fluent_patches=patches or set(),
        )

    def test_resolvable_via_data_or_model(self):
        search = self._make_search()
        search._gt_states_by_obs = {0: {0, 4}}
        pbl = ParameterBoundLiteral("holding", ("?x",), is_positive=True)
        c = Conflict(
            action_name="put_down", pbl=pbl,
            conflict_type=ConflictType.REQUIRE_EFFECT_VS_CANNOT,
            observation_index=0, component_index=3,
            grounded_fluent="(not (holding b))",
        )
        key = ("put_down", ModelPart.EFFECT, pbl)
        # No data fix (GT next), but the model fix (FORBID) is free -> resolvable.
        self.assertTrue(search._conflict_is_resolvable(c, self._node()))
        # Key occupied -> unresolvable.
        node = self._node(constraints={key: PatchOperation.REQUIRE})
        self.assertFalse(search._conflict_is_resolvable(c, node))

    def test_frame_axiom_between_gt_states_is_unresolvable(self):
        search = self._make_search()
        search._gt_states_by_obs = {0: {0, 3, 4}}
        c = Conflict(
            action_name="put_down",
            pbl=ParameterBoundLiteral("on", (), is_positive=True),
            conflict_type=ConflictType.FRAME_AXIOM,
            observation_index=0, component_index=3,
            grounded_fluent="(on a c)",
        )
        self.assertFalse(search._conflict_is_resolvable(c, self._node()))
        search._gt_states_by_obs = {0: {0, 4}}
        self.assertTrue(search._conflict_is_resolvable(c, self._node()))


class TestFrameAxiomChildBGtGate(_GtSearchTestBase):
    """End-to-end: no frame-axiom prev-fix ever lands on a GT state."""

    def test_noisy_run_with_gt_injection_never_patches_gt(self):
        # Inject noise into s2 (spurious (on a c) appears then disappears):
        # a frame-axiom violation for put_down/unstack around components 1-2.
        noisy = list(CLEAN_STATES)
        noisy[2] = noisy[2] + " (on a c)"
        obs = self._load_observation(noisy, ACTIONS, name="noisy_frame")
        gt = {0: {0, 2, 4}}  # s2 marked GT: the spurious literal is "trusted"
        search = self._make_search()
        result = search.run(observations=[obs], timeout_seconds=30, gt_states_by_obs=gt)
        self._assert_no_gt_patch(result, gt_states={0, 2, 4})


class TestPlainStillWorks(_GtSearchTestBase):
    """Regression: clean trajectory, no GT map -> conflict-free at the root."""

    def test_clean_trajectory_root_solution(self):
        obs = self._load_observation(CLEAN_STATES, ACTIONS, name="clean_plain")
        search = self._make_search()
        result = search.run(observations=[obs], timeout_seconds=30)
        self.assertEqual(result.conflicts, [])
        self.assertEqual(result.final_cost, 0.0)
        self.assertEqual(result.report["conflict_free_model_count"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
