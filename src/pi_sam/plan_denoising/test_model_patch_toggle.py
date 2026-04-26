import unittest
from typing import Dict

from src.pi_sam.noisy_pisam.typings import (
    Conflict,
    ConflictType,
    ModelPart,
    ParameterBoundLiteral,
    PatchOperation,
)
from src.pi_sam.plan_denoising.conflict_search import (
    ConflictDrivenPatchSearchBase,
    Key,
)


class _DummySearch(ConflictDrivenPatchSearchBase):
    """Minimal concrete subclass for testing patch-update logic only."""

    def _create_learner(self, domain_copy):
        raise NotImplementedError("Learner creation is not needed for these unit tests.")


def _make_conflict(conflict_type: ConflictType) -> Conflict:
    pbl = ParameterBoundLiteral(predicate_name="on", parameters=("?x", "?y"), is_positive=True)
    return Conflict(
        action_name="stack",
        pbl=pbl,
        conflict_type=conflict_type,
        observation_index=0,
        component_index=0,
        grounded_fluent="(on a b)",
    )


def _make_key() -> Key:
    pbl = ParameterBoundLiteral(predicate_name="on", parameters=("?x", "?y"), is_positive=True)
    return "stack", ModelPart.EFFECT, pbl


class TestModelPatchToggleBehavior(unittest.TestCase):
    def setUp(self):
        self.search = _DummySearch(partial_domain_template=None)
        self.key = _make_key()

    def test_builder_keeps_noop_when_op_already_matches(self):
        conflict = _make_conflict(ConflictType.REQUIRE_EFFECT_VS_CANNOT)
        constraints: Dict[Key, PatchOperation] = {self.key: PatchOperation.FORBID}

        updated = self.search._build_model_patch(conflict, constraints)

        self.assertEqual(updated, constraints)

    def test_builder_replaces_opposite_constraint(self):
        conflict = _make_conflict(ConflictType.REQUIRE_EFFECT_VS_CANNOT)
        constraints: Dict[Key, PatchOperation] = {self.key: PatchOperation.REQUIRE}

        updated = self.search._build_model_patch(conflict, constraints)

        self.assertEqual(updated[self.key], PatchOperation.FORBID)

    def test_builder_adds_missing_constraint(self):
        conflict = _make_conflict(ConflictType.FORBID_EFFECT_VS_MUST)
        constraints: Dict[Key, PatchOperation] = {}

        updated = self.search._build_model_patch(conflict, constraints)

        self.assertEqual(updated[self.key], PatchOperation.REQUIRE)


if __name__ == "__main__":
    unittest.main()
