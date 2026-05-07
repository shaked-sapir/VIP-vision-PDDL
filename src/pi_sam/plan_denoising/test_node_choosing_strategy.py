import unittest

from src.pi_sam.plan_denoising.conflict_search import ConflictDrivenPatchSearchBase
from src.pi_sam.plan_denoising.frontier import (
    HeapFrontier,
    NodeChoosingStrategy,
    SearchNode,
)


class _DummySearch(ConflictDrivenPatchSearchBase):
    def _create_learner(self, domain_copy):
        raise NotImplementedError


def _make_node(depth: int) -> SearchNode:
    return SearchNode(
        cost=1.0,
        depth=depth,
        model_constraints={},
        fluent_patches=set(),
    )


class TestNodeChoosingStrategy(unittest.TestCase):
    def test_model_patch_first_returns_model_then_fluent(self):
        search = _DummySearch(
            partial_domain_template=None,
            node_choosing_strategy=NodeChoosingStrategy.MODEL_PATCH_FIRST,
        )
        fluent_child = _make_node(depth=1)
        model_child = _make_node(depth=2)

        ordered = search._ordered_children_for_frontier(fluent_child, model_child)

        self.assertEqual(ordered, [model_child, fluent_child])

    def test_fluent_patch_first_returns_fluent_then_model(self):
        search = _DummySearch(
            partial_domain_template=None,
            node_choosing_strategy=NodeChoosingStrategy.FLUENT_PATCH_FIRST,
        )
        fluent_child = _make_node(depth=1)
        model_child = _make_node(depth=2)

        ordered = search._ordered_children_for_frontier(fluent_child, model_child)

        self.assertEqual(ordered, [fluent_child, model_child])

    def test_randomized_strategy_produces_both_orders(self):
        search = _DummySearch(
            partial_domain_template=None,
            seed=7,
            node_choosing_strategy=NodeChoosingStrategy.RANDOMIZED,
        )
        fluent_child = _make_node(depth=1)
        model_child = _make_node(depth=2)

        first_nodes = set()
        for _ in range(30):
            ordered = search._ordered_children_for_frontier(fluent_child, model_child)
            first_nodes.add(ordered[0].depth)

        self.assertEqual(first_nodes, {1, 2})

    def test_single_child_is_returned(self):
        search = _DummySearch(partial_domain_template=None)
        fluent_child = _make_node(depth=1)

        ordered = search._ordered_children_for_frontier(fluent_child, None)

        self.assertEqual(ordered, [fluent_child])

    def test_heap_frontier_tiebreak_uses_insertion_order(self):
        frontier = HeapFrontier()
        first = _make_node(depth=1)
        second = _make_node(depth=1)

        frontier.push(first)
        frontier.push(second)

        self.assertIs(frontier.pop(), first)
        self.assertIs(frontier.pop(), second)


if __name__ == "__main__":
    unittest.main()
