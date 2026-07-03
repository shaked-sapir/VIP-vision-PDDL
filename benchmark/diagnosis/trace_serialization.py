"""
Shared serialization utilities for conflict search trace events.

Used by both retrace_search.py (offline retracing) and the experiment pipeline
(online tracing via --events-tracing).
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pddl_plus_parser.models import Observation

from src.pi_sam.plan_denoising.conflict_search import NodeExpansionEvent


def _serialize_state(state) -> Dict:
    """Serialize a State into positive predicates and masked predicates lists."""
    positives = []
    masked = []
    for grounded_predicates in state.state_predicates.values():
        for p in grounded_predicates:
            rep = p.untyped_representation
            positives.append(rep)
            if p.is_masked:
                masked.append(rep)
    return {"predicates": sorted(positives), "masked": sorted(masked)}


def serialize_observations(
    ordered_observations: List[Tuple[str, Observation]],
) -> List[Dict]:
    """Serialize observations into a JSON-friendly format.

    Each observation becomes:
        {"problem": str, "components": [{"action": str, "prev": {...}, "next": {...}}, ...]}

    Each state is {"predicates": [...positive...], "masked": [...masked...]}.
    """
    result = []
    for problem_name, obs in ordered_observations:
        components = []
        for comp in obs.components:
            components.append({
                "action": str(comp.grounded_action_call),
                "prev": _serialize_state(comp.previous_state),
                "next": _serialize_state(comp.next_state),
            })
        result.append({"problem": problem_name, "components": components})
    return result


def serialize_event(e: NodeExpansionEvent) -> Dict:
    """Convert a NodeExpansionEvent to a JSON-serializable dict."""
    return {
        "index": e.node_index,
        "parent_index": e.parent_index,
        "branch_type": e.branch_type,
        "depth": e.depth,
        "cost": e.cost,
        "is_conflict_free": e.is_conflict_free,
        "cfm_index": e.cfm_index,
        "model_constraints": list(e.model_constraints),
        "fluent_patches": list(e.fluent_patches),
        "conflicts": [
            {
                "action": c.action_name,
                "predicate": str(c.pbl),
                "type": c.conflict_type.value,
                "obs": c.observation_index,
                "comp": c.component_index,
                "fluent": c.grounded_fluent,
            }
            for c in e.conflicts
        ],
        "chosen_group": [
            {
                "action": c.action_name,
                "predicate": str(c.pbl),
                "type": c.conflict_type.value,
                "obs": c.observation_index,
                "comp": c.component_index,
                "fluent": c.grounded_fluent,
            }
            for c in e.chosen_group
        ] if e.chosen_group else None,
        "children": {
            "fluent_fix": {"cost": e.child_fluent_cost, "desc": e.child_fluent_fix} if e.child_fluent_fix else None,
            "model_fix": {"cost": e.child_model_cost, "desc": e.child_model_fix} if e.child_model_fix else None,
        },
    }


def write_trace_json(
    trace_log: List[NodeExpansionEvent],
    output_path: Path,
    search_params: Dict,
    fold_dir: Path = None,
    ordered_observations: Optional[List[Tuple[str, Observation]]] = None,
) -> Path:
    """Serialize a list of NodeExpansionEvents and write to a JSON file.

    Args:
        trace_log: Collected events from on_node_expanded callback.
        output_path: Where to write the JSON file.
        search_params: Dict of search parameters used.
        fold_dir: Optional fold directory path (stored as metadata).
        ordered_observations: Optional list of (problem_name, Observation)
            to embed in the trace for the observation viewer.

    Returns:
        The output_path.
    """
    cfm_nodes = [e for e in trace_log if e.is_conflict_free]
    trace_data = {
        "fold_dir": str(fold_dir) if fold_dir else None,
        "search_params": search_params,
        "outcome": {
            "nodes_expanded": len(trace_log),
            "conflict_free_count": len(cfm_nodes),
            "best_cost": min((e.cost for e in cfm_nodes), default=None),
        },
        "nodes": [serialize_event(e) for e in trace_log],
    }

    if ordered_observations is not None:
        trace_data["observations"] = serialize_observations(ordered_observations)

    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    return output_path
