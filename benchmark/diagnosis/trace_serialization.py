"""
Shared serialization utilities for conflict search trace events.

Used by both retrace_search.py (offline retracing) and the experiment pipeline
(online tracing via --events-tracing).
"""

import json
from pathlib import Path
from typing import Dict, List

from src.pi_sam.plan_denoising.conflict_search import NodeExpansionEvent


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
) -> Path:
    """Serialize a list of NodeExpansionEvents and write to a JSON file.

    Args:
        trace_log: Collected events from on_node_expanded callback.
        output_path: Where to write the JSON file.
        search_params: Dict of search parameters used.
        fold_dir: Optional fold directory path (stored as metadata).

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

    with open(output_path, "w") as f:
        json.dump(trace_data, f, indent=2)

    return output_path
