from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


TIMEOUT_RE = re.compile(r"dfs_timeout=(\d+)s")
DATE_RE = re.compile(r"__(\d{8})$")


def _parse_timeout_from_name(experiment_id: str) -> Optional[int]:
    match = TIMEOUT_RE.search(experiment_id)
    return int(match.group(1)) if match else None


def _parse_date_from_name(experiment_id: str) -> str:
    match = DATE_RE.search(experiment_id)
    return match.group(1) if match else ""


def _safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def discover_experiments(root: Path) -> List[Dict[str, Any]]:
    """Discover experiment folders under root.

    An experiment is a directory that contains `evaluation_results`.
    """
    experiments: List[Dict[str, Any]] = []

    for exp_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        eval_dir = exp_dir / "evaluation_results"
        if not eval_dir.exists():
            continue

        run_params_path = eval_dir / "run_params.json"
        run_params = _safe_load_json(run_params_path) if run_params_path.exists() else None

        timeout_seconds = None
        model_constraint_weight = 0.0
        mode = None
        frame_axiom_mode = None
        planning_timeout_seconds = None
        learning_timeout_seconds = None
        fluent_patch_cost = None
        fluent_patch_weight = None
        model_patch_cost = None
        max_search_nodes = None
        node_choosing_strategy = None

        if run_params:
            learning_timeout_seconds = run_params.get("learning_timeout_seconds")
            timeout_seconds = learning_timeout_seconds
            planning_timeout_seconds = run_params.get("planning_timeout_seconds")
            model_constraint_weight = run_params.get("model_constraint_weight", 0.0)
            fluent_patch_cost = run_params.get("fluent_patch_cost")
            fluent_patch_weight = run_params.get("fluent_patch_weight")
            model_patch_cost = run_params.get("model_patch_cost")
            max_search_nodes = run_params.get("max_search_nodes")
            node_choosing_strategy = run_params.get("node_choosing_strategy")
            mode = run_params.get("mode")
            frame_axiom_mode = run_params.get("frame_axiom_mode")

        if timeout_seconds is None:
            timeout_seconds = _parse_timeout_from_name(exp_dir.name)

        experiments.append(
            {
                "experiment_id": exp_dir.name,
                "experiment_path": str(exp_dir),
                "evaluation_results_path": str(eval_dir),
                "run_params_path": str(run_params_path) if run_params_path.exists() else None,
                "has_run_params": run_params is not None,
                "date_tag": _parse_date_from_name(exp_dir.name),
                "timeout_seconds": timeout_seconds,
                "learning_timeout_seconds": learning_timeout_seconds,
                "planning_timeout_seconds": planning_timeout_seconds,
                "frame_axiom_mode": frame_axiom_mode,
                "fluent_patch_cost": fluent_patch_cost,
                "fluent_patch_weight": fluent_patch_weight,
                "model_patch_cost": model_patch_cost,
                "model_constraint_weight": model_constraint_weight,
                "max_search_nodes": max_search_nodes,
                "node_choosing_strategy": node_choosing_strategy,
                "mode": mode,
            }
        )

    return experiments

