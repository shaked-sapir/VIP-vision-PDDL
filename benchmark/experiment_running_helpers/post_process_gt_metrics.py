"""
Post-process conflict search outputs: load 100% GT, T, and T'; compute T vs GT and T' vs GT metrics; write metrics.json.
No GT is loaded during conflict search; this runs after conflict search returns.
"""

import re
import json
from pathlib import Path
from typing import List, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser, TrajectoryParser
from pddl_plus_parser.models import Observation

from src.utils.pddl import compare_observations


def get_gt100_path(traj_path: Path) -> Path:
    """Path to 100% GT trajectory: same dir, stem with gtrate replaced by gtrate100."""
    stem = re.sub(r"gtrate\d+", "gtrate100", traj_path.stem)
    return traj_path.parent / f"{stem}.trajectory"


def load_observations_from_trajectory_list(
    domain_path: Path, trajectory_list: List[Tuple[Path, Path, Path]]
) -> List[Observation]:
    """Load one observation per (traj_path, _, _); order preserved."""
    domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    out = []
    for traj_path, *_ in trajectory_list:
        if not traj_path.exists():
            continue
        try:
            obs = parser.parse_trajectory(traj_path)
            out.append(obs)
        except Exception:
            continue
    return out


def load_t_prime_for_indices(
    domain_path: Path,
    final_observations_dir: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    indices: List[int],
) -> List[Observation]:
    """Load T' for given indices; order preserved. Returns list of length len(indices) (None where missing)."""
    domain = DomainParser(domain_path).parse_domain()
    parser = TrajectoryParser(domain)
    out = []
    for i in indices:
        name = prepared_trajectories[i][2].stem
        path = final_observations_dir / f"final_observation_{name}.trajectory"
        if not path.exists():
            out.append(None)
            continue
        try:
            out.append(parser.parse_trajectory(path))
        except Exception:
            out.append(None)
    return out


def run_post_process_gt_metrics(
    fold_work_dir: Path,
    prepared_trajectories: List[Tuple[Path, Path, Path]],
    domain_ref_path: Path,
    gt_rate: int,
) -> None:
    """
    Load 100% GT and T from paths; for each model dir under conflict_free_models load T'; compute metrics; write metrics.json.
    """
    conflict_free_models_dir = fold_work_dir / "conflict_free_models"
    if not conflict_free_models_dir.exists():
        return

    gt100_paths = [get_gt100_path(t[0]) for t in prepared_trajectories]
    indices_with_gt = [i for i, p in enumerate(gt100_paths) if p.exists()]
    if not indices_with_gt:
        return
    gt100_list = [(gt100_paths[i], None, prepared_trajectories[i][2]) for i in indices_with_gt]
    obs_gt_all = load_observations_from_trajectory_list(domain_ref_path, gt100_list)
    traj_list_t = [prepared_trajectories[i] for i in indices_with_gt]
    obs_t_all = load_observations_from_trajectory_list(domain_ref_path, traj_list_t)
    if len(obs_gt_all) != len(obs_t_all):
        return

    def metrics_for_pair(obs_pred_list: List[Observation], obs_gt_list: List[Observation]) -> dict:
        per_traj = []
        for op, og in zip(obs_pred_list, obs_gt_list):
            cmp = compare_observations(op, og, unmasked_only=True)
            per_traj.append({"overall": cmp["overall"], "per_state": cmp["per_state"]})
        # Aggregate over all trajectories
        all_overall = [p["overall"] for p in per_traj]
        total_tp = sum(o["tp"] for o in all_overall)
        total_fp = sum(o["fp"] for o in all_overall)
        total_fn = sum(o["fn"] for o in all_overall)
        total_masked = sum(o["masked_count"] for o in all_overall)
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return {
            "T_vs_GT": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": prec, "recall": rec, "masked_count": total_masked},
            "per_trajectory": per_traj,
        }

    t_vs_gt = metrics_for_pair(obs_t_all, obs_gt_all)

    model_dirs = sorted([d for d in conflict_free_models_dir.iterdir() if d.is_dir()])
    for model_dir in model_dirs:
        final_obs_dir = model_dir / "final_observations"
        if not final_obs_dir.exists():
            continue
        t_prime_raw = load_t_prime_for_indices(
            domain_ref_path, final_obs_dir, prepared_trajectories, indices_with_gt
        )
        obs_t_prime = [o for o in t_prime_raw if o is not None]
        if len(obs_t_prime) != len(obs_gt_all):
            continue
        t_prime_vs_gt = metrics_for_pair(obs_t_prime, obs_gt_all)
        payload = {
            "T_vs_GT": t_vs_gt["T_vs_GT"],
            "T_prime_vs_GT": t_prime_vs_gt["T_vs_GT"],
            "per_trajectory_T_vs_GT": t_vs_gt["per_trajectory"],
            "per_trajectory_T_prime_vs_GT": t_prime_vs_gt["per_trajectory"],
        }
        (model_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
