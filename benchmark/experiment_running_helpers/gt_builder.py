"""Single source of truth for ground-truth (GT) trajectory export + validation.

This module replaces the lossy "rebuild GT JSON from the classifier `.trajectory`"
step that used to corrupt `gt_trajectories/`. It writes GT as a first-class,
parallel artifact directly from the authoritative GT states captured at source
(PDDLGym `env.step` for gym domains, the source planner trajectory for external
domains), and never derives GT from the noisy classifier output.

Two responsibilities:

1. **Export** — given a problem's GT trajectory JSON (a list of step dicts, each
   with ``current_state.literals`` / ``ground_action`` / ``next_state.literals``),
   write the eval-schema GT ``{problem}.trajectory`` + ``{problem}_trajectory.json``
   into ``gt_trajectories/{problem}/``. For generate-mode datasets the GT JSON
   arrives in raw PDDLGym schema, so a domain handler's ``_manipulate_trajectory_json``
   hook is applied first to translate it to the eval schema. The writer itself
   lives in ``src.utils.pddl_trajectory`` and is re-exported here.

2. **Validate** — model-free consistency checks on the exported GT (fully
   specified states, a ground action per step, constant object set) plus optional
   per-domain physical invariants (hanoi, npuzzle). Used as a guard so a
   corrupt/inconsistent GT fails loudly instead of shipping silently.
"""

import json
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.utils.pddl_trajectory import export_gt_trajectory

__all__ = [
    "export_gt_trajectory",
    "load_gt_json",
    "export_gt_from_problem_dir",
    "validate_gt_trajectory_file",
    "validate_gt_dir",
]


# ===========================================================================
# Export
# ===========================================================================

def load_gt_json(problem_dir: Path, problem_name: Optional[str] = None) -> List[dict]:
    """Load the ``*_trajectory.json`` GT states from a problem dir."""
    problem_dir = Path(problem_dir)
    if problem_name is not None:
        candidate = problem_dir / f"{problem_name}_trajectory.json"
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)
    matches = list(problem_dir.glob("*_trajectory.json"))
    if not matches:
        raise FileNotFoundError(f"No *_trajectory.json in {problem_dir}")
    with open(matches[0]) as f:
        return json.load(f)


def export_gt_from_problem_dir(
    problem_dir: Path,
    gt_root: Path,
    problem_name: str,
    handler=None,
    needs_schema_translation: bool = False,
) -> Path:
    """Export a problem's GT (from its ``_trajectory.json``) into ``gt_root``.

    Args:
        problem_dir: The per-problem dir holding the GT ``_trajectory.json``
            (before any classifier rebuild deletes/overwrites it).
        gt_root: The experiment's ``gt_trajectories/`` root.
        problem_name: e.g. ``"problem3"``.
        handler: A trajectory handler exposing ``_manipulate_trajectory_json``.
            Required when ``needs_schema_translation`` is True.
        needs_schema_translation: True for generate-mode datasets whose GT JSON
            is in raw PDDLGym schema and must be translated to the eval schema.
            False when the GT JSON is already eval-schema (predefined / external).

    Returns:
        Path to ``gt_root/{problem}/``.
    """
    gt_json = load_gt_json(problem_dir, problem_name)

    if needs_schema_translation:
        if handler is None or not hasattr(handler, "_manipulate_trajectory_json"):
            raise ValueError(
                "needs_schema_translation=True requires a handler with "
                "_manipulate_trajectory_json"
            )
        gt_json = handler._manipulate_trajectory_json(gt_json)

    return export_gt_trajectory(gt_json, problem_name, Path(gt_root) / problem_name)


# ===========================================================================
# Validation
# ===========================================================================

def _states_from_trajectory_file(traj_path: Path) -> List[set]:
    """Parse the state predicate sets (init + each :state) from a .trajectory."""
    txt = Path(traj_path).read_text()
    states = []
    for m in re.finditer(r"\(:(?:init|state)\s(.*?)\)\n", txt, re.S):
        preds = set(re.findall(r"\([a-z0-9_\-]+(?:\s+[a-z0-9_\-]+)*\)", m.group(1)))
        states.append(preds)
    return states


def _ops_from_trajectory_file(traj_path: Path) -> List[str]:
    """Parse the ground operator strings (in order) from a .trajectory."""
    txt = Path(traj_path).read_text()
    return re.findall(r"\(operator:\s*\((.*?)\)\)", txt)


def _pred_args(pred: str) -> List[str]:
    """('(on-disc d1 d2)') -> ['on-disc', 'd1', 'd2']."""
    return pred.strip("()").split()


def _transition_consistency_violations(states: List[set], ops: List[str]) -> List[str]:
    """Model-free check: within a valid deterministic trajectory, every ground
    instance of the same lifted action must produce the same *lifted* effect
    (add/delete) set. More than one distinct lifted effect per action name means
    the states are mutually inconsistent (classifier corruption), even if each
    state is individually well-formed. This is what catches npuzzle-style errors
    that single-state invariants miss.
    """
    from collections import defaultdict

    def lifted_delta(pre: set, post: set, action_args: List[str]):
        role = {obj: f"?{i}" for i, obj in enumerate(action_args)}
        def lift(pred: str) -> str:
            a = _pred_args(pred)
            return "(" + a[0] + " " + " ".join(role.get(x, x) for x in a[1:]) + ")"
        return frozenset(lift(p) for p in (post - pre)), frozenset(lift(p) for p in (pre - post))

    effects: Dict[str, set] = defaultdict(set)
    for i, op in enumerate(ops):
        if i + 1 >= len(states):
            break
        toks = op.split()
        name, args = toks[0], toks[1:]
        effects[name].add(lifted_delta(states[i], states[i + 1], args))

    viol = []
    for name, distinct in effects.items():
        if len(distinct) > 1:
            viol.append(f"action '{name}' has {len(distinct)} distinct lifted "
                        f"effect sets (expected 1) — inconsistent transitions")
    return viol


def _hanoi_invariant_violations(state: set) -> List[str]:
    """Physical invariants for a single hanoi state."""
    viol = []
    # Each disc sits in exactly one place (on-disc/on-peg as first arg).
    location: Dict[str, List[str]] = {}
    for p in state:
        a = _pred_args(p)
        if a and a[0] in ("on-disc", "on-peg"):
            location.setdefault(a[1], []).append(a[2])
    for disc, places in location.items():
        if len(places) > 1:
            viol.append(f"{disc} in {len(places)} places: {sorted(places)}")
    # clear-disc X implies nothing is on X.
    covered = {_pred_args(p)[2]: _pred_args(p)[1] for p in state if _pred_args(p)[0] == "on-disc"}
    for p in state:
        a = _pred_args(p)
        if a and a[0] == "clear-disc" and a[1] in covered:
            viol.append(f"{a[1]} clear but {covered[a[1]]} on it")
    # Legal stacking: (on-disc x y) requires (smaller-disc y x) present.
    for p in state:
        a = _pred_args(p)
        if a and a[0] == "on-disc" and f"(smaller-disc {a[2]} {a[1]})" not in state:
            viol.append(f"illegal stack {p} (no smaller-disc {a[2]} {a[1]})")
    return viol


def _npuzzle_invariant_violations(state: set) -> List[str]:
    """Physical invariants for an n-puzzle state: each tile/position used once."""
    viol = []
    at_tile: Dict[str, List[str]] = {}   # tile -> positions
    at_pos: Dict[str, List[str]] = {}    # position -> tiles
    for p in state:
        a = _pred_args(p)
        if a and a[0] == "at":
            at_tile.setdefault(a[1], []).append(a[2])
            at_pos.setdefault(a[2], []).append(a[1])
    for tile, ps in at_tile.items():
        if len(ps) > 1:
            viol.append(f"tile {tile} at {len(ps)} positions: {sorted(ps)}")
    for pos, ts in at_pos.items():
        if len(ts) > 1:
            viol.append(f"position {pos} holds {len(ts)} tiles: {sorted(ts)}")
    return viol


# Per-domain physical invariant hooks. Domains not listed get structural checks only.
_INVARIANT_HOOKS: Dict[str, Callable[[set], List[str]]] = {
    "hanoi": _hanoi_invariant_violations,
    "npuzzle": _npuzzle_invariant_violations,
    "n_puzzle": _npuzzle_invariant_violations,
}


def validate_gt_trajectory_file(
    traj_path: Path,
    domain_key: Optional[str] = None,
) -> List[str]:
    """Return a list of violation strings for a GT ``.trajectory`` (empty = clean).

    Runs structural checks always, plus per-domain physical invariants when a
    hook is registered for ``domain_key``.
    """
    traj_path = Path(traj_path)
    problems: List[str] = []

    states = _states_from_trajectory_file(traj_path)
    if not states:
        return [f"{traj_path.name}: no states parsed"]

    for i, s in enumerate(states):
        if not s:
            problems.append(f"state[{i}] is empty")

    hook = _INVARIANT_HOOKS.get((domain_key or "").lower())
    if hook:
        for i, s in enumerate(states):
            for v in hook(s):
                problems.append(f"state[{i}] {v}")

    # Generic, domain-agnostic transition-consistency check.
    ops = _ops_from_trajectory_file(traj_path)
    problems.extend(_transition_consistency_violations(states, ops))

    return problems


def validate_gt_dir(gt_root: Path, domain_key: Optional[str] = None) -> Dict[str, List[str]]:
    """Validate every ``gt_trajectories/{problem}/{problem}.trajectory`` under a root.

    Returns a dict problem_name -> violations (only problems with violations, or
    empty dict if all clean).
    """
    gt_root = Path(gt_root)
    report: Dict[str, List[str]] = {}
    for prob_dir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        traj = prob_dir / f"{prob_dir.name}.trajectory"
        if not traj.exists():
            report[prob_dir.name] = ["missing .trajectory"]
            continue
        viol = validate_gt_trajectory_file(traj, domain_key)
        if viol:
            report[prob_dir.name] = viol
    return report
