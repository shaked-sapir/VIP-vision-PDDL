"""Aggregate conflict-free model PDDL domains across an experiment.

For every ``conflict_free_model_*/model.pddl`` under an experiment's
``testing/`` tree, builds six merged domain files:

1. ``union_pre_intersect_eff.pddl`` — per action, preconditions are the
   union of literals across CFMs; effects are the intersection.
2. ``intersect_all.pddl`` — per action, both preconditions and effects
   are intersections across CFMs.
3. ``vote_0p25.pddl`` — literals in ≥ 25% of CFMs.
4. ``vote_0p50.pddl`` — literals in ≥ 50% of CFMs.
5. ``vote_0p75.pddl`` — literals in ≥ 75% of CFMs.
6. ``vote_0p90.pddl`` — literals in ≥ 90% of CFMs.

Usage:
    python -m benchmark.evaluation.cfm.cfm_domain_aggregate <experiment_dir>
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Action, Domain, Predicate

logger = logging.getLogger(__name__)

LiteralStr = str
Inequality = Tuple[str, str]

UNION_PRE_INTERSECT_EFF = "union_pre_intersect_eff"
INTERSECT_ALL = "intersect_all"
VOTE_0P25 = "vote_0p25"
VOTE_0P50 = "vote_0p50"
VOTE_0P75 = "vote_0p75"
VOTE_0P90 = "vote_0p90"

# (fraction of CFMs, output filename stem)
THRESHOLD_VOTES: List[Tuple[float, str]] = [
    (0.25, VOTE_0P25),
    (0.5, VOTE_0P50),
    (0.75, VOTE_0P75),
    (0.9, VOTE_0P90),
]


@dataclass(frozen=True)
class ActionLiterals:
    """Normalized literal sets for one action in one CFM."""

    positive_preconditions: Set[LiteralStr]
    inequalities: Set[Inequality]
    equalities: Set[Inequality]
    effects: Set[LiteralStr]


@dataclass(frozen=True)
class AggregationResult:
    """Paths and metadata from a CFM domain aggregation run."""

    experiment_dir: Path
    output_dir: Path
    source_model_count: int
    source_model_paths: List[str]
    union_pre_intersect_eff_path: Path
    intersect_all_path: Path
    vote_0p25_path: Path
    vote_0p50_path: Path
    vote_0p75_path: Path
    vote_0p90_path: Path
    summary_path: Path
    action_names: List[str]
    threshold_vote_paths: Dict[str, Path]


def find_cfm_model_paths(experiment_dir: Path) -> List[Path]:
    """Return sorted paths to legitimate CFM ``model.pddl`` files."""
    testing_dir = experiment_dir / "testing"
    if not testing_dir.exists():
        raise FileNotFoundError(
            f"Expected a 'testing/' subdirectory under {experiment_dir}, but none was found."
        )

    paths = sorted(
        testing_dir.glob("**/conflict_free_models/conflict_free_model_*/model.pddl")
    )
    if not paths:
        raise FileNotFoundError(
            f"No conflict_free_model_*/model.pddl files found under {testing_dir}"
        )
    return paths


def _extract_action_literals(action: Action) -> ActionLiterals:
    root = action.preconditions.root
    positive_preconditions = {pred.untyped_representation for pred in root.operands}
    return ActionLiterals(
        positive_preconditions=positive_preconditions,
        inequalities=set(root.inequality_preconditions),
        equalities=set(root.equality_preconditions),
        effects={eff.untyped_representation for eff in action.discrete_effects},
    )


def _build_predicate_registry(domains: Iterable[Domain]) -> Dict[LiteralStr, Predicate]:
    """Map untyped literal strings to a canonical Predicate object."""
    registry: Dict[LiteralStr, Predicate] = {}
    for domain in domains:
        for action in domain.actions.values():
            for pred in action.preconditions.root.operands:
                registry.setdefault(pred.untyped_representation, pred)
            for pred in action.discrete_effects:
                registry.setdefault(pred.untyped_representation, pred)
    return registry


def _fraction_threshold(cfm_count: int, fraction: float) -> int:
    """Minimum CFM count for a literal to qualify at the given fraction."""
    return math.ceil(cfm_count * fraction)


def _merge_fraction_vote(parts: List[ActionLiterals], fraction: float) -> ActionLiterals:
    """Keep literals that appear in at least *fraction* of the given CFM snapshots."""
    if not parts:
        return ActionLiterals(set(), set(), set(), set())

    threshold = _fraction_threshold(len(parts), fraction)

    def _qualified(field: str) -> Set:
        counts: Dict = {}
        for part in parts:
            for item in getattr(part, field):
                counts[item] = counts.get(item, 0) + 1
        return {item for item, count in counts.items() if count >= threshold}

    return ActionLiterals(
        positive_preconditions=_qualified("positive_preconditions"),
        inequalities=_qualified("inequalities"),
        equalities=_qualified("equalities"),
        effects=_qualified("effects"),
    )


def _merge_literal_sets(
    parts: List[ActionLiterals],
    precond_mode: str,
    effect_mode: str,
) -> ActionLiterals:
    if not parts:
        return ActionLiterals(set(), set(), set(), set())

    def _merge(field: str, mode: str) -> Set:
        sets = [getattr(p, field) for p in parts]
        if mode == "union":
            return set.union(*sets) if sets else set()
        return set.intersection(*sets) if sets else set()

    return ActionLiterals(
        positive_preconditions=_merge("positive_preconditions", precond_mode),
        inequalities=_merge("inequalities", precond_mode),
        equalities=_merge("equalities", precond_mode),
        effects=_merge("effects", effect_mode),
    )


def _apply_literals_to_action(
    action: Action,
    literals: ActionLiterals,
    registry: Dict[LiteralStr, Predicate],
) -> None:
    root = action.preconditions.root
    root.operands.clear()
    root.inequality_preconditions.clear()
    root.equality_preconditions.clear()

    for literal in sorted(literals.positive_preconditions):
        action.preconditions.add_condition(copy.deepcopy(registry[literal]))

    for ineq in sorted(literals.inequalities):
        root.add_equality_condition(ineq, inequality=True)

    for eq in sorted(literals.equalities):
        root.add_equality_condition(eq, inequality=False)

    action.discrete_effects = {
        copy.deepcopy(registry[literal]) for literal in sorted(literals.effects)
    }


def _build_merged_domain(
    template_domain: Domain,
    action_literals_by_name: Dict[str, ActionLiterals],
    registry: Dict[LiteralStr, Predicate],
) -> Domain:
    merged = copy.deepcopy(template_domain)
    merged.actions = {
        name: merged.actions[name]
        for name in action_literals_by_name
        if name in merged.actions
    }
    for name, literals in action_literals_by_name.items():
        _apply_literals_to_action(merged.actions[name], literals, registry)
    return merged


def _collect_action_literals(
    domains: List[Domain],
) -> Dict[str, List[ActionLiterals]]:
    by_action: Dict[str, List[ActionLiterals]] = {}
    for domain in domains:
        for name, action in domain.actions.items():
            by_action.setdefault(name, []).append(_extract_action_literals(action))
    return by_action


def aggregate_cfm_domains(
    experiment_dir: Path,
    output_dir: Path | None = None,
) -> AggregationResult:
    """Aggregate all CFM model.pddl files and write merged domain outputs."""
    experiment_dir = experiment_dir.resolve()
    model_paths = find_cfm_model_paths(experiment_dir)
    domains = [DomainParser(path).parse_domain() for path in model_paths]

    if output_dir is None:
        output_dir = experiment_dir / "aggregated_domains"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = _build_predicate_registry(domains)
    action_literals = _collect_action_literals(domains)
    template_domain = copy.deepcopy(domains[0])

    union_pre_parts = {
        name: _merge_literal_sets(parts, "union", "intersection")
        for name, parts in action_literals.items()
    }
    intersect_all_parts = {
        name: _merge_literal_sets(parts, "intersection", "intersection")
        for name, parts in action_literals.items()
    }
    threshold_parts: Dict[str, Dict[str, ActionLiterals]] = {
        stem: {
            name: _merge_fraction_vote(parts, fraction)
            for name, parts in action_literals.items()
        }
        for fraction, stem in THRESHOLD_VOTES
    }

    union_pre_domain = _build_merged_domain(template_domain, union_pre_parts, registry)
    intersect_all_domain = _build_merged_domain(
        copy.deepcopy(template_domain), intersect_all_parts, registry
    )

    union_path = output_dir / "union_pre_intersect_eff.pddl"
    intersect_path = output_dir / "intersect_all.pddl"
    summary_path = output_dir / "aggregation_summary.json"

    union_path.write_text(union_pre_domain.to_pddl())
    intersect_path.write_text(intersect_all_domain.to_pddl())

    threshold_vote_paths: Dict[str, Path] = {}
    for fraction, stem in THRESHOLD_VOTES:
        vote_domain = _build_merged_domain(
            copy.deepcopy(template_domain), threshold_parts[stem], registry
        )
        vote_path = output_dir / f"{stem}.pddl"
        vote_path.write_text(vote_domain.to_pddl())
        threshold_vote_paths[stem] = vote_path

    per_action_summary = {}
    for name in sorted(action_literals):
        cfm_count = len(action_literals[name])
        entry: Dict = {
            "cfm_count": cfm_count,
            "union_preconditions": sorted(union_pre_parts[name].positive_preconditions),
            "union_inequalities": [list(t) for t in sorted(union_pre_parts[name].inequalities)],
            "intersect_effects": sorted(union_pre_parts[name].effects),
            "intersect_preconditions": sorted(intersect_all_parts[name].positive_preconditions),
            "intersect_inequalities": [list(t) for t in sorted(intersect_all_parts[name].inequalities)],
            "intersect_effects_all": sorted(intersect_all_parts[name].effects),
        }
        for fraction, stem in THRESHOLD_VOTES:
            parts_at_threshold = threshold_parts[stem][name]
            entry[f"{stem}_threshold"] = _fraction_threshold(cfm_count, fraction)
            entry[f"{stem}_preconditions"] = sorted(
                parts_at_threshold.positive_preconditions
            )
            entry[f"{stem}_inequalities"] = [
                list(t) for t in sorted(parts_at_threshold.inequalities)
            ]
            entry[f"{stem}_effects"] = sorted(parts_at_threshold.effects)
        per_action_summary[name] = entry

    summary = {
        "experiment_dir": str(experiment_dir),
        "source_model_count": len(model_paths),
        "source_models": [str(p) for p in model_paths],
        "action_names": sorted(action_literals),
        "outputs": {
            UNION_PRE_INTERSECT_EFF: str(union_path),
            INTERSECT_ALL: str(intersect_path),
            **{stem: str(path) for stem, path in threshold_vote_paths.items()},
        },
        "threshold_votes": [
            {"fraction": fraction, "output": stem} for fraction, stem in THRESHOLD_VOTES
        ],
        "per_action": per_action_summary,
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    return AggregationResult(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        source_model_count=len(model_paths),
        source_model_paths=[str(p) for p in model_paths],
        union_pre_intersect_eff_path=union_path,
        intersect_all_path=intersect_path,
        vote_0p25_path=threshold_vote_paths[VOTE_0P25],
        vote_0p50_path=threshold_vote_paths[VOTE_0P50],
        vote_0p75_path=threshold_vote_paths[VOTE_0P75],
        vote_0p90_path=threshold_vote_paths[VOTE_0P90],
        summary_path=summary_path,
        action_names=sorted(action_literals),
        threshold_vote_paths=threshold_vote_paths,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate conflict-free model.pddl files under an experiment directory "
            "into union/intersection/majority merged domains."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Parent experiment config directory (contains a testing/ subdirectory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for merged PDDL outputs (default: <experiment_dir>/aggregated_domains).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    args = _parse_args()
    result = aggregate_cfm_domains(args.experiment_dir, args.output_dir)

    print(f"Aggregated {result.source_model_count} CFM model.pddl file(s)")
    print(f"Actions: {', '.join(result.action_names)}")
    print(f"\n1. Union preconditions / intersect effects:")
    print(f"   {result.union_pre_intersect_eff_path}")
    print(f"\n2. Intersect preconditions / intersect effects:")
    print(f"   {result.intersect_all_path}")
    for fraction, stem in THRESHOLD_VOTES:
        pct = int(fraction * 100)
        print(f"\n{3 + THRESHOLD_VOTES.index((fraction, stem))}. "
              f"Threshold vote (>={pct}% of CFMs) for preconditions and effects:")
        print(f"   {result.threshold_vote_paths[stem]}")
    print(f"\nSummary: {result.summary_path}")
