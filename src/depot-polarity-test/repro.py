"""Reproduce the depot both-polarity corruption (see README.md).

    python src/depot-polarity-test/repro.py

Parses one clean depot trajectory (mask=0.0, noise=0.0 cell), runs
``ground_observation_completely`` on it, and regenerates ``state_dump.txt``
showing every state's contradictory ``clear`` fluents. Also prints the
eq/hash-contract violation for a single pair of GroundedPredicates.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from src.utils.pddl_state import (
    get_all_possible_groundings,
    ground_observation_completely,
)

DOMAIN_PDDL = REPO_ROOT / "src/domains/depot/depot.pddl"
CELL = (
    REPO_ROOT
    / "benchmark/running_results/depot/sim_run__mask=0.0__noise=0.0"
    / "testing/fold0_numtrajs3_gtrate0"
)
TRAJECTORY = CELL / "original_observations/original_observation_problem3.trajectory"
PROBLEM_PDDL = (
    REPO_ROOT
    / "benchmark/data/depot"
    / "multi_problem_04-07-2026T01:32:08__model=gpt-5.2__steps=100"
    / "training/trajectories/problem3/problem3.pddl"
)
OUT = Path(__file__).resolve().parent / "state_dump.txt"


def _fluents(state):
    """Sorted (str, name, args, is_positive) tuples of a state's fluents."""
    rows = []
    for _key, preds in sorted(state.state_predicates.items()):
        for gp in sorted(preds, key=str):
            rows.append((str(gp), gp.name, tuple(gp.object_mapping.values()), gp.is_positive))
    return rows


def _both_polarity_keys(state):
    """(name, args) keys appearing with both polarities in the state."""
    seen = {}
    for _s, name, args, pos in _fluents(state):
        seen.setdefault((name, args), set()).add(pos)
    return sorted(k for k, v in seen.items() if len(v) > 1)


def main() -> None:
    partial = DomainParser(DOMAIN_PDDL, partial_parsing=True).parse_domain()
    problem = ProblemParser(PROBLEM_PDDL, partial).parse_problem()
    observation = TrajectoryParser(partial, problem).parse_trajectory(TRAJECTORY)
    grounded = ground_observation_completely(partial, observation)

    raw_states = [observation.components[0].previous_state] + [
        c.next_state for c in observation.components
    ]
    states = [grounded.components[0].previous_state] + [
        c.next_state for c in grounded.components
    ]

    lines = [
        f"source trajectory : {TRAJECTORY.relative_to(REPO_ROOT)}",
        f"problem pddl      : {PROBLEM_PDDL.relative_to(REPO_ROOT)}",
        f"domain pddl       : {DOMAIN_PDDL.relative_to(REPO_ROOT)}",
        "",
        "STATE-BY-STATE SUMMARY (after ground_observation_completely)",
        f"{'state':>5} {'raw fluents':>12} {'completed':>10}  both-polarity fluents",
    ]
    for idx, (raw, st) in enumerate(zip(raw_states, states)):
        both = _both_polarity_keys(st)
        both_str = ", ".join(f"{n}({','.join(a)})" for n, a in both) or "-"
        lines.append(f"{idx:>5} {len(_fluents(raw)):>12} {len(_fluents(st)):>10}  {both_str}")

    bad = set(_both_polarity_keys(states[0]))
    lines += [
        "",
        "=" * 70,
        "FULL STATE 0 (initial state) AFTER ground_observation_completely",
        "  '>>>' marks fluents present in BOTH polarities (the corruption)",
        "=" * 70,
    ]
    for s, name, args, _pos in _fluents(states[0]):
        mark = ">>> " if (name, args) in bad else "    "
        lines.append(f"{mark}{s}")

    lines += ["", "RAW STATE 0 AS PARSED FROM THE TRAJECTORY FILE (before completion)"]
    for s, *_ in _fluents(raw_states[0]):
        lines.append(f"    {s}")

    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(lines)} lines)")

    # --- the smoking gun: eq/hash contract violation on one pair -----------
    parsed = next(
        gp
        for preds in raw_states[0].state_predicates.values()
        for gp in preds
        if gp.name == "clear"
    )
    lifted_clear = partial.predicates["clear"]
    generated = next(
        g
        for g in get_all_possible_groundings(lifted_clear, observation.grounded_objects)
        if g.object_mapping == parsed.object_mapping
    )
    print(f"parsed    : {parsed}")
    print(f"generated : {generated}")
    print(
        f"eq: {parsed == generated} | same str: {str(parsed) == str(generated)} "
        f"| same hash: {hash(parsed) == hash(generated)} "
        f"| set membership (generated in {{parsed}}): {generated in {parsed}}"
    )


if __name__ == "__main__":
    main()
