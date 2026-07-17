"""Parity check: our ternary ROSAME encoder at mask=0 == AMLGym's binary encoder.

TEMPORARY — delete after confirming parity (hence the DELETE_LATER suffix).

Why the encoder (not the final model)?
    ROSAME trains a neural net for 100 epochs, so two end-to-end runs give
    different PDDL models (random init). The *encoding* — ``prepare_rosame_data``
    → (state1, action, state2) tensors — is deterministic, so that is what we
    compare. At mask=0 no fluent is masked, so our ternary encoder never emits
    0.5 and should reproduce AMLGym's binary encoder exactly.

What it does:
    - AMLGym path : TrajectoryParser(domain, problem) → base Rosame_Runner
                    (binary encoding), as AMLGym's ROSAME.learn uses it.
    - Our path    : same parse → ground_observation_completely (no masking) →
                    our PORosame_Runner (ternary encoding).
    - Compares the per-step state1/state2 vectors; reports PASS or the first
      (step, proposition) mismatch.

Usage:
    python -m benchmark.diagnosis.check_rosame_encoding_parity_DELETE_LATER \
        --domain  path/to/domain.pddl \
        --problem path/to/problemN.pddl \
        --trajectory path/to/problemN.trajectory

Requires the full stack (torch / amlgym / pddl_plus_parser) — run on your machine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser

from amlgym.algorithms.rosame.experiment_runner.rosame_runner import Rosame_Runner
from benchmark.algorithm_adapters.po_rosame_runner import PORosame_Runner
from src.utils.pddl import ground_observation_completely


def _encode_amlgym(domain_path, problem, trajectory_path, partial_domain):
    """AMLGym's actual path: problem-aware parse, binary base encoder."""
    obs = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_path)
    runner = Rosame_Runner(str(domain_path))
    runner.add_problem(problem)
    runner.ground_new_trajectory()
    state1, action, state2 = runner.prepare_rosame_data(obs)
    return runner.rosame.propositions, state1, state2


def _encode_ours(domain_path, problem, trajectory_path, partial_domain):
    """Our path: problem-aware parse, ground completely, no masking, ternary encoder."""
    obs = TrajectoryParser(partial_domain, problem).parse_trajectory(trajectory_path)
    grounded = ground_observation_completely(partial_domain, obs)
    runner = PORosame_Runner(str(domain_path))
    runner.add_problem(problem)
    runner.ground_new_trajectory()
    state1, action, state2 = runner.prepare_rosame_data(grounded)
    return runner.rosame.propositions, state1, state2


def _compare(props, name, ours, base) -> bool:
    """Compare two per-step encodings element-wise. Returns True if identical."""
    if len(ours) != len(base):
        print(f"  ✗ {name}: step count differs (ours={len(ours)} vs amlgym={len(base)})")
        return False
    ok = True
    for step, (row_ours, row_base) in enumerate(zip(ours, base)):
        if len(row_ours) != len(row_base):
            print(f"  ✗ {name}[step {step}]: vector length differs "
                  f"(ours={len(row_ours)} vs amlgym={len(row_base)})")
            ok = False
            continue
        for idx, (vo, vb) in enumerate(zip(row_ours, row_base)):
            if float(vo) != float(vb):
                prop = props[idx] if idx < len(props) else f"<idx {idx}>"
                print(f"  ✗ {name}[step {step}] proposition '{prop}': "
                      f"ours={vo} vs amlgym={vb}")
                ok = False
                break  # first mismatch per step is enough
    return ok


def main():
    parser = argparse.ArgumentParser(description="ROSAME encoder parity (ternary@mask=0 vs binary).")
    parser.add_argument("--domain", required=True, type=Path)
    parser.add_argument("--problem", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    args = parser.parse_args()

    partial_domain = DomainParser(args.domain, partial_parsing=True).parse_domain()
    problem = ProblemParser(args.problem, partial_domain).parse_problem()

    props_base, s1_base, s2_base = _encode_amlgym(args.domain, problem, args.trajectory, partial_domain)
    props_ours, s1_ours, s2_ours = _encode_ours(args.domain, problem, args.trajectory, partial_domain)

    print(f"propositions: ours={len(props_ours)}, amlgym={len(props_base)}, "
          f"same order={list(props_ours) == list(props_base)}")

    ok_pre = _compare(list(props_ours), "state1 (pre)", s1_ours, s1_base)
    ok_next = _compare(list(props_ours), "state2 (next)", s2_ours, s2_base)

    if ok_pre and ok_next:
        print("\n✓ PASS — ternary encoding at mask=0 matches AMLGym's binary encoding.")
    else:
        print("\n✗ FAIL — encodings diverge (see mismatches above).")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
