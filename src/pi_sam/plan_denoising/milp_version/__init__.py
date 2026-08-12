"""MILP formulation of the trajectory-repair problem CDPS solves by search.

The encoder here is shared by two very different callers, which is why it lives
under ``src/`` rather than next to either of them:

  - ``cdps_milp_*`` (our learner family) — the MILP *is* the denoiser: one solve
    produces repaired trajectories T', and PI-SAM learns the returned model from
    them. See ``docs/cdps-milp-denoiser-design.md``.
  - ``rosame_milp*`` (competitor baselines, ``benchmark/baselines/``) — the MILP
    regularizes a neural learner's soft predictions inside a training loop.

The two differ only in the *rule set* handed to the encoder (see
:class:`encoding_config.MilpEncodingConfig` presets) and in what the caller does
with the solution — never in the constraint machinery itself.

Layout:
  vendor/                  — vendored from the ROSAME repo (see vendor/UPSTREAM.md)
  encoding_config.py       — the toggle-able constraint/objective families
  encoder.py               — CP-SAT encoder with observed (fixed) actions
  converter.py             — our pddl_plus_parser observations -> planning_structs
  config.py                — the validated ``cdps_milp`` config block
  trajectory_extraction.py — solved MILP -> repaired (re-masked) observations T'
  single_round.py          — the ``cdps_milp_single_round`` driver

The vendored packages keep their upstream top-level names (``planning_structs``,
``constraint_opt``), loaded via the sys.path insertion below so upstream's
absolute imports work unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
