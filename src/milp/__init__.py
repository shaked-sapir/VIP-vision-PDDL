"""Action-model learning as one CP-SAT program: the encoder and its vocabulary.

Nothing here knows who is calling. Two unrelated learners share it, and they
agree on nothing except the constraint machinery:

  - ``pisam_milp_*`` (``src/plan_denoising/milp_denoiser/``) — the solve
    *is* the denoiser: it repairs the observed trajectories, and PI-SAM learns
    from the result.
  - ``rosame_milp*`` (``benchmark/baselines/``) — the solve regularizes a neural
    learner's soft predictions inside a training loop.

They differ in the rule set handed to the encoder (see
:class:`encoding_config.MilpEncodingConfig` presets) and in what they do with
the solution — never in the encoding itself.

Layout:
  vendor/            — vendored from the ROSAME repo (see vendor/UPSTREAM.md)
  encoding_config.py — the toggle-able constraint/objective families
  encoder.py         — CP-SAT encoder with observed (fixed) actions
  converter.py       — pddl_plus_parser observations -> planning_structs

The vendored packages keep their upstream top-level names (``planning_structs``,
``constraint_opt``), loaded via the sys.path insertion below so upstream's own
absolute imports work unchanged. Importing this package is therefore the
precondition for any bare ``from planning_structs...`` elsewhere in the project.
"""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = str(Path(__file__).resolve().parent / "vendor")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
