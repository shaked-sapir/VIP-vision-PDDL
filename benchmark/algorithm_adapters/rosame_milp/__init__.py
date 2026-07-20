"""ROSAME+MILP adapter — the ICAPS-26 MILP "solution fixer" with observed actions.

Two baselines are built on this package (registered in ``benchmark.baselines``):
  - ``rosame_milp``      — the iterative training loop (paper-faithful V2):
                           warmup epochs, then a MILP solve every
                           ``mip_interval`` epochs whose model pseudo-labels
                           supervise further ROSAME training; output = decode
                           of the final MILP solution.
  - ``rosame_milp_base`` — the one-shot variant (V1): train ROSAME fully,
                           solve one MILP, decode.

Layout:
  vendor/            — code vendored from the ROSAME repo (see vendor/UPSTREAM.md)
  encoder.py         — our CP-SAT encoder variant with observed (fixed) actions
  converter.py       — our observations/problems -> planning_structs inputs
  model_bridge.py    — trained ROSAME <-> ObservationM / labels / PDDL decode
  milp_loop.py       — the V2 training loop (pooled schedule + model-CE rounds)

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
