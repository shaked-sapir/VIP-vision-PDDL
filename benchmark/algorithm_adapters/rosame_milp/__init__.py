"""ROSAME+MILP adapter — the ICAPS-26 MILP "solution fixer" with observed actions.

Two baselines are built on this package (registered in ``benchmark.baselines``):
  - ``rosame_milp_24``     — the iterative training loop: warmup epochs, then a
                             MILP solve every ``mip_interval`` epochs whose model
                             pseudo-labels supervise further ROSAME training;
                             output = decode of the final MILP solution.
  - ``rosame_milp_24_tag`` — the same loop under the ``tag()`` rule-set.

Layout:
  model_bridge.py    — trained ROSAME <-> ObservationM / labels / PDDL decode
  milp_loop.py       — the V2 training loop (pooled schedule + model-CE rounds)

The encoder, the converter, the encoding rule-sets and the vendored
``planning_structs``/``constraint_opt`` packages live in ``src/milp/`` — they are
shared with the ``pisam_milp_*`` learners, which drive the same encoder under a
different rule-set preset. Only the genuinely ROSAME-specific pieces (anything
that reads the network's ``forward()`` rows or drives torch training) are here.

Importing this package also performs the vendor sys.path insertion, so
``from planning_structs...`` keeps working for our modules and for upstream's
own absolute imports.
"""

from __future__ import annotations

# Side effect: inserts the vendored packages' directory into sys.path.
import src.milp  # noqa: F401
