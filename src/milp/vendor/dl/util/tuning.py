"""Stub replacing upstream's hyperparameter-search harness.

VENDOR MODIFICATION (see ../../UPSTREAM.md). Upstream's ``dl/util/tuning.py`` is
655 lines of grid/genetic search that spawns subprocesses and re-runs whole
experiments; ``train_common.py`` describes it as "if the value is a list, it is
interpreted as a hyperparameter choice... a separate experiment will be run and
recorded". None of that may run inside a fold.

Only one symbol is reachable from the code we do vendor: ``dl/main/normalization``
does ``from ..util.tuning import parameters`` and uses the dict as a cache for
the per-pixel image mean/std. That cache is unkeyed upstream and carries the
image shape, so a single process touching two domains reuses a wrong-shaped
array (plan §4.4); our adapter owns the keyed, on-disk cache and writes through
this dict.
"""

from typing import Any, Dict

parameters: Dict[str, Any] = {}
