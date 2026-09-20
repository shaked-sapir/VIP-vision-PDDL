"""The lowest-training-loss checkpoint of a ROSAME network's schema heads.

Both ROSAME networks (ICAPS-24 and ICAPS-26) expose their learnable model as
``action_schemas``, a list of modules, and the emitted PDDL is a function of
those alone, so their ``state_dict``s are the whole checkpoint. Selection reads
training loss only and never touches test data.
"""

from __future__ import annotations

import copy
from typing import List, Optional


class BestModelTracker:
    """Snapshots ``model.action_schemas`` at the best loss seen, and restores it.

    Inactive, :meth:`observe` and :meth:`restore` are no-ops, so a caller can
    keep one code path whether or not it emits the best-loss model.
    """

    def __init__(self, active: bool = True) -> None:
        self.active = active
        self.best_loss: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._state: Optional[List[dict]] = None
        self._model = None

    def bind(self, model) -> None:
        """Attach the model whose schema heads are snapshotted."""
        self._model = model

    def reset(self) -> None:
        """Forget the best seen so far; the bound model stays bound."""
        self.best_loss = None
        self.best_epoch = None
        self._state = None

    def observe(self, loss: float, epoch: int) -> None:
        """Snapshot if ``loss`` is the best seen since the last reset."""
        if not self.active or self._model is None:
            return
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self._state = copy.deepcopy(
                [schema.state_dict() for schema in self._model.action_schemas]
            )

    def restore(self, model) -> None:
        """Load the best snapshot back into ``model``, if one was taken."""
        if not self.active or self._state is None:
            return
        for schema, state in zip(model.action_schemas, self._state):
            schema.load_state_dict(state)
