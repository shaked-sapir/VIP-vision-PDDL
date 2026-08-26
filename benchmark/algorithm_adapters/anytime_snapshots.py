"""Periodic model snapshots for anytime curves.

An anytime curve is a claim about what a learner had on the table at each point
in time, so it needs the intermediate models, not just the final one. CDPS gets
this for free — it produces one model per round and the loop already writes them
(``milp_loop_round_models/``). ROSAME does not: it produces exactly one model, at
the end. This module supplies the missing half, so the two can be plotted on the
same axes.

The recorded clock **excludes** the cost of snapshotting itself. That is not
fussiness: on a real blocksworld fold one epoch costs ~6.4 ms while one snapshot
costs ~0.8 ms, and on depot it is 8.9 ms against 1.5 ms — 12% and 17%. Charging
that to the learner would shift the curve right by more than the gaps the plot
exists to show, and it would penalise ROSAME precisely in proportion to how
densely we chose to measure it.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional


@dataclass(frozen=True)
class SnapshotRecord:
    """One captured model and the training progress it corresponds to."""

    index: int
    step: int
    trajectory: int
    epoch: int
    elapsed_seconds: float
    path: str
    loss: Optional[float] = None


class SnapshotWriter:
    """Writes ``snapshot_{i}.pddl`` every ``interval`` steps, plus an index.

    Rendering is passed in as a callable rather than an object, so this stays
    agnostic about what is being snapshotted; the ROSAME adapter hands it
    ``self.rosame_to_pddl``.
    """

    INDEX_NAME = "snapshots.json"

    def __init__(
        self,
        out_dir: Path,
        interval: int = 1,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if interval < 1:
            raise ValueError(f"snapshot interval must be >= 1, got {interval}")
        self.out_dir = Path(out_dir)
        self.interval = interval
        self._clock = clock
        self._records: List[SnapshotRecord] = []
        self._overhead = 0.0
        self._t0: Optional[float] = None

    # ------------------------------------------------------------------ clock

    def start(self) -> None:
        """Mark the start of the timed region. Call once, before training."""
        self._t0 = self._clock()
        self._overhead = 0.0
        self._records = []

    @property
    def overhead_seconds(self) -> float:
        """Wall-clock spent rendering and writing snapshots."""
        return self._overhead

    def elapsed_seconds(self) -> float:
        """Time since :meth:`start`, net of snapshot overhead."""
        if self._t0 is None:
            raise RuntimeError("SnapshotWriter.start() was never called")
        return self._clock() - self._t0 - self._overhead

    # ------------------------------------------------------------- capturing

    def maybe_capture(
        self,
        *,
        step: int,
        trajectory: int,
        epoch: int,
        render: Callable[[], str],
        loss: Optional[float] = None,
    ) -> None:
        """Capture a snapshot if ``step`` falls on the interval.

        Args:
            step: Global step counter, 1-based, monotonic across the whole run.
            trajectory: Index of the trace being trained, or -1 when pooled.
            epoch: Epoch within that trace.
            render: Produces the model text. Called only when capturing.
            loss: Training loss at this step, recorded in the index. ``None``
                leaves the field absent, which is what a caller that does not
                track a loss gets.
        """
        if step % self.interval != 0:
            return
        self.capture(
            step=step, trajectory=trajectory, epoch=epoch, render=render,
            loss=loss,
        )

    def capture(
        self,
        *,
        step: int,
        trajectory: int,
        epoch: int,
        render: Callable[[], str],
        loss: Optional[float] = None,
    ) -> SnapshotRecord:
        """Unconditionally capture a snapshot and return its record."""
        elapsed = self.elapsed_seconds()
        t_before = self._clock()

        index = len(self._records)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"snapshot_{index:04d}.pddl"
        path.write_text(render())

        self._overhead += self._clock() - t_before

        record = SnapshotRecord(
            index=index,
            step=step,
            trajectory=trajectory,
            epoch=epoch,
            elapsed_seconds=elapsed,
            path=path.name,
            loss=loss,
        )
        self._records.append(record)
        return record

    # ---------------------------------------------------------------- output

    def close(self) -> Path:
        """Write the index file and return its path."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.out_dir / self.INDEX_NAME
        payload = {
            "interval": self.interval,
            "num_snapshots": len(self._records),
            "overhead_seconds": self._overhead,
            "timing_note": (
                "elapsed_seconds excludes snapshot rendering and I/O; it is the "
                "time the run would have taken uninstrumented"
            ),
            "snapshots": [asdict(r) for r in self._records],
        }
        index_path.write_text(json.dumps(payload, indent=2))
        return index_path
