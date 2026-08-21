"""ROSAME-I (26) baseline runner — the ICAPS-26 network, imaged, DL-only.

The fourth of the six ROSAME arms and the first that is not the ICAPS-24
network. It composes three pieces that already exist and adds no learning of its
own:

* :func:`~benchmark.baselines.rosame26_data.build_fold_batch` turns the fold into
  the padded tensors, the corpus-wide grounding and the head-column maps;
* :class:`~src.milp.rosame26_training.Rosame26Trainer` runs the loop, with no
  ``mip_repairer`` injected — which is what makes this arm DL-only and
  ``rosame_i_milp_26`` the same code with one;
* :func:`~src.milp.rosame26_emitter.emit_pddl` writes the learned schemas back in
  the reference domain's own argument order, guarded by
  :func:`~src.milp.rosame26_emitter.check_not_degenerate`.

WHAT DIFFERS FROM THE 24 ARM, and is not a matter of implementation, is the
table in ``rosame_i_26_fixing_PROCESS.md`` §4 Phase 4. Eight things move between
the two arms, and a 24-vs-26 delta is not an architecture comparison unless that
table travels with it. The two this file decides are named in
:data:`_EPOCHS` and :data:`_RESIZE`.

Simulation-mode cells have no images; the runner reports the skip and returns
``(None, {})`` so the harness records a null row, exactly as the 24 arm does.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Domain

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.baselines.image_fold_inputs import infer_bench_key, resolve_fold_inputs
from benchmark.baselines.rosame_i_runner import (
    ResizeSpec,
    _RESIZE_FROM_TABLE,
    _resize_tag,
)

#: Epochs per domain. 5000 is the ICAPS-26 code default and is what gate 7
#: measures against outside the cell timeout; the grid runs the calibrated
#: value the §1.2 pre-flight settles on. Written per domain rather than left to
#: one constant so an off-default budget is visible where the experiment is read.
_EPOCH_DEFAULT: int = 750
_EPOCHS: Dict[str, int] = {
    "blocksworld": 750,
    "hanoi": 750,
    "npuzzle": 750,
    "gripper": 750,
    "depot": 750,
}

#: Image resize, as the 24 arm's table. Kept equal to it on purpose: resolution
#: is one of the eight movable factors, and holding it fixed is what keeps the
#: 24-vs-26 delta about the network.
_RESIZE_DEFAULT: ResizeSpec = 64
_RESIZE: Dict[str, ResizeSpec] = {
    "blocksworld": 64,
    "hanoi": 64,
    "npuzzle": 64,
    "gripper": 64,
    "depot": 64,
}


class Rosame26BaselineRunner(BaselineRunner):
    """ROSAME-I (26): the ICAPS-26 network on images, without the MILP."""

    #: Row label before any suffix.
    _base_name: str = "ROSAME-I_26"

    def __init__(
        self,
        device: Optional[str] = None,
        seed: int = 8800,
        resize: ResizeSpec = _RESIZE_FROM_TABLE,
        epochs: Optional[int] = None,
        batch_size: int = 128,
    ) -> None:
        """
        Args:
            device: Torch device; defaults to cuda when available.
            seed: Seeds weight init and the loader's shuffle.
            resize: Explicit override, else the per-domain table.
            epochs: Explicit override, else the per-domain table. Gate 7 runs
                this at 5000 outside the cell timeout.
            batch_size: Capped at the fold size by the trainer.
        """
        self.device = device
        self.seed = seed
        self.resize = resize
        self.epochs = epochs
        self.batch_size = batch_size
        self._bench_cache: Dict[Path, Tuple[str, Domain]] = {}

    # ------------------------------------------------------------- identity

    @property
    def name(self) -> str:
        return self._base_name

    @property
    def display_name(self) -> str:
        return "ROSAME-I (26)"

    @property
    def color(self) -> str:
        return "#7b4fb5"

    def row_name(self, domain_path: Path) -> str:
        """Row label for this domain, carrying an off-default resize or budget."""
        bench, _ = self._bench_and_domain(domain_path)
        return f"{self._base_name}{self._suffix(bench)}"

    def _suffix(self, bench: str) -> str:
        """``__res=`` / ``__ep=`` for whatever this domain runs off-default.

        Keyed on the *effective* value, not on whether an override was passed:
        two budgets averaged under one row name is the failure the 24 arm's
        resize suffix exists to prevent, and an epoch sweep would repeat it.
        """
        parts: List[str] = []
        if _resize_tag(self._resolve_resize(bench)) != _resize_tag(_RESIZE_DEFAULT):
            parts.append(f"__res={_resize_tag(self._resolve_resize(bench))}")
        if self._resolve_epochs(bench) != _EPOCH_DEFAULT:
            parts.append(f"__ep={self._resolve_epochs(bench)}")
        return "".join(parts)

    # ------------------------------------------------------------- settings

    def _bench_and_domain(self, domain_path: Path) -> Tuple[str, Domain]:
        """Parse ``domain_path`` and derive its bench key, memoized per instance."""
        key = Path(domain_path).resolve()
        if key not in self._bench_cache:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            self._bench_cache[key] = (
                infer_bench_key(domain_path, partial_domain),
                partial_domain,
            )
        return self._bench_cache[key]

    def _resolve_resize(self, bench: str) -> ResizeSpec:
        """Explicit override if given, else the per-domain table, else the default."""
        if self.resize is not _RESIZE_FROM_TABLE:
            return self.resize
        return _RESIZE.get(bench, _RESIZE_DEFAULT)

    def _resolve_epochs(self, bench: str) -> int:
        """Explicit override if given, else the per-domain table, else the default."""
        if self.epochs is not None:
            return int(self.epochs)
        return _EPOCHS.get(bench, _EPOCH_DEFAULT)

    # ---------------------------------------------------------------- learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        """Train one fold and return its emitted domain.

        Returns:
            ``(pddl string | None, report)``. ``None`` on a simulation-mode cell
            or a degenerate model, both of which are reported rather than raised
            so one bad cell does not kill a grid.
        """
        bench, partial_domain = self._bench_and_domain(domain_path)
        resolved = resolve_fold_inputs(partial_domain, prepared_trajectories, bench)
        if not resolved:
            print("  [ROSAME-I 26] skipping: no images (simulation-mode cell?)")
            return None, {}

        # Imported lazily so a torch-less environment can still import the registry.
        from benchmark.baselines.rosame26_data import build_fold_batch
        from src.milp.rosame26_emitter import (
            DegenerateModelError,
            check_not_degenerate,
            emit_pddl,
        )
        from src.milp.rosame26_training import Rosame26Trainer, default_parameters

        run_dir = Path(work_dir) / "baseline_models" / self.name
        run_dir.mkdir(parents=True, exist_ok=True)

        start = time.perf_counter()
        fold = build_fold_batch(
            resolved,
            partial_domain,
            bench,
            assets_root=run_dir / "grounding",
            resize=self._resolve_resize(bench),
        )

        epochs = self._resolve_epochs(bench)
        parameters = default_parameters(
            domain=bench,
            domain_assets_root=fold.grounding.assets_root,
            epoch=epochs,
            batch_size=self.batch_size,
            # DL-only: pre_mip_epoch >= epoch is upstream's own way to say so.
            pre_mip_epoch=epochs,
            device=self.device,
            seed=self.seed,
        )
        trainer = Rosame26Trainer(run_dir / "network", parameters, mip_repairer=None)
        history = trainer.train(fold.batch)

        report: Dict = {
            "epochs": epochs,
            "resize": self._resolve_resize(bench),
            "batch_size": self.batch_size,
            "seed": self.seed,
            "device": parameters["device"],
            "n_traces": len(fold.kept),
            "kept_traces": fold.kept,
            "dropped_traces": fold.dropped,
            "prop_dim": len(fold.grounding.proposition_index),
            "action_dim": len(fold.grounding.action_index),
            "final_loss": history[-1].get("total_loss") if history else None,
            "training_seconds": time.perf_counter() - start,
            "timeout_seconds": timeout_seconds,
            "loss_history": history,
        }

        model = emit_pddl(trainer.domain_model, bench)
        try:
            report["effect_counts"] = check_not_degenerate(model)
        except DegenerateModelError as error:
            # Gate 4. Written out anyway, so the collapse can be inspected.
            (run_dir / "degenerate_model.pddl").write_text(model)
            print(f"  [ROSAME-I 26] gate 4: {error}")
            report["degenerate"] = str(error)
            return None, report

        (run_dir / "model.pddl").write_text(model)
        return model, report
