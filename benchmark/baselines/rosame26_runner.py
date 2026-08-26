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
from src.milp.rosame26_budget import PER_EPOCH_DL_SECONDS, BudgetMode

#: Epochs per domain, before the §1.2 pre-flight is allowed to lower it. 5000 is
#: the ICAPS-26 code default and is what gate 7 measures against *outside* the
#: cell timeout. 600 is the plan's conservative round number: it is not derived,
#: it is a ceiling the pre-flight lowers against the cell's actual budget, and
#: each row records the count it actually ran. Written per domain rather than
#: left to one constant so an off-default budget is visible where the experiment
#: is read.
_EPOCH_DEFAULT: int = 600
_EPOCHS: Dict[str, int] = {
    "blocksworld": 600,
    "hanoi": 600,
    "npuzzle": 600,
    "gripper": 600,
    "depot": 600,
}

#: Epochs the timing probe runs before the budget is re-projected (§1.2). Small
#: enough to be a rounding error against the budget, large enough that build and
#: first-batch overhead do not dominate the per-epoch figure it yields.
_PROBE_EPOCHS: int = 20

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


class _BestModelTracker:
    """Keeps the lowest-training-loss snapshot of the schema heads.

    The emitted PDDL is a function of ``domain_model.action_schemas`` alone, so
    their ``state_dict`` is the whole checkpoint. Selection reads training loss
    only and never touches test data.

    Inactive outside ``converge`` mode: :meth:`observe` and :meth:`restore`
    become no-ops, so the other two modes keep emitting the final epoch.
    """

    def __init__(self, active: bool) -> None:
        self.active = active
        self.best_loss: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._state = None
        self._model = None

    def bind(self, domain_model) -> None:
        """Attach the model whose schema heads are snapshotted."""
        self._model = domain_model

    def observe(self, loss: float, epoch: int) -> None:
        """Snapshot if ``loss`` is the best seen so far."""
        if not self.active or self._model is None:
            return
        if self.best_loss is None or loss < self.best_loss:
            import copy

            self.best_loss = loss
            self.best_epoch = epoch
            self._state = copy.deepcopy(
                [schema.state_dict() for schema in self._model.action_schemas]
            )

    def restore(self, domain_model) -> None:
        """Load the best snapshot back, if one was taken."""
        if not self.active or self._state is None:
            return
        for schema, state in zip(domain_model.action_schemas, self._state):
            schema.load_state_dict(state)


class Rosame26BaselineRunner(BaselineRunner):
    """ROSAME-I (26): the ICAPS-26 network on images, without the MILP."""

    #: Row label before any suffix.
    _base_name: str = "ROSAME-I_26"

    def __init__(
        self,
        n_seeds: int = 3,
        device: Optional[str] = None,
        base_seed: int = 8800,
        resize: ResizeSpec = _RESIZE_FROM_TABLE,
        epochs: Optional[int] = None,
        batch_size: int = 128,
        budget_mode: str = BudgetMode.PREFLIGHT.value,
    ) -> None:
        """
        Args:
            n_seeds: Independent models trained; the one with the lowest final
                training loss is kept, as the 24 arm does. It multiplies the
                §1.2 projection, so the pre-flight sees it.
            device: Torch device; defaults to cuda when available.
            base_seed: Seeds weight init and the loader's shuffle; seed ``i`` is
                ``base_seed + i``.
            resize: Explicit override, else the per-domain table.
            epochs: Explicit override, else the per-domain table. Gate 7 runs
                this at 5000, with ``respect_budget=False``.
            batch_size: Capped at the fold size by the trainer.
            budget_mode: One of :class:`~src.milp.rosame26_budget.BudgetMode`.
                ``preflight`` lets the §1.2 check lower ``epochs`` to fit the
                cell timeout (the grid's mode); ``fixed`` runs ``epochs``
                whatever the projection says (gate 7's control, outside the
                timeout); ``converge`` early-stops on a training-loss plateau
                with ``epochs`` as a ceiling, and is the only mode that emits
                the best-loss model rather than the final epoch's.

        Raises:
            ValueError: on an unknown ``budget_mode``.
        """
        try:
            self.budget_mode = BudgetMode(budget_mode)
        except ValueError:
            raise ValueError(
                f"unknown budget_mode {budget_mode!r}; expected one of "
                f"{[m.value for m in BudgetMode]}"
            ) from None
        self.n_seeds = n_seeds
        self.device = device
        self.base_seed = base_seed
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
    def input_kind(self) -> str:
        return "imaged"

    @property
    def paper(self) -> str:
        return "26"

    @property
    def uses_milp(self) -> bool:
        return False

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

        Note what is *not* suffixed: an epoch count the §1.2 pre-flight lowered.
        That is the configured budget meeting the cell's own timeout, the same
        for every arm, and it is recorded per row rather than in the label.
        Gate 7's control cell is a different arm — it opts out of the budget —
        and it is the configured count that labels it.
        """
        parts: List[str] = []
        if _resize_tag(self._resolve_resize(bench)) != _resize_tag(_RESIZE_DEFAULT):
            parts.append(f"__res={_resize_tag(self._resolve_resize(bench))}")
        if self._resolve_epochs(bench) != _EPOCH_DEFAULT:
            parts.append(f"__ep={self._resolve_epochs(bench)}")
        if self.budget_mode is not BudgetMode.PREFLIGHT:
            parts.append(f"__mode={self.budget_mode.value}")
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

    def _budgeted_epochs(
        self, bench: str, timeout_seconds: float
    ) -> Tuple[int, Dict]:
        """The epoch count to run, and what the §1.2 pre-flight said about it.

        The configured count is a ceiling, not a derived constant: the pre-flight
        lowers it to whatever fits the cell, and the row records what actually
        ran. With ``respect_budget=False`` the count stands whatever the
        projection says — the control-cell setting.

        Returns:
            ``(epochs, projection report)``.
        """
        from src.milp.rosame26_budget import project

        requested = self._resolve_epochs(bench)
        projection = project(
            epochs=requested,
            # DL-only, so nothing solves and the DL term is the whole cost.
            pre_mip_epoch=requested,
            mip_interval=1,
            timeout_seconds=timeout_seconds,
            n_seeds=self.n_seeds,
        )
        report = {
            "mode": self.budget_mode.value,
            "requested_epochs": requested,
            "projected_seconds": projection.seconds,
            "budget_seconds": projection.budget,
            "fits": projection.fits,
        }
        if projection.fits or self.budget_mode is not BudgetMode.PREFLIGHT:
            return requested, report

        print(
            f"  [ROSAME-I 26] budget: {requested} epochs x {self.n_seeds} seed(s) "
            f"project to {projection.seconds:.0f} s, past the "
            f"{projection.budget:.0f} s of this cell's {timeout_seconds:.0f} s; "
            f"running {projection.max_epochs}"
        )
        return projection.max_epochs, report

    def _reprojected_epochs(
        self,
        bench: str,
        fold: object,
        epochs: int,
        timeout_seconds: float,
        run_dir: Path,
    ) -> Tuple[int, Dict]:
        """Time a short probe and re-project the budget against it (§1.2).

        The seeded per-epoch constant is one domain's, and the domains differ by
        roughly 2x, so it is slack by up to ~2.6x — the right direction for a
        refuse-to-start guard, the wrong one for an epoch budget. Timing
        :data:`_PROBE_EPOCHS` on this fold recovers the difference without a
        per-domain table.

        The probe is thrown away rather than continued: the trainer builds its
        own network per run and continuing one would mean a partially-trained
        seed 0, which the lowest-final-loss selection would then compare against
        fully-trained peers.

        Returns:
            ``(epochs, probe report)``. The count is only ever *raised* — a
            probe that finds the run more expensive than projected leaves the
            already-conservative count alone rather than shrinking a budget the
            pre-flight has agreed to.
        """
        from src.milp.rosame26_budget import reproject
        from src.milp.rosame26_training import Rosame26Trainer, default_parameters

        if (
            self.budget_mode is not BudgetMode.PREFLIGHT
            or epochs >= self._resolve_epochs(bench)
        ):
            return epochs, {"ran": False, "reason": "budget not binding"}

        start = time.perf_counter()
        try:
            parameters = default_parameters(
                domain=bench,
                domain_assets_root=fold.grounding.assets_root,
                epoch=_PROBE_EPOCHS,
                batch_size=self.batch_size,
                pre_mip_epoch=_PROBE_EPOCHS,
                device=self.device,
                seed=self.base_seed,
            )
            Rosame26Trainer(
                run_dir / "probe", parameters, mip_repairer=None
            ).train(fold.batch)
        except Exception as error:  # a probe failure is not a cell failure
            print(f"  [ROSAME-I 26] probe failed, keeping {epochs} epochs: {error}")
            return epochs, {"ran": False, "reason": str(error)}
        measured = time.perf_counter() - start

        projection = reproject(
            epochs=self._resolve_epochs(bench),
            measured_seconds=measured,
            measured_epochs=_PROBE_EPOCHS,
            pre_mip_epoch=self._resolve_epochs(bench),
            mip_interval=1,
            timeout_seconds=timeout_seconds,
            elapsed_seconds=measured * self.n_seeds,
            n_seeds=self.n_seeds,
        )
        probe = {
            "ran": True,
            "probe_epochs": _PROBE_EPOCHS,
            "probe_seconds": measured,
            "measured_per_epoch": measured / _PROBE_EPOCHS,
            "seeded_per_epoch": PER_EPOCH_DL_SECONDS,
            "reprojected_epochs": projection.max_epochs,
        }
        if projection.max_epochs <= epochs:
            return epochs, probe

        print(
            f"  [ROSAME-I 26] probe: {measured / _PROBE_EPOCHS:.3f} s/epoch "
            f"against the seeded {PER_EPOCH_DL_SECONDS:.3f}; raising "
            f"{epochs} epochs to {projection.max_epochs}"
        )
        return projection.max_epochs, probe

    def pre_mip_epoch(self, epochs: int) -> int:
        """The first epoch at which the MILP may run.

        ``>= epochs`` is upstream's own way of asking for a DL-only run
        (``network.py``'s gate never opens), which is what makes this arm and
        ``rosame_i_milp_26`` the same code.
        """
        return epochs

    def parameter_overrides(self) -> Dict:
        """Extra ``default_parameters`` entries a subclass needs. Empty here."""
        return {}

    def make_repairer(self, trainer, fold, bench: str, traces):
        """The solve collaborator, or ``None`` for a DL-only run.

        Called once per seed, after the trainer exists and before it trains, so
        an implementation may close over ``trainer.domain_model``.
        """
        return None

    def _stop_check(self, tracker: "_BestModelTracker", trainer_ref: Dict):
        """The per-epoch hook the trainer calls: track the best, then decide.

        Returns ``None`` outside ``converge`` mode, so ``preflight`` and
        ``fixed`` run their full epoch count and keep emitting the final model —
        which is what their already-recorded numbers mean.
        """
        if self.budget_mode is not BudgetMode.CONVERGE:
            return None

        from src.milp.rosame26_budget import has_converged

        def check(history: List[Dict[str, float]]) -> bool:
            # Bound here rather than before `train`: `domain_model` is created by
            # `build()` inside it, and this hook first fires after epoch 0.
            tracker.bind(trainer_ref["trainer"].domain_model)
            losses = [float(record["total_loss"]) for record in history]
            tracker.observe(losses[-1], len(losses) - 1)
            return has_converged(losses)

        return check

    # ---------------------------------------------------------------- learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        """Train one fold over ``n_seeds`` models and return the best one's domain.

        The kept seed is the one with the lowest final *training* loss, which
        never touches test data — the 24 arm's selection rule.

        Returns:
            ``(pddl string | None, report)``. ``None`` on a simulation-mode cell,
            a budget that fits no epoch at all, or a degenerate model. All three
            are reported rather than raised, so one bad cell does not kill a grid.
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

        epochs, budget = self._budgeted_epochs(bench, timeout_seconds)
        if epochs < 1:
            print(f"  [ROSAME-I 26] skipping: no epoch fits {timeout_seconds:.0f} s")
            return None, {"budget": budget}

        start = time.perf_counter()
        fold = build_fold_batch(
            resolved,
            partial_domain,
            bench,
            assets_root=run_dir / "grounding",
            resize=self._resolve_resize(bench),
        )

        report: Dict = {
            "epochs": epochs,
            "budget": budget,
            "resize": self._resolve_resize(bench),
            "batch_size": self.batch_size,
            "n_seeds": self.n_seeds,
            "base_seed": self.base_seed,
            "n_traces": len(fold.kept),
            "kept_traces": fold.kept,
            "dropped_traces": fold.dropped,
            "prop_dim": len(fold.grounding.proposition_index),
            "action_dim": len(fold.grounding.action_index),
            "timeout_seconds": timeout_seconds,
        }

        epochs, probe = self._reprojected_epochs(
            bench, fold, epochs, timeout_seconds, run_dir
        )
        report["epochs"] = epochs
        report["budget"] = dict(budget, probe=probe)

        seed_losses: Dict[int, float] = {}
        seed_models: Dict[int, str] = {}
        seed_counts: Dict[int, Dict[str, int]] = {}
        seed_histories: Dict[int, List[Dict[str, float]]] = {}
        seed_stops: Dict[int, Dict] = {}
        seed_rounds: Dict[int, List[Dict]] = {}
        degenerate: Dict[int, str] = {}

        for index in range(self.n_seeds):
            seed = self.base_seed + index
            parameters = default_parameters(
                domain=bench,
                domain_assets_root=fold.grounding.assets_root,
                epoch=epochs,
                batch_size=self.batch_size,
                pre_mip_epoch=self.pre_mip_epoch(epochs),
                device=self.device,
                seed=seed,
            )
            parameters.update(self.parameter_overrides())
            tracker = _BestModelTracker(self.budget_mode is BudgetMode.CONVERGE)
            trainer_ref: Dict = {}
            trainer = Rosame26Trainer(
                run_dir / f"seed_{seed}",
                parameters,
                mip_repairer=None,
                stop_check=self._stop_check(tracker, trainer_ref),
            )
            trainer_ref["trainer"] = trainer
            trainer.on_built = lambda built: setattr(
                built, "mip_repairer", self.make_repairer(built, fold, bench, resolved)
            )
            try:
                history = trainer.train(fold.batch)
            except Exception as error:  # keep one bad seed from killing the cell
                print(f"  [ROSAME-I 26] seed {seed} failed: {error}")
                continue

            # Only "converge" restores a checkpoint; the other two modes emit the
            # final epoch, which is what their recorded numbers already mean.
            tracker.restore(trainer.domain_model)
            model = emit_pddl(trainer.domain_model, bench)
            (run_dir / f"seed_{seed}" / "model.pddl").write_text(model)
            try:
                seed_counts[seed] = check_not_degenerate(model)
            except DegenerateModelError as error:
                # Gate 4, per seed. The collapsed model stays on disk above.
                print(f"  [ROSAME-I 26] seed {seed} gate 4: {error}")
                degenerate[seed] = str(error)
                continue

            seed_losses[seed] = (
                tracker.best_loss
                if tracker.best_loss is not None
                else float(history[-1]["total_loss"])
            )
            seed_models[seed] = model
            seed_histories[seed] = history
            rounds = getattr(trainer.mip_repairer, "rounds", None)
            if rounds:
                seed_rounds[seed] = rounds
            if trainer.stopped_early or tracker.best_epoch is not None:
                seed_stops[seed] = {
                    "stopped_early": trainer.stopped_early,
                    "ran_epochs": len(history),
                    "emitted_epoch": tracker.best_epoch
                    if tracker.best_epoch is not None
                    else len(history) - 1,
                }

        report["training_seconds"] = time.perf_counter() - start
        report["device"] = default_parameters(
            domain=bench,
            domain_assets_root=fold.grounding.assets_root,
            epoch=epochs,
            device=self.device,
        )["device"]
        if degenerate:
            report["degenerate_seeds"] = degenerate

        if not seed_models:
            print("  [ROSAME-I 26] no seed produced a usable model")
            return None, report

        chosen = min(seed_losses, key=seed_losses.get)
        history = seed_histories[chosen]
        losses = [float(record["total_loss"]) for record in history]
        best = min(losses)
        report.update(
            seeds=seed_losses,
            chosen_seed=chosen,
            chosen_final_loss=seed_losses[chosen],
            effect_counts=seed_counts[chosen],
            loss_history=history,
            best_loss=best,
            best_loss_epoch=losses.index(best),
        )
        if seed_stops:
            report["early_stopping"] = seed_stops
        if seed_rounds:
            report["milp_rounds"] = seed_rounds
        (run_dir / "model.pddl").write_text(seed_models[chosen])
        return seed_models[chosen], report
