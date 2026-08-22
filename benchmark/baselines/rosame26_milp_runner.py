"""ROSAME-I+MILP (26) baseline runner — the ICAPS-26 network with the solver on.

The sixth and last of the ROSAME arms, and the same code as ``rosame_i_26``
with one collaborator injected. §7 of the plan asks for exactly that: the two
arms differ in whether ``pre_mip_epoch`` lets the solve gate ever open, so
:class:`Rosame26MilpRunner` overrides two hooks and nothing else.

* :meth:`pre_mip_epoch` returns the configured warm-up instead of ``epochs``,
  which is what opens upstream's gate (``network.py:272``, ``:303``).
* :meth:`make_repairer` builds a
  :class:`~src.milp.rosame26_repairer.Rosame26MipRepairer` over
  :mod:`src.milp.encoder`, per §6.1 — not the vendored ``Convertor``, whose
  solvers take ``max_t`` from the first trace of a ragged bundle.

MILP CADENCE (§6.2). ``mip_interval`` is passed through and the per-solve limit
is the **constant** ``mip_time_limit``, as upstream passes it. Our own
``_solve_time_limit`` rationing is deliberately not reused: at this arm's solve
counts its 5 s floor would force ``mip_interval`` wider on every cell, and §1.2's
pre-flight owns the budget instead.

Every solve is recorded in ``milp_rounds`` on the row, so ``mip_gt_dist`` and the
solver's own exit status are visible per round rather than only in aggregate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from benchmark.baselines.rosame26_data import canonical
from benchmark.baselines.rosame26_runner import Rosame26BaselineRunner
from benchmark.baselines.rosame_i_runner import ResizeSpec, _RESIZE_FROM_TABLE

from src.milp.encoding_config import MilpEncodingConfig
from src.milp.rosame26_budget import BudgetMode

#: Upstream's warm-up before the first solve (``train_common.py``).
PRE_MIP_EPOCH: int = 50

#: Traces one solve covers, and the interval between solves (§6.2, upstream).
MIP_TRACES: int = 3
MIP_INTERVAL: int = 1

#: Seconds per solve, passed as the constant upstream passes (§6.2).
MIP_TIME_LIMIT: float = 60.0


class Rosame26MilpRunner(Rosame26BaselineRunner):
    """ROSAME-I (26) with the MILP pseudo-label loop switched on."""

    _base_name: str = "ROSAME-I_MILP_26"

    def __init__(
        self,
        n_seeds: int = 3,
        device: Optional[str] = None,
        base_seed: int = 8800,
        resize: ResizeSpec = _RESIZE_FROM_TABLE,
        epochs: Optional[int] = None,
        batch_size: int = 128,
        budget_mode: str = BudgetMode.PREFLIGHT.value,
        pre_mip_epochs: int = PRE_MIP_EPOCH,
        mip_traces: int = MIP_TRACES,
        mip_time_limit: float = MIP_TIME_LIMIT,
        encoding_config: Optional[MilpEncodingConfig] = None,
    ) -> None:
        """
        Args:
            pre_mip_epochs: Epochs of DL-only warm-up before the first solve.
            mip_traces: Traces per solve; the FIFO selector takes the first this
                many offered in an epoch.
            mip_time_limit: Seconds per solve, passed as a constant.
            encoding_config: The MILP rule set; defaults to upstream's.
        """
        super().__init__(
            n_seeds=n_seeds,
            device=device,
            base_seed=base_seed,
            resize=resize,
            epochs=epochs,
            batch_size=batch_size,
            budget_mode=budget_mode,
        )
        self.pre_mip_epochs = pre_mip_epochs
        self.mip_traces = mip_traces
        self.mip_time_limit = mip_time_limit
        self.encoding_config = encoding_config or MilpEncodingConfig.upstream()

    @property
    def display_name(self) -> str:
        return "ROSAME-I+MILP (26)"

    @property
    def color(self) -> str:
        return "#2f8f6f"

    def parameter_overrides(self) -> Dict:
        """The two solve settings §6.2 pins, passed as constants."""
        return {
            "mip_interval": MIP_INTERVAL,
            "mip_time_limit": self.mip_time_limit,
            "mip_traces": self.mip_traces,
        }

    # ------------------------------------------------------------- the hooks

    def pre_mip_epoch(self, epochs: int) -> int:
        """The warm-up, capped so a short run still solves at least once.

        A budget the pre-flight lowered below the configured warm-up would
        otherwise never open the gate, and the arm would silently be its DL-only
        sibling under a different row name.
        """
        return min(self.pre_mip_epochs, max(0, epochs - 1))

    def make_repairer(self, trainer, fold, bench: str, traces):
        """A :class:`Rosame26MipRepairer` over this fold's grounding."""
        from src.milp.rosame26_repairer import (
            Rosame26MipRepairer,
            reference_action_model,
        )

        return Rosame26MipRepairer(
            domain_model=trainer.domain_model,
            ps_domain=self._ps_domain(bench),
            contexts=self._contexts(fold, traces, bench),
            proposition_index=fold.grounding.proposition_index,
            config=self.encoding_config,
            mip_traces=self.mip_traces,
            # GT-facing, and the diagnostic ONLY: mip_gt_dist is reported, never
            # selected on. Upstream's second model_permutation call, minus the
            # search (plan §0.1a).
            gt_action_model=reference_action_model(bench),
            lengths=fold.batch.lengths,
        )

    # -------------------------------------------------------------- contexts

    def _ps_domain(self, bench: str):
        from src.milp.domain_assets import build_domain

        return build_domain(bench)

    def _contexts(self, fold, traces, bench: str) -> Dict[int, object]:
        """``{batch row: TraceContext}``, keyed as the fold's tensors are.

        Both endpoints come from the GT trajectory, never from the degraded
        states: the same anchors every imaged arm uses (§4.3).
        """
        from src.milp.converter import build_ps_instance
        from src.milp.rosame26_repairer import TraceContext

        ps_domain = self._ps_domain(bench)
        _, partial_domain = self._bench_and_domain(self._domain_path)
        by_name = {trace.problem_name: trace for trace in traces}

        contexts: Dict[int, object] = {}
        for index, name in enumerate(fold.kept):
            trace = by_name[name]
            calls = [
                (call.split()[0], call.split()[1:])
                for call in (canonical(a) for a in trace.action_strings)
            ]
            goal = {
                (fluent.split()[0], tuple(fluent.split()[1:]))
                for fluent in (canonical(p) for p in trace.gt_final_predicates)
            }
            contexts[index] = TraceContext(
                instance=build_ps_instance(ps_domain, partial_domain, trace.problem),
                calls=calls,
                init_fluents=_init_fluents(trace.problem),
                goal_fluents=goal or None,
            )
        return contexts

    # ----------------------------------------------------------------- learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        """As the DL-only arm, with the solve schedule recorded on the row."""
        # `_contexts` needs the reference domain, which only `learn` receives.
        self._domain_path = domain_path
        model, report = super().learn(
            domain_path, prepared_trajectories, work_dir, timeout_seconds
        )
        if report:
            report.setdefault("pre_mip_epochs", self.pre_mip_epochs)
            report.setdefault("mip_traces", self.mip_traces)
            report.setdefault("mip_interval", MIP_INTERVAL)
            report.setdefault("mip_time_limit", self.mip_time_limit)
        return model, report


def _init_fluents(problem) -> set:
    """Positive fluents of a problem's ``:init``, in the underscored dialect."""
    fluents = set()
    for grounded in problem.initial_state_predicates.values():
        for one in grounded:
            if not getattr(one, "is_positive", True):
                continue
            fluents.add(
                (
                    canonical(one.name),
                    tuple(one.object_mapping[key] for key in one.signature.keys()),
                )
            )
    return fluents
