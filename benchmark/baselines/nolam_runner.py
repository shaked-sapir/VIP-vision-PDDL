"""NOLAM baseline runner (Lamanna and Serafini, ICAPS 2024).

NOLAM learns from fully observed, noisy states: per (operator, atom) it counts
the before/after truth values across transitions and takes the MAP hypothesis
under a symmetric flip probability ``e`` that it is *given*. It has no notion
of a hidden atom, so it is a competitor only in cells with ``masking_p == 0``,
and it is handed the flip rate the fold's data actually carries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from pddl_plus_parser.lisp_parsers import DomainParser

from benchmark.algorithm_adapters.lamanna import RealisedRates
from benchmark.algorithm_adapters.lamanna.nolam_glue import learn_nolam
from benchmark.baselines.lamanna_runner_base import LamannaBaselineRunner
from benchmark.baselines.regime import DegradationRegime

ORACLE = "oracle"
NoiseSpec = Union[str, float]

_TRACE_ACTION = re.compile(r"\(:action \(([\w-]+)")
_MODEL_ACTION = re.compile(r"\(:action\s+([\w-]+)")


def observed_operators(trace_paths: List[Path]) -> List[str]:
    """The operator names that occur in the trace files, sorted."""
    names = set()
    for path in trace_paths:
        names.update(_TRACE_ACTION.findall(Path(path).read_text()))
    return sorted(names)


def _parse_noise(value: NoiseSpec) -> NoiseSpec:
    """``"oracle"`` or a flip probability in ``[0, 1)``."""
    if isinstance(value, str):
        if value.strip().lower() == ORACLE:
            return ORACLE
        value = float(value)
    value = float(value)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"nolam_noise must be 'oracle' or a float in [0, 1), got {value}")
    return value


class NOLAMRunner(LamannaBaselineRunner):
    """NOLAM over the fold's frozen observations, in a child process.

    Args:
        nolam_noise: ``"oracle"`` hands NOLAM the fold's realised flip rate
            (the paper's protocol: the noise level is an input); a float pins
            one ``e`` for every cell.
        nolam_allow_neg_precs: ``True`` is the paper's ``MAP`` variant,
            ``False`` (default) its ``MAP_pre+``, which learns no negative
            preconditions.
        nolam_seed: NumPy seed for MAP tie-breaking.
        regime: The cell this runner is bound to; set by :meth:`for_regime`.
    """

    workspace_dirname = "nolam_workspace"

    def __init__(
        self,
        nolam_noise: NoiseSpec = ORACLE,
        nolam_allow_neg_precs: bool = False,
        nolam_seed: int = 0,
        regime: Optional[DegradationRegime] = None,
    ) -> None:
        super().__init__(regime)
        self.nolam_noise = _parse_noise(nolam_noise)
        self.nolam_allow_neg_precs = bool(nolam_allow_neg_precs)
        self.nolam_seed = int(nolam_seed)

    # ------------------------------------------------------------------ #
    # Identity
    # ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return "NOLAM"

    def row_name(self, domain_path: Path) -> str:
        """``NOLAM``, suffixed when a knob makes this a different algorithm."""
        parts = []
        if self.nolam_noise != ORACLE:
            parts.append(f"e={self.nolam_noise:g}")
        if self.nolam_allow_neg_precs:
            parts.append("negprecs")
        return self.name + ("__" + "__".join(parts) if parts else "")

    @property
    def display_name(self) -> str:
        return "NOLAM (ICAPS-24)"

    @property
    def color(self) -> str:
        return "#7b3fa0"

    @property
    def paper(self) -> str:
        return "icaps24"

    def run_params(self) -> Dict[str, object]:
        return {
            "nolam_noise": self.nolam_noise,
            "nolam_allow_neg_precs": self.nolam_allow_neg_precs,
            "nolam_seed": self.nolam_seed,
        }

    # ------------------------------------------------------------------ #
    # Regime
    # ------------------------------------------------------------------ #

    def supports(self, regime: DegradationRegime) -> Tuple[bool, str]:
        if not regime.is_simulated:
            return False, "NOLAM assumes fully observed states; image cells mask fluents"
        if not regime.mask_free:
            return False, f"NOLAM assumes fully observed states; masking_p={regime.masking_p:g}"
        return True, ""

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def _resolve_e(self, rates: Optional[RealisedRates]) -> Tuple[float, str]:
        """``(e, source)``: the flip probability NOLAM is given and where it came from."""
        if self.nolam_noise != ORACLE:
            return float(self.nolam_noise), "pinned"
        if rates is not None:
            return rates.noise_rate, "realised"
        if self.regime is not None and self.regime.noising_p is not None:
            return float(self.regime.noising_p), "nominal"
        raise ValueError("nolam_noise='oracle' needs a bound regime (see for_regime)")

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        workspace, domain_copy, trace_paths, rates = self._prepare(
            domain_path, prepared_trajectories, work_dir
        )
        e, e_source = self._resolve_e(rates)
        domain = DomainParser(Path(domain_path), partial_parsing=True).parse_domain()
        seen = observed_operators(trace_paths)
        unobserved = sorted(set(domain.actions) - set(seen))
        report = self._base_report(rates, len(trace_paths))
        report.update({
            "nolam_e_source": e_source,
            "nolam_e_used": e,
            "unobserved_operators": unobserved,
        })
        print(f"  [NOLAM] e={e:.4f} ({e_source}), {len(trace_paths)} traces, "
              f"allow_neg_precs={self.nolam_allow_neg_precs}"
              + (f", unobserved operators: {', '.join(unobserved)}" if unobserved else ""))
        model_text, report = self._run(
            learn_nolam,
            {
                "domain_path": str(domain_copy),
                "trace_paths": [str(p) for p in trace_paths],
                "e": e,
                "allow_neg_precs": self.nolam_allow_neg_precs,
                "seed": self.nolam_seed,
            },
            workspace,
            timeout_seconds,
            report,
        )
        emitted = _MODEL_ACTION.findall(model_text) if model_text else []
        report["n_operators_emitted"] = len(emitted)
        report["operators_dropped_empty"] = sorted(set(domain.actions) - set(emitted)) if model_text else []
        return model_text, report
