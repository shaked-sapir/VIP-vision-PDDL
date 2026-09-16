"""OffLAM baseline runner (Lamanna, Serafini, Saetti, Gerevini, Traverso, AIJ 2025).

OffLAM learns from partial traces by applying completion rules to a fixpoint
and emits its cautious model. It treats every literal it sees as true, so it
is a competitor only in cells with ``noising_p == 0``: the paper's setting 1,
partial states with fully observed actions. It has no parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from benchmark.algorithm_adapters.lamanna.offlam_glue import learn_offlam
from benchmark.baselines.lamanna_runner_base import LamannaBaselineRunner
from benchmark.baselines.regime import DegradationRegime


class OffLAMRunner(LamannaBaselineRunner):
    """OffLAM over the fold's frozen observations, in a child process."""

    workspace_dirname = "offlam_workspace"

    @property
    def name(self) -> str:
        return "OffLAM"

    @property
    def display_name(self) -> str:
        return "OffLAM (AIJ-25)"

    @property
    def color(self) -> str:
        return "#1f9e89"

    @property
    def paper(self) -> str:
        return "aij25"

    def supports(self, regime: DegradationRegime) -> Tuple[bool, str]:
        if not regime.is_simulated:
            return False, "OffLAM assumes every observed literal is true; image cells flip fluents"
        if not regime.noise_free:
            return False, f"OffLAM assumes every observed literal is true; noising_p={regime.noising_p:g}"
        return True, ""

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
        report = self._base_report(rates, len(trace_paths))
        report["omega_nominal"] = (
            None if self.regime is None or self.regime.masking_p is None
            else 1.0 - self.regime.masking_p
        )
        print(f"  [OffLAM] {len(trace_paths)} traces, timeout {timeout_seconds}s")
        return self._run(
            learn_offlam,
            {"domain_path": str(domain_copy), "trace_paths": [str(p) for p in trace_paths]},
            workspace,
            timeout_seconds,
            report,
        )
