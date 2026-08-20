"""ROSAME-I baseline runner (imaged mode only).

Trains the joint CV + ROSAME model (:class:`RosameI_Runner`) directly from each
trajectory's *state images*, using the observed action sequence and the GT final
state as supervision — never the (VLM-inferred / degraded) trajectory states.
This is the ICAPS-24 ROSAME-I baseline; see
``docs/rosame-i-implementation-plan.md`` for the full design.

Traces are trained pooled, one shuffled pass over all of them per epoch, as in
ICAPS-24 ``train.py``. To mitigate small-data variance, ``n_seeds`` independent
models are trained and the one with the lowest final training loss is kept (a
selection rule that never touches test data).

On simulation-mode cells (no images on disk) the runner emits a clear skip
message and returns ``(None, {})`` so the harness records a null row.
"""

from __future__ import annotations

import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Domain, Problem, State

from benchmark.baselines.base_runner import BaselineRunner
from benchmark.experiment_running_helpers.normalize import _normalize_hyphens

# An image resize setting: shorter-edge int, explicit (h, w), or None for native.
ResizeSpec = Union[int, Sequence[int], None]

# Paper per-domain defaults (epochs, lambda_, gamma), keyed by our bench names.
_HYPERPARAMS: Dict[str, Dict[str, float]] = {
    "blocksworld": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
    "hanoi": {"epochs": 70, "lambda_": 0.2, "gamma": 10},
    "npuzzle": {"epochs": 300, "lambda_": 0.4, "gamma": 10},
    "gripper": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
    "depot": {"epochs": 100, "lambda_": 0.2, "gamma": 10},
}
_DEFAULT_HYPERPARAMS = {"epochs": 100, "lambda_": 0.2, "gamma": 10}

# PDDL domain-name / path-part → bench key (domain names differ from bench keys).
_DOMAIN_ALIASES: Dict[str, str] = {
    "blocks": "blocksworld",
    "blocksworld": "blocksworld",
    "hanoi": "hanoi",
    "n_puzzle": "npuzzle",
    "n_puzzle_typed": "npuzzle",
    "npuzzle": "npuzzle",
    "gripper": "gripper",
    "depot": "depot",
}

# Domains whose renderings are horizontal-flip invariant (paper augments these).
_AUGMENT_DOMAINS = {"blocksworld"}

# Per-domain image resize, applied by every pixel arm (ROSAME-I, ROSAME-I+MILP).
# Written out per domain rather than left to a code default: resolution is an
# experimental variable here, so it belongs where the experiment is read.
#   int          -> Resize(n), shorter edge, aspect PRESERVED (ICAPS-24 form)
#   (h, w)       -> Resize((h, w)), forced, aspect distorted -- torchvision order
#                   is (height, width), not (width, height)
#   None         -> no resize, native size (ICAPS-26 form)
# An entry differing from _RESIZE_DEFAULT suffixes that domain's row name; a new
# suffix needs a matching key in benchmark/evaluation/cfm/dashboard_config.yaml
# or the series will not render.
# See docs/rosame-i-milp-26-implementation-plan.md §4.6 / §4.6.1.
_RESIZE_DEFAULT: ResizeSpec = 64
_RESIZE: Dict[str, ResizeSpec] = {
    "blocksworld": 64,
    "hanoi": 64,
    "npuzzle": 64,
    "gripper": 64,
    "depot": 64,  # candidate for 224 -- see docs/algorithm_comparison_analysis.md §5.1
}

class _ResizeFromTable:
    """Sentinel: "no explicit override, use the per-domain table".

    A bare ``object()`` is **not** usable here: ``backfill_baseline`` forwards
    this value to a ``ProcessPoolExecutor`` worker, and pickling a plain
    ``object()`` yields a *different* instance on the far side, so the ``is``
    check silently fails and the sentinel itself is passed on as a resize value.
    This singleton round-trips through pickle to the same instance.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __reduce__(self):
        return (_ResizeFromTable, ())

    def __repr__(self) -> str:
        return "RESIZE_FROM_TABLE"


_RESIZE_FROM_TABLE = _ResizeFromTable()


def _resize_tag(resize: ResizeSpec) -> str:
    """Canonical form of a resize setting (``64``, ``64x96``, ``native``).

    Doubles as the equality test between two settings: tags compare equal iff
    the settings name the same torchvision operation, so ``(64, 64)`` and
    ``[64, 64]`` agree while ``64`` and ``[64, 64]`` — a preserved aspect ratio
    versus a forced square — do not.
    """
    if resize is None:
        return "native"
    if isinstance(resize, int):
        return str(resize)
    return "x".join(str(int(v)) for v in resize)


# Staging directory names that sit one level above a problem's trajectory dir.
_TRAJECTORY_DIR_NAMES = {"trajectories", "trajectories_normalized"}


class RosameIBaselineRunner(BaselineRunner):
    """ROSAME-I (imaged-mode CV + ROSAME) baseline."""

    #: Row label before the resize suffix is appended.
    _base_name: str = "ROSAME-I_24"

    def __init__(
        self,
        n_seeds: int = 3,
        device: Optional[str] = None,
        base_seed: int = 8800,
        resize: ResizeSpec = _RESIZE_FROM_TABLE,
    ) -> None:
        self.n_seeds = n_seeds
        self.device = device
        self.base_seed = base_seed
        self.resize = resize
        self._bench_cache: Dict[Path, Tuple[str, Domain]] = {}

    def _bench_and_domain(self, domain_path: Path) -> Tuple[str, Domain]:
        """Parse ``domain_path`` and derive its bench key, memoized per instance.

        The single derivation shared by :meth:`row_name` and :meth:`learn`, so a
        row can never be labelled for one domain's resolution and trained at
        another's.
        """
        key = Path(domain_path).resolve()
        if key not in self._bench_cache:
            partial_domain = DomainParser(domain_path, partial_parsing=True).parse_domain()
            self._bench_cache[key] = (
                self._infer_domain_name(domain_path, partial_domain), partial_domain,
            )
        return self._bench_cache[key]

    def _resolve_resize(self, bench: str) -> ResizeSpec:
        """Explicit override if given, else the per-domain table, else the default."""
        if self.resize is not _RESIZE_FROM_TABLE:
            return self.resize
        return _RESIZE.get(bench, _RESIZE_DEFAULT)

    def _resize_suffix(self, bench: str) -> str:
        """``__res=<tag>`` when this domain's effective resize is off-default.

        Keyed on the effective value rather than on whether an override was
        passed: a per-domain table entry that diverges from ``_RESIZE_DEFAULT``
        must be labelled just as an override is, or two resolutions end up
        averaged under one row name -- the failure the ``__gt=none`` suffix rule
        exists to prevent.
        """
        tag = _resize_tag(self._resolve_resize(bench))
        if tag == _resize_tag(_RESIZE_DEFAULT):
            return ""
        return f"__res={tag}"

    def row_name(self, domain_path: Path) -> str:
        """Row label for this domain, carrying an off-default resize."""
        bench, _ = self._bench_and_domain(domain_path)
        return f"{self._base_name}{self._resize_suffix(bench)}"

    @property
    def name(self) -> str:
        return self._base_name

    @property
    def display_name(self) -> str:
        return "ROSAME-I (24)"

    @property
    def color(self) -> str:
        return "#d55181"

    # ------------------------------------------------------------------ learn

    def learn(
        self,
        domain_path: Path,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        work_dir: Path,
        timeout_seconds: int = 60,
    ) -> Tuple[Optional[str], Dict]:
        bench, partial_domain = self._bench_and_domain(domain_path)
        hp = _HYPERPARAMS.get(bench, _DEFAULT_HYPERPARAMS)
        augment = bench in _AUGMENT_DOMAINS
        resize = self._resolve_resize(bench)

        prepared_problems, _gt_paths = self._resolve_inputs(
            partial_domain, prepared_trajectories, bench
        )
        if not prepared_problems:
            print("  [ROSAME-I] skipping: no images (simulation-mode cell?)")
            return None, {}

        # Imported lazily so a torch-less environment can still import the registry.
        from benchmark.algorithm_adapters.rosame_i_runner import RosameI_Runner

        start = time.perf_counter()

        def timeout_check() -> bool:
            return (time.perf_counter() - start) > timeout_seconds

        seed_losses: Dict[int, float] = {}
        seed_models: Dict[int, str] = {}
        skipped_seeds: List[int] = []

        for i in range(self.n_seeds):
            seed = self.base_seed + i
            if seed_models and timeout_check():
                skipped_seeds.append(seed)
                continue
            try:
                runner = RosameI_Runner(
                    str(domain_path), device=self.device, seed=seed, resize=resize
                )
                final_loss = runner.learn_full(
                    prepared_problems,
                    epochs=int(hp["epochs"]),
                    gamma=float(hp["gamma"]),
                    lambda_=float(hp["lambda_"]),
                    augment=augment,
                    timeout_check=timeout_check,
                )
            except Exception as e:  # keep one bad seed from killing the cell
                print(f"  [ROSAME-I] seed {seed} failed: {e}")
                continue

            if final_loss is None:
                continue
            model = runner.to_pddl()
            if not model or ":action" not in model:
                print(f"  [ROSAME-I] seed {seed}: invalid model, skipping")
                continue

            seed_losses[seed] = final_loss
            seed_models[seed] = model
            seed_dir = Path(work_dir) / "baseline_models" / self.name / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            (seed_dir / "model.pddl").write_text(model)

        if not seed_models:
            print("  [ROSAME-I] no seed produced a valid model")
            return None, {}

        chosen = min(seed_losses, key=seed_losses.get)
        extra_info: Dict = {
            "seeds": seed_losses,
            "chosen_seed": chosen,
            "chosen_final_loss": seed_losses[chosen],
            # Distinguishes these rows from ones written before the continual
            # schedule was removed, which carry "train_per_trajectory": true.
            "schedule": "pooled",
            "resize": resize,
        }
        if skipped_seeds:
            extra_info["skipped_seeds_timeout"] = skipped_seeds
        return seed_models[chosen], extra_info

    # ------------------------------------------------------------------ inputs

    def _resolve_inputs(
        self,
        partial_domain: Domain,
        prepared_trajectories: List[Tuple[Path, Path, Path]],
        bench: str,
    ) -> Tuple[List[Tuple[object, List[Path], List[str], List[str]]], List[Path]]:
        """Per trajectory, resolve (problem, image paths, action strings, GT final).

        Trajectories with no images on disk are dropped; an empty result means a
        simulation-mode cell (handled by the caller).

        Returns:
            ``(prepared_problems, gt_trajectory_paths)``, the latter positionally
            aligned with the former and holding the GT trajectory each final-state
            anchor was read from.

        Raises:
            FileNotFoundError: if a kept trajectory has no GT counterpart.
        """
        prepared_problems: List[Tuple[object, List[Path], List[str], List[str]]] = []
        gt_trajectory_paths: List[Path] = []

        for traj_path, _masking_path, problem_pddl_path, *_ in prepared_trajectories:
            problem_dir = problem_pddl_path.parent
            problem_name = problem_pddl_path.stem

            image_paths = self._resolve_images(problem_dir, bench, problem_name)
            if not image_paths:
                continue

            problem = self._parse_problem_normalized(partial_domain, problem_pddl_path)

            # Actions only (states here are degraded and must NOT be used).
            observation = self._parse_trajectory_normalized(partial_domain, problem, traj_path)
            action_strings = [
                str(component.grounded_action_call)[1:-1]
                for component in observation.components
            ]

            gt_path = self.resolve_final_state_path(problem_dir, problem_name)
            final_preds = self._resolve_final_state(
                partial_domain, problem, gt_path, problem_name
            )
            prepared_problems.append((problem, image_paths, action_strings, final_preds))
            gt_trajectory_paths.append(gt_path)

        return prepared_problems, gt_trajectory_paths

    @staticmethod
    def _resolve_images(problem_dir: Path, bench: str, problem_name: str) -> List[Path]:
        """Numerically-sorted ``state_*.png`` frames for one trajectory.

        Looks next to the problem PDDL first (PDDLGym-generated domains keep
        frames in the data dir), then falls back to the external-image domain
        layout ``src/domains/<bench>/problems/<problem>/`` (e.g. depot, gripper),
        whose frames never get copied into ``benchmark/data``.
        """
        repo_root = Path(__file__).resolve().parents[2]
        candidates = [
            problem_dir,
            repo_root / "src" / "domains" / bench / "problems" / problem_name,
        ]
        for cand in candidates:
            images = list(cand.glob("state_*.png"))
            if images:
                return sorted(images, key=lambda p: int(re.search(r"\d+", p.stem).group()))
        return []

    @staticmethod
    def resolve_final_state_path(problem_dir: Path, problem_name: str) -> Path:
        """Path of the GT trajectory the final-state anchor is read from.

        The anchor must come from
        ``<data_dir>/gt_trajectories/<problem>/<problem>.trajectory``; the
        degraded trajectory next to the problem PDDL is never substituted.

        Raises:
            FileNotFoundError: if ``problem_dir`` is not in the
                ``<data_dir>/training/<staging_dir>/<problem>/`` layout, or the
                GT trajectory is absent.
        """
        if problem_dir.parent.name not in _TRAJECTORY_DIR_NAMES:
            raise FileNotFoundError(
                f"cannot locate gt_trajectories/ for '{problem_name}': expected "
                f"'{problem_dir}' to sit under one of {sorted(_TRAJECTORY_DIR_NAMES)}"
            )
        data_dir = problem_dir.parents[2]
        gt_path = data_dir / "gt_trajectories" / problem_name / f"{problem_name}.trajectory"
        if not gt_path.exists():
            raise FileNotFoundError(
                f"no GT trajectory for '{problem_name}': expected '{gt_path}'. "
                f"ROSAME-I anchors its last predicted state on the GT final state, "
                f"so run benchmark/generate_gt_trajectories.py for '{data_dir}' "
                f"before using an imaged ROSAME-I arm on it."
            )
        return gt_path

    def _resolve_final_state(
        self,
        partial_domain: Domain,
        problem,
        final_state_path: Path,
        problem_name: str,
    ) -> List[str]:
        """GT final-state positive predicate strings from an already-resolved path.

        Raises:
            ValueError: if the GT trajectory holds no transition, so there is no
                final state to anchor on.
        """
        observation = self._parse_trajectory_normalized(
            partial_domain, problem, final_state_path
        )
        if not observation.components:
            raise ValueError(
                f"GT trajectory '{final_state_path}' for '{problem_name}' parsed to "
                f"zero transitions, so it carries no final state to anchor on"
            )
        return self._state_positive_predicates(observation.components[-1].next_state)

    # Problem PDDLs and gt_trajectories on disk keep whichever dialect they were
    # written in, which need not match the reference domain: an image experiment
    # that ran normalize_experiment_data has an underscored domain, one that
    # skipped it keeps the raw domain. Parsing an input against a domain in the
    # other dialect raises ``illegal state component``, so inputs are conformed to
    # the dialect the domain itself declares.

    @staticmethod
    def _domain_uses_hyphens(partial_domain: Domain) -> bool:
        """True if any identifier the domain declares contains a hyphen."""
        declared = (
            list(partial_domain.predicates)
            + list(partial_domain.actions)
            + list(getattr(partial_domain, "types", None) or {})
            + list(getattr(partial_domain, "constants", None) or {})
        )
        return any("-" in name for name in declared)

    @classmethod
    def _conformed_tempfile(
        cls, source: Path, suffix: str, partial_domain: Domain
    ) -> Path:
        """Write a copy of ``source`` in ``partial_domain``'s dialect to a temp file.

        Hyphens are rewritten to underscores unless the domain declares hyphenated
        identifiers, in which case ``source`` is copied verbatim.
        """
        text = source.read_text()
        if not cls._domain_uses_hyphens(partial_domain):
            text = _normalize_hyphens(text)
        with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as tmp:
            tmp.write(text)
            return Path(tmp.name)

    @classmethod
    def _parse_problem_normalized(cls, partial_domain: Domain, problem_pddl_path: Path) -> Problem:
        """Parse a problem PDDL, conforming its dialect to the domain's."""
        tmp_path = cls._conformed_tempfile(problem_pddl_path, ".pddl", partial_domain)
        try:
            return ProblemParser(tmp_path, partial_domain).parse_problem()
        finally:
            tmp_path.unlink(missing_ok=True)

    @classmethod
    def _parse_trajectory_normalized(
        cls, partial_domain: Domain, problem: Problem, trajectory_path: Path
    ):
        """Parse a trajectory, conforming its dialect to the domain's."""
        tmp_path = cls._conformed_tempfile(trajectory_path, ".trajectory", partial_domain)
        try:
            return TrajectoryParser(partial_domain, problem).parse_trajectory(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _state_positive_predicates(state: State) -> List[str]:
        """Untyped predicate strings (no parens) of a state's positive fluents."""
        predicates: List[str] = []
        for grounded_set in state.state_predicates.values():
            for pred in grounded_set:
                predicates.append(pred.untyped_representation[1:-1])
        return predicates

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _infer_domain_name(domain_path: Path, partial_domain: Domain) -> str:
        """Map a PDDL domain name / experiment path to a bench key."""
        domain_name = (partial_domain.name or "").lower()
        if domain_name in _DOMAIN_ALIASES:
            return _DOMAIN_ALIASES[domain_name]
        for part in Path(domain_path).resolve().parts:
            key = part.lower()
            if key in _HYPERPARAMS:
                return key
            if key in _DOMAIN_ALIASES:
                return _DOMAIN_ALIASES[key]
        return domain_name
