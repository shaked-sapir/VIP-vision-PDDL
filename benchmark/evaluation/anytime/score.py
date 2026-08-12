"""Score a fold's checkpoints offline, against one shared set of observations.

Two rules make the resulting numbers comparable, and both are choices rather
than conveniences:

**One reference, not one per arm.** Every arm is scored against the fold-level
``original_observations/`` -- the plain degraded traces. ``cdps_anchored`` keeps
its own differently-prepped copy, and scoring it against that would ask a
different question of it than of its neighbours. Anchoring is part of the arm,
not part of the yardstick.

**Ground-truth-free.** :func:`observations_reconstruction_score` only ever sees
the observations, so the curve measures the same thing the loop optimises. The
GT-based metrics stay where they belong, in the results table.

Identical models are scored once. Consecutive ROSAME snapshots frequently agree
to the character, and a loop round that fails to improve re-emits its incumbent,
so on a dense run most checkpoints are repeats of their predecessor.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pddl_plus_parser.lisp_parsers import DomainParser
from pddl_plus_parser.models import Observation

from benchmark.evaluation.anytime.checkpoints import Checkpoint
from src.pi_sam.plan_denoising.evaluator import (
    EvaluationWeights,
    observations_reconstruction_score,
)
from src.utils.masking import load_masked_observation

ORIGINAL_OBS_DIR = "original_observations"
DOMAIN_REFERENCE = "domain_reference.pddl"


@dataclass(frozen=True)
class ScoredCheckpoint:
    """A checkpoint plus the score it earned. ``None`` score means unparseable."""

    checkpoint: Checkpoint
    success_rate: Optional[float]
    v_raw: Optional[float]
    v_per_transition: Optional[float]
    model_digest: str
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """Flatten for the JSON sidecar the plotter reads."""
        return {
            "arm": self.checkpoint.arm,
            "index": self.checkpoint.index,
            "elapsed_seconds": self.checkpoint.elapsed_seconds,
            "model_path": str(self.checkpoint.model_path),
            "model_digest": self.model_digest,
            "success_rate": self.success_rate,
            "v_raw": self.v_raw,
            "v_per_transition": self.v_per_transition,
            "error": self.error,
        }


def _canonical_model_text(text: str) -> str:
    """Model text with its ``:requirements`` line dropped and whitespace folded.

    ``LearnerDomain.to_pddl`` rebuilds ``:requirements`` from a *set*, so two
    renders of one model can differ only in that line's ordering. Hashing the raw
    text would therefore miss most genuine repeats -- which is the whole reason
    the loop's round log carries its own structural ``model_hash``. This digest
    is for caching only; it is never used to claim two models are the same model
    in any reported number.
    """
    without_requirements = re.sub(r"\(:requirements[^)]*\)", "", text)
    return " ".join(without_requirements.split())


def load_fold_observations(fold_dir: Path) -> List[Observation]:
    """The fold's frozen degraded observations, masked exactly as learned from.

    Raises:
        FileNotFoundError: if the fold has no ``original_observations/`` or no
            ``domain_reference.pddl``. Both are written by every run, so their
            absence means the path is not a fold, not that the fold is odd.
    """
    fold_dir = Path(fold_dir)
    obs_dir = fold_dir / ORIGINAL_OBS_DIR
    domain_path = fold_dir / DOMAIN_REFERENCE
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"no {ORIGINAL_OBS_DIR}/ under {fold_dir}")
    if not domain_path.exists():
        raise FileNotFoundError(f"no {DOMAIN_REFERENCE} under {fold_dir}")

    domain = DomainParser(domain_path, partial_parsing=True).parse_domain()

    observations: List[Observation] = []
    for trajectory in sorted(obs_dir.glob("*.trajectory")):
        masking = trajectory.with_suffix(".masking_info")
        if not masking.exists():
            print(f"  Warning: {trajectory.name} has no .masking_info; skipping")
            continue
        observations.append(load_masked_observation(trajectory, masking, domain))
    return observations


def score_checkpoints(
    checkpoints: Sequence[Checkpoint],
    observations: Sequence[Observation],
    weights: Optional[EvaluationWeights] = None,
) -> List[ScoredCheckpoint]:
    """Score each checkpoint's model against ``observations``, reusing repeats."""
    weights = weights or EvaluationWeights()
    cache: Dict[str, ScoredCheckpoint] = {}
    scored: List[ScoredCheckpoint] = []

    for checkpoint in checkpoints:
        text = checkpoint.model_path.read_text()
        digest = hashlib.sha256(_canonical_model_text(text).encode()).hexdigest()[:16]

        hit = cache.get(digest)
        if hit is not None:
            scored.append(
                ScoredCheckpoint(
                    checkpoint, hit.success_rate, hit.v_raw,
                    hit.v_per_transition, digest, hit.error,
                )
            )
            continue

        result = _score_one(checkpoint, observations, weights, digest)
        cache[digest] = result
        scored.append(result)

    return scored


def _score_one(
    checkpoint: Checkpoint,
    observations: Sequence[Observation],
    weights: EvaluationWeights,
    digest: str,
) -> ScoredCheckpoint:
    """Parse and score one model; a broken model becomes a recorded gap, not a crash.

    A model that will not parse is a real outcome for a mid-run snapshot -- an
    arm can and does emit garbage before it converges -- so it is carried through
    as a scored point with no score rather than dropped, which would quietly
    flatter the arm that produced it.
    """
    try:
        domain = DomainParser(checkpoint.model_path).parse_domain()
    except Exception as exc:  # noqa: BLE001 - any parse failure is the same outcome here
        print(f"  Warning: {checkpoint.arm} #{checkpoint.index} did not parse: {exc}")
        return ScoredCheckpoint(checkpoint, None, None, None, digest, error=str(exc))

    evaluation = observations_reconstruction_score(domain, observations, weights=weights)
    return ScoredCheckpoint(
        checkpoint,
        success_rate=evaluation.success_rate,
        v_raw=evaluation.v_raw,
        v_per_transition=evaluation.v_per_transition,
        model_digest=digest,
    )


def score_fold(
    fold_dir: Path,
    checkpoints_by_arm: Dict[str, List[Checkpoint]],
    weights: Optional[EvaluationWeights] = None,
) -> Dict[str, List[ScoredCheckpoint]]:
    """Score every arm of one fold against that fold's shared observations."""
    observations = load_fold_observations(fold_dir)
    weights = weights or EvaluationWeights()
    return {
        arm: score_checkpoints(checkpoints, observations, weights)
        for arm, checkpoints in checkpoints_by_arm.items()
    }


def write_scores(
    fold_dir: Path,
    scored_by_arm: Dict[str, List[ScoredCheckpoint]],
    weights: Optional[EvaluationWeights] = None,
    filename: str = "anytime_scores.json",
) -> Path:
    """Write the scored stream next to the fold's other diagnostics."""
    weights = weights or EvaluationWeights()
    payload = {
        "weights": {
            "effect_mismatch": weights.effect_mismatch,
            "inapplicability": weights.inapplicability,
        },
        "reference": f"{ORIGINAL_OBS_DIR}/ (fold-level, shared by every arm)",
        "arms": {
            arm: [s.as_dict() for s in scored]
            for arm, scored in scored_by_arm.items()
        },
    }
    path = Path(fold_dir) / filename
    path.write_text(json.dumps(payload, indent=2))
    return path
