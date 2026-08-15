"""Turn a solved MILP into repaired trajectories T' for PI-SAM.

The MILP assigns a truth value to *every* fluent of *every* state (variable
``hol[i, t, p]``). We deliberately do NOT hand all of that to PI-SAM. Per
``docs/pisam-milp-denoiser-design.md`` §4.2:

- **Observed (unmasked) fluents** take the MILP's value. Where that differs from
  the original observation, the difference is a *repair* — exactly the quantity
  the objective minimizes, and exactly what a CDPS fluent patch would have done.
- **Masked fluents stay masked.** The MILP's completion of them is one arbitrary
  member of a whole family of consistent completions; committing to it would
  hand PI-SAM evidence the data never contained and would break the partial-
  observability contract the safety theorem rests on. The completion is written
  out as ``milp_masked_completion.json`` for diagnosis only.

Because masks are preserved and only polarities of already-present predicates
change, "re-masking" needs no extra step — it is what copying-then-editing does.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pddl_plus_parser.models import Observation, State

from src.milp.converter import proposition_of
from src.utils.pddl_state import copy_observation_linked, get_state_grounded_predicates


@dataclass(frozen=True)
class FluentFlip:
    """One observed fluent whose polarity the MILP changed."""

    observation_index: int
    state_index: int  # 0-based state index (0 = initial state)
    fluent: str
    original_is_positive: bool

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observation_index": self.observation_index,
            "state_index": self.state_index,
            "fluent": self.fluent,
            "from": self.original_is_positive,
            "to": not self.original_is_positive,
        }

    def as_patch_dict(self) -> Dict[str, Any]:
        """The same flip in CDPS's ``FluentLevelPatch`` JSON shape.

        CDPS addresses a state by (component, ``prev``/``next``); we address it
        by state index. State 0 is component 0's ``prev``, state s>0 is
        component s-1's ``next`` — the same state either way, because
        ``comp[i].next_state is comp[i+1].previous_state``.
        """
        if self.state_index == 0:
            component_index, state_type = 0, "prev"
        else:
            component_index, state_type = self.state_index - 1, "next"
        return {
            "observation_index": self.observation_index,
            "component_index": component_index,
            "state_type": state_type,
            "fluent": self.fluent,
        }


@dataclass
class ExtractionResult:
    """Repaired observations plus the bookkeeping the reports need."""

    observations: List[Observation]
    flips: List[FluentFlip] = field(default_factory=list)
    masked_completion: List[Dict[str, Any]] = field(default_factory=list)
    unmapped_fluents: int = 0

    @property
    def repair_cost(self) -> int:
        """Number of observed-fluent flips — the ``cost`` of this repair."""
        return len(self.flips)

    def as_stats(self) -> Dict[str, Any]:
        return {
            "repair_cost": self.repair_cost,
            "n_masked_completed": sum(
                len(entry["fluents"]) for entry in self.masked_completion
            ),
            "n_unmapped_fluents": self.unmapped_fluents,
        }


def _observation_states(observation: Observation) -> List[State]:
    """The N+1 states of an observation, in order (state 0 = initial)."""
    components = observation.components
    return [components[0].previous_state] + [c.next_state for c in components]


def _grounded_arg_names(grounded_predicate) -> List[str]:
    mapping = grounded_predicate.object_mapping
    return [mapping[k] for k in grounded_predicate.signature.keys()]


def extract_repaired_observations(
    encoder,
    observations: Sequence[Observation],
) -> ExtractionResult:
    """Build T' from a *solved* encoder.

    Args:
        encoder: a solved :class:`encoder.CPSATObservedActions`.
        observations: the original observations, aligned 1:1 with
            ``encoder.traces.obs_t`` (the converter may drop traces, so the
            caller is responsible for keeping the two lists in step).

    Returns:
        An :class:`ExtractionResult` holding deep copies — the originals are the
        frozen evaluation reference and are never mutated.
    """
    traces = encoder.traces.obs_t
    if len(observations) != len(traces):
        raise ValueError(
            f"observations ({len(observations)}) and traces ({len(traces)}) must "
            "be aligned 1:1"
        )

    result = ExtractionResult(observations=[])
    for obs_idx, (observation, trace) in enumerate(zip(observations, traces)):
        repaired = copy_observation_linked(observation)
        instance = trace.instance
        completion: Dict[str, bool] = {}

        for state_idx, state in enumerate(_observation_states(repaired)):
            for grounded_predicate in get_state_grounded_predicates(state):
                milp_value = _solved_value(
                    encoder, obs_idx + 1, state_idx + 1, instance, grounded_predicate
                )
                if milp_value is None:
                    result.unmapped_fluents += 1
                    continue
                if grounded_predicate.is_masked:
                    completion[grounded_predicate.untyped_representation] = milp_value
                    continue
                if grounded_predicate.is_positive != milp_value:
                    result.flips.append(FluentFlip(
                        observation_index=obs_idx,
                        state_index=state_idx,
                        fluent=grounded_predicate.untyped_representation,
                        original_is_positive=grounded_predicate.is_positive,
                    ))
                    grounded_predicate.is_positive = milp_value

        result.observations.append(repaired)
        if completion:
            result.masked_completion.append(
                {"observation_index": obs_idx, "fluents": completion}
            )

    return result


def _solved_value(encoder, trace_index: int, time_index: int, instance, grounded_predicate):
    """The MILP's truth value for one grounded predicate, or ``None`` when it has
    no MILP counterpart (unknown predicate/object — should never happen)."""
    proposition = proposition_of(
        instance, grounded_predicate.name, _grounded_arg_names(grounded_predicate)
    )
    if proposition is None:
        return None
    variable = encoder.hol.get((trace_index, time_index, proposition))
    if variable is None:
        return None
    return bool(variable.value())


def save_extraction_artifacts(result: ExtractionResult, output_dir: Path) -> None:
    """Write the two diagnostics: masked completion and the repair log."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "milp_masked_completion.json").write_text(json.dumps({
        "note": "MILP's completion of MASKED fluents. Diagnostic only — these "
                "values are NOT given to PI-SAM (see trajectory_extraction.py).",
        "observations": result.masked_completion,
    }, indent=2))

    (output_dir / "milp_repair_log.json").write_text(json.dumps({
        "repair_cost": result.repair_cost,
        "flips": [flip.as_dict() for flip in result.flips],
    }, indent=2))
