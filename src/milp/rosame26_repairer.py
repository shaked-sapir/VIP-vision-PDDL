"""The MILP half of the ICAPS-26 loop, as an injected :class:`MipRepairer`.

:class:`Rosame26MipRepairer` implements the Protocol
:mod:`src.milp.rosame26_training` defines — ``clear`` / ``update`` /
``run_fixer`` / ``pseudo_labels`` — over :mod:`src.milp.encoder` rather than the
vendored ``Convertor``. Upstream splits the same work across a ``TraceSelector``
and a ``Convertor``; one object here, because our encoder replaces both and
which traces a solve covers is an implementation detail of the selection.

WHY NOT THE VENDORED CONVERTOR (plan §6.1, DECIDED). Its solvers take
``max_t = traces.obs_t[0].step + 1`` — the *first* trace's length applied to the
whole bundle. Upstream's corpora are length-homogeneous; a fold of ours is
ragged (a real hanoi fold is 9/8/6), so that either raises or silently encodes a
short trace against a long horizon. ``src/milp/encoder.py`` reads ``_steps(i)``
per trace and already fixes it.

WHY THE RAGGED LABEL PROBLEM DISSOLVES. §6.1 warns that
``translator.extract_sol_label`` sizes its outputs from one shared
``problem.max_t``, and prescribes pad-and-mask end to end. That is true of the
*vendored translator*, which this module does not call: like the ICAPS-24 arm,
it reads :meth:`~src.milp.encoder.CPSATObservedActions.repaired_states` per
trace, which is sized from ``_steps(i)``. The arithmetic then lands exactly —
``repaired_states`` returns one frame per image (``N``), dropping both endpoints
leaves ``N - 2 = T`` rows, and ``z`` is ``[T, S]``. Phase 3's ``lengths`` mask
covers the padded batch dimension; no second padding layer is needed.

THE THREE PERMUTATION ENDS MEET HERE. The model channel writes into rows of
``schema()``, which are in the head's sorted-type order while the CP domain's
bindings are in PDDL order — silent, since both have the same width.
:mod:`src.milp.schema_row_alignment` is what makes the two comparable, and
plan §0.1a is why identity does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch

import src.milp  # noqa: F401  (vendor sys.path bootstrap)
from src.milp.converter import cv_predictions_to_trace, proposition_of
from src.milp.encoder import CPSATObservedActions
from src.milp.encoding_config import MilpEncodingConfig
from src.milp.schema_row_alignment import schema_row_keys

from convertor.pseudo_label import PseudoLabels
from planning_structs.traces import ObservationM, Traces

#: What the encoder optimises. ``state`` and ``model`` are the two channels
#: option B leaves it: the action is observed, so there is nothing to infer.
OBJECTIVES: Set[str] = {"state", "model"}

#: Probability floor, matching ``cv_predictions_to_trace``'s own clamp.
EPS: float = 1e-4


@dataclass
class TraceContext:
    """Everything about one trace the encoder needs that the network does not.

    Attributes:
        instance: The CP grounding this trace is expressed over.
        calls: The observed grounded actions, as ``(name, args)``.
        init_fluents: GT initial-state positive fluents.
        goal_fluents: GT final-state positive fluents.
    """

    instance: Any
    calls: List[Tuple[str, Sequence[str]]]
    init_fluents: Set[Tuple[str, Tuple[str, ...]]]
    goal_fluents: Optional[Set[Tuple[str, Tuple[str, ...]]]]


@dataclass
class _Collected:
    """One batch row offered to the next solve."""

    index: int
    probs: List[List[float]]


def state_probabilities(z_row: torch.Tensor, live: int) -> List[List[float]]:
    """One trace's ``z`` as the ``probs`` rows ``cv_predictions_to_trace`` takes.

    ``z`` covers the ``T`` interior frames; ``probs`` is indexed by image and
    must have ``T + 2`` rows, of which only ``probs[1..T]`` are read — the
    endpoints are replaced by hard rows built from the GT states. The two
    placeholder rows are therefore never read and their value is immaterial;
    ``0.5`` is used because it is the encoder's own zero-weight value.
    """
    interior = z_row[:live].detach().cpu().tolist()
    width = len(interior[0]) if interior else 0
    placeholder = [0.5] * width
    return [placeholder, *interior, placeholder]


def model_labels_in_head_order(
    solution: ObservationM, domain_model: object, ps_domain: object
) -> Dict[str, torch.Tensor]:
    """The solved action model as ``{schema: [rows, 4]}`` in *head row* order.

    The 4-way encoding is upstream's: 0 irrelevant, 1 add, 2 precondition,
    3 precondition-and-delete, with precedence add > delete > precondition.

    Every row is placed by its ``(predicate, PDDL positions)`` key rather than by
    position, which is what :mod:`src.milp.schema_row_alignment` exists for.

    Raises:
        ValueError: if a schema's rows are not a bijection onto the CP bindings.
    """
    keys = schema_row_keys(domain_model, ps_domain)
    labels: Dict[str, torch.Tensor] = {}
    for schema in domain_model.action_schemas:
        rows = keys[schema.name]
        label = torch.zeros(len(rows), 4)
        ps_schema = ps_domain.get_action_schema(schema.name)
        for index, (predicate_name, args) in enumerate(rows):
            key = (ps_schema, ps_domain.get_predicate(predicate_name), args)
            if bool(solution.add.get(key, 0)):
                label[index, 1] = 1.0
            elif bool(solution.dele.get(key, 0)):
                label[index, 3] = 1.0
            elif bool(solution.pre.get(key, 0)):
                label[index, 2] = 1.0
            else:
                label[index, 0] = 1.0
        labels[schema.name] = label
    return labels


def state_labels_for(
    instance: Any, repaired: Sequence[Set[Any]], proposition_index: Dict[str, int]
) -> torch.Tensor:
    """Interior repaired frames as a ``[T, S]`` binary target aligned with ``z``.

    ``repaired`` holds one set of true propositions per *image*; the endpoints
    are hard-fixed in the encoder and are dropped, leaving exactly ``T`` rows.

    A head column with no counterpart in ``instance`` reads 0.0. That is a
    *supervised false*, not an abstention, and it is correct: the head is
    grounded over the corpus object union while each trace's instance covers only
    its own problem, so such a column names a proposition over an object this
    problem does not have. The union grounding therefore hands the head free,
    correct supervision a per-problem grounding could not have given — the
    benefit Phase 2 identified alongside the phantom columns' zero objective cost
    (process log §2.5, plan §4.2a′). The reverse direction, head → encoder, is
    where a phantom instead defaults to ``0.5`` and carries no weight.
    """
    interior = list(repaired)[1:-1]
    width = len(proposition_index)
    label = torch.zeros(len(interior), width)
    columns: Dict[Any, int] = {}
    for key, column in proposition_index.items():
        parts = key.split()
        prop = proposition_of(instance, parts[0], parts[1:]) if parts else None
        if prop is not None:
            columns[prop] = column
    for row, frame in enumerate(interior):
        for prop in frame:
            if prop in columns:
                label[row, columns[prop]] = 1.0
    return label


def reference_action_model(bench: str) -> ObservationM:
    """The domain's own action model as an ``ObservationM``, for ``mip_gt_dist``.

    Built from ``src/domains/<bench>.pddl`` through the project's own parser
    rather than the vendored ``util.pddl_parsing.parse_pddl_domain``, which needs
    the ``lifted_pddl`` package this project does not depend on and which is
    reachable only from ``Convertor.__init__``.

    Keys are ``(schema, predicate, 1-based PDDL argument positions)`` — the CP
    domain's own scheme, which is what makes it comparable to a solved
    ``action_model_sol`` without a permutation search (plan §0.1a).

    Raises:
        KeyError: if the domain PDDL names a predicate the CP domain does not.
    """
    from pddl_plus_parser.lisp_parsers import DomainParser

    from src.milp.domain_assets import build_domain, source_pddl_path

    ps_domain = build_domain(bench)
    parsed = DomainParser(source_pddl_path(bench), partial_parsing=False).parse_domain()

    pre: Dict[Any, float] = {}
    add: Dict[Any, float] = {}
    dele: Dict[Any, float] = {}
    for schema in ps_domain.action_schemas:
        for predicate in ps_domain.predicates:
            for binding in ps_domain.predicate_arguments.get((schema, predicate), []):
                key = (schema, predicate, tuple(binding))
                pre[key] = 0.0
                add[key] = 0.0
                dele[key] = 0.0

    for name, action in parsed.actions.items():
        schema = ps_domain.get_action_schema(name)
        if schema is None:
            continue
        positions = {
            parameter: index + 1 for index, parameter in enumerate(action.signature)
        }
        for literal in action.preconditions.root.operands:
            key = _literal_key(ps_domain, schema, str(literal), positions)
            if key is not None:
                pre[key] = 1.0
        for effect in action.discrete_effects:
            text = str(effect).strip()
            negated = text.startswith("(not ")
            if negated:
                text = text[len("(not ") : -1].strip()
            key = _literal_key(ps_domain, schema, text, positions)
            if key is None:
                continue
            if negated:
                dele[key] = 1.0
            else:
                add[key] = 1.0
    return ObservationM(pre, add, dele)


def _literal_key(ps_domain, schema, text: str, positions: Dict[str, int]):
    """``"(on ?x - block ?y - block)"`` as a CP binding key, or ``None``."""
    tokens = text.strip().strip("()").split()
    if not tokens:
        return None
    predicate = ps_domain.get_predicate(tokens[0])
    if predicate is None:
        return None
    args = tuple(
        positions[token] for token in tokens[1:] if token.startswith("?")
    )
    return (schema, predicate, args)


class Rosame26MipRepairer:
    """``src/milp/encoder.py`` in the shape ``_run_training`` drives it.

    Args:
        domain_model: The grounded vendored ``Domain_Model`` being trained.
        ps_domain: The CP domain the encoder is built over.
        contexts: ``{trace index: TraceContext}``, indexed as the fold's rows.
        proposition_index: ``{"on a b": head column}``, the run's shared map.
        config: The MILP rule set.
        mip_traces: How many traces one solve covers. Upstream's FIFO rule:
            the first ``mip_traces`` offered in an epoch, so batch order decides.
        gt_action_model: A reference ``ObservationM`` for the ``mip_gt_dist``
            diagnostic. ``None`` reports no distance.
        lengths: The fold's per-trace ``T``, indexed as ``contexts`` is. The one
            source of truth for how much of a padded ``z`` row is real;
            :mod:`src.milp.trace_tensors` owns that arithmetic. ``None``
            re-derives it from the action count, which is the same number
            (``T = len(calls) - 1``) and is checked against ``lengths`` when both
            are available.
    """

    def __init__(
        self,
        domain_model: object,
        ps_domain: object,
        contexts: Dict[int, TraceContext],
        proposition_index: Dict[str, int],
        config: Optional[MilpEncodingConfig] = None,
        mip_traces: int = 3,
        gt_action_model: Optional[ObservationM] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> None:
        self.domain_model = domain_model
        self.ps_domain = ps_domain
        self.contexts = contexts
        self.proposition_index = proposition_index
        self.config = config
        self.mip_traces = mip_traces
        self.gt_action_model = gt_action_model
        self.lengths = lengths
        self._labels = PseudoLabels()
        self._collected: List[_Collected] = []
        self.rounds: List[Dict[str, Any]] = []
        #: The most recent solved encoder. Kept so a caller can recompute a
        #: label, read ``repaired_states`` or render the solution without
        #: re-solving; gate 3's parity check is the reason it exists.
        self._last_encoder: Optional[CPSATObservedActions] = None

    @property
    def pseudo_labels(self) -> PseudoLabels:
        """The store :meth:`Rosame26Goal.loss` reads."""
        return self._labels

    def clear(self) -> None:
        """Drop the traces collected for the previous solve.

        The *labels* survive: upstream decays them by ``pseudo_weight_decay``
        each epoch rather than discarding them, so a trace keeps teaching until
        it is relabelled or its weight decays away.
        """
        self._collected = []

    def update(
        self,
        indices: torch.Tensor,
        z: torch.Tensor,
        a: torch.Tensor,
        inits: torch.Tensor,
        goals: torch.Tensor,
    ) -> None:
        """Offer a batch's predicted states for the next solve.

        ``a``, ``inits`` and ``goals`` are accepted to match the Protocol and are
        unused: the actions are observed and both endpoints are GT, so all three
        already live in this repairer's :class:`TraceContext`.

        FIFO to ``mip_traces``, which is upstream's rule (``network.py``'s
        ``TraceSelector`` fills from the first batches of the epoch).
        """
        for row, index in enumerate(indices.tolist()):
            if len(self._collected) >= self.mip_traces:
                return
            if index not in self.contexts:
                continue
            self._collected.append(
                _Collected(
                    index=index,
                    probs=state_probabilities(z[row], self._live(index)),
                )
            )

    def run_fixer(self, time_limit: float) -> Optional[float]:
        """Solve over the collected traces and write pseudo-labels.

        Returns:
            ``mip_gt_dist`` when a reference model was given and the solve
            succeeded; ``None`` otherwise, which is what upstream logs when the
            solve produced nothing.
        """
        names = sorted(self.proposition_index, key=self.proposition_index.get)
        observations = []
        solved: List[_Collected] = []
        for item in self._collected:
            context = self.contexts[item.index]
            trace = cv_predictions_to_trace(
                context.instance, names, item.probs, context.calls,
                context.init_fluents, context.goal_fluents,
            )
            if trace is None:
                continue
            observations.append(trace)
            solved.append(item)

        if not observations:
            self.rounds.append({"status": "NO_TRACES"})
            return None

        encoder = CPSATObservedActions(
            self.ps_domain,
            Traces(instance=None, obs_m=self._observation_m(), obs_t=observations),
            OBJECTIVES,
            config=self.config,
        )
        if not encoder.solve(time_limit=time_limit):
            self.rounds.append(dict(encoder.solve_stats, status="NO_SOLUTION"))
            return None

        self._last_encoder = encoder
        solution = encoder.action_model_sol()
        self._labels.model = model_labels_in_head_order(
            solution, self.domain_model, self.ps_domain
        )
        for position, item in enumerate(solved, start=1):
            context = self.contexts[item.index]
            state_label = state_labels_for(
                context.instance, encoder.repaired_states(position),
                self.proposition_index,
            )
            self._labels.traces[item.index] = (
                1,
                state_label,
                torch.zeros(state_label.shape[0], dtype=torch.long),
            )

        distance = self._gt_distance(solution)
        self.rounds.append(
            dict(
                encoder.solve_stats,
                status="OK",
                traces=[item.index for item in solved],
                mip_gt_dist=distance,
            )
        )
        return distance

    def _live(self, index: int) -> int:
        """Trace ``index``'s real step count, ``T``.

        Raises:
            ValueError: if ``lengths`` and the action count disagree, which means
                the fold's tensors and its contexts were built from different
                trace lists.
        """
        from_calls = len(self.contexts[index].calls) - 1
        if self.lengths is None:
            return from_calls
        from_lengths = int(self.lengths[index].item())
        if from_lengths != from_calls:
            raise ValueError(
                f"trace {index} has lengths[{index}]={from_lengths} but "
                f"{len(self.contexts[index].calls)} actions, which implies "
                f"T={from_calls}; the tensors and the contexts disagree"
            )
        return from_lengths

    # ------------------------------------------------------------- internals

    def _observation_m(self) -> ObservationM:
        """The head's current action model, as the encoder's reference channel."""
        keys = schema_row_keys(self.domain_model, self.ps_domain)
        pre: Dict[Any, float] = {}
        add: Dict[Any, float] = {}
        dele: Dict[Any, float] = {}
        for schema in self.domain_model.action_schemas:
            probabilities = self.domain_model.activation(schema()).detach().cpu()
            ps_schema = self.ps_domain.get_action_schema(schema.name)
            for index, (predicate_name, args) in enumerate(keys[schema.name]):
                key = (ps_schema, self.ps_domain.get_predicate(predicate_name), args)
                row = probabilities[index]
                # 0 irrelevant, 1 add, 2 precondition, 3 precondition+delete.
                pre[key] = float(row[2] + row[3])
                add[key] = float(row[1])
                dele[key] = float(row[3])
        return ObservationM(pre, add, dele)

    def _gt_distance(self, solution: ObservationM) -> Optional[float]:
        """Fraction of bindings on which the solution differs from the reference.

        Upstream reports ``1 - highest_agree`` from ``model_permutation``. That
        search is bypassed here (plan §0.1a), so this is the same quantity
        without the permutation: both models are already in the CP domain's own
        argument order.
        """
        if self.gt_action_model is None:
            return None
        keys = list(self.gt_action_model.pre)
        if not keys:
            return None
        differing = sum(
            1
            for key in keys
            if (
                bool(solution.pre.get(key, 0)) != bool(self.gt_action_model.pre[key])
                or bool(solution.add.get(key, 0)) != bool(self.gt_action_model.add[key])
                or bool(solution.dele.get(key, 0))
                != bool(self.gt_action_model.dele[key])
            )
        )
        return differing / len(keys)
