"""NOLAM's MAP rule with the posterior evaluated in log space.

``nolam.algorithm.Learner.Learner.learn`` scores nine hypotheses per (operator,
atom) from four transition counts: each likelihood is a multinomial coefficient
times a product of probabilities raised to the counts. The coefficient is
computed as an integer division into a float and raises ``OverflowError`` once
it passes the largest float; the product underflows to zero at about the same
counts. :class:`LogSpaceLearner` keeps the library's counting, priors,
per-hypothesis probabilities and choice rule, and evaluates the same posterior
from sums of logarithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from nolam.algorithm import Configuration
from nolam.algorithm.ActionModel import ActionModel
from nolam.algorithm.Learner import Learner

Hypothesis = Tuple[str, str]

# (precondition, effect), in the order of the library's ``hypothesis`` array.
HYPOTHESES: Tuple[Hypothesis, ...] = (
    ("none", "none"),
    ("none", "pos"),
    ("none", "neg"),
    ("pos", "none"),
    ("neg", "none"),
    ("pos", "pos"),
    ("neg", "pos"),
    ("pos", "neg"),
    ("neg", "neg"),
)

# The library's posterior for every hypothesis when no likelihood is positive.
NO_EVIDENCE_POSTERIOR = 1e-10


@dataclass(frozen=True)
class TransitionCounts:
    """How often an atom was seen false/true before and after an operator."""

    neg_neg: int
    neg_pos: int
    pos_neg: int
    pos_pos: int

    @property
    def total(self) -> int:
        return self.neg_neg + self.neg_pos + self.pos_neg + self.pos_pos

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """``(neg-neg, neg-pos, pos-neg, pos-pos)``."""
        return self.neg_neg, self.neg_pos, self.pos_neg, self.pos_pos


def transition_probabilities(e: float) -> Dict[Hypothesis, Tuple[float, float, float, float]]:
    """Per hypothesis, the probability of observing each transition under flip rate ``e``.

    Each tuple is ``(neg-neg, neg-pos, pos-neg, pos-pos)``; the expressions are
    the library's ``PrNmm`` / ``PrNmp`` / ``PrNpm`` / ``PrNpp``, term for term.
    """
    k_neg = 1 / 2
    return {
        ("none", "none"): (
            k_neg * ((1 - e) ** 2) + (1 - k_neg) * (e ** 2),
            k_neg * (1 - e) * e + (1 - k_neg) * e * (1 - e),
            k_neg * e * (1 - e) + (1 - k_neg) * (1 - e) * e,
            k_neg * (e ** 2) + (1 - k_neg) * ((1 - e) ** 2),
        ),
        ("none", "pos"): (
            k_neg * (1 - e) * e + (1 - k_neg) * (e ** 2),
            k_neg * ((1 - e) ** 2) + (1 - k_neg) * (1 - e) * e,
            (1 - k_neg) * (1 - e) * e + k_neg * (e ** 2),
            (1 - k_neg) * ((1 - e) ** 2) + k_neg * (1 - e) * e,
        ),
        ("none", "neg"): (
            k_neg * ((1 - e) ** 2) + (1 - k_neg) * (1 - e) * e,
            k_neg * (1 - e) * e + (1 - k_neg) * (e ** 2),
            (1 - k_neg) * ((1 - e) ** 2) + k_neg * (1 - e) * e,
            (1 - k_neg) * (1 - e) * e + k_neg * (e ** 2),
        ),
        ("pos", "none"): (e ** 2, e * (1 - e), (1 - e) * e, (1 - e) ** 2),
        ("neg", "none"): ((1 - e) ** 2, (1 - e) * e, e * (1 - e), e ** 2),
        ("pos", "pos"): (e ** 2, e * (1 - e), (1 - e) * e, (1 - e) ** 2),
        ("neg", "pos"): ((1 - e) * e, (1 - e) ** 2, e ** 2, e * (1 - e)),
        ("pos", "neg"): (e * (1 - e), e ** 2, (1 - e) ** 2, (1 - e) * e),
        ("neg", "neg"): ((1 - e) ** 2, (1 - e) * e, e * (1 - e), e ** 2),
    }


def hypothesis_priors(counts: TransitionCounts, allow_prec_neg: bool) -> Dict[Hypothesis, float]:
    """The library's count-based prior of each hypothesis; ``counts.total`` must be positive."""
    total = counts.total
    prior_prepos = ((counts.pos_pos + counts.pos_neg) / total) * 2 / 3
    prior_preneg = ((counts.neg_pos + counts.neg_neg) / total) * 2 / 3
    prior_prenone = 1 / 3
    if not allow_prec_neg:
        prior_preneg = 0
        tot = prior_prepos + prior_prenone
        prior_prepos = prior_prepos / tot
        prior_prenone = prior_prenone / tot
    precondition = {"none": prior_prenone, "pos": prior_prepos, "neg": prior_preneg}
    effect = {
        "none": (counts.neg_neg + counts.pos_pos) / total,
        "pos": counts.neg_pos / total,
        "neg": counts.pos_neg / total,
    }
    return {(pre, eff): precondition[pre] * effect[eff] for pre, eff in HYPOTHESES}


def _log(value: float) -> float:
    return math.log(value) if value > 0 else -math.inf


def log_likelihood(counts: TransitionCounts, probabilities: Tuple[float, ...]) -> float:
    """Log of the product of ``probabilities`` raised to ``counts``, without the coefficient."""
    return sum(
        count * _log(probability)
        for count, probability in zip(counts.as_tuple(), probabilities)
        if count > 0
    )


def posteriors(counts: TransitionCounts, e: float, allow_prec_neg: bool) -> List[float]:
    """The posterior of each hypothesis, in :data:`HYPOTHESES` order.

    Every entry is ``NO_EVIDENCE_POSTERIOR`` when no hypothesis has a positive
    prior and likelihood, which is what the library returns in that case.
    """
    probabilities = transition_probabilities(e)
    priors = hypothesis_priors(counts, allow_prec_neg)
    scores = [
        log_likelihood(counts, probabilities[hypothesis]) + _log(priors[hypothesis])
        for hypothesis in HYPOTHESES
    ]
    best = max(scores)
    if best == -math.inf:
        return [NO_EVIDENCE_POSTERIOR] * len(HYPOTHESES)
    weights = [math.exp(score - best) for score in scores]
    normaliser = sum(weights)
    return [weight / normaliser for weight in weights]


def _add_to_operator(operator, atom: str, hypothesis: Hypothesis) -> None:
    """Record ``atom`` in ``operator`` as the library does for the chosen hypothesis."""
    if hypothesis == ("none", "pos"):
        operator.eff_pos.add(atom)
    elif hypothesis == ("none", "neg"):
        operator.eff_neg.add(atom)
    elif hypothesis == ("pos", "none"):
        operator.precs_pos.add(atom)
    elif hypothesis == ("neg", "none"):
        operator.precs_neg.add(atom)
    elif hypothesis == ("neg", "pos"):
        operator.precs_neg.add(atom)
        operator.eff_pos.add(atom)
    elif hypothesis == ("pos", "neg"):
        operator.precs_pos.add(atom)
        operator.eff_neg.add(atom)


class LogSpaceLearner(Learner):
    """The library's learner with :func:`posteriors` in place of its float likelihoods."""

    def learn(self, input_file: str, trace_names: List[str], e: float) -> ActionModel:
        """Learn a model by MAP; same arguments and result as ``Learner.learn``.

        Raises:
            NotImplementedError: If the library is configured to sample a
                hypothesis instead of taking the maximum.
        """
        if Configuration.SAMPLING:
            raise NotImplementedError("LogSpaceLearner implements the MAP criterion only")
        domain_learned = ActionModel(input_file)
        domain_learned.empty()
        domain_learned.init_prec_eff()
        self.op_stats = {
            o.operator_name: {
                p: {"pos-pos": 0, "pos-neg": 0, "neg-pos": 0, "neg-neg": 0} for p in o.eff_pos
            }
            for o in domain_learned.operators
        }
        op_stats = self.count_traces(trace_names, domain_learned)
        domain_learned.empty()

        for operator in domain_learned.operators:
            for atom, stats in op_stats[operator.operator_name].items():
                counts = TransitionCounts(
                    neg_neg=stats["neg-neg"], neg_pos=stats["neg-pos"],
                    pos_neg=stats["pos-neg"], pos_pos=stats["pos-pos"],
                )
                if counts.total == 0:
                    continue
                b = np.array(posteriors(counts, e, Configuration.ALLOW_PREC_NEG))
                chosen = np.random.choice(np.flatnonzero(b == b.max()))
                _add_to_operator(operator, atom, HYPOTHESES[chosen])
        return domain_learned
