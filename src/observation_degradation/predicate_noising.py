import random
from typing import Set

from pddl_plus_parser.models import GroundedPredicate

from src.observation_degradation.noising import NoisingType, RandomNoisingStrategy, PercentageNoisingStrategy
from src.observation_degradation.noising.noising_strategies import NoisingStrategy


class PredicateNoiser:
    """Seeded wrapper around a noising strategy.

    Mirrors the interface of ``PredicateMasker``.  Callers supply the
    candidate predicates (typically the *unmasked* predicates of a single
    state) and receive back the subset selected for polarity flipping.
    No predicate objects are mutated here.
    """

    _noising_strategy: NoisingType
    _noising_kwargs: dict

    _strategies: dict[NoisingType, NoisingStrategy] = {
        NoisingType.RANDOM: RandomNoisingStrategy(),
        NoisingType.PERCENTAGE: PercentageNoisingStrategy(),
    }

    def __init__(
        self,
        seed: int = 42,
        noising_strategy: NoisingType = NoisingType.PERCENTAGE,
        noising_kwargs: dict = None,
    ) -> None:
        self.seed = seed
        self._call_count: int = 0
        self.set_noising_strategy(
            noising_strategy,
            **(noising_kwargs or self._default_params_for(noising_strategy)),
        )

    @staticmethod
    def _default_params_for(strategy: NoisingType) -> dict:
        if strategy == NoisingType.RANDOM:
            return {"noise_proba": 0.1}
        if strategy == NoisingType.PERCENTAGE:
            return {"noise_ratio": 0.1}
        return {}

    def set_noising_strategy(self, noising_strategy: NoisingType, **kwargs) -> None:
        """Set the active noising strategy and validate its parameters.

        Args:
            noising_strategy: Which strategy to use.
            **kwargs: Parameters forwarded to the strategy's ``noise`` method.
        """
        self._noising_strategy = noising_strategy
        self._noising_kwargs = kwargs or self._default_params_for(noising_strategy)
        self._strategies[noising_strategy].validate_strategy_kwargs(self._noising_kwargs)

    def reset(self) -> None:
        """Reset the internal call counter so the next noise() call reproduces the original sequence."""
        self._call_count = 0

    def noise(self, predicates: Set[GroundedPredicate]) -> Set[GroundedPredicate]:
        """Return the subset of predicates selected for flipping.

        Uses a derived seed (base seed + call counter) so each invocation
        produces different random draws while remaining reproducible
        across runs.

        Args:
            predicates: Candidate predicates (typically the unmasked
                subset of a single state).

        Returns:
            The subset selected for polarity flipping.
        """
        random.seed(self.seed + self._call_count)
        self._call_count += 1
        return self._strategies[self._noising_strategy].noise(predicates, **self._noising_kwargs)
