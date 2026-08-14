import inspect
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Set

from pddl_plus_parser.models import GroundedPredicate


class NoisingType(str, Enum):
    RANDOM = "random"
    PERCENTAGE = "percentage"


class NoisingStrategy(ABC):
    """Base class for selecting a subset of predicates to flip.

    A noising strategy is a *selector*: it returns the subset of the
    supplied predicates that should have their polarity flipped.  It does
    not mutate any predicate object — the caller is responsible for the
    actual flip (see ``flip_fluent_in_state`` in ``src/utils/pddl_state``).

    Callers are expected to pass only *unmasked* predicates; the strategy
    has no awareness of masking.
    """

    name: str

    @abstractmethod
    def noise(self, predicates: Set[GroundedPredicate],
              *args, **kwargs) -> Set[GroundedPredicate]:
        """Select the subset of predicates whose polarity should be flipped.

        Args:
            predicates: Candidate predicates (typically the unmasked subset
                of a single state).

        Returns:
            The subset selected for flipping.  May be empty.
        """
        raise NotImplementedError

    def validate_strategy_kwargs(self, strategy_kwargs: dict) -> None:
        """Validate kwargs against the concrete ``noise`` signature.

        Args:
            strategy_kwargs: Keyword arguments to validate.

        Raises:
            ValueError: If any unknown parameter names are supplied.
        """
        sig = inspect.signature(self.noise)
        valid_params = set(sig.parameters.keys()) - {"self", "predicates", "args", "kwargs"}
        unknown_params = set(strategy_kwargs.keys()) - valid_params
        if unknown_params:
            raise ValueError(
                f"Invalid parameter(s) for strategy '{self.name}': {unknown_params}. "
                f"Expected one or more of: {valid_params}"
            )


class RandomNoisingStrategy(NoisingStrategy):
    """Select each predicate for flipping independently with probability p."""

    name: str = NoisingType.RANDOM

    def noise(self, predicates: Set[GroundedPredicate],
              noise_proba: float = 0.1,
              *args, **kwargs) -> Set[GroundedPredicate]:
        print(f"using {self.name} noising strategy with probability {noise_proba}")
        # Sort by str for a deterministic iteration order (set order depends on
        # PYTHONHASHSEED, which would make seeded runs non-reproducible).
        return {p for p in sorted(predicates, key=str) if random.random() < noise_proba}


class PercentageNoisingStrategy(NoisingStrategy):
    """Select exactly p% of the predicates for flipping."""

    name: str = NoisingType.PERCENTAGE

    def noise(self, predicates: Set[GroundedPredicate],
              noise_ratio: float = 0.1,
              *args, **kwargs) -> Set[GroundedPredicate]:
        print(f"using {self.name} noising strategy with ratio {noise_ratio}")
        if not predicates or noise_ratio == 0:
            return set()
        sample_size = max(1, round(len(predicates) * noise_ratio))
        # Sort by str for a deterministic sampling pool (set order depends on
        # PYTHONHASHSEED, which would make seeded runs non-reproducible).
        return set(random.sample(sorted(predicates, key=str), sample_size))
