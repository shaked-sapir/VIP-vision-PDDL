import random
from abc import ABC, abstractmethod
from enum import Enum

from pddl_plus_parser.models import GroundedPredicate


class MaskingType(str, Enum):
    RANDOM = "random"
    PERCENTAGE = "percentage"


class MaskingStrategy(ABC):
    """
    This class serves as a base class for masking a set of predicates based on some logic/strategy.
    """
    @abstractmethod
    def mask(self, predicates: set[GroundedPredicate], *args, **kwargs) -> set[GroundedPredicate]:
        raise NotImplementedError


class RandomMasking(MaskingStrategy):
    """
    this class masks a certain predicate with probability p, and leaves it as it is with probability (1 - p).
    """
    def mask(self, predicates: set[GroundedPredicate], masking_proba: float = 0.3,
             *args, **kwargs) -> set[GroundedPredicate]:
        for predicate in predicates:
            if random.random() < masking_proba:
                predicate.is_masked = True
        return predicates


class PercentageMasking(MaskingStrategy):
    """
    This class masks some p percent of predicates from the set
    """
    def mask(self, predicates: set[GroundedPredicate], masking_ratio: float = 0.75, *args, **kwargs) -> set[GroundedPredicate]:
        sample_size = max(1, round(len(predicates) * masking_ratio))  # Ensure at least 1 element if p > 0
        sample = set(random.sample(list(predicates), sample_size))
        for predicate in sample:
            predicate.is_masked = True
        return predicates


# create a class which gets a set of predicates and masks them according to the masking strategy
class PredicateMasker:
    """
    This class is used to mask predicates based on a given masking strategy.
    """
    def __init__(self, seed: int = 42):
        self.masking_strategies = {
            MaskingType.RANDOM: RandomMasking(),
            MaskingType.PERCENTAGE: PercentageMasking()
        }
        self.seed = seed

    def mask(self, predicates: set[GroundedPredicate], masking_strategy: MaskingType = MaskingType.RANDOM,
             *args, **kwargs) -> set[GroundedPredicate]:
        random.seed(self.seed) # Ensure reproducibility of masking
        return self.masking_strategies[masking_strategy].mask(predicates, *args, **kwargs)
