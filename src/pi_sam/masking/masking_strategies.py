import inspect
import random
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple

from pddl_plus_parser.models import GroundedPredicate

from src.fluent_classification.base_fluent_classifier import PredicateTruthValue


class MaskingType(str, Enum):
    RANDOM = "random"
    PERCENTAGE = "percentage"
    UNCERTAIN = "uncertain"


class MaskingStrategy(ABC):
    """
    This class serves as a base class for masking a set of predicates based on some logic/strategy.
    """
    name: str

    @abstractmethod
    def mask(self, predicates: set[GroundedPredicate],
             *args, **kwargs) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        """
            Masks a set of predicates based on the specific masking strategy.
            :param predicates: The set of grounded predicates from which to mask.
            :param args: Additional positional arguments for the specific masking strategy.
            :param kwargs: Additional keyword arguments for the specific masking strategy.
            :return: A tuple containing the set of masked predicates and the set of unmasked predicates:
                - The first set contains the predicates that were masked.
                - The second set contains the predicates that were not masked.
        """
        raise NotImplementedError

    @staticmethod
    def _split_masked_and_unmasked(
            predicates: set[GroundedPredicate]) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        """
        Splits the predicates into masked and unmasked sets based on their `is_masked` attribute.

        :param predicates: The set of grounded predicates to split.
        :return: A tuple containing two sets:
            - The first set contains the masked predicates.
            - The second set contains the unmasked predicates.
        """
        masked = {predicate for predicate in predicates if predicate.is_masked}
        unmasked = {predicate for predicate in predicates if not predicate.is_masked}
        return masked, unmasked

    def validate_strategy_kwargs(self, strategy_kwargs: dict):
        """
        Validates the parameters provided for the masking strategy against the expected parameters of the mask method.

        :param strategy_kwargs: The keyword arguments to validate.
        :raises ValueError: If any unknown parameters are provided.
        """

        sig = inspect.signature(self.mask)

        valid_params = set(sig.parameters.keys()) - {"self", "predicates", "args", "kwargs"}
        unknown_params = set(strategy_kwargs.keys()) - valid_params

        if unknown_params:
            raise ValueError(
                f"Invalid parameter(s) for strategy '{self.name}': {unknown_params}. "
                f"Expected one or more of: {valid_params}"
            )


class RandomMaskingStrategy(MaskingStrategy):
    """
    this class masks a certain predicate with probability p, and leaves it as it is with probability (1 - p).
    """
    name: str = MaskingType.RANDOM

    def mask(self, predicates: set[GroundedPredicate], masking_proba: float = 0.3,
             *args, **kwargs) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        print(f"using {self.name} masking strategy with probability {masking_proba}")
        for predicate in predicates:
            if random.random() < masking_proba:
                predicate.is_masked = True
        return self._split_masked_and_unmasked(predicates)


class PercentageMaskingStrategy(MaskingStrategy):
    """
    This class masks some p percent of predicates from the set
    """
    name: str = MaskingType.PERCENTAGE

    def mask(self, predicates: set[GroundedPredicate], masking_ratio: float = 0.75,
             *args, **kwargs) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        print(f"using {self.name} masking strategy with ratio {masking_ratio}")
        sample_size = max(1, round(len(predicates) * masking_ratio))  # Ensure at least 1 element if p > 0
        sample = set(random.sample(list(predicates), sample_size))
        for predicate in sample:
            predicate.is_masked = True
        return self._split_masked_and_unmasked(predicates)


class UncertainMaskingStrategy(MaskingStrategy):
    """
    This class gets the probability for each predicate to be true,
    and a threshold for uncertainty, and masks the predicate if its probability is within
    the threshold.
    """
    name: str = MaskingType.UNCERTAIN

    def mask(self, predicates: set[GroundedPredicate], predicate_truth_values: dict[str, PredicateTruthValue] = None,
             *args, **kwargs) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        if predicate_truth_values is None:
            raise ValueError("predicate_truth_values must be provided for UncertainMaskingStrategy")

        print(f"using {self.name} masking strategy")
        for predicate in predicates:
            if predicate_truth_values[predicate.lifted_untyped_representation] == PredicateTruthValue.UNCERTAIN:
                predicate.is_masked = True
        return self._split_masked_and_unmasked(predicates)
