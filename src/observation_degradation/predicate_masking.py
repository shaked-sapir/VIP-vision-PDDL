import random
from typing import List, Set, Tuple

from pddl_plus_parser.models import GroundedPredicate, Observation, State

from src.observation_degradation.masking import (
    MaskingType,
    PercentageMaskingStrategy,
    RandomMaskingStrategy,
    UncertainMaskingStrategy,
)
from src.utils.pddl import get_state_grounded_predicates


class PredicateMasker:
    """
    This class is used to mask predicates based on a given masking strategy.
    """

    _masking_strategy: MaskingType
    _masking_kwargs: dict

    masking_strategies = {
        MaskingType.RANDOM: RandomMaskingStrategy(),
        MaskingType.PERCENTAGE: PercentageMaskingStrategy(),
        MaskingType.UNCERTAIN: UncertainMaskingStrategy(),
    }

    def __init__(self, seed: int = 42, masking_strategy: MaskingType = MaskingType.RANDOM,
                 masking_kwargs: dict = None):
        self.seed = seed
        self._call_count: int = 0
        # TODO: consider making the kwargs with * as I did with the sam_learning
        self.set_masking_strategy(masking_strategy, **(masking_kwargs or self._default_params_for(masking_strategy)))

    @staticmethod
    def _default_params_for(strategy: MaskingType) -> dict:
        if strategy == MaskingType.RANDOM:
            return {"masking_proba": 0.3}
        elif strategy == MaskingType.PERCENTAGE:
            return {"masking_ratio": 0.3}
        else:
            return {}

    def set_masking_strategy(self, masking_strategy: MaskingType, **kwargs):
        """
        Sets the masking strategy to be used for masking predicates.

        :param masking_strategy: The strategy to use for masking.
        :param kwargs: Additional parameters for the selected masking strategy.
        """
        self._masking_strategy = masking_strategy
        self._masking_kwargs = kwargs or self._default_params_for(masking_strategy)
        self.masking_strategies[masking_strategy].validate_strategy_kwargs(self._masking_kwargs)

    def reset(self) -> None:
        """Reset the internal call counter so the next mask() call reproduces the original sequence."""
        self._call_count = 0

    def mask(self, predicates: set[GroundedPredicate]) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        random.seed(self.seed + self._call_count)
        self._call_count += 1
        return self.masking_strategies[self._masking_strategy].mask(predicates, **self._masking_kwargs)

    def mask_state(self, state: State) -> Tuple[Set[GroundedPredicate], Set[GroundedPredicate]]:
        """
        Mask predicates in *state* according to the configured strategy.

        Selected predicates are marked in-place (``is_masked=True`` on the
        state's ``GroundedPredicate`` objects).

        :param state: The state whose predicates should be masked.
        :return: ``(masked_predicates, unmasked_predicates)`` after applying
            the strategy.
        """
        grounded_predicates = get_state_grounded_predicates(state)
        return self.mask(grounded_predicates)

    def mask_observation(self, observation: Observation) -> List[set[GroundedPredicate]]:
        """
        Masks the predicates in the observation's states based on the masking strategy.
        Note that for each 2 consecutive components (c, c'), it holds that c.next_state == c'.previous_state,
        so they should be masked in the same way. Therefore, we generate the masking info only once for each component.

        :param observation: The observation containing predicates to be masked.
        :return: The observation with masked predicates.
        """

        # Mask the initial state
        masked, _ = self.mask_state(observation.components[0].previous_state)
        masking_info = [masked]

        # Mask the next state for each component in the observation
        for i in range(len(observation.components)):
            masked, _ = self.mask_state(observation.components[i].next_state)
            masking_info.append(masked)

        return masking_info
