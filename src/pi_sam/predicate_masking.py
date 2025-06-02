import random
from typing import Set, List, Tuple

from pddl_plus_parser.models import GroundedPredicate, Observation, State

from src.pi_sam.masking import MaskingType, PercentageMaskingStrategy, RandomMaskingStrategy
from src.utils.pddl import get_state_grounded_predicates


class PredicateMasker:
    """
    This class is used to mask predicates based on a given masking strategy.
    """

    _masking_strategy: MaskingType
    _masking_kwargs: dict

    masking_strategies = {
        MaskingType.RANDOM: RandomMaskingStrategy(),
        MaskingType.PERCENTAGE: PercentageMaskingStrategy()
    }

    def __init__(self, seed: int = 42, masking_strategy: MaskingType = MaskingType.RANDOM,
                 masking_kwargs: dict = None):
        self.seed = seed
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

    def mask(self, predicates: set[GroundedPredicate]) -> Tuple[set[GroundedPredicate], set[GroundedPredicate]]:
        random.seed(self.seed)  # Ensures reproducibility of masking
        return self.masking_strategies[self._masking_strategy].mask(predicates, **self._masking_kwargs)

    def mask_state(self, state: State) -> Set[GroundedPredicate]:
        """
        Masks the predicates in the state based on the masking strategy.

        :param state: The state containing predicates to be masked.
        :return: The state with masked predicates and the set of masked predicates.
        """
        grounded_predicates = get_state_grounded_predicates(state)
        masked_predicates, unmasked_predicates = self.mask(grounded_predicates)
        return masked_predicates

    def mask_observation(self, observation: Observation) -> List[set[GroundedPredicate]]:
        """
        Masks the predicates in the observation's states based on the masking strategy.
        Note that for each 2 consecutive components (c, c'), it holds that c.next_state == c'.previous_state,
        so they should be masked in the same way. Therefore, we generate the masking info only once for each component.

        :param observation: The observation containing predicates to be masked.
        :return: The observation with masked predicates.
        """

        # Mask the initial state
        masking_info = [self.mask_state(observation.components[0].previous_state)]

        # Mask the next state for each component in the observation
        for i in range(len(observation.components)):
            masking_info.append(self.mask_state(observation.components[i].next_state))

        return masking_info
