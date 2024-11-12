from typing import List, Sequence

from src.action_model.action_model import ActionModel
from src.fluent_classification.fluent_mapping import FluentMapping
from src.types import Image, Action, Image, State
from src.types.image_action_triplet import ImageActionTriplet


class StateConstructor:
    """
    This class is responsible for constructing states out of image sequences, using
    the actions observed for altering between "states" represented by the images.

    Attributes:
        - fluent_mapping (FluentMapping): holds, for each image, which fluents are T/F,
        and it is being updated by every round of the containing simulator.
        - action_model (ActionModel): holds the mostly updated action model currently known
        by the containing simulator.

    Methods:
        - construct_states (action_sequences, image_sequence): generates the states out of the
        given actions and images using the fluent mapping and the action model.
    """
    def __init__(self, fluent_mapping: FluentMapping, action_model: ActionModel):
        # TODO: consider not setting these as class attributes becuase in each fluent mapping/AM update it has to
        # be reflected in the class so it doesn't fit the purpose of being class props.
        # moreover - the fluent_mapping and AM should be props of Main Alg class, and being injected to the rest of the algorithms.
        self.fluent_mapping = fluent_mapping
        self.action_model = action_model

    def construct_states(self, image_trajectory: Sequence[ImageActionTriplet]) -> Sequence[State]:
        """
        This function constructs states out of the given image and action sequences,
        using the current fluent mapping and the current action model.
        :param image_trajectory: the sequence of images representing states in the current trajectory
        """

        states: List[State] = []
        for triplet in image_trajectory:
            prev_state = State(self.fluent_mapping.get_fluents(triplet.prev_image)) #TODO: the get_fluent mechanism should be implemented

            # TODO: we currently skip this stage as we are going to use SAM as a partial action model, and SAM could only be confident on effects and not on preconditions
            # state_i.add_fluents(self.action_model.get(triplet.action).preconditions)

            next_state = State(self.fluent_mapping.get_fluents(triplet.next_image))
            next_state.add_fluents(self.action_model.get(triplet.action).add_effects)
            next_state.del_fluents(self.action_model.get(triplet.action).del_effects)

        #TODO: check if we must have the initial state so we can construct all the states
        #TODO: iteratively - in particular, check if we can somehow generate the initial state

        #TODO: handle the updateing of the same state in consecutive stages of the algotihm

        # right now it ain't returning anything
        return states
