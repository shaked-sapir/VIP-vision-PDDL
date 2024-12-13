from typing import List, Sequence

from src.action_model.action_model import ActionModel
from src.fluent_classification.fluent_mapping import FluentMapping
from src.trajectory_handlers.image_trajectory_handler import ImageTrajectoryHandler
from src.types import State, Image, StateActionTriplet, ImageStatePair, ImageActionTriplet, Action


def get_initial_action_model() -> ActionModel:
    """
    This function aims to being the inital action model.
    This initial model might be an empty model which we have to deduce solely from the videos,
    and it can be partially defined to start with some knowledge about the domain.
    """
    raise NotImplementedError

def extract_images_from_video(video_path: str) -> Sequence[Image]:
    """
    this function extracts images from a video, in a pre-determined fps to represent the video with proper images
    containing much data to preserve video's information.
    :param video_path: the path to the video file.
    #TODO: right now, we can assume that the video is in a .mp4 format, later on we might want to support other formats as well
    #TODO: use the implementation of gym for this one, it should be pretty good, we may also use their output as a reference to the image class properties (and methods)
    """
    raise NotImplementedError


def construct_states(action_model: ActionModel, image_trajectory: Sequence[ImageActionTriplet], fluent_mapping: FluentMapping) -> Sequence[State]:
    """
    This function is the first function in the Notability file, constructing the states out of the current given action model,
    the first state and the images from the video and the currently-learnt fluent mapping.
    :param action_model: the PAM to work with
    :param image_trajectory: a sequence of image_action triplets, representing the observed trajectoru
    :param fluent_mapping: the fluent mapping learnt so far by the classifiers
    """
    raise NotImplementedError


def learn_fluents(image_state_pairs: Sequence[ImageStatePair]=None) -> FluentMapping:
    #TODO later: in a case of a classifier, we can make a binary classifier for each binary fluent, or at least for each "dynamic" fluent (meaning, fluent which can change its state during an episode)
    """
    this function learns fluents existence for each image in the image_stream, in order to be used in the process
    to create states out of images.
    This one might be a simple fluent mapping (extracting manually all the fluents from an image in a pre-known
    domain) or, for example, a smarter classifier based on image and object detection.
    :param image_state_pairs:  pairs of image and the fluentic state that we believe representing the image
    """
    # TODO: this is the algorithm pseudo-code, will need to be implemented properly. now these are placeholders
    for pair in image_state_pairs:
        img, state = pair.get_pair()
        img_fluents = classify_fluents(img)
        for fluent in state:
            if fluent in img_fluents:
                continue
            elif fluent not in img_fluents and not(fluent) not in img_fluents:
                img_fluents.append(fluent)
            else: # conflict
                resolve_conflict(fluent, img_fluents, state)





def construct_action_model(action_triplets: Sequence[StateActionTriplet], current_action_model: ActionModel = None) -> ActionModel:
    """
    #TODO later: decide whether we want this action model to be safe, e.g. using SAM
    :param action_triplets: a list of ActionTriplets representing the video trajectory's transitions.
    :param current_action_model: the current action model, whose behavior could be expanded using the action triplets.
           defaults to None
    """
    raise NotImplementedError


def enrich_action_triplets(action_triplets: Sequence[StateActionTriplet], action_model: ActionModel) -> Sequence[StateActionTriplet]:
    """
    this function gets action triplets as an input, feeds them into an action model and updates the states of
    each triplet using the action model's information (for example, if it can somehow affect the fluents of at least
    one of the states in the triplet)
    :rtype: List[ActionTriplet], the (potentially) updated triplets
    :param action_triplets: the action triplets to be enriched
    :param action_model: the action model to be used for the update
    """
    raise NotImplementedError


def update_image_state_pairs(image_state_pairs: Sequence[ImageStatePair], action_triplets: List[StateActionTriplet]) -> Sequence[ImageStatePair]:
    raise NotImplementedError

#TODO: we may not need the initial state, though it is good we have it for some initials/sanity checks from the fluent mapping
def learn_action_model(initial_state: State, video_path: str, action_stream: Sequence[Action]) -> ActionModel:
    """
    This is the main function which aims to learn the action model of the provided domain solely from the videos
    filmed in the domain.
    We have the initial state in our hands to hold the initial observations of the specific problem in the domain.
    aside of that, we have the video from which we have to deduce the changes in the domain over time and together with
    the provided actions - construct the rest of the states representing the trajectory captured by the video.

    We aim to start with the following:
    s_0, a_1, img_1, a_2, img_2, a_3, img_3, ... , a_n, img_n
    where each image represents a state which we want to represent in PDDL language,
    a_i is the action moved the domain from s_(i-1) to s_i.
    so at the end we have:
    s_0, a_1, s_1, a_2, s_2, a_3, s_3, ... , a_n, s_n.

    RECALL that actions[0] is actually the first action that should move us from s_0 to s_1, so the indices
    are being shifted according to that.


    :param initial_action_model: partial action model to start with, allowing us to deduce preconditions and effects
    for some of the states.
    :param initial_state: The initial state of the problem, allowing us to know the fluents holding at timestep 0
    to construct the rest of the states in the video.
    :param video_path: The path to the video holding the information about the trajectory we examine.
    Each image extracted from the video represents a state.
    :param actions: the actions moving us between two consecutive frames in the video.
    """

    #TODO later: we may want to inject it as a parameter, so we can "get an un-optimized action model as a parameter and our goal is to improve it
    image_stream: List[Image] = extract_images_from_video(video_path)

    # The following stages of the algorithm are numbered as they are in the research proposal, not in the Notability.
    #0: preparation
    action_model: ActionModel = get_initial_action_model()
    fluent_mapping: FluentMapping = learn_fluents()
    action_model_changes = True
    trajectory_handler: ImageTrajectoryHandler = ImageTrajectoryHandler()

    # TODO:  stick to the convention of sequence instead of list, as the resulted sequences should be read-only
    T_img: Sequence[ImageActionTriplet] = trajectory_handler.build_trajectory(image_stream, action_stream)

    # 1
    # the cycle runs until "convergence", meaning the action model has been fixed
    while action_model_changes:
        # 2
        constructed_states: Sequence[State] = construct_states(action_model=action_model,
                                                               image_trajectory=T_img,
                                                               fluent_mapping=fluent_mapping)

        # 3
        action_triplets: Sequence[StateActionTriplet] = trajectory_handler.build_trajectory(constructed_states, action_stream)

        # 4
        image_state_pairs: Sequence[ImageStatePair] = [ImageStatePair(image, state) for image, state in
                                                       zip(image_stream, constructed_states)]

        """
        These stages are nice-to-have but are not part of the original framework. TODO LATER: see if they are actually needed.

        # TODO: this is optional and not necessary for the baseline (and the entire algorithm) to work, its a NiceToHave
        action_triplets = enrich_action_triplets(action_triplets, action_model)

        # TODO: this is optional and not necessary for the baseline (and the entire algorithm) to work, its a NiceToHave
        image_state_pairs = update_image_state_pairs(image_state_pairs, action_triplets)
        """

        # 5
        # TODO: decide how to involve the classifers in this - as a variable injected to the function, or maybe
        #      turn the simulator into a class which holds the classifiers as a propoerty.
        #      note: this is the "enrich_classifiers_from_PAM" in the proposal. actually, it is not connected directly
        #      to the action model but to its resulted states... maybe it should be renamed properly.
        fluent_mapping = learn_fluents(image_state_pairs)

        # 6
        new_action_model: ActionModel = construct_action_model(action_triplets, action_model)

        # 7 TODO later: implement an "equals" method for ActionModel objects
        if new_action_model == action_model:
            # 8
            action_model_changes = False
            break

        # 11 (9-10 are endif + else)
        action_model = new_action_model
    # 14
    return action_model


if __name__ == '__main__':
    print("Starting simulator")
