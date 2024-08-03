


def get_initial_action_model() -> ActionModel:
    """
    This function aims to being the inital action model.
    This initial model might be an empty model which we have to deduce solely from the videos,
    and it can be partially defined to start with some knowledge about the domain.
    """
    raise NotImplementedError

def extract_images_from_video(video_path: str) -> List[Image]:
    """
    this function extracts images from a video, in a pre-determined fps to represent the video with proper images
    containing much data to preserve video's information.
    :param video_path: the path to the video file.
    #TODO: right now, we can assume that the video is in a .mp4 format, later on we might want to support other formats as well
    #TODO: use the implementation of gym for this one, it should be pretty good, we may also use their output as a reference to the image class properties (and methods)
    """
    raise NotImplementedError


def construct_states(action_model: ActionModel, action_stream: List[Action], image_stream: List[Image], fluent_mapping: FluentMapping) -> List[State]:
    """
    This function is the first function in the Notability file, constructing the states out of the current given action model,
    the first state and the images from the video and the currently-learnt fluent mapping.
    :param action_model: the PAM to work with
    :param action_stream: the sequence of actions in the current trajectory
    :param image_stream: the sequence of images representing states in the current trajectory
    :param fluent_mapping: the fluent mapping learnt so far by the classifiers
    """
    raise NotImplementedError


def learn_fluents(image_state_pairs: List[ImageStatePair]=None) -> FluentMapping:
    #TODO later: in a case of a classifier, we can make a binary classifier for each binary fluent, or at least for each "dynamic" fluent (meaning, fluent which can change its state during an episode)
    """
    this function learns fluents existence for each image in the image_stream, in order to be used in the process
    to create states out of images.
    This one might be a simple fluent mapping (extracting manually all the fluents from an image in a pre-known
    domain) or, for example, a smarter classifier based on image and object detection.
    :param image_state_pairs:  pairs of image and the fluentic state that we believe representing the image
    """
    raise NotImplementedError



def construct_action_model(action_triplets: List[ActionTriplet]) -> ActionModel:
    """
    #TODO later: decide whether we want this action model to be safe, e.g. using SAM
    :param action_triplets: a list of ActionTriplets representing the video trajectory's transitions.
    """
    raise NotImplementedError


def enrich_action_triplets(action_triplets: List[ActionTriplet], action_model: ActionModel) -> List[ActionTriplet]:
    """
    this function gets action triplets as an input, feeds them into an action model and updates the states of
    each triplet using the action model's information (for example, if it can somehow affect the fluents of at least
    one of the states in the triplet)
    :rtype: List[ActionTriplet], the (potentially) updated triplets
    :param action_triplets: the action triplets to be enriched
    :param action_model: the action model to be used for the update
    """
    raise NotImplementedError



def update_image_state_pairs(image_state_pairs: List[ImageStatePair], action_triplets: List[ActionTriplet]) -> List[ImageStatePair]:
    raise NotImplementedError

#TODO: we may not need the initial state, though it is good we have it for some initials/sanity checks from the fluent mapping
def learn_action_model(initial_state: State, video_path: str, action_stream: List[Action]) -> ActionModel:
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

    #1
    action_model: ActionModel = get_initial_action_model()
    fluent_mapping: FluentMapping = learn_fluents()
    model_changes = True

    #2
    #the cycle runs until "convergence", meaning the action model has been fixed
    #TODO later actually we can just run "while True" and if the action models are equal - just break. without the var..
    while model_changes:
        #3
        constructed_states: List[State] = construct_states(action_model=action_model,
                                                           action_stream=action_stream,
                                                           image_stream=image_stream,
                                                           fluent_mapping=fluent_mapping)

        #4 TODO extract this one to a function named "extract_action_triplets"
        action_triplets: List[ActionTriplet] = [ActionTriplet(constructed_states[i], action_stream[i], constructed_states[i+1]) for i in range(len(action_stream))]

        #5
        new_action_model: ActionModel = construct_action_model(action_triplets)

        #6 TODO later: implement an "equals" method for ActionModel objects
        if new_action_model == action_model:
            #7
            model_changes = False
            #8
            break

        #9
        action_model = new_action_model

        #10
        image_state_pairs: List[ImageStatePair] = [ImageStatePair(image, state) for image, state in zip(image_stream, constructed_states)]

        #11 TODO: this is optional and not necessary for the baseline (and the entire algorithm) to work, its a NiceToHave
        action_triplets = enrich_action_triplets(action_triplets, action_model)

        #12 TODO: this is optional and not necessary for the baseline (and the entire algorithm) to work, its a NiceToHave
        image_state_pairs = update_image_state_pairs(image_state_pairs, action_triplets)

        #13
        fleunt_mapping = learn_fluents(image_state_pairs)

    #14
    return action_model




if __name__ == '__main__':
