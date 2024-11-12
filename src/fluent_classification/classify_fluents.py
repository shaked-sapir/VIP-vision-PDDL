from typing import Set, List, Any

from PIL import Image

from src.types import Fluent


# from src.types import Fluent


# TODO: define the type of ObjectDetector with a class and an interface.
# TODO: further define the Fluent class as we are going to need it in this method.
def classify_fluents(image: Image, object_detectors: List[Any], prop_map) -> Set[Fluent]:
    """
    This is the main algorithm for classifying fluents non/existence in an image using ad-hoc classifers
    and a predefined object detector.
    We assume that for each object type in the domain, we have a pretrained property detection model for each
    property of interest in that type, e.g. for the type "Box" we would have a pretrained model which is able
    to distinguish between open and closed boxes, so we can use it for the Open(Box) boolean state.
    :param image: the image to be classified
    :param object_detectors: the model for detecting objects in the picture
    :param prop_map: a container holding all the classifiers for the properties of interest for each object type
           in the domain.
           It is of the form {"type": {"prop": Classifier}}
    :return:
    """
    fluents = {}
    objects = [detector.detect_objects(image) for detector in object_detectors]
    for obj in objects:
        for prop, classifier in prop_map[obj.type]:
            if classifier.predict(prop):
                fluents.append(prop)
            else:
                fluents.append(not(prop))

    return fluents

