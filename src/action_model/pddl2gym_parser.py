from src.fluent_classification.base_fluent_classifier import PredicateTruthValue

NEGATION_PREFIX = "Not"
UNKNWON_PREFIX = "Unknown"


def parse_image_predicate_to_gym(predicate_str: str, is_holding_in_image: PredicateTruthValue) -> str:
    """
    Parse a predicate extracted from an image to pddlgym format.
    :param predicate_str: the string representing the predicate, e.g. `holding(e:block)`
    :param is_holding_in_image: whether the predicate holds in the image it was extracted from
    :return: updated string representing the predicate
    """
    return predicate_str if is_holding_in_image == PredicateTruthValue.TRUE\
        else f"{NEGATION_PREFIX}{predicate_str}" if is_holding_in_image == PredicateTruthValue.FALSE\
        else f"{UNKNWON_PREFIX}{predicate_str}"


def is_positive_gym_predicate(predicate_str: str) -> bool:
    return NEGATION_PREFIX not in predicate_str and UNKNWON_PREFIX not in predicate_str


def is_unknown_gym_predicate(predicate_str: str) -> bool:
    return UNKNWON_PREFIX in predicate_str
