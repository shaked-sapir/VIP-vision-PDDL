NEGATION_PREFIX = "Not"


def parse_image_predicate_to_gym(predicate_str: str, is_holding_in_image: bool) -> str:
    """
    Parse a predicate extracted from an image to pddlgym format.
    :param predicate_str: the string representing the predicate, e.g. `holding(e:block)`
    :param is_holding_in_image: whether the predicate holds in the image it was extracted from
    :return: updated string representing the predicate
    """
    return predicate_str if is_holding_in_image else f"{NEGATION_PREFIX}{predicate_str}"


def is_positive_gym_predicate(predicate_str: str) -> bool:
    return NEGATION_PREFIX not in predicate_str
