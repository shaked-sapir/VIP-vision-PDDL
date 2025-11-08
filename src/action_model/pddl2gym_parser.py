import re

from src.fluent_classification.base_fluent_classifier import PredicateTruthValue

NEGATION_PREFIX = "NOT"
UNKNOWN_PREFIX = "UNK"


def parse_image_predicate_to_gym(predicate_str: str, is_holding_in_image: PredicateTruthValue) -> str:
    """
    Parse a predicate extracted from an image to pddlgym format.
    :param predicate_str: the string representing the predicate, e.g. `holding(e:block)`
    :param is_holding_in_image: whether the predicate holds in the image it was extracted from
    :return: updated string representing the predicate
    """
    return predicate_str if is_holding_in_image == PredicateTruthValue.TRUE\
        else f"{NEGATION_PREFIX} {predicate_str}" if is_holding_in_image == PredicateTruthValue.FALSE\
        else f"{UNKNOWN_PREFIX} {predicate_str}"


def is_positive_gym_predicate(predicate_str: str) -> bool:
    return NEGATION_PREFIX not in predicate_str and UNKNOWN_PREFIX not in predicate_str


def is_unknown_gym_predicate(predicate_str: str) -> bool:
    return UNKNOWN_PREFIX in predicate_str


def get_predicate_base_form(predicate_str: str) -> str:
    return predicate_str.replace(f"{NEGATION_PREFIX} ", "").replace(f"{UNKNOWN_PREFIX} ", "")


def pddlplus_to_gym_predicate(s: str) -> str:
    # example for formats: (on a - block b - block) -> on(a:block,b:block)
    pred = re.search(r'^\(\s*([^\s()]+)', s).group(1)
    args = re.findall(r'([^\s()]+)\s*-\s*([^\s()]+)', s)
    return f"{pred}({','.join(f'{n}:{t}' for n, t in args)})"


def negate_str_predicate(predicate_str: str) -> str:
    if '(not ' in predicate_str:
        return predicate_str.replace('(not ', '')[:-1]
    else:
        return f"(not {predicate_str})"
