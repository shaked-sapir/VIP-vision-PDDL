from enum import Enum


class PredicateTruthValue(str, Enum):
    """Three-valued truth of a grounded predicate in a single observed state.

    ``UNCERTAIN`` is not a synonym for ``FALSE``: it suspends the closed-world
    assumption for that predicate and is what drives masking downstream.
    """

    TRUE = "true"
    FALSE = "false"
    UNCERTAIN = "uncertain"
