import abc
import logging
from typing import List


class AlgorithmAdapter(abc.ABC):
    """Local lightweight replacement for AMLGym's AlgorithmAdapter.

    This avoids importing `amlgym.algorithms` package-level side effects
    (which eagerly imports optional algorithms and extra dependencies).
    """

    logger = logging.getLogger("AlgorithmAdapter")

    def __init__(self, **kwargs):
        self.logger = kwargs.pop("logger", logging.getLogger(self.__class__.__name__))
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abc.abstractmethod
    def learn(self, domain_path: str, trajectory_paths: List[str], use_problems: bool = False):
        raise NotImplementedError
