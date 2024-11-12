from gym import Env, logger, spaces, utils

from src.types import Action, State


class Gym2PddlDomainParser:
    def __init__(self, env: Env):
        self.env = env

    def parse_action(self, action: Action):
        pass

    def parse_state(self, state: State):
        pass
