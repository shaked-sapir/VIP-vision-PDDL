from gym import Env, logger, spaces, utils

class Gym2PddlDomainParser:
    def __init__(self, env: Env):
        self.env = env

    def parse_action(self, action: Action):

    def parse_state(self, state: State):
