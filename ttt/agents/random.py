
import random
from ttt.env.state import TTTEnv
from ttt.agents.base import Agent


class RandomAgent(Agent):
    def select_action(self, env: TTTEnv) -> int:
        return random.choice(env.legal_actions()), [], []
