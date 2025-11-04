
import random
from uttt.env.state import UTTTEnv
from uttt.agents.base import Agent


class RandomAgent(Agent):
    def select_action(self, env: UTTTEnv) -> int:
        return (random.choice(env.legal_actions())), [], []
