import random
from uttt.env.state import UTTTEnv


class RandomAgent:
    def select_action(self, env: UTTTEnv) -> int:
        return random.choice(env.legal_actions())
