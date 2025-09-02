from uttt.env.state import UTTTEnv
from uttt.agents.random import RandomAgent


def test_random_vs_random_finishes():
    env = UTTTEnv(seed=123)
    a1, a2 = RandomAgent(), RandomAgent()
    step = None
    for _ in range(400):
        step = env.step(a1.select_action(env))
        if step.terminated:
            break
        step = env.step(a2.select_action(env))
        if step.terminated:
            break
    assert step is not None and step.terminated
