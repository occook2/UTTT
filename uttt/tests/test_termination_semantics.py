import pytest
import numpy as np
from uttt.env.state import UTTTEnv

rc = lambda r, c: r * 9 + c

def _step(env, a):
    out = env.step(a)
    if len(out) == 5:
        obs, r, term, trunc, info = out
        term = bool(term or trunc)
        return obs, r, term, info
    return out

@pytest.mark.xfail(reason="Enable once you add a tiny state setter or a deterministic quick-win helper")
def test_win_is_terminal_and_reward_sign():
    """
    This test expects a quick state-construction helper (e.g., env.set_state(...))
    that yields a position where X has a forced immediate win on the next move.
    Then: step -> terminated True, reward == +1 if X, -1 if O.
    """
    env = UTTTEnv(seed=5)
    # Example (pseudo):
    # env.set_state(board_x=..., board_o=..., player=+1, last_move=...)
    # obs, r, term, _ = _step(env, winning_action)
    # assert term is True
    # assert r == +1
    assert False, "Provide or use a helper to build a one-move win position"

@pytest.mark.xfail(reason="Enable after the terminal test above is reliable")
def test_step_after_terminal_raises():
    env = UTTTEnv(seed=6)
    # Build a terminal position first (see test above), then:
    # with pytest.raises(AssertionError):
    #     _step(env, some_action)
    assert False, "Call step after terminal and expect an assertion/exception"
