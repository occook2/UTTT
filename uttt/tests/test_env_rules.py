import pytest
import numpy as np
from uttt.env.state import UTTTEnv

# handy helper for linear <-> (r,c)
rc = lambda r, c: r * 9 + c


def test_reset_and_legal_mask():
    env = UTTTEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (7, 9, 9)

    mask = env.legal_actions_mask()
    assert mask.shape == (9, 9)
    assert mask.sum() == 81  # first move anywhere


def test_uttt_redirect_rule_center_targets_1_1():
    """
    Play (4,4) so the next player is redirected to micro (1,1) i.e., rows 3..5, cols 3..5.
    Since (4,4) is now occupied, exactly 8 legals should remain there.
    """
    env = UTTTEnv(seed=0)
    env.reset()

    env.step(rc(4, 4))  # (4,4) -> target micro-board (1,1)

    mask = env.legal_actions_mask().reshape(9, 9)
    allowed = mask[3:6, 3:6]  # the (1,1) micro-board region
    assert allowed.sum() == 8
    assert mask.sum() == 8


def test_illegal_on_occupied_square_raises():
    """Trying to play the same square twice should raise ValueError."""
    env = UTTTEnv(seed=1)
    env.reset()

    a = rc(4, 4)
    env.step(a)  # occupy center once

    with pytest.raises(ValueError):
        env.step(a)  # attempt to play on occupied square again


def test_illegal_outside_target_micro_raises():
    """
    After (4,4), only the (1,1) micro is legal. Pick any empty square outside (3..5,3..5)
    and ensure the env rejects it.
    """
    env = UTTTEnv(seed=2)
    env.reset()
    env.step(rc(4, 4))  # redirects to (1,1)

    mask = env.legal_actions_mask().reshape(9, 9)
    # find any empty legal-looking coordinate OUTSIDE the target micro
    # (we actually want an *illegal* attempt, so choose a clearly outside cell)
    r, c = 0, 0  # top-left corner, definitely not in rows 3..5, cols 3..5
    bad_action = rc(r, c)

    # Sanity: it should be illegal by mask
    assert mask[r, c] == 0

    with pytest.raises(ValueError):
        env.step(bad_action)
