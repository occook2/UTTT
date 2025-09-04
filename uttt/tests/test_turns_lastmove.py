# uttt/tests/test_turns_lastmove.py
import numpy as np
from uttt.env.state import UTTTEnv, rc_to_action, StepResult

rc = rc_to_action  # rc(r, c) -> action id

def _step(env: UTTTEnv, a: int):
    """Normalize StepResult to (obs, reward, terminated, info)."""
    out = env.step(a)
    assert isinstance(out, StepResult)
    return out.obs, out.reward, out.terminated, out.info

def _mask2d(mask: np.ndarray) -> np.ndarray:
    # Your env returns a (9,9) int8 already, but keep this robust.
    return mask if mask.ndim == 2 else mask.reshape(9, 9)

def test_player_flips_and_last_move_tracks():
    env = UTTTEnv(seed=123)
    obs0 = env.reset()
    p0 = env.player

    # First move at center
    a1 = rc(4, 4)
    obs1, r1, term1, info1 = _step(env, a1)
    assert term1 is False, "Center move at start should not be terminal"
    assert r1 == 0.0
    assert env.player == -p0, "player should flip after a legal non-terminal move"
    assert env.last_move == (4, 4), "last_move should equal the action just played"

    # Next move must be in target micro (1,1): rows 3..5, cols 3..5
    mask = _mask2d(env.legal_actions_mask())
    rr, cc = np.where(mask[3:6, 3:6] == 1)
    # Pick the first legal that isn't (4,4)
    chosen = None
    for i in range(len(rr)):
        r, c = 3 + int(rr[i]), 3 + int(cc[i])
        if (r, c) != (4, 4):
            chosen = (r, c)
            break
    assert chosen is not None, "Expected at least one legal move in (1,1) other than (4,4)"
    r2, c2 = chosen

    obs2, r2w, term2, info2 = _step(env, rc(r2, c2))
    assert term2 is False
    assert r2w == 0.0
    assert env.player == p0, "player should flip back after second legal move"
    assert env.last_move == (r2, c2), "last_move should track the most recent move"
