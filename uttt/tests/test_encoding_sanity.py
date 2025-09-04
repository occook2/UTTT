# uttt/tests/test_encoding_sanity.py
import numpy as np
from uttt.env.state import UTTTEnv, rc_to_action, StepResult

# Plane indices match your encode(): [curr, X, O, legal, macroX, macroO, last]
PLANE_CURR  = 0
PLANE_X     = 1
PLANE_O     = 2
PLANE_LEGAL = 3
PLANE_MX    = 4
PLANE_MO    = 5
PLANE_LAST  = 6

rc = rc_to_action

def _step(env: UTTTEnv, a: int):
    """Normalize StepResult to (obs, reward, terminated, info)."""
    out = env.step(a)
    assert isinstance(out, StepResult)
    return out.obs, out.reward, out.terminated, out.info

def _mask2d(mask: np.ndarray) -> np.ndarray:
    # Your env returns (9,9) int8, but keep robust to flat.
    return mask if mask.ndim == 2 else mask.reshape(9, 9)

def test_obs_shape_and_curr_player_plane():
    env = UTTTEnv(seed=42)
    obs = env.reset()
    assert obs.shape == (7, 9, 9)

    if env.player == +1:
        assert np.all(obs[PLANE_CURR] == 1), "PLANE_CURR should be all ones for X to move"
    else:
        assert np.all(obs[PLANE_CURR] == 0), "PLANE_CURR should be all zeros for O to move"

def test_legal_plane_matches_mask_and_lastmove_updates():
    env = UTTTEnv(seed=99)
    obs0 = env.reset()

    # legal plane equals legal_actions_mask
    mask0 = _mask2d(env.legal_actions_mask())
    # obs[PLANE_LEGAL] is float32; mask is int8 → compare values, not dtype
    assert np.array_equal(obs0[PLANE_LEGAL], mask0.astype(obs0.dtype))

    # last-move plane all zeros right after reset
    assert np.count_nonzero(obs0[PLANE_LAST]) == 0

    # Play a move and verify last-move one-hot
    a = rc(0, 8)
    obs1, r1, term1, info1 = _step(env, a)
    r, c = 0, 8
    assert term1 is False
    assert r1 == 0.0
    assert obs1[PLANE_LAST, r, c] == 1
    assert np.count_nonzero(obs1[PLANE_LAST]) == 1

    # Legal plane keeps matching after a step
    mask_after = _mask2d(env.legal_actions_mask())
    assert np.array_equal(obs1[PLANE_LEGAL], mask_after.astype(obs1.dtype))
