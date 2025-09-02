from uttt.env.state import UTTTEnv


def test_reset_and_legal_mask():
    env = UTTTEnv(seed=0)
    obs = env.reset()
    assert obs.shape == (7, 9, 9)
    mask = env.legal_actions_mask()
    assert mask.sum() == 81  # first move anywhere


def test_uttt_redirect_rule():
    env = UTTTEnv(seed=0)
    env.reset()
    env.step(4 * 9 + 4)  # (4,4) -> target micro-board (1,1)
    mask = env.legal_actions_mask()
    allowed = mask[3:6, 3:6]
    assert allowed.sum() == 8
    assert mask.sum() == 8
