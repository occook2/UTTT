from uttt.env.state import UTTTEnv, action_to_rc, rc_to_action

def test_action_mapping_bijection():
    env = UTTTEnv(seed=7)
    env.reset()
    for a in range(81):
        r, c = action_to_rc(a)
        a2 = rc_to_action(r, c)
        assert a2 == a, f"rc<->action bijection failed at {a} -> ({r},{c}) -> {a2}"
