# uttt/tests/test_az_agent_play_game.py
"""
Integration test: AlphaZeroAgent plays a full game against RandomAgent.
Verifies that all moves are legal and the game completes.
"""
import pytest
from uttt.env.state import UTTTEnv
from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.random import RandomAgent


def test_az_agent_vs_random_full_game():
    env = UTTTEnv()
    from uttt.mcts.base import MCTSConfig
    az_agent = AlphaZeroAgent(mcts_config=MCTSConfig(n_simulations=2))
    random_agent = RandomAgent()

    agents = [az_agent, random_agent]
    move_history = []

    while not env.terminated:
        current_agent = agents[(env.player - 1) % 2]
        action = current_agent.select_action(env)
        legal_actions = env.legal_actions()
        assert action in legal_actions, f"Agent selected illegal move: {action}"
        env.step(action)
        move_history.append(action)

    # Game should terminate
    assert env.terminated, "Game did not terminate"
    # There should be at least one move
    assert len(move_history) > 0, "No moves were made"
    # Winner should be valid (0 = draw, 1 = player 1, 2 = player 2)
    winner = env._macro_winner()
    assert winner in [0, 1, 2], f"Invalid winner: {winner}"
    print(f"Game finished in {len(move_history)} moves. Winner: {winner}")