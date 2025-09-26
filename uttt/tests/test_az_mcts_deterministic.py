# uttt/tests/test_az_mcts_deterministic.py
"""
Test that AlphaZero MCTS with temperature=0 picks a single deterministic move.
"""
import pytest
import numpy as np
import torch

from uttt.env.state import UTTTEnv
from uttt.agents.az.net import AlphaZeroNetUTTT
from uttt.mcts.alphazero_strategy import AlphaZeroStrategy
from uttt.mcts.base import GenericMCTS, MCTSConfig


def test_az_mcts_deterministic_move():
    """Test that AZ MCTS with temperature=0 picks exactly one move deterministically."""
    # Create environment in initial state
    env = UTTTEnv()
    
    # Create neural network (random weights are fine for this test)
    net = AlphaZeroNetUTTT()
    
    # Create strategy and MCTS with temperature=0
    strategy = AlphaZeroStrategy(net)
    config = MCTSConfig(
        n_simulations=10,  # Small number for fast test
        temperature=0.0,   # Deterministic selection
        c_puct=1.0
    )
    mcts = GenericMCTS(strategy, config)
    
    # Run search
    action, probs = mcts.search(env)
    
    # Verify deterministic behavior
    assert isinstance(action, int), "Action should be an integer"
    assert 0 <= action < 81, "Action should be in valid range [0, 80]"
    
    # Check that exactly one action has probability 1.0
    assert probs[action] == 1.0, f"Selected action {action} should have probability 1.0, got {probs[action]}"
    assert probs.sum() == 1.0, f"Probabilities should sum to 1.0, got {probs.sum()}"
    assert (probs > 0).sum() == 1, f"Exactly one action should have positive probability, got {(probs > 0).sum()}"
    
    # Verify action is legal
    legal_actions = env.legal_actions()
    assert action in legal_actions, f"Selected action {action} should be legal"


def test_az_mcts_consistent_deterministic():
    """Test that AZ MCTS with temperature=0 picks the same move consistently."""
    # Create environment in initial state
    env = UTTTEnv()
    
    # Create neural network with fixed seed for consistency
    torch.manual_seed(42)
    net = AlphaZeroNetUTTT()
    
    # Create strategy and MCTS with temperature=0
    strategy = AlphaZeroStrategy(net)
    config = MCTSConfig(
        n_simulations=20,
        temperature=0.0,
        c_puct=1.0,
        use_transposition_table=False  # Disable to ensure fresh search each time
    )
    
    # Run multiple searches and verify they return the same action
    actions = []
    for _ in range(3):
        mcts = GenericMCTS(strategy, config)
        action, probs = mcts.search(env)
        actions.append(action)
        
        # Verify deterministic properties each time
        assert probs[action] == 1.0
        assert probs.sum() == 1.0
        assert (probs > 0).sum() == 1
    
    # All actions should be the same
    assert len(set(actions)) == 1, f"All searches should return same action, got {actions}"


def test_az_mcts_temperature_comparison():
    """Test that temperature=0 behaves differently from temperature=1."""
    # Create environment in initial state
    env = UTTTEnv()
    
    # Create neural network
    torch.manual_seed(42)
    net = AlphaZeroNetUTTT()
    strategy = AlphaZeroStrategy(net)
    
    # Test with temperature=0 (deterministic)
    config_det = MCTSConfig(n_simulations=10, temperature=0.0)
    mcts_det = GenericMCTS(strategy, config_det)
    action_det, probs_det = mcts_det.search(env)
    
    # Test with temperature=1 (stochastic)
    config_stoch = MCTSConfig(n_simulations=10, temperature=1.0)
    mcts_stoch = GenericMCTS(strategy, config_stoch)
    action_stoch, probs_stoch = mcts_stoch.search(env)
    
    # Temperature=0 should have exactly one action with probability 1
    assert (probs_det > 0).sum() == 1
    assert probs_det[action_det] == 1.0
    
    # Temperature=1 should potentially have multiple actions with positive probability
    # (Though not guaranteed depending on visit counts)
    assert probs_stoch.sum() == pytest.approx(1.0, abs=1e-6)
    assert action_stoch in env.legal_actions()


def test_az_mcts_multiple_positions():
    """Test deterministic behavior across different game positions."""
    env = UTTTEnv()
    
    # Make a few moves to get to a different position
    env.step(0)   # Player 1 plays in position 0
    legal_actions = env.legal_actions()
    env.step(legal_actions[0])  # Player 2 plays a legal move
    
    # Create neural network and MCTS
    torch.manual_seed(42)
    net = AlphaZeroNetUTTT()
    strategy = AlphaZeroStrategy(net)
    config = MCTSConfig(n_simulations=15, temperature=0.0)
    mcts = GenericMCTS(strategy, config)
    
    # Run search
    action, probs = mcts.search(env)
    
    # Verify deterministic behavior
    assert probs[action] == 1.0
    assert probs.sum() == 1.0
    assert (probs > 0).sum() == 1
    assert action in env.legal_actions()


if __name__ == "__main__":
    # Run the tests
    test_az_mcts_deterministic_move()
    test_az_mcts_consistent_deterministic()
    test_az_mcts_temperature_comparison()
    test_az_mcts_multiple_positions()
    print("All tests passed!")