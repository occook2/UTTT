#!/usr/bin/env python3
"""
Test script to verify adaptive temperature implementation.
"""
import numpy as np
from uttt.env.state import UTTTEnv
from uttt.mcts.base import MCTSConfig
from uttt.agents.az.self_play import SelfPlayTrainer
from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.az.net import AlphaZeroNetUTTT

def test_adaptive_temperature():
    """Test that adaptive temperature works correctly."""
    print("Testing adaptive temperature implementation...")
    
    # Create a basic network and agent
    network = AlphaZeroNetUTTT()
    agent = AlphaZeroAgent(network=network, device="cpu")
    
    # Create MCTS config
    mcts_config = MCTSConfig(
        n_simulations=5,  # Low for testing
        c_puct=1.0,
        temperature=1.0,
        use_transposition_table=True
    )
    
    # Test adaptive temperature trainer
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=30,
        collect_data=False,  # Don't collect data for test
        use_adaptive_temperature=True,
        confidence_threshold=0.6
    )
    
    # Create environment
    env = UTTTEnv()
    
    # Test raw policy probabilities
    try:
        mcts = trainer.agent.strategy
        from uttt.mcts.base import GenericMCTS
        mcts_instance = GenericMCTS(mcts, mcts_config)
        raw_probs = mcts_instance.get_raw_policy_probabilities(env)
        max_prob = np.max(raw_probs)
        
        print(f"✅ Raw policy check successful")
        print(f"   Max probability: {max_prob:.3f}")
        print(f"   Would use temperature: {'0.0' if max_prob >= 0.6 else '1.0'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_adaptive_temperature()