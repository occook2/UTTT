# example_usage.py
"""
Example showing how to use the new generic MCTS framework with different strategies.
"""

from uttt.env.state import UTTTEnv
from uttt.agents.az.net import AlphaZeroNetUTTT
from uttt.agents.heuristic import HeuristicAgent
from uttt.mcts.base import GenericMCTS, MCTSConfig
from uttt.mcts.alphazero_strategy import AlphaZeroStrategy, RandomStrategy
from uttt.mcts.heuristic_strategy import HeuristicStrategy


def example_alphazero_mcts():
    """Example using MCTS with neural network evaluation"""
    print("=== AlphaZero MCTS Example ===")
    
    # Create environment and neural network
    env = UTTTEnv()
    network = AlphaZeroNetUTTT()
    
    # Create AlphaZero strategy and MCTS
    strategy = AlphaZeroStrategy(network, device="cpu")
    config = MCTSConfig(n_simulations=50, c_puct=1.5, temperature=1.0)
    mcts = GenericMCTS(strategy, config)
    
    # Run search
    action, policy = mcts.search(env)
    print(f"Selected action: {action}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")


def example_heuristic_mcts():
    """Example using MCTS with heuristic evaluation"""
    print("\n=== Heuristic MCTS Example ===")
    
    # Create environment and heuristic agent
    env = UTTTEnv()
    heuristic_agent = HeuristicAgent()
    
    # Create heuristic strategy and MCTS
    strategy = HeuristicStrategy(heuristic_agent, rollout_depth=3, temperature=2.0)
    config = MCTSConfig(n_simulations=100, c_puct=0.8, temperature=0.5)
    mcts = GenericMCTS(strategy, config)
    
    # Run search
    action, policy = mcts.search(env)
    print(f"Selected action: {action}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")


def example_random_mcts():
    """Example using MCTS with random evaluation (baseline)"""
    print("\n=== Random MCTS Example ===")
    
    # Create environment
    env = UTTTEnv()
    
    # Create random strategy and MCTS
    strategy = RandomStrategy(seed=42)
    config = MCTSConfig(n_simulations=25, c_puct=1.0, temperature=1.0)
    mcts = GenericMCTS(strategy, config)
    
    # Run search
    action, policy = mcts.search(env)
    print(f"Selected action: {action}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {policy.sum():.3f}")


def compare_strategies():
    """Compare different MCTS strategies on the same position"""
    print("\n=== Strategy Comparison ===")
    
    # Create a test environment
    env = UTTTEnv()
    # Make a few moves to get an interesting position
    env.step(40)  # Center of board
    env.step(20)  # Some response
    
    print(f"Position after 2 moves, player {env.player} to move")
    
    strategies = {
        "Random": RandomStrategy(seed=42),
        "Heuristic": HeuristicStrategy(HeuristicAgent(), rollout_depth=2),
        "Neural": AlphaZeroStrategy(AlphaZeroNetUTTT(), device="cpu")
    }
    
    config = MCTSConfig(n_simulations=30, c_puct=1.0, temperature=0.1)  # Low temp for more deterministic
    
    for name, strategy in strategies.items():
        mcts = GenericMCTS(strategy, config)
        action, policy = mcts.search(env)
        
        # Find top 3 actions
        top_actions = policy.argsort()[-3:][::-1]
        top_probs = policy[top_actions]
        
        print(f"{name:10}: action={action:2d}, top_3={top_actions}, probs={top_probs}")


if __name__ == "__main__":
    example_alphazero_mcts()
    example_heuristic_mcts() 
    example_random_mcts()
    compare_strategies()