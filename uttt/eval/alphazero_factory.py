"""
Factory functions for creating AlphaZero agents with different checkpoints.
"""
import os
from typing import Optional, Dict, Any
from uttt.agents.az.agent import AlphaZeroAgent


def create_alphazero_agent(
    checkpoint_path: Optional[str] = None,
    mcts_simulations: int = 100,
    temperature: float = 0.0,
    device: str = "cpu"
) -> AlphaZeroAgent:
    """
    Create an AlphaZero agent, optionally loading from a checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file (None for random weights)
        mcts_simulations: Number of MCTS simulations per move
        temperature: Temperature for action selection (0.0 = deterministic)
        device: Device to run on ('cpu' or 'cuda')
    
    Returns:
        Configured AlphaZero agent
    """
    agent = AlphaZeroAgent(device=device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load_weights(checkpoint_path)
        print(f"Loaded AlphaZero agent from {checkpoint_path}")
    elif checkpoint_path:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random weights")
    else:
        print("Created AlphaZero agent with random weights")
    
    # Configure MCTS
    agent.set_temperature(temperature)
    agent.set_simulations(mcts_simulations)
    
    return agent


def alphazero_agent_factory(checkpoint_path: str, **kwargs):
    """
    Factory function that returns a callable for creating AlphaZero agents.
    This is compatible with the tournaments.py callable agent pattern.
    """
    def create_agent(**extra_kwargs):
        # Merge the factory kwargs with any extra kwargs passed at creation time
        # Factory kwargs take precedence
        merged_kwargs = {**extra_kwargs, **kwargs}
        return create_alphazero_agent(checkpoint_path, **merged_kwargs)
    
    # Add metadata for identification
    create_agent.checkpoint_path = checkpoint_path
    create_agent.agent_type = "AlphaZero"
    
    return create_agent


def get_alphazero_agent_name(checkpoint_path: str, **kwargs) -> str:
    """
    Generate a readable name for an AlphaZero agent configuration.
    
    Args:
        checkpoint_path: Path to checkpoint file
        **kwargs: Additional agent parameters
    
    Returns:
        Human-readable agent name
    """
    # Extract meaningful checkpoint identifier
    if checkpoint_path:
        # Extract just the filename without extension
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        
        # Clean up common checkpoint patterns
        if checkpoint_name.startswith('alphazero_'):
            checkpoint_name = checkpoint_name[10:]  # Remove 'alphazero_' prefix
        
        if checkpoint_name == 'final':
            checkpoint_name = 'Final'
        elif checkpoint_name.startswith('epoch_'):
            epoch_num = checkpoint_name[6:]  # Remove 'epoch_' prefix
            checkpoint_name = f"Epoch{epoch_num}"
        elif checkpoint_name == 'best_model':
            checkpoint_name = 'Best'
    else:
        checkpoint_name = 'Random'
    
    # Add MCTS configuration if specified
    simulations = kwargs.get('mcts_simulations', 100)
    temperature = kwargs.get('temperature', 0.0)
    
    name = f"AZ-{checkpoint_name}"
    
    # Add simulation count if non-standard
    if simulations != 100:
        name += f"-{simulations}sim"
    
    # Add temperature if non-zero
    if temperature > 0:
        name += f"-T{temperature:.1f}"
    
    return name


def discover_alphazero_checkpoints(checkpoint_dir: str = "checkpoints") -> Dict[str, str]:
    """
    Discover available AlphaZero checkpoint files.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Dict mapping readable names to checkpoint paths
    """
    checkpoints = {}
    
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    # Find all .pt files
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(checkpoint_dir, filename)
            readable_name = get_alphazero_agent_name(filepath)
            checkpoints[readable_name] = filepath
    
    return checkpoints