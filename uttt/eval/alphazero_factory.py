"""
Factory functions for creating AlphaZero agents with different checkpoints.
"""
import os
import torch
from typing import Optional, Dict, Any
from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.az.net import AlphaZeroNetUTTT, AZNetConfig


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
    network_config = None
    network = None

    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Try to get network config from checkpoint
        if 'config' in checkpoint and hasattr(checkpoint['config'], 'network'):
            cfg = checkpoint['config'].network
            # If it's a dict, convert to AZNetConfig
            if isinstance(cfg, dict):
                network_config = AZNetConfig(**cfg)
            else:
                network_config = cfg
        else:
            # Fallback to default config
            print("Warning: No network config found in checkpoint, using default")
            network_config = AZNetConfig()
        
        # Create network with correct config and load weights
        network = AlphaZeroNetUTTT(network_config).to(device)
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        
    elif checkpoint_path:
        print(f"Warning: Checkpoint {checkpoint_path} not found, using random weights")
        network_config = AZNetConfig()
        network = AlphaZeroNetUTTT(network_config).to(device)
    else:
        print("Created AlphaZero agent with random weights")
        network_config = AZNetConfig()
        network = AlphaZeroNetUTTT(network_config).to(device)
    
    # Create MCTS config without noise for evaluation
    from uttt.mcts.base import MCTSConfig
    mcts_config = MCTSConfig(
        n_simulations=mcts_simulations,
        c_puct=1.0,
        temperature=temperature,
        use_transposition_table=True,
        add_noise=False,           # NO noise during evaluation
        noise_alpha=0.3,
        noise_epsilon=0.25,
        noise_moves=10
    )
    
    # Create agent with the properly configured network and MCTS config
    agent = AlphaZeroAgent(network=network, mcts_config=mcts_config, device=device)
    
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