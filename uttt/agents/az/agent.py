# uttt/agents/az/agent.py
"""
Bare-bones AlphaZero agent that uses neural network + MCTS for move selection.
"""
from typing import Optional
import torch

from uttt.agents.base import Agent
from uttt.env.state import UTTTEnv
from uttt.agents.az.net import AlphaZeroNetUTTT
from uttt.mcts.alphazero_strategy import AlphaZeroStrategy
from uttt.mcts.base import GenericMCTS, MCTSConfig


class AlphaZeroAgent(Agent):
    """
    AlphaZero agent that uses neural network + MCTS for move selection.
    """
    
    def __init__(
        self, 
        network: Optional[AlphaZeroNetUTTT] = None,
        mcts_config: Optional[MCTSConfig] = None,
        device: str = "cpu"
    ):
        """
        Initialize AlphaZero agent.
        
        Args:
            network: Neural network for evaluation. If None, creates a new one.
            mcts_config: MCTS configuration. If None, uses default.
            device: Device to run computations on ("cpu" or "cuda").
        """
        self.device = device
        
        # Initialize neural network
        if network is None:
            self.network = AlphaZeroNetUTTT()
        else:
            self.network = network
        
        self.network.to(self.device)
        self.network.eval()  # Set to evaluation mode
        
        # Initialize MCTS config
        if mcts_config is None:
            self.mcts_config = MCTSConfig(
                n_simulations=2, # Must be greater than 1 to avoid errors
                c_puct=1.0,
                temperature=0.0,  # Deterministic play
                use_transposition_table=True
            )
        else:
            self.mcts_config = mcts_config
        
        # Create evaluation strategy
        self.strategy = AlphaZeroStrategy(self.network, device=self.device)
    
    def select_action(self, env: UTTTEnv) -> int:
        """
        Select an action using neural network + MCTS.
        
        Args:
            env: Current environment state
            
        Returns:
            Selected action (integer)
        """
        if env.terminated:
            raise ValueError("Cannot select action from a terminated environment")
        
        # Create fresh MCTS instance for this search
        mcts = GenericMCTS(self.strategy, self.mcts_config)
        
        # Run MCTS search
        action, _ = mcts.search(env)
        return action
    
    def load_weights(self, checkpoint_path: str):
        """
        Load neural network weights from checkpoint.
        
        Args:
            checkpoint_path: Path to saved model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.network.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.network.load_state_dict(checkpoint)
        self.network.eval()
    
    def save_weights(self, checkpoint_path: str):
        """
        Save neural network weights to checkpoint.
        
        Args:
            checkpoint_path: Path to save model checkpoint
        """
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'mcts_config': self.mcts_config
        }, checkpoint_path)
    
    def set_temperature(self, temperature: float):
        """
        Set MCTS temperature for move selection.
        
        Args:
            temperature: Temperature value (0.0 = deterministic, >0 = stochastic)
        """
        self.mcts_config.temperature = temperature
    
    def set_simulations(self, n_simulations: int):
        """
        Set number of MCTS simulations.
        
        Args:
            n_simulations: Number of simulations to run
        """
        self.mcts_config.n_simulations = n_simulations
