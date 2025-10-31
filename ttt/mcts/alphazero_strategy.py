# ttt/mcts/alphazero_strategy.py
"""
AlphaZero evaluation strategy for MCTS.
Uses neural networks to evaluate positions and provide action priors.
"""
from typing import Dict, Tuple
import torch
import numpy as np

from ttt.env.state import TTTEnv
from ttt.mcts.base import EvaluationStrategy
from ttt.agents.az.net import AlphaZeroNetTTT, infer


class AlphaZeroStrategy(EvaluationStrategy):
    """
    Evaluation strategy that uses a neural network (AlphaZero style).
    """
    
    def __init__(self, network: AlphaZeroNetTTT, device: str = "cpu"):
        self.network = network
        self.device = device
        self.network.to(device)
        self.network.eval()
    
    def evaluate_and_expand(self, env: TTTEnv) -> Tuple[float, Dict[int, float]]:
        """
        Use neural network to evaluate position and get action priors.
        
        Args:
            env: Current environment state
            
        Returns:
            Tuple of (value_estimate, action_priors_dict)
        """
        # Get current state encoding
        obs = env._encode()  # Shape: (5, 3, 3)
        legal_mask = env.legal_actions_mask()  # Shape: (3, 3)
        
        # Convert to tensors and add batch dimension
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        legal_mask_tensor = torch.from_numpy(legal_mask.flatten()).unsqueeze(0).bool().to(self.device)
        
        # Get neural network predictions
        with torch.no_grad():
            priors_tensor, value_tensor = infer(self.network, obs_tensor, legal_mask_tensor)
            
        # Convert back to numpy
        priors = priors_tensor.squeeze().cpu().numpy()  # Shape: (9,)
        value = value_tensor.squeeze().cpu().item()  # Scalar
        
        # Create action priors dictionary for legal actions only
        legal_actions = env.legal_actions()
        action_priors = {}
        
        for action in legal_actions:
            action_priors[action] = float(priors[action])
            
        # Normalize priors to ensure they sum to 1
        total_prior = sum(action_priors.values())
        if total_prior > 0:
            action_priors = {a: p / total_prior for a, p in action_priors.items()}
        else:
            # Fallback to uniform if all priors are 0
            uniform_prior = 1.0 / len(legal_actions)
            action_priors = {a: uniform_prior for a in legal_actions}
        
        return value, action_priors
    
    def is_deterministic(self) -> bool:
        """Neural networks are deterministic given the same input"""
        return True


class RandomStrategy(EvaluationStrategy):
    """
    Simple random evaluation strategy for testing/baseline.
    """
    
    def __init__(self, seed: int = None):
        self.rng = np.random.RandomState(seed)
    
    def evaluate_and_expand(self, env: TTTEnv) -> Tuple[float, Dict[int, float]]:
        """
        Random evaluation and uniform action priors.
        """
        # Random value in [-1, 1]
        value = self.rng.uniform(-1, 1)
        
        # Uniform priors over legal actions
        legal_actions = env.legal_actions()
        uniform_prior = 1.0 / len(legal_actions) if legal_actions else 1.0
        action_priors = {action: uniform_prior for action in legal_actions}
        
        return value, action_priors
    
    def is_deterministic(self) -> bool:
        return False