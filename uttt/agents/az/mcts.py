from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple, Optional, Hashable
import torch
import torch.nn as nn
import torch.nn.functional as F
from uttt.agents.az.net import infer, AlphaZeroNetUTTT
from uttt.agents.mcts.transpo import state_key, clone_env
from uttt.mcts.base import GenericMCTS, MCTSConfig
from uttt.mcts.alphazero_strategy import AlphaZeroStrategy


@dataclass
class AlphaZeroMCTSConfig(MCTSConfig):
    """AlphaZero-specific MCTS configuration"""
    n_simulations: int = 96
    c_puct: float = 1.5
    temperature: float = 1.0
    add_noise: bool = True
    noise_alpha: float = 0.3
    noise_epsilon: float = 0.25


class AlphaZeroMCTS:
    """
    AlphaZero MCTS wrapper that uses the generic MCTS with neural network evaluation.
    """
    
    def __init__(self, network: AlphaZeroNetUTTT, config: AlphaZeroMCTSConfig = None, device: str = "cpu"):
        self.network = network
        self.config = config or AlphaZeroMCTSConfig()
        self.device = device
        
        # Create evaluation strategy and generic MCTS
        self.strategy = AlphaZeroStrategy(network, device)
        self.mcts = GenericMCTS(self.strategy, self.config)
    
    def run(self, env, net=None, cfg=None) -> Tuple[int, np.ndarray]:
        """
        Compatibility method for existing code.
        
        Args:
            env: Environment to search from
            net: Neural network (ignored, uses self.network)
            cfg: Configuration (ignored, uses self.config)
            
        Returns:
            Tuple of (action, policy_vector)
        """
        return self.search(env)
    
    def search(self, env) -> Tuple[int, np.ndarray]:
        """
        Run MCTS search and return best action and policy.
        
        Args:
            env: Environment to search from
            
        Returns:
            Tuple of (best_action, action_probabilities)
        """
        return self.mcts.search(env)


# Legacy compatibility classes to maintain existing interface
@dataclass  
class MCTSConfig:
    n_simulations: int = 96
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_moves: int = 10
    temperature: float = 1.0
    temperature_after: float = 0.05


class MCTS:
    """Legacy MCTS class for backwards compatibility"""
    
    def __init__(self, config: MCTSConfig = None):
        self.config = config or MCTSConfig()
        self._mcts_impl = None  # Will be set when we have a network
    
    def run(self, env, net, cfg: MCTSConfig = None) -> Tuple[int, np.ndarray]:
        """
        Run MCTS with the given network.
        
        Args:
            env: Environment to search from
            net: Neural network to use
            cfg: Configuration to use (defaults to self.config)
            
        Returns:
            Tuple of (action, policy_vector)
        """
        cfg = cfg or self.config
        
        # Create AlphaZero MCTS config
        az_config = AlphaZeroMCTSConfig(
            n_simulations=cfg.n_simulations,
            c_puct=cfg.c_puct,
            temperature=cfg.temperature,
            add_noise=True,
            noise_alpha=cfg.dirichlet_alpha,
            noise_epsilon=cfg.dirichlet_epsilon
        )
        
        # Create and run AlphaZero MCTS
        az_mcts = AlphaZeroMCTS(net, az_config)
        return az_mcts.search(env)
