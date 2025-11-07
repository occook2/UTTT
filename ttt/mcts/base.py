# uttt/mcts/base.py
"""
Generic MCTS framework that can be used with different evaluation strategies.
This separates the MCTS algorithm from the specific evaluation methods.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Protocol
import numpy as np
import time
from collections import defaultdict

from ttt.env.state import TTTEnv
from ttt.mcts.transpo import state_key, clone_env


@dataclass
class MCTSConfig:
    """Generic MCTS configuration"""
    n_simulations: int = 200 # Can cause slowdowns if too high
    c_puct: float = 1.0  # UCB exploration constant
    use_transposition_table: bool = True
    temperature: float = 1.0  # For action selection
    add_noise: bool = False  # Dirichlet noise for root
    noise_alpha: float = 0.3
    noise_epsilon: float = 0.25
    noise_moves: int = 10  # Only apply noise to first N moves


class EvaluationStrategy(ABC):
    """Abstract base class for different evaluation strategies"""
    
    @abstractmethod
    def evaluate_and_expand(self, env: TTTEnv) -> Tuple[float, Dict[int, float]]:
        """
        Evaluate a position and get action priors.
        
        Args:
            env: The environment state to evaluate
            
        Returns:
            Tuple of (value_estimate, action_priors_dict)
            - value_estimate: float in [-1, 1] from current player's perspective
            - action_priors_dict: {action: prior_probability} for legal actions
        """
        pass
    
    @abstractmethod
    def is_deterministic(self) -> bool:
        """Whether this strategy is deterministic (affects caching behavior)"""
        pass


class Node:
    """MCTS tree node"""
    
    def __init__(self, state_key: tuple, parent: Optional[Node] = None, action: Optional[int] = None, prior: float = 0.0):
        self.state_key = state_key
        self.parent = parent
        self.action = action  # Action that led to this node
        self.prior = prior
        
        # Visit statistics
        self.visit_count = 0
        self.value_sum = 0.0
        self.mean_value = 0.0
        
        # Children
        self.children: Dict[int, Node] = {}
        self.is_expanded = False
        
    def is_root(self) -> bool:
        return self.parent is None
        
    def select_child_ucb(self, c_puct: float) -> Node:
        """Select child with highest UCB score"""
        def ucb_score(child: Node) -> float:
            q_value = -child.mean_value
            u_value = c_puct * child.prior * np.sqrt(self.visit_count) / (1 + child.visit_count)
            return q_value + u_value
        
        return max(self.children.values(), key=ucb_score)
    
    def add_child(self, action: int, prior: float, state_key: tuple) -> Node:
        """Add a child node"""
        child = Node(state_key=state_key, parent=self, action=action, prior=prior)
        self.children[action] = child
        return child
    
    def backup(self, value: float):
        """Backup value up the tree"""
        self.visit_count += 1
        self.value_sum += value
        self.mean_value = self.value_sum / self.visit_count
        
    
    def get_action_probabilities(self, temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities based on visit counts"""
        probs = np.zeros(9)  # TTT has 9 actions
        
        if not self.children:
            return probs
            
        if temperature == 0:
            # Deterministic - pick most visited
            best_action = max(self.children.keys(), key=lambda a: self.children[a].visit_count)
            probs[best_action] = 1.0
        else:
            # Apply temperature
            counts = np.array([self.children[a].visit_count for a in sorted(self.children.keys())])
            actions = np.array(sorted(self.children.keys()))
            
            if temperature != 1.0:
                counts = counts ** (1.0 / temperature)
            
            counts_sum = counts.sum()
            if counts_sum > 0:
                normalized_counts = counts / counts_sum
                probs[actions] = normalized_counts
                
        return probs


class GenericMCTS:
    """
    Generic Monte Carlo Tree Search implementation.
    Uses an EvaluationStrategy to handle different types of position evaluation.
    """
    
    def __init__(self, evaluation_strategy: EvaluationStrategy, config: MCTSConfig = None):
        self.strategy = evaluation_strategy
        self.config = config or MCTSConfig()
        
        # Transposition table: state_key -> Node
        self.transposition_table: Dict[tuple, Node] = {}
        
        # Statistics
        self.stats = defaultdict(int)
        
    def search(self, env: TTTEnv) -> Tuple[int, np.ndarray]:
        """
        Run MCTS search and return best action and action probabilities.
        
        Args:
            env: Environment to search from
            
        Returns:
            Tuple of (best_action, action_probabilities)
        """
        if env.terminated:
            raise ValueError("Cannot search from terminal position")
            
        # Get or create root node
        root_key = state_key(env)
        if root_key in self.transposition_table and self.config.use_transposition_table:
            root = self.transposition_table[root_key]
        else:
            root = Node(state_key=root_key)
            if self.config.use_transposition_table:
                self.transposition_table[root_key] = root

        # Expand root if needed and add Dirichlet noise BEFORE simulations
        if not root.is_expanded:
            # Need to expand root first to get children for noise
            value, priors = self.strategy.evaluate_and_expand(env)
            legal_actions = env.legal_actions()
            for action in legal_actions:
                child_env = clone_env(env)
                child_env.step(action)
                child_key = state_key(child_env)
                prior = priors.get(action, 1.0 / len(legal_actions))
                root.add_child(action, prior, child_key)
            root.is_expanded = True
            root.visit_count = 1  # Initialize specifically first move to 1, all expanded children will be initialized to 1
        
        # Add Dirichlet noise to root BEFORE running simulations
        # Only apply noise during early moves of the game
        if self.config.add_noise and root.children:
            # Count moves played so far
            moves_played = int(np.count_nonzero(env.board))
            if moves_played < self.config.noise_moves:
                self._add_dirichlet_noise(root)
        
        # Run simulations
        for _ in range(self.config.n_simulations):
            self._simulate(root, clone_env(env))
        
        # Get action probabilities
        action_probs = root.get_action_probabilities(self.config.temperature)
        
        # Get raw policy probabilities (before MCTS) for confidence checking
        raw_policy_probs = self.get_raw_policy_probabilities(env)
        
        # Select best action
        if self.config.temperature == 0:
            best_action = np.argmax(action_probs)
        else:
            # Sample according to probabilities
            legal_actions = [a for a in range(9) if action_probs[a] > 0]
            legal_probs = action_probs[legal_actions]
            legal_probs = legal_probs / legal_probs.sum()  # Renormalize
            best_action = np.random.choice(legal_actions, p=legal_probs)
            
        return int(best_action), action_probs, raw_policy_probs
    
    def _simulate(self, root: Node, env: TTTEnv):
        """Run one simulation from root"""
        node = root
        path = [node]
        
        # Selection: traverse down the tree
        while node.is_expanded and not env.terminated:
            if not node.children:
                break
                
            node = node.select_child_ucb(self.config.c_puct)
            path.append(node)
            env.step(node.action)
            
        # Terminal check
        if env.terminated:
            # Terminal node - get actual game result
            winner = env._macro_winner()
            # Convert to current player's perspective
            current_player = env.player
            if winner == 0:
                value = 0.0
            elif winner == current_player:
                value = -1.0
            else:
                value = 1.0
        else:
            # Expansion and evaluation
            value, priors = self.strategy.evaluate_and_expand(env)
            
            # Expand the node
            if not node.is_expanded:
                legal_actions = env.legal_actions()
                for action in legal_actions:
                    child_env = clone_env(env)
                    child_env.step(action)
                    child_key = state_key(child_env)
                    prior = priors.get(action, 1.0 / len(legal_actions))
                    node.add_child(action, prior, child_key)
                node.is_expanded = True
        
        # Backup
        for i, path_node in enumerate(reversed(path)):
            # Flip value sign as we go up the tree
            backup_value = value if i % 2 == 0 else -value
            path_node.backup(backup_value)
    
    def get_raw_policy_probabilities(self, env: TTTEnv) -> np.ndarray:
        """Get raw neural network policy probabilities (before MCTS)"""
        _, priors = self.strategy.evaluate_and_expand(env)
        
        # Convert to numpy array
        probs = np.zeros(9)
        for action, prob in priors.items():
            probs[action] = prob
            
        # Mask illegal actions and renormalize
        legal_mask = env.legal_actions_mask().flatten()
        probs = probs * legal_mask
        
        # Renormalize to sum to 1
        if probs.sum() > 0:
            probs = probs / probs.sum()
            
        return probs
    
    def _add_dirichlet_noise(self, root: Node):
        """Add Dirichlet noise to root node priors"""
        if not root.children:
            return
            
        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.config.noise_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1 - self.config.noise_epsilon) * child.prior + self.config.noise_epsilon * noise[i]