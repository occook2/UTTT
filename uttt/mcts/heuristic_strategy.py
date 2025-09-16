# uttt/mcts/heuristic_strategy.py
"""
Heuristic evaluation strategy for MCTS.
Uses your existing heuristic agent to evaluate positions and provide action preferences.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from uttt.env.state import UTTTEnv
from uttt.mcts.base import EvaluationStrategy
from uttt.agents.heuristic import HeuristicAgent
from uttt.mcts.transpo import clone_env


class HeuristicStrategy(EvaluationStrategy):
    """
    Evaluation strategy that uses heuristic evaluation and action preferences.
    """
    
    def __init__(self, heuristic_agent: HeuristicAgent = None, rollout_depth: int = 5, temperature: float = 2.0):
        self.heuristic = heuristic_agent or HeuristicAgent()
        self.rollout_depth = rollout_depth
        self.temperature = temperature  # Controls how "sharp" the action preferences are
    
    def evaluate_and_expand(self, env: UTTTEnv) -> Tuple[float, Dict[int, float]]:
        """
        Use heuristic evaluation and action scoring.
        
        Args:
            env: Current environment state
            
        Returns:
            Tuple of (value_estimate, action_priors_dict)
        """
        # Evaluate current position
        value = self._evaluate_position(env)
        
        # Get action preferences based on heuristic scoring
        action_priors = self._get_action_preferences(env)
        
        return value, action_priors
    
    def _evaluate_position(self, env: UTTTEnv) -> float:
        """
        Evaluate the position using a combination of heuristic and short rollouts.
        """
        if env.terminated:
            winner = env._macro_winner()
            if winner == 0:
                return 0.0
            elif winner == env.player:
                return 1.0
            else:
                return -1.0
        
        # Quick heuristic evaluation
        heuristic_value = self._quick_heuristic_value(env)
        
        # Optional: Add some rollout-based evaluation
        if self.rollout_depth > 0:
            rollout_value = self._rollout_evaluation(env)
            # Combine heuristic and rollout (weight heuristic more heavily)
            value = 0.7 * heuristic_value + 0.3 * rollout_value
        else:
            value = heuristic_value
        
        return np.clip(value, -1.0, 1.0)
    
    def _quick_heuristic_value(self, env: UTTTEnv) -> float:
        """
        Quick heuristic evaluation similar to your existing implementation.
        """
        macro = env.macro_wins
        player = env.player
        opp = -player
        
        # Count macro wins
        player_wins = int((macro == player).sum())
        opp_wins = int((macro == opp).sum())
        
        # Base value from macro advantage
        base_value = (player_wins - opp_wins) / 3.0
        
        # Add tactical bonuses
        tactical_bonus = 0.0
        
        # Check for immediate macro win/block opportunities
        if env.last_move is not None:
            lr, lc = env.last_move
            target_micro = (lr % 3, lc % 3)
            if env._micro_is_open(*target_micro):
                # Check for immediate wins in target micro
                r0, c0 = target_micro[0] * 3, target_micro[1] * 3
                sub_board = env.board[r0:r0+3, c0:c0+3]
                
                if self._has_immediate_win(sub_board, player):
                    tactical_bonus += 0.3
                elif self._has_immediate_win(sub_board, opp):
                    tactical_bonus -= 0.3
        
        return np.clip(base_value + tactical_bonus, -1.0, 1.0)
    
    def _has_immediate_win(self, sub_board: np.ndarray, player: int) -> bool:
        """Check if player has an immediate win in the 3x3 sub-board"""
        for r in range(3):
            for c in range(3):
                if sub_board[r, c] == 0:
                    # Try placing here
                    sub_board[r, c] = player
                    win = self._check_local_win(sub_board, player)
                    sub_board[r, c] = 0  # Undo
                    if win:
                        return True
        return False
    
    def _check_local_win(self, sub_board: np.ndarray, player: int) -> bool:
        """Check if player has won the 3x3 sub-board"""
        # Check rows, columns, diagonals
        for i in range(3):
            if np.sum(sub_board[i, :] == player) == 3:  # Row
                return True
            if np.sum(sub_board[:, i] == player) == 3:  # Column
                return True
        
        if np.sum(np.diag(sub_board) == player) == 3:  # Main diagonal
            return True
        if np.sum(np.diag(np.fliplr(sub_board)) == player) == 3:  # Anti-diagonal
            return True
        
        return False
    
    def _rollout_evaluation(self, env: UTTTEnv) -> float:
        """
        Perform a short rollout using heuristic moves.
        """
        rollout_env = clone_env(env)
        original_player = env.player
        
        for _ in range(self.rollout_depth):
            if rollout_env.terminated:
                break
                
            legal_actions = rollout_env.legal_actions()
            if not legal_actions:
                break
                
            # Use heuristic to pick action
            if hasattr(self.heuristic, 'best_of'):
                action = self.heuristic.best_of(rollout_env, legal_actions)
            else:
                action = self.heuristic.select_action(rollout_env)
                
            rollout_env.step(action)
        
        # Evaluate final position
        if rollout_env.terminated:
            winner = rollout_env._macro_winner()
            if winner == 0:
                return 0.0
            elif winner == original_player:
                return 1.0
            else:
                return -1.0
        else:
            # Use heuristic evaluation of final position
            return self._quick_heuristic_value(rollout_env)
    
    def _get_action_preferences(self, env: UTTTEnv) -> Dict[int, float]:
        """
        Get action preferences by evaluating each legal action with the heuristic.
        """
        legal_actions = env.legal_actions()
        if not legal_actions:
            return {}
        
        action_values = []
        
        # Evaluate each legal action
        for action in legal_actions:
            child_env = clone_env(env)
            child_env.step(action)
            
            if child_env.terminated:
                # Terminal - check if it's a win
                winner = child_env._macro_winner()
                if winner == env.player:
                    value = 1.0  # Winning move
                elif winner == -env.player:
                    value = -1.0  # Losing move (shouldn't happen)
                else:
                    value = 0.0  # Draw
            else:
                # Use heuristic evaluation from opponent's perspective (so flip)
                value = -self._quick_heuristic_value(child_env)
            
            action_values.append((action, value))
        
        # Convert to preferences using softmax with temperature
        actions, values = zip(*action_values)
        values = np.array(values)
        
        # Apply temperature and softmax
        if self.temperature > 0:
            exp_values = np.exp(values / self.temperature)
            probs = exp_values / np.sum(exp_values)
        else:
            # Deterministic - all weight on best action
            probs = np.zeros(len(values))
            probs[np.argmax(values)] = 1.0
        
        return {action: float(prob) for action, prob in zip(actions, probs)}
    
    def is_deterministic(self) -> bool:
        """Heuristic evaluation is deterministic"""
        return True