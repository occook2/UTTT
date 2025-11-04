"""
Self-play training loop for AlphaZero agent.
Generates training data by having agents play against themselves.
"""
import random
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import Pool
import copy

from ttt.env.state import TTTEnv
from ttt.agents.az.agent import AlphaZeroAgent
from ttt.mcts.base import MCTSConfig, GenericMCTS


def _play_single_game_worker_shared(args):
    """
    Worker function for multiprocessing self-play games using shared memory.
    
    Args:
        args: Tuple containing (shared_network, net_config, mcts_config, temperature_threshold, game_idx, device, use_adaptive_temperature, confidence_threshold)
    
    Returns:
        Tuple of (training_examples, game_stats)
    """
    shared_network, net_config, mcts_config, temperature_threshold, game_idx, device, use_adaptive_temperature, confidence_threshold = args
    
    # The network weights are already shared in memory, so no copying needed
    shared_network.eval()
    
    # Create agent using the shared network
    agent = AlphaZeroAgent(
        network=shared_network,
        mcts_config=mcts_config,
        device=device
    )
    
    # Create trainer for this game
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=temperature_threshold,
        collect_data=True,
        use_adaptive_temperature=use_adaptive_temperature,
        confidence_threshold=confidence_threshold
    )
    
    # Play the game and return results
    return trainer.play_game()


def _share_network_memory(network):
    """
    Share the network's parameters across processes using torch.multiprocessing.
    This allows multiple processes to access the same network weights in memory.
    """
    # Share all parameters and buffers
    for param in network.parameters():
        param.share_memory_()
    
    for buffer in network.buffers():
        buffer.share_memory_()
    
    return network
@dataclass
class TrainingExample:
    """Single training example from self-play."""
    state: np.ndarray  # Board state
    policy: np.ndarray  # MCTS policy (visit counts normalized)
    value: float  # Game outcome from current player's perspective
    agent_value: float = 0.0  # Agent's network evaluation of the position


class SelfPlayTrainer:
    """
    Self-play trainer for AlphaZero agent.
    """
    
    def __init__(
        self,
        agent: AlphaZeroAgent,
        mcts_config: MCTSConfig,
        temperature_threshold: int = 30,  # Switch to deterministic after this many moves
        collect_data: bool = True,
        use_adaptive_temperature: bool = False,  # Use confidence-based temperature
        confidence_threshold: float = 0.6  # Switch to deterministic if policy confidence > threshold
    ):
        """
        Initialize self-play trainer.
        
        Args:
            agent: AlphaZero agent to use for self-play
            mcts_config: MCTS configuration for training games
            temperature_threshold: Move number after which to use temperature=0
            collect_data: Whether to collect training examples
        """
        self.agent = agent
        self.mcts_config = mcts_config
        self.temperature_threshold = temperature_threshold
        self.collect_data = collect_data
        self.use_adaptive_temperature = use_adaptive_temperature
        self.confidence_threshold = confidence_threshold
        
        # Training data storage
        self.training_examples: List[TrainingExample] = []
        
    def play_game(self) -> Tuple[List[TrainingExample], Dict[str, Any]]:
        """
        Play a single self-play game and collect training data.
        
        Returns:
            Tuple of (training_examples, game_stats)
        """
        env = TTTEnv()
        examples = []
        move_count = 0
        winner = None
        
        # Game statistics
        game_stats = {
            'moves': 0,
            'winner': None,
            'game_length': 0
        }
        
        while not env.terminated:
            move_count += 1
            
            # Get current state for training data
            if self.collect_data:
                current_state = env._encode()
            
            # Create MCTS instance and run search
            mcts = GenericMCTS(self.agent.strategy, self.mcts_config)
            
            # Determine temperature based on adaptive or threshold strategy
            if self.use_adaptive_temperature:
                # First, get raw policy probabilities to check confidence
                raw_policy_probs = mcts.get_raw_policy_probabilities(env)
                max_policy_prob = np.max(raw_policy_probs)
                temperature = 0.0 if max_policy_prob >= self.confidence_threshold else 1.0
            else:
                # Use traditional move-count threshold
                temperature = 1.0 if move_count <= self.temperature_threshold else 0.0
                raw_policy_probs = None
            
            # Run MCTS search with determined temperature
            # Temporarily override the MCTS config temperature
            original_temp = mcts.config.temperature
            mcts.config.temperature = temperature
            action, action_probs, _ = mcts.search(env)
            mcts.config.temperature = original_temp  # Restore original
            
            # Convert policy to numpy array (normalized visit counts)
            if self.collect_data:
                # action_probs is already a normalized probability array
                policy = action_probs.copy()
                
                # Get agent's network evaluation of this position
                agent_evaluation = self.agent.evaluate_position(env)
                
                # DEBUG: Log the first few evaluations per game (uncomment when debugging)
                # if move_count <= 3:
                #     print(f"  Move {move_count}: Agent evaluation = {agent_evaluation:.6f}")
                
                # Store training example (value will be filled in later)
                examples.append(TrainingExample(
                    state=current_state,
                    policy=policy,
                    value=0.0,  # Placeholder, will be updated with game outcome
                    agent_value=agent_evaluation  # Agent's evaluation of this position
                ))
            
            # Make the move
            step_result = env.step(action)
            winner = step_result.info.get('winner', None)
        
        # Update game statistics
        game_stats['moves'] = move_count
        game_stats['game_length'] = move_count
        game_stats['winner'] = winner
        
        # Update training examples with final game outcome
        if self.collect_data:
            self._update_examples_with_outcome(examples, winner)
        
        return examples, game_stats
    
    def generate_training_data(self, n_games: int, show_progress: bool = False) -> List[TrainingExample]:
        """
        Generate training data from multiple self-play games.
        
        Args:
            n_games: Number of games to play
            show_progress: Whether to show progress (used by standalone sessions)
            
        Returns:
            List of training examples
        """
        all_examples = []
        game_results = {'wins_p1': 0, 'wins_p2': 0, 'draws': 0}
        
        for game_idx in range(n_games):
            examples, stats = self.play_game()
            all_examples.extend(examples)
            
            # Track game results
            if stats['winner'] == 1:
                game_results['wins_p1'] += 1
            elif stats['winner'] == 2:
                game_results['wins_p2'] += 1
            else:
                game_results['draws'] += 1
            
            # Only show progress if explicitly requested (for standalone use)
            if show_progress and (game_idx + 1) % 10 == 0:
                print(f"Completed {game_idx + 1}/{n_games} games. "
                      f"Results: P1={game_results['wins_p1']}, "
                      f"P2={game_results['wins_p2']}, "
                      f"Draws={game_results['draws']}")
        
        # Only show final results if explicitly requested
        if show_progress:
            print(f"Self-play complete! Generated {len(all_examples)} training examples.")
            print(f"Final results: {game_results}")
        
        # Store examples for later use
        self.training_examples.extend(all_examples)
        
        return all_examples
    
    def generate_training_data_parallel(self, n_games: int, num_processes: int = None, show_progress: bool = False) -> List[TrainingExample]:
        """
        Generate training data from multiple self-play games using multiprocessing with shared memory.
        
        Args:
            n_games: Number of games to play
            num_processes: Number of parallel processes (default: CPU count)
            show_progress: Whether to show progress (used by standalone sessions)
            
        Returns:
            List of training examples
        """
        if num_processes is None:
            num_processes = min(mp.cpu_count(), n_games)  # Don't use more processes than games
        
        if show_progress:
            print(f"Starting parallel self-play: {n_games} games using {num_processes} processes")
            print("Using shared memory for network weights...")
        
        try:
            # Create a CPU copy of the network for sharing across processes
            # This avoids CUDA tensor sharing issues
            cpu_network = copy.deepcopy(self.agent.network).cpu()
            
            # Share the CPU network's memory across processes
            cpu_network = _share_network_memory(cpu_network)
            
            # Get network configuration for agent creation in workers
            net_config = self.agent.network.cfg
            device = "cpu"  # Force CPU for multiprocessing
            
            # Prepare arguments for each game
            # We pass the shared network directly - no copying or serialization!
            args_list = [
                (cpu_network, net_config, self.mcts_config, self.temperature_threshold, i, device, 
                 self.use_adaptive_temperature, self.confidence_threshold)
                for i in range(n_games)
            ]
            
            # Use torch.multiprocessing for better PyTorch integration
            with mp.Pool(num_processes) as pool:
                results = pool.map(_play_single_game_worker_shared, args_list)
            
        except Exception as e:
            if show_progress:
                print(f"Shared memory approach failed: {e}")
                print("Falling back to sequential execution...")
            # Fallback to sequential execution
            return self.generate_training_data(n_games, show_progress)
        
        # Collect all examples and statistics
        all_examples = []
        game_results = {'wins_p1': 0, 'wins_p2': 0, 'draws': 0}
        
        for examples, stats in results:
            all_examples.extend(examples)
            
            # Track game results
            if stats['winner'] == 1:
                game_results['wins_p1'] += 1
            elif stats['winner'] == -1:
                game_results['wins_p2'] += 1
            else:
                game_results['draws'] += 1
        
        # Only show final results if explicitly requested
        if show_progress:
            print(f"Parallel self-play complete! Generated {len(all_examples)} training examples.")
            print(f"Final results: {game_results}")
        
        # Store examples for later use
        self.training_examples.extend(all_examples)
        
        return all_examples
    
    def _policy_dict_to_array(self, policy_dict: Dict[int, int], env: TTTEnv) -> np.ndarray:
        """
        Convert MCTS policy dictionary to normalized numpy array.
        
        Args:
            policy_dict: Dictionary mapping actions to visit counts
            env: Current environment state
            
        Returns:
            Normalized policy array
        """
        # Get all legal actions
        legal_actions = env.legal_actions()
        
        # Create policy array
        policy = np.zeros(9)  # 3x3 grid = 9 possible positions
        
        # Fill in visit counts for legal actions
        total_visits = sum(policy_dict.values()) if len(policy_dict) > 0 else 1
        
        for action in legal_actions:
            visits = policy_dict.get(action, 0)
            policy[action] = visits / total_visits
        
        return policy
    
    def _update_examples_with_outcome(self, examples: List[TrainingExample], winner: int):
        """
        Update training examples with game outcome.
        
        Args:
            examples: List of training examples to update
            winner: Game winner (1, -1, or 0 for draw)
        """
        # Update examples with alternating perspective
        for i, example in enumerate(examples):
            # Player 1 moves on even indices (0, 2, 4, ...)
            # Player 2 moves on odd indices (1, 3, 5, ...)
            current_player = 1 if i % 2 == 0 else -1
            
            if winner == 0:  # Draw
                example.value = 0.0
            elif winner == current_player:  # Current player won
                example.value = 1.0
            else:  # Current player lost
                example.value = -1.0
    
    def get_training_examples(self) -> List[TrainingExample]:
        """Get all collected training examples."""
        return self.training_examples
    
    def clear_training_examples(self):
        """Clear stored training examples."""
        self.training_examples.clear()
    
    def save_training_data(self, filepath: str):
        """
        Save training examples to file.
        
        Args:
            filepath: Path to save training data
        """
        data = {
            'examples': [(ex.state, ex.policy, ex.value, getattr(ex, 'agent_value', 0.0)) for ex in self.training_examples],
            'config': {
                'mcts_simulations': self.mcts_config.n_simulations,
                'temperature_threshold': self.temperature_threshold
            }
        }
        torch.save(data, filepath)
        print(f"Saved {len(self.training_examples)} training examples to {filepath}")
    
    def load_training_data(self, filepath: str):
        """
        Load training examples from file.
        
        Args:
            filepath: Path to load training data from
        """
        data = torch.load(filepath)
        self.training_examples = []
        
        for example_data in data['examples']:
            if len(example_data) == 4:
                # New format with agent_value
                state, policy, value, agent_value = example_data
                self.training_examples.append(TrainingExample(
                    state=state, policy=policy, value=value, agent_value=agent_value
                ))
            else:
                # Old format without agent_value
                state, policy, value = example_data
                self.training_examples.append(TrainingExample(
                    state=state, policy=policy, value=value, agent_value=0.0
                ))
        print(f"Loaded {len(self.training_examples)} training examples from {filepath}")


def run_self_play_session(
    agent: AlphaZeroAgent,
    n_games: int = 100,
    n_simulations: int = 400,
    temperature_threshold: int = 30,
    use_multiprocessing: bool = True,
    num_processes: int = None
) -> List[TrainingExample]:
    """
    Convenience function to run a self-play training session.
    
    Args:
        agent: AlphaZero agent to train
        n_games: Number of games to play
        n_simulations: MCTS simulations per move
        temperature_threshold: Move threshold for deterministic play
        
    Returns:
        List of training examples
    """
    # Configure MCTS for training
    mcts_config = MCTSConfig(
        n_simulations=n_simulations,
        c_puct=1.0,
        temperature=1.0,  # Will be adjusted during play
        use_transposition_table=True
    )
    
    # Create trainer and generate data
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=temperature_threshold
    )
    
    print(f"Starting self-play session: {n_games} games, {n_simulations} simulations/move")
    
    if use_multiprocessing and n_games > 1:
        training_examples = trainer.generate_training_data_parallel(
            n_games=n_games, 
            num_processes=num_processes,
            show_progress=True
        )
    else:
        training_examples = trainer.generate_training_data(n_games, show_progress=True)
    
    return training_examples
