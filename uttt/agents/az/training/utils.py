"""
Utility functions for AlphaZero training.
"""
import os
import json
import yaml
from datetime import datetime
from typing import List, Dict, Any

from uttt.agents.az.self_play import TrainingExample
from .config import TrainingConfig


def make_run_directory() -> str:
    """
    Create a timestamped run directory for organizing training artifacts.
    
    Returns:
        str: Path to the created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Create subdirectories within the run directory
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    config_dir = os.path.join(run_dir, "config")
    metrics_dir = os.path.join(run_dir, "metrics")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    print(f"Created run directory: {run_dir}")
    print(f"Created checkpoints directory: {checkpoints_dir}")
    print(f"Created config directory: {config_dir}")
    print(f"Created metrics directory: {metrics_dir}")
    return run_dir


def save_config_to_run(config: TrainingConfig, config_path: str, run_dir: str):
    """
    Save a copy of the training configuration to the run directory.
    
    Args:
        config: The training configuration object
        config_path: Path to the original config file (if it exists)
        run_dir: Path to the run directory
    """
    config_save_dir = os.path.join(run_dir, "config")
    
    # Save the config as YAML (preserving original format if it came from a file)
    if os.path.exists(config_path):
        # Copy the original config file
        import shutil
        original_filename = os.path.basename(config_path)
        dest_path = os.path.join(config_save_dir, original_filename)
        shutil.copy2(config_path, dest_path)
        print(f"Saved original config file: {dest_path}")
    
    # Also save the final config as used (in case of any modifications)
    final_config_path = os.path.join(config_save_dir, "final_config.yaml")
    config_dict = {
        'n_epochs': config.n_epochs,
        'games_per_epoch': config.games_per_epoch,
        'batch_size': config.batch_size,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'mcts_simulations': config.mcts_simulations,
        'temperature_threshold': config.temperature_threshold,
        'use_multiprocessing': config.use_multiprocessing,
        'num_processes': config.num_processes,
        'save_every': config.save_every,
        'checkpoint_dir': config.checkpoint_dir,
        'device': config.device,
        'max_training_samples': config.max_training_samples,
        'use_symmetry_augmentation': config.use_symmetry_augmentation,
        'save_ui_data': config.save_ui_data,
        'network': {
            'in_planes': config.network.in_planes,
            'channels': config.network.channels,
            'blocks': config.network.blocks,
            'board_n': config.network.board_n,
            'policy_reduce': config.network.policy_reduce,
            'value_hidden': config.network.value_hidden
        }
    }
    
    with open(final_config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    print(f"Saved final config: {final_config_path}")


def save_training_games_for_ui(examples: List[TrainingExample], epoch_num: int, config: TrainingConfig, run_dir: str):
    """
    Save training examples in a UI-friendly format that reconstructs games.
    This is separate from the actual training data and used only for inspection.
    
    Args:
        examples: Raw training examples from self-play
        epoch_num: Current training epoch
        config: Training configuration
        run_dir: Path to the run directory (not checkpoints)
    """
    # Create training examples directory in the run folder (not checkpoints)
    ui_data_dir = os.path.join(run_dir, "training_ui_data")
    os.makedirs(ui_data_dir, exist_ok=True)
    
    # Reconstruct games from training examples
    games = reconstruct_games_from_examples(examples)
    
    # Create UI-friendly data structure
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    ui_data = {
        "meta": {
            "epoch": epoch_num,
            "timestamp": timestamp,
            "total_examples": len(examples),
            "total_games": len(games),
            "mcts_simulations": config.mcts_simulations,
            "use_symmetry_augmentation": getattr(config, 'use_symmetry_augmentation', True),
            "temperature_threshold": getattr(config, 'temperature_threshold', 30)
        },
        "games": games
    }
    
    # Save to JSON file
    filename = f"training_games_epoch_{epoch_num}_{timestamp}.json"
    filepath = os.path.join(ui_data_dir, filename)
    
    with open(filepath, 'w') as f:
        json.dump(ui_data, f, indent=2)
    
    print(f"Saved {len(games)} games ({len(examples)} examples) for UI inspection: {filename}")


def reconstruct_games_from_examples(examples: List[TrainingExample]) -> List[Dict[str, Any]]:
    """
    Reconstruct individual games from flat list of training examples.
    
    The key insight is that games are stored sequentially, and we can detect
    game boundaries by looking for significant drops in piece count between
    consecutive examples (indicating a new game started).
    
    Args:
        examples: Flat list of training examples
        
    Returns:
        List of game dictionaries, each containing moves and metadata
    """
    import numpy as np
    
    if not examples:
        return []
    
    games = []
    current_game = []
    
    for i, example in enumerate(examples):
        # Convert example to move data
        piece_count = int(np.sum(example.state[1]) + np.sum(example.state[2]))
        move_data = {
            "move_number": len(current_game) + 1,
            "state": example.state.tolist(),  # Convert numpy to list for JSON
            "policy": example.policy.tolist(),
            "value": float(example.value),
            "player": 1 if len(current_game) % 2 == 0 else -1,  # Alternating players
            "piece_count": piece_count,
            "policy_max": float(np.max(example.policy)),
            "policy_sum": float(np.sum(example.policy))
        }
        
        # Check if this should start a new game
        should_start_new_game = False
        
        if current_game:
            # Get piece count from previous move
            prev_piece_count = current_game[-1]["piece_count"]
            
            # If current move has significantly fewer pieces, it's likely a new game
            if piece_count < prev_piece_count - 1:  # Allow for 1 piece difference due to normal play
                should_start_new_game = True
            
            # Also check if we've been building up pieces consistently and suddenly dropped
            elif len(current_game) >= 3:
                # Look at piece count trend
                recent_counts = [move["piece_count"] for move in current_game[-3:]]
                if all(recent_counts[i] <= recent_counts[i+1] for i in range(len(recent_counts)-1)):
                    # Pieces were increasing, now dropped significantly
                    if piece_count < min(recent_counts):
                        should_start_new_game = True
        
        # If we should start a new game, save the current one first
        if should_start_new_game and current_game:
            # Finalize current game
            final_value = current_game[-1]["value"]
            winner = None
            if final_value > 0.1:
                winner = 1 if (len(current_game) - 1) % 2 == 0 else -1
            elif final_value < -0.1:
                winner = -1 if (len(current_game) - 1) % 2 == 0 else 1
            else:
                winner = 0  # Draw
            
            game_data = {
                "game_id": len(games),
                "moves": current_game,
                "game_length": len(current_game),
                "winner": winner,
                "final_value": float(final_value),
                "total_pieces": current_game[-1]["piece_count"]
            }
            
            games.append(game_data)
            current_game = []
        
        # Add current move to game
        current_game.append(move_data)
    
    # Add the final game
    if current_game:
        final_value = current_game[-1]["value"]
        winner = None
        if final_value > 0.1:
            winner = 1 if (len(current_game) - 1) % 2 == 0 else -1
        elif final_value < -0.1:
            winner = -1 if (len(current_game) - 1) % 2 == 0 else 1
        else:
            winner = 0  # Draw
        
        game_data = {
            "game_id": len(games),
            "moves": current_game,
            "game_length": len(current_game),
            "winner": winner,
            "final_value": float(final_value),
            "total_pieces": current_game[-1]["piece_count"]
        }
        
        games.append(game_data)
    
    return games