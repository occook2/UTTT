"""
Configuration management for AlphaZero training.
"""
import yaml
from dataclasses import dataclass
import torch


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""
    # Training parameters
    n_epochs: int = 100
    games_per_epoch: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    
    # Self-play parameters
    mcts_simulations: int = 400
    temperature_threshold: int = 30
    use_multiprocessing: bool = True  # Enable parallel self-play
    num_processes: int = 5  # None = use all CPUs
    
    # Model saving
    save_every: int = 10  # Save model every N epochs
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training data management
    max_training_samples: int = 50000  # Keep only the most recent samples
    
    # Symmetry augmentation
    use_symmetry_augmentation: bool = True  # Apply 4-fold rotational symmetry
    
    # UI data saving (separate from training)
    save_ui_data: bool = True


def load_config_from_yaml(filepath: str) -> TrainingConfig:
    """Load training configuration from a YAML file."""
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)