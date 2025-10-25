"""
Configuration management for AlphaZero training.
"""
import yaml
from dataclasses import dataclass
import torch


@dataclass
class NetworkConfig:
    """Configuration for neural network architecture."""
    # Network architecture parameters (matching AZNetConfig)
    in_planes: int = 5        # Input state planes (5×3×3)
    channels: int = 96        # Trunk width (try 64–128)
    blocks: int = 2           # Number of residual blocks
    board_n: int = 3          # Spatial size (3x3)
    policy_reduce: int = 32   # Internal feature width in policy head
    value_hidden: int = 256   # MLP hidden size for value head


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
    temperature: float = 1.0
    c_puct: float = 1.0
    use_multiprocessing: bool = True  # Enable parallel self-play
    num_processes: int = 5  # None = use all CPUs
    
    # Dirichlet noise parameters
    add_dirichlet_noise: bool = False
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    dirichlet_noise_moves: int = 10  # Only apply noise to first N moves
    
    # Adaptive temperature based on policy confidence
    use_adaptive_temperature: bool = False
    confidence_threshold: float = 0.6  # Switch to deterministic if max policy > threshold
    
    # Adaptive learning rate parameters
    use_lr_scheduler: bool = False
    lr_scheduler_patience: int = 3  # Epochs to wait before reducing LR
    lr_scheduler_factor: float = 0.5  # Factor to multiply LR by
    lr_scheduler_threshold: float = 0.01  # Minimum decrease to be significant
    
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
    
    # Neural network architecture
    network: NetworkConfig = None
    
    def __post_init__(self):
        """Initialize network config if not provided."""
        if self.network is None:
            self.network = NetworkConfig()


def load_config_from_yaml(filepath: str) -> TrainingConfig:
    """Load training configuration from a YAML file."""
    with open(filepath, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Handle nested network configuration
    if 'network' in config_dict and isinstance(config_dict['network'], dict):
        network_dict = config_dict.pop('network')
        network_config = NetworkConfig(**network_dict)
        config_dict['network'] = network_config
    
    return TrainingConfig(**config_dict)