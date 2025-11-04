"""
AlphaZero training loop implementation.
Combines self-play data generation with neural network training.
"""
import os

# Suppress TensorFlow oneDNN optimization warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from .training import (
    TrainingConfig, 
    NetworkConfig,
    load_config_from_yaml,
    AlphaZeroTrainer,
    make_run_directory,
    save_config_to_run
)


def main():
    """Main training function."""
    
    # Create timestamped run directory
    run_dir = make_run_directory()
    
    # Load config from file (or fallback to defaults)
    config_path = 'uttt/uttt_config.yaml'
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        config = load_config_from_yaml(config_path)
    else:
        print(f"Config file {config_path} not found, using default values")
        config = TrainingConfig(
            n_epochs=5,           # More epochs for better training
            games_per_epoch=25,    # Moderate number of games
            mcts_simulations=3,  # Good balance of strength vs speed
            batch_size=32,
            learning_rate=0.001,
            save_every=1,          # Save every epoch to track progress
            checkpoint_dir="checkpoints",
            use_multiprocessing=True,  # Enable parallel self-play
            num_processes=5,    # Use all available CPUs
            network=NetworkConfig(
                in_planes=7,      # Standard UTTT state representation
                channels=64,      # Smaller network for testing
                blocks=3,         # Fewer blocks to avoid vanishing gradients
                board_n=9,        # UTTT board size
                policy_reduce=32, # Policy head reduction
                value_hidden=128  # Smaller value head
            )
        )
    

    
    # Update config to use the run-specific checkpoints directory
    config.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    
    # Save config files to the run directory
    save_config_to_run(config, config_path, run_dir)
    
    # Create trainer and start training
    trainer = AlphaZeroTrainer(config, run_dir)
    
    trainer.train()


if __name__ == "__main__":
    main()
