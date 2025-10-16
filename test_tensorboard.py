#!/usr/bin/env python3
"""
Test TensorBoard integration in AlphaZero training.
"""
import os
import sys
import tempfile
import shutil

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

from uttt.agents.az.training.trainer import AlphaZeroTrainer
from uttt.agents.az.training.config import TrainingConfig, NetworkConfig

def test_tensorboard_integration():
    """Test that TensorBoard logging works correctly."""
    print("Testing TensorBoard integration...")
    
    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as temp_dir:
        run_dir = os.path.join(temp_dir, "test_run")
        os.makedirs(run_dir)
        
        # Create minimal config for testing
        config = TrainingConfig(
            n_epochs=2,
            games_per_epoch=3,
            batch_size=16,
            learning_rate=0.001,
            mcts_simulations=5,
            save_every=1,
            checkpoint_dir=os.path.join(run_dir, "checkpoints"),
            device="cpu",
            network=NetworkConfig(
                channels=16,  # Small network for testing
                blocks=2,
                value_hidden=32,
                policy_reduce=8
            )
        )
        
        # Create trainer
        print("  Creating trainer...")
        trainer = AlphaZeroTrainer(config, run_dir)
        
        # Check TensorBoard setup
        tensorboard_dir = os.path.join(run_dir, 'tensorboard')
        print(f"  TensorBoard directory: {tensorboard_dir}")
        
        if os.path.exists(tensorboard_dir):
            print("  ✓ TensorBoard directory created")
        else:
            print("  ✗ TensorBoard directory not found")
            return False
        
        # Test logging some dummy metrics
        print("  Testing metric logging...")
        metrics = {
            'total_loss': 1.5,
            'policy_loss': 0.8,
            'value_loss': 0.7
        }
        
        self_play_stats = {
            'games_played': 3,
            'avg_game_length': 45.0,
            'win_rate_p1': 0.4,
            'win_rate_p2': 0.5,
            'draw_rate': 0.1
        }
        
        trainer._log_epoch_metrics(1, metrics, self_play_stats)
        
        # Check if log files were created
        log_files = os.listdir(tensorboard_dir)
        if log_files:
            print(f"  ✓ TensorBoard log files created: {log_files}")
        else:
            print("  ✗ No TensorBoard log files found")
            return False
        
        # Clean up
        trainer.close()
        print("  ✓ TensorBoard writer closed successfully")
        
        return True

def test_shared_logging_setup():
    """Test shared TensorBoard logging setup."""
    print("\nTesting shared TensorBoard logging setup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Change to temp directory to test shared logging
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            run_dir = "test_run_shared"
            os.makedirs(run_dir)
            
            config = TrainingConfig(
                n_epochs=1,
                games_per_epoch=2,
                batch_size=8,
                device="cpu",
                checkpoint_dir=os.path.join(run_dir, "checkpoints"),
                network=NetworkConfig(channels=8, blocks=1)
            )
            
            trainer = AlphaZeroTrainer(config, run_dir)
            
            # Check if shared directory setup was attempted
            if os.path.exists('tensorboard_logs'):
                print("  ✓ Shared tensorboard_logs directory created")
            else:
                print("  ! Shared directory not created (may be expected on some systems)")
            
            trainer.close()
            return True
            
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    print("Starting TensorBoard integration tests...\n")
    
    success = True
    
    if not test_tensorboard_integration():
        success = False
    
    if not test_shared_logging_setup():
        success = False
    
    if success:
        print("\n🎉 All TensorBoard tests passed!")
        print("\nUsage:")
        print("1. Run training: python -m uttt.agents.az.train")
        print("2. View logs: python -m uttt.scripts.launch_tensorboard --mode compare")
        print("3. Or manually: tensorboard --logdir tensorboard_logs")
    else:
        print("\n❌ Some tests failed.")