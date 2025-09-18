"""
Test script for AlphaZero training pipeline.
This is a minimal test to verify that all components work together.
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from uttt.agents.az.train import TrainingConfig, AlphaZeroTrainer


def test_training_pipeline():
    """Test the AlphaZero training pipeline with minimal configuration."""
    print("Testing AlphaZero training pipeline...")
    
    # Create a minimal training configuration for testing
    config = TrainingConfig(
        n_epochs=2,           # Just 2 epochs for testing
        games_per_epoch=3,    # Only 3 games per epoch
        batch_size=2,         # Small batch size
        learning_rate=0.01,   # Higher learning rate for faster convergence in test
        mcts_simulations=5,   # Very few simulations for speed
        save_every=1,         # Save every epoch for testing
        checkpoint_dir="test_checkpoints",
        max_training_samples=20  # Keep dataset small
    )
    
    print(f"Configuration:")
    print(f"  Epochs: {config.n_epochs}")
    print(f"  Games per epoch: {config.games_per_epoch}")
    print(f"  MCTS simulations: {config.mcts_simulations}")
    print(f"  Device: {config.device}")
    
    # Create trainer
    trainer = AlphaZeroTrainer(config)
    
    # Run training - this should complete without exceptions
    trainer.train()
    
    print("✅ Training pipeline test completed successfully!")
    
    # Verify that checkpoints were created
    checkpoint_dir = config.checkpoint_dir
    assert os.path.exists(checkpoint_dir), f"Checkpoint directory {checkpoint_dir} should exist"
    
    files = os.listdir(checkpoint_dir)
    assert len(files) > 0, "At least one checkpoint file should be created"
    print(f"✅ Created {len(files)} checkpoint files: {files}")
    
    # Verify training statistics were recorded
    assert len(trainer.training_stats['epoch_losses']) > 0, "Training losses should be recorded"
    assert trainer.training_stats['games_played'] > 0, "Games should have been played"
    
    print("✅ All assertions passed!")


if __name__ == "__main__":
    # When run directly, execute the test function
    try:
        test_training_pipeline()
        print("\n🎉 All tests passed! The AlphaZero training pipeline is working.")
    except Exception as e:
        print(f"\n💥 Tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)