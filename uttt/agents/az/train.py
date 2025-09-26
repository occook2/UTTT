"""
AlphaZero training loop implementation.
Combines self-play data generation with neural network training.
"""
import os
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.az.net import AlphaZeroNetUTTT
from uttt.agents.az.loss import AlphaZeroLoss
from uttt.agents.az.self_play import SelfPlayTrainer, TrainingExample, run_self_play_session
from uttt.mcts.base import MCTSConfig


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
    
    # Model saving
    save_every: int = 10  # Save model every N epochs
    checkpoint_dir: str = "checkpoints"
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Training data management
    max_training_samples: int = 50000  # Keep only the most recent samples


class AlphaZeroDataset(Dataset):
    """Dataset for AlphaZero training examples."""
    
    def __init__(self, training_examples: List[TrainingExample]):
        self.examples = training_examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            'state': torch.FloatTensor(example.state),
            'policy': torch.FloatTensor(example.policy),
            'value': torch.FloatTensor([example.value])
        }


class AlphaZeroTrainer:
    """Main trainer for AlphaZero agent."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Initialize neural network
        self.network = AlphaZeroNetUTTT()
        self.network.to(self.config.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Initialize loss function
        self.loss_fn = AlphaZeroLoss()
        
        # Initialize agent for self-play
        self.agent = AlphaZeroAgent(
            network=self.network,
            device=self.config.device
        )
        
        # Training data storage
        self.training_examples: List[TrainingExample] = []
        
        # Training statistics
        self.training_stats = {
            'epoch_losses': [],
            'policy_losses': [],
            'value_losses': [],
            'games_played': 0
        }
        
        # Create checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def train(self):
        """Run the complete AlphaZero training loop."""
        print(f"Starting AlphaZero training: {self.config.n_epochs} epochs")
        print(f"Games per epoch: {self.config.games_per_epoch}")
        print(f"MCTS simulations: {self.config.mcts_simulations}")
        print(f"Device: {self.config.device}")
        print("-" * 60)
        
        for epoch in range(1, self.config.n_epochs + 1):
            epoch_start_time = time.time()
            
            # Generate self-play data
            new_examples = self._generate_self_play_data(epoch)
            
            # Add to training dataset and manage size
            self._update_training_data(new_examples)
            
            # Train neural network on collected data
            if len(self.training_examples) >= self.config.batch_size:
                print("Training neural network...")
                epoch_loss, policy_loss, value_loss = self._train_network()
                
                # Record statistics
                self.training_stats['epoch_losses'].append(epoch_loss)
                self.training_stats['policy_losses'].append(policy_loss)
                self.training_stats['value_losses'].append(value_loss)
                
                print(f"Epoch {epoch} - Loss: {epoch_loss:.4f} "
                      f"(Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
            else:
                print(f"Not enough training data yet ({len(self.training_examples)} samples)")
            
            # Save model checkpoint
            if epoch % self.config.save_every == 0:
                self._save_checkpoint(epoch)
            
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            print(f"Total games played: {self.training_stats['games_played']}")
            print(f"Training examples: {len(self.training_examples)}")
            print("-" * 60)
        
        # Save final model
        self._save_final_model()
        print("Training completed!")
    
    def _generate_self_play_data(self, epoch_num: int) -> List[TrainingExample]:
        """Generate training data through self-play with clean progress output."""
        mcts_config = MCTSConfig(
            n_simulations=self.config.mcts_simulations,
            c_puct=1.0,
            temperature=1.0,
            use_transposition_table=True
        )
        
        trainer = SelfPlayTrainer(
            agent=self.agent,
            mcts_config=mcts_config,
            temperature_threshold=self.config.temperature_threshold,
            collect_data=True
        )
        
        examples = []
        total_games = self.config.games_per_epoch
        
        # Show initial progress line
        print(f"Running Epoch {epoch_num}: Completed 0/{total_games} games", end='', flush=True)
        
        for i in range(total_games):
            game_examples, _ = trainer.play_game()
            examples.extend(game_examples)
            
            # Update progress in-place
            print(f"\rRunning Epoch {epoch_num}: Completed {i+1}/{total_games} games", end='', flush=True)
        
        # Move to next line after completion
        print()
        
        self.training_stats['games_played'] += total_games
        return examples
    
    def _update_training_data(self, new_examples: List[TrainingExample]):
        """Add new examples and manage training data size."""
        self.training_examples.extend(new_examples)
        
        # Keep only the most recent samples to prevent memory issues
        if len(self.training_examples) > self.config.max_training_samples:
            excess = len(self.training_examples) - self.config.max_training_samples
            self.training_examples = self.training_examples[excess:]
            print(f"Trimmed training data to {len(self.training_examples)} samples")
    
    def _train_network(self) -> tuple[float, float, float]:
        """Train the neural network on collected data."""
        # Create dataset and dataloader
        dataset = AlphaZeroDataset(self.training_examples)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Use 0 for Windows compatibility
        )
        
        self.network.train()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0
        
        for batch in dataloader:
            # Move data to device
            states = batch['state'].to(self.config.device)
            target_policies = batch['policy'].to(self.config.device)
            target_values = batch['value'].to(self.config.device).squeeze()
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            pred_policies, pred_values = self.network(states)
            pred_values = pred_values.squeeze()
            
            # Compute loss
            loss, policy_loss, value_loss = self.loss_fn(
                pred_policies, target_policies,
                pred_values, target_values
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1
        
        # Return average losses
        avg_loss = total_loss / n_batches
        avg_policy_loss = total_policy_loss / n_batches
        avg_value_loss = total_value_loss / n_batches
        
        return avg_loss, avg_policy_loss, avg_value_loss
    
    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"alphazero_epoch_{epoch}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save the final trained model."""
        final_path = os.path.join(self.config.checkpoint_dir, "alphazero_final.pt")
        
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'training_stats': self.training_stats,
            'config': self.config
        }, final_path)
        
        print(f"Saved final model: {final_path}")
    
    def _print_training_summary(self):
        """Print summary of training results."""
        print("\nTraining Summary:")
        print(f"Total games played: {self.training_stats['games_played']}")
        print(f"Final training examples: {len(self.training_examples)}")
        
        if self.training_stats['epoch_losses']:
            print(f"Final loss: {self.training_stats['epoch_losses'][-1]:.4f}")
            print(f"Best loss: {min(self.training_stats['epoch_losses']):.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint to resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


def main():
    """Main training function."""
    # Create training configuration
    config = TrainingConfig(
        n_epochs=20,           # More epochs for better training
        games_per_epoch=25,    # Moderate number of games
        mcts_simulations=200,  # Good balance of strength vs speed
        batch_size=32,
        learning_rate=0.001,
        save_every=1,          # Save every epoch to track progress
        checkpoint_dir="checkpoints"
    )
    
    # Create trainer and start training
    trainer = AlphaZeroTrainer(config)
    
    # Optional: Load from checkpoint to resume training
    # start_epoch = trainer.load_checkpoint("checkpoints/alphazero/alphazero_epoch_10.pt")
    
    trainer.train()


if __name__ == "__main__":
    main()
