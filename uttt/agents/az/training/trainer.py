"""
AlphaZero trainer implementation.
"""
import os
import time
import shutil
import torch
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import List, Dict, Any

from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.az.net import AlphaZeroNetUTTT, AZNetConfig
from uttt.agents.az.loss import AlphaZeroLoss
from uttt.agents.az.self_play import SelfPlayTrainer, TrainingExample
from uttt.agents.az.symmetry import augment_examples_with_rotations
from uttt.mcts.base import MCTSConfig

from .config import TrainingConfig
from .dataset import AlphaZeroDataset
from .metrics import log_training_metrics, log_gradient_metrics, log_parameter_metrics
from .utils import save_training_games_for_ui


class AlphaZeroTrainer:
    """Main trainer for AlphaZero agent."""
    
    def __init__(self, config: TrainingConfig, run_dir: str = None):
        self.config = config
        self.run_dir = run_dir  # Store run directory for UI data saving
        
        # Set up TensorBoard logging
        self.tensorboard_dir = os.path.join(run_dir, 'tensorboard') if run_dir else None
        self.writer = None
        if self.tensorboard_dir:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
            self._setup_shared_tensorboard_logging()
        
        # Training step counter for TensorBoard
        self.training_step = 0
        
        # Convert NetworkConfig to AZNetConfig
        net_config = AZNetConfig(
            in_planes=config.network.in_planes,
            channels=config.network.channels,
            blocks=config.network.blocks,
            board_n=config.network.board_n,
            policy_reduce=config.network.policy_reduce,
            value_hidden=config.network.value_hidden
        )
        
        self.network = AlphaZeroNetUTTT(net_config)
        self.network.to(self.config.device)
        
        # Create necessary directories
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        if self.run_dir:
            os.makedirs(os.path.join(self.run_dir, "metrics"), exist_ok=True)
        
        # Log initial network parameters (epoch 0) to track weight changes
        if self.run_dir:
            log_parameter_metrics(self.run_dir, 0, self.network)
        
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
    
    def _setup_shared_tensorboard_logging(self):
        """Set up shared TensorBoard logging for cross-run comparison."""
        if not self.run_dir:
            return
            
        shared_tensorboard_dir = os.path.join('tensorboard_logs', os.path.basename(self.run_dir))
        
        try:
            os.makedirs('tensorboard_logs', exist_ok=True)
            
            # On Windows, try to create a junction, otherwise copy logs later
            if os.name == 'nt':  # Windows
                try:
                    import subprocess
                    subprocess.run([
                        'mklink', '/J', 
                        os.path.abspath(shared_tensorboard_dir),
                        os.path.abspath(self.tensorboard_dir)
                    ], shell=True, check=True, capture_output=True)
                    print(f"Created TensorBoard junction: {shared_tensorboard_dir}")
                except:
                    self.shared_tensorboard_dir = shared_tensorboard_dir
                    print(f"Will copy TensorBoard logs to {shared_tensorboard_dir} after training")
            else:  # Unix/Linux/Mac
                try:
                    os.symlink(
                        os.path.abspath(self.tensorboard_dir),
                        os.path.abspath(shared_tensorboard_dir)
                    )
                    print(f"Created TensorBoard symlink: {shared_tensorboard_dir}")
                except:
                    self.shared_tensorboard_dir = shared_tensorboard_dir
                    print(f"Will copy TensorBoard logs to {shared_tensorboard_dir} after training")
        except Exception as e:
            print(f"Warning: Could not set up shared TensorBoard logging: {e}")
    
    def _log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], self_play_stats: Dict[str, Any] = None):
        """Log training metrics to TensorBoard."""
        if not self.writer:
            return
            
        self.training_step = epoch
        
        # Log losses
        self.writer.add_scalar('Loss/Policy', metrics['policy_loss'], self.training_step)
        self.writer.add_scalar('Loss/Value', metrics['value_loss'], self.training_step)
        self.writer.add_scalar('Loss/Total', metrics['total_loss'], self.training_step)
        
        # Log learning rate
        self.writer.add_scalar('Training/Learning_Rate', self.optimizer.param_groups[0]['lr'], self.training_step)
        
        # Log self-play statistics if available
        if self_play_stats:
            self.writer.add_scalar('SelfPlay/Games_Played', self_play_stats.get('games_played', 0), self.training_step)
            self.writer.add_scalar('SelfPlay/Avg_Game_Length', self_play_stats.get('avg_game_length', 0), self.training_step)
            self.writer.add_scalar('SelfPlay/Win_Rate_P1', self_play_stats.get('win_rate_p1', 0.5), self.training_step)
            self.writer.add_scalar('SelfPlay/Win_Rate_P2', self_play_stats.get('win_rate_p2', 0.5), self.training_step)
            self.writer.add_scalar('SelfPlay/Draw_Rate', self_play_stats.get('draw_rate', 0), self.training_step)
        
        # Log training data size
        self.writer.add_scalar('Training/Dataset_Size', len(self.training_examples), self.training_step)
        
        # Flush the writer
        self.writer.flush()
    
    def _copy_logs_to_shared_dir(self):
        """Copy TensorBoard logs to shared directory for comparison."""
        if not hasattr(self, 'shared_tensorboard_dir') or not self.tensorboard_dir:
            return
            
        try:
            if not os.path.exists(self.shared_tensorboard_dir):
                shutil.copytree(self.tensorboard_dir, self.shared_tensorboard_dir)
                print(f"Copied TensorBoard logs to {self.shared_tensorboard_dir}")
        except Exception as e:
            print(f"Warning: Could not copy TensorBoard logs: {e}")
    
    def close(self):
        """Clean up TensorBoard writer and copy logs if needed."""
        if self.writer:
            self.writer.close()
            self.writer = None
        
        # Copy logs to shared directory if needed
        self._copy_logs_to_shared_dir()
    
    def __del__(self):
        """Ensure TensorBoard writer is closed."""
        try:
            self.close()
        except:
            pass
    
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
            
            # Apply symmetry augmentation to get 4x training data
            aug_examples = augment_examples_with_rotations(new_examples)
            print(f"Generated {len(new_examples)} examples, augmented to {len(aug_examples)} examples (8x with rotations and reflections)")
            
            # Save UI-friendly data for inspection (separate from training)
            if getattr(self.config, 'save_ui_data', True):
                save_training_games_for_ui(aug_examples, epoch, self.config, self.run_dir)
            
            # Add to training dataset and manage size
            self._update_training_data(aug_examples)
            # Train neural network on collected data
            if len(self.training_examples) >= self.config.batch_size:
                print("Training neural network...")
                epoch_loss, policy_loss, value_loss = self._train_network()
                
                # Record statistics
                self.training_stats['epoch_losses'].append(epoch_loss)
                self.training_stats['policy_losses'].append(policy_loss)
                self.training_stats['value_losses'].append(value_loss)
                
                # Prepare metrics for logging
                metrics = {
                    'total_loss': epoch_loss,
                    'policy_loss': policy_loss,
                    'value_loss': value_loss
                }
                
                # Calculate self-play statistics
                self_play_stats = {
                    'games_played': self.config.games_per_epoch,
                    'avg_game_length': self._calculate_avg_game_length(new_examples),
                    'win_rate_p1': self._calculate_win_rates(new_examples)['p1'],
                    'win_rate_p2': self._calculate_win_rates(new_examples)['p2'],
                    'draw_rate': self._calculate_win_rates(new_examples)['draws']
                }
                
                # Log metrics to TensorBoard
                self._log_epoch_metrics(epoch, metrics, self_play_stats)
                
                # Log metrics to CSV files
                if self.run_dir:
                    log_training_metrics(self.run_dir, epoch, epoch_loss, policy_loss, value_loss)
                    log_gradient_metrics(self.run_dir, epoch, self.network)
                    log_parameter_metrics(self.run_dir, epoch, self.network)
                
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
        
        # Close TensorBoard writer
        self.close()
        
        print("Training completed!")
    
    def _calculate_avg_game_length(self, examples: List[TrainingExample]) -> float:
        """Calculate average game length from training examples."""
        if not examples:
            return 0.0
        
        # Count examples per game (each game contributes multiple examples)
        # Estimate game count by assuming roughly 30-60 moves per game
        estimated_games = max(1, len(examples) // 45)  # Rough estimate
        return len(examples) / estimated_games
    
    def _calculate_win_rates(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """Calculate win rates from training examples."""
        if not examples:
            return {'p1': 0.5, 'p2': 0.5, 'draws': 0.0}
        
        # Count outcomes (values are from perspective of current player)
        wins = sum(1 for ex in examples if ex.value == 1.0)
        losses = sum(1 for ex in examples if ex.value == -1.0)
        draws = sum(1 for ex in examples if ex.value == 0.0)
        
        total = len(examples)
        if total == 0:
            return {'p1': 0.5, 'p2': 0.5, 'draws': 0.0}
        
        # Since examples alternate between players, roughly estimate
        return {
            'p1': wins / total,
            'p2': losses / total,
            'draws': draws / total
        }
    
    def _generate_self_play_data(self, epoch_num: int) -> List[TrainingExample]:
        """Generate training data through self-play with clean progress output."""
        mcts_config = MCTSConfig(
            n_simulations=self.config.mcts_simulations,
            c_puct=self.config.c_puct,
            temperature=self.config.temperature,
            use_transposition_table=True
        )
        
        trainer = SelfPlayTrainer(
            agent=self.agent,
            mcts_config=mcts_config,
            temperature_threshold=self.config.temperature_threshold,
            collect_data=True
        )
        
        total_games = self.config.games_per_epoch
        
        if self.config.use_multiprocessing and total_games > 1:
            # Use parallel self-play
            num_processes = self.config.num_processes
            if num_processes is None:
                num_processes = min(mp.cpu_count(), total_games)
            
            print(f"Running Epoch {epoch_num}: Generating {total_games} games using {num_processes} processes...")
            
            try:
                examples = trainer.generate_training_data_parallel(
                    n_games=total_games,
                    num_processes=num_processes,
                    show_progress=False  # We handle progress here
                )
                print(f"Epoch {epoch_num}: Completed all {total_games} games in parallel")
            except Exception as e:
                print(f"\nParallel execution failed, falling back to sequential: {e}")
                # Fallback to sequential execution
                examples = self._generate_sequential_games(trainer, epoch_num, total_games)
        else:
            # Use sequential self-play
            examples = self._generate_sequential_games(trainer, epoch_num, total_games)
        
        self.training_stats['games_played'] += total_games
        return examples
    
    def _generate_sequential_games(self, trainer, epoch_num: int, total_games: int) -> List[TrainingExample]:
        """Generate games sequentially with progress updates."""
        examples = []
        
        # Show initial progress line
        print(f"Running Epoch {epoch_num}: Completed 0/{total_games} games", end='', flush=True)
        
        for i in range(total_games):
            game_examples, _ = trainer.play_game()
            examples.extend(game_examples)
            
            # Update progress in-place
            print(f"\rRunning Epoch {epoch_num}: Completed {i+1}/{total_games} games", end='', flush=True)
        
        # Move to next line after completion
        print()
        
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
            
            
            pred_policies, pred_values = self.network(states)
            pred_values = pred_values.squeeze()
            
            # Compute loss
            loss, policy_loss, value_loss = self.loss_fn(
                pred_policies, target_policies,
                pred_values, target_values
            )

            # Backward pass
            self.optimizer.zero_grad()
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