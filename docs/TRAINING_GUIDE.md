# Training Guide

This guide provides detailed instructions for training your AlphaZero agent on Ultimate Tic-Tac-Toe.

## Table of Contents
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Monitoring Progress](#monitoring-progress)
- [Advanced Training](#advanced-training)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Basic Training
```bash
# Start training with default settings
python -m uttt.agents.az.train

# Train with custom configuration
python -m uttt.agents.az.train --config my_config.yaml

# Resume from checkpoint
python -m uttt.agents.az.train --resume-from runs/run_YYYYMMDD_HHMMSS/checkpoints/alphazero_epoch_10.pt
```

### 2. Training Phases
1. **Self-Play**: Generates training data by playing games
2. **Neural Network Training**: Updates the network on collected data
3. **Evaluation**: Tests the new network against the previous version
4. **Repeat**: Continues until convergence or max epochs

## Configuration

### Key Training Parameters

```yaml
training:
  n_epochs: 50              # Number of training epochs
  games_per_epoch: 1000     # Self-play games per epoch
  batch_size: 32            # Training batch size
  lr: 0.001                 # Learning rate
  weight_decay: 1e-4        # L2 regularization
  save_interval: 5          # Save checkpoint every N epochs
  
  # Checkpoint resuming
  resume_from_checkpoint: null  # Path to checkpoint file
  
  # Training data
  max_examples_buffer: 100000   # Max training examples to keep
  augment_data: true           # Use 8x data augmentation
```

### Model Architecture

```yaml
model:
  n_filters: 128            # ResNet filters
  n_blocks: 10              # ResNet blocks
  dropout: 0.3              # Dropout rate
  l2_reg: 1e-4              # L2 regularization
```

### MCTS Parameters

```yaml
mcts:
  n_simulations: 800        # MCTS simulations per move
  c_puct: 1.0              # UCB exploration constant
  temp_threshold: 30        # Temperature threshold for exploration
  dirichlet_alpha: 0.3     # Dirichlet noise parameter
  dirichlet_epsilon: 0.25  # Dirichlet noise weight
```

## Training Process

### Epoch Structure
Each training epoch consists of:

1. **Self-Play Phase** (5-10 minutes)
   - Generates `games_per_epoch` games
   - Uses current network for MCTS
   - Collects training examples (state, policy, value)

2. **Training Phase** (2-5 minutes)
   - Trains network on collected examples
   - Uses data augmentation (8x multiplier)
   - Tracks loss and dead ReLU metrics

3. **Evaluation Phase** (1-2 minutes)
   - Plays games against previous network
   - Decides whether to accept new network
   - Updates Elo ratings

### Data Collection
- **Training Examples**: (board_state, mcts_policy, game_outcome)
- **Augmentation**: 8 rotations/reflections of the board
- **Buffer Management**: Keeps most recent examples up to max_examples_buffer

### Neural Network Updates
- **Loss Function**: Policy cross-entropy + Value MSE + L2 regularization
- **Optimizer**: Adam with configurable learning rate
- **Batch Training**: Processes data in configurable batch sizes

## Monitoring Progress

### 1. TensorBoard
```bash
# Launch TensorBoard
python -m uttt.scripts.launch_tensorboard

# Then open: http://localhost:6006
```

**Available Metrics:**
- Loss curves (total, policy, value)
- Learning rate schedule
- Dead ReLU percentages
- Policy entropy
- Training accuracy

### 2. Training Data Viewer
```bash
# View training examples
python -m uttt.scripts.view_training_data
```

**Features:**
- Browse training examples by epoch
- Visualize board states and policies
- Check policy entropy and value predictions
- Validate data quality

### 3. File Structure
```
runs/run_YYYYMMDD_HHMMSS/
├── checkpoints/          # Model checkpoints
├── config/              # Training configuration
├── metrics/             # Training metrics (CSV)
├── tensorboard/         # TensorBoard logs
└── training_ui_data/    # UI viewer data
```

### 4. Metrics Files
- `training_metrics.csv`: Loss, accuracy, dead ReLUs
- `self_play_metrics.csv`: Game statistics
- `evaluation_metrics.csv`: Network comparisons

## Advanced Training

### Custom Training Loops

```python
from uttt.agents.az.training.trainer import AlphaZeroTrainer
from uttt.agents.az.training.config import load_config

# Load configuration
config = load_config('config.yaml')

# Custom modifications
config['training']['lr'] = 0.0005
config['training']['n_epochs'] = 100

# Initialize trainer
trainer = AlphaZeroTrainer(config)

# Custom training logic
for epoch in range(50):
    # Custom self-play
    trainer.self_play_step()
    
    # Custom training with learning rate decay
    if epoch % 10 == 0:
        trainer.lr *= 0.9
    
    trainer.training_step()
    
    # Custom evaluation
    if epoch % 5 == 0:
        trainer.evaluation_step()
```

### Curriculum Learning

```python
# Start with easier opponents
config['mcts']['n_simulations'] = 200  # Lower for weak opponents

# Gradually increase difficulty
for phase in range(5):
    config['mcts']['n_simulations'] += 200
    trainer = AlphaZeroTrainer(config)
    trainer.train(n_epochs=10)
```

### Transfer Learning

```python
# Load pretrained model
checkpoint = torch.load('pretrained_model.pt')
trainer.network.load_state_dict(checkpoint['model_state_dict'])

# Fine-tune with lower learning rate
config['training']['lr'] = 0.0001
trainer.train(n_epochs=20)
```

### Distributed Training

For multiple GPUs or machines:

```python
# Initialize with distributed settings
config['training']['distributed'] = True
config['training']['world_size'] = 4
config['training']['rank'] = 0  # Set per process

# Launch with: python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## Performance Optimization

### Memory Management
- **Batch Size**: Reduce if getting OOM errors
- **Buffer Size**: Limit max_examples_buffer for memory constraints
- **Augmentation**: Disable if memory is tight

### Speed Optimization
- **CUDA**: Ensure PyTorch uses GPU acceleration
- **Batch Size**: Increase for better GPU utilization
- **Workers**: Use multiple processes for data loading

### Training Tips
- **Start Small**: Begin with fewer simulations and smaller networks
- **Monitor Metrics**: Watch for overfitting and dead ReLUs
- **Checkpoint Frequently**: Save every few epochs
- **Validate Data**: Use training data viewer to check quality

## Troubleshooting

### Common Issues

#### 1. Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solutions:**
- Reduce `batch_size`
- Reduce `n_simulations`
- Reduce `max_examples_buffer`
- Enable gradient checkpointing

#### 2. Network Not Learning
**Symptoms:** Loss not decreasing, random play
**Solutions:**
- Check learning rate (try 0.001 to 0.01)
- Verify data augmentation is working
- Increase network capacity
- Check for dead ReLUs

#### 3. Training Too Slow
**Solutions:**
- Enable CUDA if available
- Increase batch size
- Reduce MCTS simulations for self-play
- Use fewer training epochs

#### 4. Unstable Training
**Symptoms:** Loss oscillating, performance degrading
**Solutions:**
- Reduce learning rate
- Add gradient clipping
- Increase L2 regularization
- Use learning rate scheduling

#### 5. Dead ReLUs
**Symptoms:** High dead ReLU percentage (>50%)
**Solutions:**
- Reduce learning rate
- Add batch normalization
- Use ReLU alternatives (LeakyReLU, ELU)
- Reduce network depth

### Performance Debugging

#### Check GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

#### Profile Training
```python
import torch.profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    trainer.training_step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

#### Memory Profiling
```python
import torch

# Monitor memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Best Practices

### 1. Experiment Tracking
- Use descriptive run names
- Document configuration changes
- Track validation metrics
- Save important checkpoints

### 2. Incremental Training
- Start with small networks
- Gradually increase complexity
- Use curriculum learning
- Monitor overfitting

### 3. Data Quality
- Validate training examples
- Check policy distributions
- Monitor game statistics
- Use data viewer regularly

### 4. Model Selection
- Compare multiple checkpoints
- Use tournament evaluation
- Track Elo ratings
- Test against baselines

### 5. Reproducibility
- Set random seeds
- Document exact configurations
- Save complete environments
- Version control everything

## Next Steps

After successful training:
1. **Evaluate Performance**: Run tournaments against baselines
2. **Analyze Play Style**: Review game logs and decision patterns  
3. **Fine-tune**: Adjust hyperparameters based on results
4. **Scale Up**: Increase network size and training time
5. **Experiment**: Try different architectures and training methods

For more advanced topics, see:
- [Tournament Guide](TOURNAMENT_GUIDE.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Examples](EXAMPLES.md)