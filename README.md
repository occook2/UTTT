# Ultimate Tic-Tac-Toe AlphaZero

An implementation of the AlphaZero algorithm for Ultimate Tic-Tac-Toe, featuring Monte Carlo Tree Search (MCTS), neural network training, and comprehensive evaluation tools.

## Project Overview

This project implements:
- **AlphaZero agent** with deep neural network and MCTS
- **Self-play training** with data augmentation
- **Tournament system** for agent evaluation
- **Training data visualization** for debugging
- **Comprehensive metrics** including policy entropy and dead ReLU tracking

## Features

- 🧠 **Deep Neural Network**: Residual CNN architecture for board evaluation
- 🌳 **Monte Carlo Tree Search**: Efficient game tree exploration
- 🔄 **Self-Play Training**: Automated data generation and neural network improvement
- 📊 **Rich Metrics**: Loss tracking, policy entropy, dead ReLU percentage
- 🏆 **Tournament System**: Head-to-head agent evaluation with rating calculations
- 🎮 **Interactive Viewer**: Visualize training games and analyze agent decisions
- ⚡ **GPU Support**: CUDA acceleration for training and inference

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for training)

### Setup
1. Clone the repository:

2. Create and activate virtual environment:

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, use the setup script:
```bash
python setup.py
```

## Quick Start

### 1. Train an AlphaZero Agent

Configure training in `config.yaml`:
```yaml
# Training configuration
n_epochs: 50
games_per_epoch: 10
mcts_simulations: 25
learning_rate: 0.001

# To resume from checkpoint:
# resume_from_checkpoint: "runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt"
```

Start training:
```bash
python -m uttt.agents.az.train
```

**Output**: Creates `runs/run_YYYYMMDD_HHMMSS/` with:
- `checkpoints/`: Model checkpoints every N epochs
- `metrics/`: Training loss and metrics (CSV files)
- `training_ui_data/`: Visualization data for games

### 2. View Training Progress

Visualize training games and metrics:
```bash
python -m uttt.scripts.view_training_data
```

This opens an interactive viewer showing:
- Game board states and move sequences
- Policy probabilities and value estimates
- MCTS visit distributions
- Policy entropy and confidence metrics

### 3. Run Tournaments

Compare different agents:

**AlphaZero vs Random:**
```bash
python -m uttt.scripts.alphazero_tournament vs-baseline runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt --baseline random --games 100
```

**AlphaZero vs AlphaZero:**
```bash
python -m uttt.scripts.alphazero_tournament vs-alphazero runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt runs/run_20251017_212855/checkpoints/alphazero_epoch_28.pt --games 20
```

**Output**: Creates `tournaments/` with JSON files containing detailed game records.

### 4. Compute Ratings

Calculate ELO ratings from tournament results:
```bash
python -m uttt.eval.compute_ratings tournaments/
```

**Output**: 
- ELO ratings for all agents
- Statistical analysis of performance
- Confidence intervals

## Configuration Guide

### Training Configuration (`config.yaml`)

```yaml
# Core training parameters
n_epochs: 50                    # Number of training epochs
games_per_epoch: 10             # Self-play games per epoch
mcts_simulations: 25            # MCTS simulations per move

# Network architecture
network:
  channels: 64                  # CNN channel width
  blocks: 3                     # Number of ResNet blocks
  policy_reduce: 16             # Policy head reduction
  value_hidden: 64              # Value head hidden units

# Training hyperparameters
learning_rate: 0.001            # Adam learning rate
weight_decay: 0.0001            # L2 regularization
batch_size: 32                  # Training batch size

# MCTS parameters
c_puct: 1.0                     # Exploration constant
temperature: 1.0                # Move selection temperature
dirichlet_alpha: 0.3            # Root noise parameter

# System
device: "cuda"                  # "cuda" or "cpu"
use_multiprocessing: true       # Parallel self-play
num_processes: 5                # Number of processes

# Resume training (optional)
# resume_from_checkpoint: "runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt"
```

### Key Parameters

- **`n_epochs`**: When resuming, this is **additional** epochs to train
- **`mcts_simulations`**: More simulations = stronger play but slower training
- **`temperature`**: Controls randomness in move selection (0.0 = deterministic)
- **`c_puct`**: Higher values encourage more exploration in MCTS

## Project Structure

```
UTTT/
├── uttt/                       # Main package
│   ├── agents/                 # Agent implementations
│   │   ├── az/                 # AlphaZero agent
│   │   ├── random.py           # Random baseline
│   │   └── heuristic.py        # Heuristic baseline
│   ├── env/                    # Game environment
│   ├── mcts/                   # MCTS implementation
│   ├── eval/                   # Tournament and evaluation
│   └── scripts/                # Utility scripts
├── runs/                       # Training outputs
├── tournaments/                # Tournament results
├── config.yaml                 # Training configuration
├── requirements.txt            # Dependencies
└── docs/                       # Documentation
```

## Understanding the Output

### Training Metrics

The training process tracks several key metrics:

- **Total Loss**: Combined policy and value loss
- **Policy Loss**: Cross-entropy loss for move predictions
- **Value Loss**: MSE loss for position evaluation
- **Dead ReLU %**: Percentage of inactive neurons (health metric)
- **Policy Entropy**: Measure of move uncertainty (bits)

### Training Data Viewer

For each move, you can see:
- **Board State**: Current game position
- **Policy**: MCTS-computed move probabilities
- **Value**: Position evaluation (-1 to +1)
- **Policy Max**: Confidence in best move
- **Policy Entropy**: Uncertainty measure

### Tournament Results

Tournament files include:
- Complete game records with move sequences
- Final outcomes (win/loss/draw)
- Agent configurations and parameters
- Timing information

## Performance Tips

### Training
- Use GPU for faster training (`device: "cuda"`)
- Start with fewer epochs to test setup
- Monitor dead ReLU percentage (should be < 30%)
- Higher `mcts_simulations` improves data quality but slows training

### Evaluation
- Use opening books for fair comparisons
- Run tournaments with sufficient games (100+) for statistical significance
- Compare agents trained for different numbers of epochs

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `mcts_simulations`
2. **Training very slow**: Reduce `games_per_epoch` or `mcts_simulations`
3. **High dead ReLU %**: Lower learning rate or check network initialization
4. **Training data viewer crashes**: Check if training completed and data exists

### File Sizes
- Training runs can be large (1-2 GB per run due to UI data)
- Checkpoints are smaller (~10-50 MB each)
- Consider cleaning old runs periodically

## Example Results

With the provided example checkpoint (`alphazero_epoch_24.pt`):
- Trained for 24 epochs on Ultimate Tic-Tac-Toe
- Achieves >95% win rate against random play
- Shows structured strategic thinking in opening moves