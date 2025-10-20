# Checkpoint Resuming Guide

## Overview
You can now resume AlphaZero training from any saved checkpoint. All configuration is done through the `config.yaml` file.

## How to Resume Training

### Step 1: Find Available Checkpoints
Check your training runs for available checkpoints:
```
runs/
  run_20251017_212855/
    checkpoints/
      alphazero_epoch_24.pt  ← Latest checkpoint
      alphazero_epoch_22.pt
      alphazero_epoch_20.pt
      ...
```

### Step 2: Update config.yaml
Edit your `config.yaml` file and uncomment the `resume_from_checkpoint` line:

```yaml
# Model saving
save_every: 2
checkpoint_dir: "checkpoints"
resume_from_checkpoint: "runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt"
```

### Step 3: Start Training
Run training normally:
```bash
python -m uttt.agents.az.train
```

The training will automatically:
1. Load the neural network weights from the checkpoint
2. Restore the optimizer state (learning rate, momentum, etc.)
3. Continue from the next epoch (e.g., if you load epoch 24, it starts at epoch 25)
4. Run for `n_epochs` **additional** epochs from the checkpoint
5. Preserve all training statistics and scheduler state

**Important**: When resuming, `n_epochs` means "run N additional epochs", not "stop at epoch N".
- Example: Resume from epoch 24 with `n_epochs: 5` → runs epochs 25, 26, 27, 28, 29

## What Gets Saved/Loaded

Each checkpoint contains:
- **Neural network weights**: The trained model parameters
- **Optimizer state**: Adam optimizer momentum, learning rates, etc.
- **Training statistics**: Loss history, training progress
- **Epoch number**: Which epoch the checkpoint was saved at
- **Scheduler state**: Learning rate scheduler state (if enabled)

## Example Output

When resuming, you'll see output like:
```
Loading config from config.yaml
Resuming training from checkpoint: runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt
Loaded checkpoint from runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt
Resuming from epoch 25
Starting AlphaZero training: epochs 25 to 29  # If n_epochs: 5
Games per epoch: 5
MCTS simulations: 10
Device: cuda
------------------------------------------------------------
```

**Note**: The end epoch (29) = checkpoint epoch (24) + n_epochs (5)

## Fresh Training

To start fresh training, simply comment out or remove the `resume_from_checkpoint` line:
```yaml
# Model saving
save_every: 2
checkpoint_dir: "checkpoints"
# resume_from_checkpoint: "runs/run_20251017_212855/checkpoints/alphazero_epoch_24.pt"
```

## Tips

1. **Always use absolute or relative paths**: Ensure the checkpoint path is correct
2. **Check file exists**: If the checkpoint file doesn't exist, training will start fresh with a warning
3. **Compatible architecture**: Only resume from checkpoints with the same network architecture
4. **Backup checkpoints**: Keep copies of important checkpoints before experimenting

## Architecture Compatibility

Make sure your `network` configuration in `config.yaml` matches the checkpoint you're loading:
```yaml
network:
  in_planes: 7          # Must match checkpoint
  channels: 32          # Must match checkpoint  
  blocks: 3             # Must match checkpoint
  board_n: 9            # Must match checkpoint
  policy_reduce: 16     # Must match checkpoint
  value_hidden: 64      # Must match checkpoint
```

Mismatched architectures will cause loading errors.