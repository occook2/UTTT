# Examples

This document provides practical examples for using the Ultimate Tic-Tac-Toe AlphaZero implementation.

## Table of Contents
- [Quick Start Examples](#quick-start-examples)
- [Training Examples](#training-examples)
- [Evaluation Examples](#evaluation-examples)
- [Custom Agent Examples](#custom-agent-examples)
- [Analysis Examples](#analysis-examples)
- [Integration Examples](#integration-examples)

## Quick Start Examples

### 1. Basic Training
```bash
# Train with default settings
python -m uttt.agents.az.train

# Monitor progress with TensorBoard
python -m uttt.scripts.launch_tensorboard
# Open http://localhost:6006
```

### 2. Play Against Your Trained Agent
```bash
# Interactive play
python -m uttt.scripts.play_vs_agent

# Specify agents
python -m uttt.scripts.play_vs_agent --agent1 human --agent2 alphazero
```

### 3. Quick Tournament
```bash
# Run tournament between different agents
python -m uttt.scripts.alphazero_tournament

# View results
python -m uttt.scripts.view_tournament
```

## Training Examples

### 1. Custom Training Configuration
```python
# custom_config.py
import yaml
from uttt.agents.az.train import main

# Create custom configuration
config = {
    'alphazero': {
        'lr': 0.0005,               # Lower learning rate
        'batch_size': 64,           # Larger batches
        'n_epochs': 100,            # More training
        'games_per_epoch': 2000,    # More self-play
    },
    'model': {
        'n_filters': 256,           # Bigger network
        'n_blocks': 20,             # Deeper network
        'dropout': 0.2,             # Less dropout
    },
    'mcts': {
        'n_simulations': 1600,      # More simulations
        'c_puct': 1.5,             # More exploration
    }
}

# Save configuration
with open('big_model_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Train with custom config
if __name__ == '__main__':
    main(['--config', 'big_model_config.yaml'])
```

### 2. Resume Training from Checkpoint
```python
# resume_training.py
import yaml

# Load existing config and modify
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Set checkpoint to resume from
config['training']['resume_from_checkpoint'] = 'runs/run_20241018_143012/checkpoints/alphazero_epoch_30.pt'

# Continue training for more epochs
config['training']['n_epochs'] = 50  # Additional epochs

# Save updated config
with open('resume_config.yaml', 'w') as f:
    yaml.dump(config, f)

# Resume training
from uttt.agents.az.train import main
main(['--config', 'resume_config.yaml'])
```

### 3. Curriculum Learning
```python
# curriculum_training.py
import yaml
from uttt.agents.az.train import main

def train_curriculum():
    """Train with increasing difficulty curriculum"""
    
    # Phase 1: Fast training with weak MCTS
    config_phase1 = {
        'mcts': {'n_simulations': 200},
        'training': {'n_epochs': 20, 'games_per_epoch': 1000}
    }
    
    with open('phase1_config.yaml', 'w') as f:
        yaml.dump(config_phase1, f)
    
    print("Phase 1: Basic training...")
    main(['--config', 'phase1_config.yaml'])
    
    # Phase 2: Medium strength
    config_phase2 = {
        'mcts': {'n_simulations': 600},
        'training': {
            'n_epochs': 30,
            'resume_from_checkpoint': 'runs/latest/checkpoints/alphazero_epoch_20.pt'
        }
    }
    
    with open('phase2_config.yaml', 'w') as f:
        yaml.dump(config_phase2, f)
    
    print("Phase 2: Intermediate training...")
    main(['--config', 'phase2_config.yaml'])
    
    # Phase 3: Full strength
    config_phase3 = {
        'mcts': {'n_simulations': 1200},
        'training': {
            'n_epochs': 50,
            'resume_from_checkpoint': 'runs/latest/checkpoints/alphazero_epoch_30.pt'
        }
    }
    
    with open('phase3_config.yaml', 'w') as f:
        yaml.dump(config_phase3, f)
    
    print("Phase 3: Advanced training...")
    main(['--config', 'phase3_config.yaml'])

if __name__ == '__main__':
    train_curriculum()
```

### 4. Multi-GPU Training
```python
# distributed_training.py
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from uttt.agents.az.training.trainer import AlphaZeroTrainer

def train_worker(rank, world_size, config):
    """Worker function for distributed training"""
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    
    # Create trainer with distributed settings
    config['training']['distributed'] = True
    config['training']['rank'] = rank
    config['training']['world_size'] = world_size
    
    trainer = AlphaZeroTrainer(config, device=device)
    trainer.train()
    
    # Cleanup
    dist.destroy_process_group()

def main():
    """Launch distributed training"""
    world_size = torch.cuda.device_count()
    
    if world_size < 2:
        print("Need at least 2 GPUs for distributed training")
        return
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Launch processes
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )

if __name__ == '__main__':
    main()
```

## Evaluation Examples

### 1. Comprehensive Agent Evaluation
```python
# evaluate_agent.py
import json
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent
from uttt.eval.tournaments import RoundRobinTournament

def evaluate_training_progress():
    """Evaluate multiple checkpoints against baselines"""
    
    # Load different epochs
    checkpoints = [
        ('Epoch-10', 'runs/run_20241018_143012/checkpoints/alphazero_epoch_10.pt'),
        ('Epoch-20', 'runs/run_20241018_143012/checkpoints/alphazero_epoch_20.pt'),
        ('Epoch-30', 'runs/run_20241018_143012/checkpoints/alphazero_epoch_30.pt'),
    ]
    
    # Create agents
    agents = {}
    
    # Add AlphaZero variants
    for name, checkpoint_path in checkpoints:
        agent = create_alphazero_agent(
            checkpoint_path=checkpoint_path,
            mcts_simulations=800
        )
        agents[name] = agent
    
    # Add baselines
    agents['Random'] = RandomAgent()
    agents['Heuristic'] = HeuristicAgent()
    
    # Run tournament
    tournament = RoundRobinTournament(
        agents=agents,
        games_per_matchup=100,
        show_progress=True
    )
    
    results = tournament.run()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results.save(f'tournaments/training_progress_{timestamp}.json')
    
    # Print summary
    print("Tournament Results:")
    print("=" * 50)
    for agent_name, stats in results.agent_stats.items():
        print(f"{agent_name:15} | Win Rate: {stats.win_rate:.1%} | Elo: {stats.elo:.0f}")
    
    return results

if __name__ == '__main__':
    results = evaluate_training_progress()
```

### 2. Performance vs Computation Analysis
```python
# performance_analysis.py
import matplotlib.pyplot as plt
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.heuristic import HeuristicAgent
from uttt.eval.tournaments import HeadToHeadTournament

def analyze_compute_performance():
    """Analyze performance vs computational budget"""
    
    simulation_counts = [50, 100, 200, 400, 800, 1600]
    win_rates = []
    times = []
    
    baseline = HeuristicAgent()
    
    for n_sims in simulation_counts:
        print(f"Testing {n_sims} simulations...")
        
        # Create agent with specific simulation count
        agent = create_alphazero_agent(
            checkpoint_path='runs/latest/checkpoints/alphazero_epoch_30.pt',
            mcts_simulations=n_sims
        )
        
        # Run tournament
        tournament = HeadToHeadTournament(
            agent1=('AlphaZero', agent),
            agent2=('Heuristic', baseline),
            games=50
        )
        
        import time
        start_time = time.time()
        results = tournament.run()
        end_time = time.time()
        
        win_rate = results.win_rate_agent1
        avg_time = (end_time - start_time) / 50  # Time per game
        
        win_rates.append(win_rate)
        times.append(avg_time)
        
        print(f"  Win rate: {win_rate:.1%}, Time/game: {avg_time:.2f}s")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Win rate vs simulations
    ax1.plot(simulation_counts, win_rates, 'o-')
    ax1.set_xlabel('MCTS Simulations')
    ax1.set_ylabel('Win Rate vs Heuristic')
    ax1.set_title('Performance vs Computation')
    ax1.grid(True)
    
    # Time vs simulations
    ax2.plot(simulation_counts, times, 'o-', color='red')
    ax2.set_xlabel('MCTS Simulations')
    ax2.set_ylabel('Time per Game (seconds)')
    ax2.set_title('Computation Time')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return simulation_counts, win_rates, times

if __name__ == '__main__':
    analyze_compute_performance()
```

### 3. Opening Analysis
```python
# opening_analysis.py
from uttt.env.state import UTTTState
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.heuristic import HeuristicAgent

def analyze_opening_preferences():
    """Analyze agent's opening move preferences"""
    
    agent = create_alphazero_agent(
        checkpoint_path='runs/latest/checkpoints/alphazero_epoch_30.pt',
        mcts_simulations=1000
    )
    
    # Starting position
    state = UTTTState()
    
    # Get move probabilities
    action_probs = agent.mcts.search(state)
    
    # Convert to board coordinates
    opening_preferences = {}
    for action, prob in enumerate(action_probs):
        if prob > 0.01:  # Only show significant probabilities
            big_board = action // 9
            small_pos = action % 9
            
            # Convert to coordinates
            big_row, big_col = big_board // 3, big_board % 3
            small_row, small_col = small_pos // 3, small_pos % 3
            
            coord_str = f"Board({big_row},{big_col}) Pos({small_row},{small_col})"
            opening_preferences[coord_str] = prob
    
    # Sort by preference
    sorted_openings = sorted(opening_preferences.items(), key=lambda x: x[1], reverse=True)
    
    print("Opening Move Preferences:")
    print("=" * 50)
    for move, prob in sorted_openings[:10]:  # Top 10
        print(f"{move:30} | {prob:.1%}")
    
    # Analyze center vs corner preference
    center_actions = [40]  # Center of center board
    corner_actions = [0, 2, 6, 8, 72, 74, 78, 80]  # Corners of corner boards
    edge_actions = [1, 3, 5, 7, 36, 37, 38, 42, 43, 44]  # Various edge positions
    
    center_prob = sum(action_probs[a] for a in center_actions if a < len(action_probs))
    corner_prob = sum(action_probs[a] for a in corner_actions if a < len(action_probs))
    edge_prob = sum(action_probs[a] for a in edge_actions if a < len(action_probs))
    
    print(f"\nStrategic Preferences:")
    print(f"Center plays: {center_prob:.1%}")
    print(f"Corner plays: {corner_prob:.1%}")
    print(f"Edge plays:   {edge_prob:.1%}")
    
    return opening_preferences

if __name__ == '__main__':
    analyze_opening_preferences()
```

## Custom Agent Examples

### 1. Hybrid Agent
```python
# hybrid_agent.py
from uttt.agents.base import Agent
from uttt.agents.heuristic import HeuristicAgent
from uttt.eval.alphazero_factory import create_alphazero_agent

class HybridAgent(Agent):
    """Agent that switches between heuristic and AlphaZero based on position"""
    
    def __init__(self, alphazero_checkpoint, complexity_threshold=20):
        self.alphazero = create_alphazero_agent(alphazero_checkpoint, mcts_simulations=400)
        self.heuristic = HeuristicAgent()
        self.complexity_threshold = complexity_threshold
    
    def get_action(self, state):
        complexity = self._calculate_complexity(state)
        
        if complexity > self.complexity_threshold:
            # Use AlphaZero for complex positions
            return self.alphazero.get_action(state)
        else:
            # Use heuristic for simple positions
            return self.heuristic.get_action(state)
    
    def _calculate_complexity(self, state):
        """Calculate position complexity heuristic"""
        # Simple complexity measure: number of active small boards
        active_boards = 0
        for i in range(3):
            for j in range(3):
                if state.big_board[i, j] == 0:  # Board still active
                    active_boards += 1
        
        # Add move count factor
        complexity = active_boards * 3 + state.move_count * 0.1
        return complexity
    
    def reset(self):
        self.alphazero.reset()
        self.heuristic.reset()

# Test the hybrid agent
if __name__ == '__main__':
    from uttt.agents.random import RandomAgent
    from uttt.eval.tournaments import HeadToHeadTournament
    
    hybrid = HybridAgent('runs/latest/checkpoints/alphazero_epoch_30.pt')
    baseline = RandomAgent()
    
    tournament = HeadToHeadTournament(
        agent1=('Hybrid', hybrid),
        agent2=('Random', baseline),
        games=50
    )
    
    results = tournament.run()
    print(f"Hybrid agent win rate: {results.win_rate_agent1:.1%}")
```

### 2. Learning Agent
```python
# learning_agent.py
import pickle
from collections import defaultdict
from uttt.agents.base import Agent

class QLearningAgent(Agent):
    """Simple Q-Learning agent for Ultimate Tic-Tac-Toe"""
    
    def __init__(self, learning_rate=0.1, epsilon=0.1, discount=0.9):
        self.lr = learning_rate
        self.epsilon = epsilon
        self.discount = discount
        self.q_values = defaultdict(lambda: defaultdict(float))
        self.last_state = None
        self.last_action = None
    
    def get_action(self, state):
        state_key = self._state_to_key(state)
        valid_actions = state.get_valid_actions()
        
        # Epsilon-greedy action selection
        if np.random.random() < self.epsilon:
            action = np.random.choice(valid_actions)
        else:
            # Choose action with highest Q-value
            q_values = [self.q_values[state_key][a] for a in valid_actions]
            best_action_idx = np.argmax(q_values)
            action = valid_actions[best_action_idx]
        
        self.last_state = state_key
        self.last_action = action
        
        return action
    
    def update(self, reward, next_state=None):
        """Update Q-values based on reward"""
        if self.last_state is None or self.last_action is None:
            return
        
        current_q = self.q_values[self.last_state][self.last_action]
        
        if next_state is None or next_state.is_terminal():
            # Terminal state
            target = reward
        else:
            # Get best next action value
            next_state_key = self._state_to_key(next_state)
            next_actions = next_state.get_valid_actions()
            next_q_values = [self.q_values[next_state_key][a] for a in next_actions]
            max_next_q = max(next_q_values) if next_q_values else 0
            target = reward + self.discount * max_next_q
        
        # Q-learning update
        self.q_values[self.last_state][self.last_action] = current_q + self.lr * (target - current_q)
    
    def _state_to_key(self, state):
        """Convert state to hashable key"""
        # Simple state representation (can be improved)
        return (
            tuple(state.big_board.flatten()),
            tuple(state.small_boards.flatten()),
            state.current_player,
            state.active_board
        )
    
    def reset(self):
        self.last_state = None
        self.last_action = None
    
    def save(self, filename):
        """Save Q-values to file"""
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_values), f)
    
    def load(self, filename):
        """Load Q-values from file"""
        with open(filename, 'rb') as f:
            self.q_values = defaultdict(lambda: defaultdict(float), pickle.load(f))

# Training the Q-Learning agent
def train_qlearning_agent(episodes=10000):
    from uttt.env.state import UTTTState
    from uttt.agents.random import RandomAgent
    
    agent = QLearningAgent()
    opponent = RandomAgent()
    
    wins = 0
    
    for episode in range(episodes):
        state = UTTTState()
        agent.reset()
        opponent.reset()
        
        # Alternate who goes first
        agents = [agent, opponent] if episode % 2 == 0 else [opponent, agent]
        current_agent_idx = 0
        
        while not state.is_terminal():
            current_agent = agents[current_agent_idx]
            action = current_agent.get_action(state)
            new_state = state.make_move(action)
            
            # Update Q-learning agent if it's their turn
            if current_agent == agent:
                if new_state.is_terminal():
                    result = new_state.get_result()
                    reward = result * state.current_player  # Reward from agent's perspective
                    agent.update(reward)
                    if reward > 0:
                        wins += 1
                else:
                    agent.update(0, new_state)  # Intermediate reward
            
            state = new_state
            current_agent_idx = 1 - current_agent_idx
        
        # Decay epsilon
        if episode % 1000 == 0:
            agent.epsilon *= 0.95
            print(f"Episode {episode}, Wins: {wins}, Win Rate: {wins/(episode+1):.2%}, Epsilon: {agent.epsilon:.3f}")
    
    agent.save('qlearning_agent.pkl')
    return agent

if __name__ == '__main__':
    trained_agent = train_qlearning_agent()
```

### 3. Ensemble Agent
```python
# ensemble_agent.py
from uttt.agents.base import Agent
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.heuristic import HeuristicAgent
import numpy as np

class EnsembleAgent(Agent):
    """Agent that combines multiple agents via voting"""
    
    def __init__(self, agent_configs):
        """
        agent_configs: list of (agent, weight) tuples
        """
        self.agents = []
        self.weights = []
        
        for agent, weight in agent_configs:
            self.agents.append(agent)
            self.weights.append(weight)
        
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()  # Normalize
    
    def get_action(self, state):
        valid_actions = state.get_valid_actions()
        action_scores = np.zeros(81)
        
        # Get action preferences from each agent
        for agent, weight in zip(self.agents, self.weights):
            if hasattr(agent, 'get_action_probabilities'):
                # Agent can provide full probability distribution
                probs = agent.get_action_probabilities(state)
                action_scores += weight * probs
            else:
                # Agent only provides single action - give it full weight
                action = agent.get_action(state)
                action_scores[action] += weight
        
        # Mask invalid actions
        for i in range(81):
            if i not in valid_actions:
                action_scores[i] = 0
        
        # Select action with highest score
        return np.argmax(action_scores)
    
    def reset(self):
        for agent in self.agents:
            agent.reset()

# Create ensemble of different strength AlphaZero agents
def create_ensemble():
    agents = [
        (create_alphazero_agent('checkpoints/epoch_10.pt', mcts_simulations=200), 0.2),
        (create_alphazero_agent('checkpoints/epoch_20.pt', mcts_simulations=400), 0.3),
        (create_alphazero_agent('checkpoints/epoch_30.pt', mcts_simulations=800), 0.4),
        (HeuristicAgent(), 0.1)
    ]
    
    return EnsembleAgent(agents)

if __name__ == '__main__':
    ensemble = create_ensemble()
    
    # Test against individual agents
    from uttt.eval.tournaments import HeadToHeadTournament
    from uttt.agents.random import RandomAgent
    
    tournament = HeadToHeadTournament(
        agent1=('Ensemble', ensemble),
        agent2=('Random', RandomAgent()),
        games=50
    )
    
    results = tournament.run()
    print(f"Ensemble win rate: {results.win_rate_agent1:.1%}")
```

## Analysis Examples

### 1. Training Progress Analysis
```python
# training_analysis.py
import pandas as pd
import matplotlib.pyplot as plt

def analyze_training_progress(run_path):
    """Analyze training metrics and visualize progress"""
    
    # Load training metrics
    metrics_path = f"{run_path}/metrics/training_metrics.csv"
    df = pd.read_csv(metrics_path)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(df['epoch'], df['total_loss'], label='Total Loss')
    axes[0, 0].plot(df['epoch'], df['policy_loss'], label='Policy Loss')
    axes[0, 0].plot(df['epoch'], df['value_loss'], label='Value Loss')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Dead ReLUs
    axes[0, 1].plot(df['epoch'], df['dead_relu_pct'])
    axes[0, 1].set_title('Dead ReLU Percentage')
    axes[0, 1].set_ylabel('Percentage')
    axes[0, 1].grid(True)
    
    # Policy entropy
    if 'policy_entropy' in df.columns:
        axes[0, 2].plot(df['epoch'], df['policy_entropy'])
        axes[0, 2].set_title('Policy Entropy')
        axes[0, 2].set_ylabel('Bits')
        axes[0, 2].grid(True)
    
    # Learning rate (if available)
    if 'learning_rate' in df.columns:
        axes[1, 0].plot(df['epoch'], df['learning_rate'])
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
    
    # Training accuracy
    if 'policy_accuracy' in df.columns:
        axes[1, 1].plot(df['epoch'], df['policy_accuracy'])
        axes[1, 1].set_title('Policy Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True)
    
    # Value prediction accuracy
    if 'value_mse' in df.columns:
        axes[1, 2].plot(df['epoch'], df['value_mse'])
        axes[1, 2].set_title('Value MSE')
        axes[1, 2].set_ylabel('MSE')
        axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{run_path}/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("Training Summary:")
    print("=" * 50)
    print(f"Total epochs: {df['epoch'].max()}")
    print(f"Final total loss: {df['total_loss'].iloc[-1]:.4f}")
    print(f"Final policy loss: {df['policy_loss'].iloc[-1]:.4f}")
    print(f"Final value loss: {df['value_loss'].iloc[-1]:.4f}")
    print(f"Final dead ReLU %: {df['dead_relu_pct'].iloc[-1]:.1f}%")
    
    if 'policy_entropy' in df.columns:
        print(f"Final policy entropy: {df['policy_entropy'].iloc[-1]:.2f} bits")

if __name__ == '__main__':
    analyze_training_progress('runs/run_20241018_143012')
```

### 2. Game Analysis
```python
# game_analysis.py
import json
from uttt.env.state import UTTTState

def analyze_game(game_file):
    """Analyze a specific game for patterns and mistakes"""
    
    with open(game_file, 'r') as f:
        game_data = json.load(f)
    
    moves = game_data['moves']
    state = UTTTState()
    
    print(f"Game Analysis: {game_file}")
    print("=" * 50)
    
    game_length = len(moves)
    print(f"Game length: {game_length} moves")
    
    # Analyze move distribution
    move_heatmap = np.zeros((9, 9))
    for move in moves:
        action = move['action']
        big_board = action // 9
        small_pos = action % 9
        big_row, big_col = big_board // 3, big_board % 3
        small_row, small_col = small_pos // 3, small_pos % 3
        
        # Map to 9x9 heatmap
        row = big_row * 3 + small_row
        col = big_col * 3 + small_col
        move_heatmap[row, col] += 1
    
    print("\nMove heatmap (most played positions):")
    print(move_heatmap)
    
    # Analyze game phases
    opening_moves = moves[:10] if len(moves) >= 10 else moves
    print(f"\nOpening moves ({len(opening_moves)}):")
    for i, move in enumerate(opening_moves):
        print(f"  {i+1}. Action {move['action']} (Player {move['player']})")
    
    # Check for forced moves vs free moves
    forced_moves = 0
    free_moves = 0
    
    for move in moves:
        if move.get('forced', False):
            forced_moves += 1
        else:
            free_moves += 1
    
    print(f"\nMove types:")
    print(f"  Forced moves: {forced_moves}")
    print(f"  Free moves: {free_moves}")
    print(f"  Force ratio: {forced_moves/len(moves):.1%}")
    
    return {
        'length': game_length,
        'move_heatmap': move_heatmap,
        'forced_ratio': forced_moves/len(moves)
    }

def analyze_tournament_games(tournament_file):
    """Analyze all games in a tournament"""
    
    with open(tournament_file, 'r') as f:
        tournament_data = json.load(f)
    
    game_stats = []
    
    for matchup in tournament_data['matchups']:
        if 'games' in matchup:
            for game in matchup['games']:
                stats = analyze_game_data(game)
                stats['agent1'] = matchup['agent1']
                stats['agent2'] = matchup['agent2']
                game_stats.append(stats)
    
    # Aggregate statistics
    df = pd.DataFrame(game_stats)
    
    print("Tournament Game Analysis:")
    print("=" * 50)
    print(f"Total games: {len(game_stats)}")
    print(f"Average game length: {df['length'].mean():.1f} moves")
    print(f"Game length std: {df['length'].std():.1f}")
    print(f"Shortest game: {df['length'].min()} moves")
    print(f"Longest game: {df['length'].max()} moves")
    print(f"Average forced move ratio: {df['forced_ratio'].mean():.1%}")
    
    # Plot game length distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['length'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Game Length (moves)')
    plt.ylabel('Frequency')
    plt.title('Game Length Distribution')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return df

if __name__ == '__main__':
    analyze_tournament_games('tournaments/latest_tournament.json')
```

## Integration Examples

### 1. Web Interface
```python
# web_interface.py
from flask import Flask, request, jsonify, render_template_string
from uttt.env.state import UTTTState
from uttt.eval.alphazero_factory import create_alphazero_agent

app = Flask(__name__)

# Load agent
agent = create_alphazero_agent('runs/latest/checkpoints/alphazero_epoch_30.pt')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Ultimate Tic-Tac-Toe vs AI</title>
    <style>
        .big-board { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; width: 600px; margin: 20px auto; }
        .small-board { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2px; border: 2px solid #333; }
        .cell { width: 60px; height: 60px; border: 1px solid #ccc; display: flex; align-items: center; justify-content: center; font-size: 24px; cursor: pointer; }
        .cell:hover { background-color: #f0f0f0; }
        .won { background-color: #90EE90; }
        .active { border-color: #ff4444; }
    </style>
</head>
<body>
    <h1>Ultimate Tic-Tac-Toe vs AI</h1>
    <div id="game-board"></div>
    <div id="game-status"></div>
    <button onclick="newGame()">New Game</button>
    
    <script>
        let gameState = null;
        
        async function newGame() {
            const response = await fetch('/new_game', { method: 'POST' });
            gameState = await response.json();
            renderBoard();
        }
        
        async function makeMove(action) {
            const response = await fetch('/make_move', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action: action })
            });
            gameState = await response.json();
            renderBoard();
        }
        
        function renderBoard() {
            // Render the game board based on gameState
            // This is a simplified version - full implementation would be more complex
            const boardDiv = document.getElementById('game-board');
            boardDiv.innerHTML = '<p>Game state updated</p>';
            
            document.getElementById('game-status').innerText = gameState.status || 'Your turn';
        }
        
        // Initialize
        newGame();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/new_game', methods=['POST'])
def new_game():
    global current_state
    current_state = UTTTState()
    return jsonify({
        'board': current_state.to_dict(),
        'status': 'Your turn',
        'valid_actions': current_state.get_valid_actions()
    })

@app.route('/make_move', methods=['POST'])
def make_move():
    global current_state
    
    data = request.get_json()
    action = data['action']
    
    # Make human move
    current_state = current_state.make_move(action)
    
    if current_state.is_terminal():
        result = current_state.get_result()
        status = 'You win!' if result == 1 else 'You lose!' if result == -1 else 'Draw!'
    else:
        # Make AI move
        ai_action = agent.get_action(current_state)
        current_state = current_state.make_move(ai_action)
        
        if current_state.is_terminal():
            result = current_state.get_result()
            status = 'You win!' if result == 1 else 'You lose!' if result == -1 else 'Draw!'
        else:
            status = 'Your turn'
    
    return jsonify({
        'board': current_state.to_dict(),
        'status': status,
        'valid_actions': current_state.get_valid_actions() if not current_state.is_terminal() else []
    })

if __name__ == '__main__':
    app.run(debug=True)
```

### 2. API Server
```python
# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
from uttt.env.state import UTTTState
from uttt.eval.alphazero_factory import create_alphazero_agent

app = FastAPI(title="Ultimate Tic-Tac-Toe AI API")

# Load AI agent
ai_agent = create_alphazero_agent('runs/latest/checkpoints/alphazero_epoch_30.pt')

# Game state storage
games = {}

class GameRequest(BaseModel):
    difficulty: Optional[str] = "medium"  # easy, medium, hard

class MoveRequest(BaseModel):
    game_id: str
    action: int

class GameResponse(BaseModel):
    game_id: str
    board: dict
    valid_actions: List[int]
    status: str
    is_terminal: bool

@app.post("/new_game", response_model=GameResponse)
def create_game(request: GameRequest):
    """Create a new game"""
    game_id = str(uuid.uuid4())
    state = UTTTState()
    
    # Adjust AI strength based on difficulty
    if request.difficulty == "easy":
        ai_agent.mcts.config['n_simulations'] = 100
    elif request.difficulty == "medium":
        ai_agent.mcts.config['n_simulations'] = 400
    else:  # hard
        ai_agent.mcts.config['n_simulations'] = 1200
    
    games[game_id] = {
        'state': state,
        'difficulty': request.difficulty
    }
    
    return GameResponse(
        game_id=game_id,
        board=state.to_dict(),
        valid_actions=state.get_valid_actions(),
        status="Your turn",
        is_terminal=False
    )

@app.post("/make_move", response_model=GameResponse)
def make_move(request: MoveRequest):
    """Make a move in the game"""
    if request.game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[request.game_id]
    state = game['state']
    
    # Validate move
    if request.action not in state.get_valid_actions():
        raise HTTPException(status_code=400, detail="Invalid move")
    
    # Make human move
    state = state.make_move(request.action)
    
    if state.is_terminal():
        status = _get_game_status(state)
    else:
        # Make AI move
        ai_action = ai_agent.get_action(state)
        state = state.make_move(ai_action)
        status = _get_game_status(state) if state.is_terminal() else "Your turn"
    
    # Update game state
    games[request.game_id]['state'] = state
    
    return GameResponse(
        game_id=request.game_id,
        board=state.to_dict(),
        valid_actions=state.get_valid_actions() if not state.is_terminal() else [],
        status=status,
        is_terminal=state.is_terminal()
    )

@app.get("/game/{game_id}", response_model=GameResponse)
def get_game(game_id: str):
    """Get current game state"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    state = games[game_id]['state']
    
    return GameResponse(
        game_id=game_id,
        board=state.to_dict(),
        valid_actions=state.get_valid_actions() if not state.is_terminal() else [],
        status=_get_game_status(state) if state.is_terminal() else "Your turn",
        is_terminal=state.is_terminal()
    )

@app.delete("/game/{game_id}")
def delete_game(game_id: str):
    """Delete a game"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    del games[game_id]
    return {"message": "Game deleted"}

def _get_game_status(state: UTTTState) -> str:
    """Get human-readable game status"""
    if not state.is_terminal():
        return "Game in progress"
    
    result = state.get_result()
    if result == 1:
        return "You win!"
    elif result == -1:
        return "AI wins!"
    else:
        return "Draw!"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 3. Batch Processing
```python
# batch_processor.py
import asyncio
import aiofiles
import json
from pathlib import Path
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.random import RandomAgent
from uttt.env.state import UTTTState

class BatchGameProcessor:
    """Process many games in parallel for analysis"""
    
    def __init__(self, agent1, agent2, output_dir="batch_results"):
        self.agent1 = agent1
        self.agent2 = agent2
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    async def play_game(self, game_id):
        """Play a single game asynchronously"""
        state = UTTTState()
        agents = [self.agent1, self.agent2]
        current_agent_idx = game_id % 2  # Alternate who goes first
        
        moves = []
        
        while not state.is_terminal():
            agent = agents[current_agent_idx]
            action = agent.get_action(state)
            
            moves.append({
                'player': state.current_player,
                'action': action,
                'move_number': len(moves) + 1
            })
            
            state = state.make_move(action)
            current_agent_idx = 1 - current_agent_idx
        
        result = state.get_result()
        
        game_data = {
            'game_id': game_id,
            'moves': moves,
            'result': result,
            'length': len(moves),
            'first_player': (game_id % 2) + 1
        }
        
        # Save game data
        async with aiofiles.open(self.output_dir / f"game_{game_id:06d}.json", 'w') as f:
            await f.write(json.dumps(game_data, indent=2))
        
        return game_data
    
    async def process_batch(self, n_games, batch_size=100):
        """Process games in batches"""
        results = []
        
        for start_idx in range(0, n_games, batch_size):
            end_idx = min(start_idx + batch_size, n_games)
            batch_games = range(start_idx, end_idx)
            
            print(f"Processing games {start_idx} to {end_idx-1}...")
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*[
                self.play_game(game_id) for game_id in batch_games
            ])
            
            results.extend(batch_results)
            
            # Save batch summary
            batch_summary = {
                'batch_start': start_idx,
                'batch_end': end_idx - 1,
                'games': batch_results
            }
            
            async with aiofiles.open(self.output_dir / f"batch_{start_idx:06d}.json", 'w') as f:
                await f.write(json.dumps(batch_summary, indent=2))
        
        # Save overall summary
        overall_summary = {
            'total_games': len(results),
            'agent1_wins': sum(1 for r in results if r['result'] == 1),
            'agent2_wins': sum(1 for r in results if r['result'] == -1),
            'draws': sum(1 for r in results if r['result'] == 0),
            'avg_game_length': sum(r['length'] for r in results) / len(results)
        }
        
        async with aiofiles.open(self.output_dir / "summary.json", 'w') as f:
            await f.write(json.dumps(overall_summary, indent=2))
        
        return results, overall_summary

async def main():
    """Run batch processing"""
    agent1 = create_alphazero_agent('runs/latest/checkpoints/alphazero_epoch_30.pt')
    agent2 = RandomAgent()
    
    processor = BatchGameProcessor(agent1, agent2)
    results, summary = await processor.process_batch(n_games=1000, batch_size=50)
    
    print("Batch Processing Complete!")
    print("=" * 50)
    print(f"Total games: {summary['total_games']}")
    print(f"Agent 1 wins: {summary['agent1_wins']} ({summary['agent1_wins']/summary['total_games']:.1%})")
    print(f"Agent 2 wins: {summary['agent2_wins']} ({summary['agent2_wins']/summary['total_games']:.1%})")
    print(f"Draws: {summary['draws']} ({summary['draws']/summary['total_games']:.1%})")
    print(f"Average game length: {summary['avg_game_length']:.1f} moves")

if __name__ == "__main__":
    asyncio.run(main())
```

These examples demonstrate various ways to use and extend the Ultimate Tic-Tac-Toe AlphaZero implementation. You can adapt these patterns for your specific needs and combine them to create more complex applications.

For more advanced usage patterns, see:
- [Training Guide](TRAINING_GUIDE.md) - Advanced training techniques
- [Tournament Guide](TOURNAMENT_GUIDE.md) - Comprehensive evaluation methods
- [Architecture Guide](ARCHITECTURE.md) - Implementation details and extension points