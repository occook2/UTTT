# Tournament Guide

This guide explains how to evaluate and compare different agents using the tournament system.

## Table of Contents
- [Quick Start](#quick-start)
- [Tournament Types](#tournament-types)
- [Agent Management](#agent-management)
- [Running Tournaments](#running-tournaments)
- [Results Analysis](#results-analysis)
- [Advanced Evaluation](#advanced-evaluation)

## Quick Start

### Basic Tournament
```bash
# Run a quick tournament between agents
python -m uttt.scripts.alphazero_tournament

# View tournament results
python -m uttt.scripts.view_tournament

# Compute Elo ratings
python -m uttt.eval.compute_ratings
```

### Quick Evaluation
```bash
# Test your trained agent against baselines
python -m uttt.scripts.play_vs_agent --agent1 alphazero --agent2 random
python -m uttt.scripts.play_vs_agent --agent1 alphazero --agent2 heuristic
```

## Tournament Types

### 1. Round Robin Tournament
All agents play against all other agents.

```python
from uttt.eval.tournaments import RoundRobinTournament

# Setup agents
agents = {
    'AlphaZero-Epoch30': az_agent_30,
    'AlphaZero-Epoch20': az_agent_20,
    'Heuristic': heuristic_agent,
    'Random': random_agent
}

# Run tournament
tournament = RoundRobinTournament(agents, games_per_matchup=50)
results = tournament.run()
```

### 2. Swiss System Tournament
Pairs agents with similar ratings.

```python
from uttt.eval.tournaments import SwissTournament

tournament = SwissTournament(agents, rounds=10, games_per_round=20)
results = tournament.run()
```

### 3. Elimination Tournament
Single or double elimination brackets.

```python
from uttt.eval.tournaments import EliminationTournament

tournament = EliminationTournament(agents, bracket_type='single')
results = tournament.run()
```

## Agent Management

### Available Agent Types

#### 1. AlphaZero Agents
```python
from uttt.eval.alphazero_factory import create_alphazero_agent

# Load from checkpoint
agent = create_alphazero_agent(
    checkpoint_path='runs/run_20241018_143012/checkpoints/alphazero_epoch_30.pt',
    config_path='runs/run_20241018_143012/config/config.yaml',
    mcts_simulations=800
)

# Load from config
agent = create_alphazero_agent(
    config_path='config.yaml',
    mcts_simulations=400  # Faster for tournaments
)
```

#### 2. Baseline Agents
```python
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent

random_agent = RandomAgent()
heuristic_agent = HeuristicAgent()
```

#### 3. Human Agent
```python
from uttt.agents.human import HumanAgent

human_agent = HumanAgent()  # For interactive play
```

### Agent Configuration

#### MCTS Settings for Tournaments
```python
# Fast evaluation (quick tournaments)
fast_agent = create_alphazero_agent(
    checkpoint_path='model.pt',
    mcts_simulations=200,    # Reduced for speed
    temp_threshold=0         # No exploration
)

# Strong evaluation (final testing)
strong_agent = create_alphazero_agent(
    checkpoint_path='model.pt',
    mcts_simulations=1600,   # More simulations
    temp_threshold=0         # Deterministic play
)

# Exploration evaluation (diverse play)
explore_agent = create_alphazero_agent(
    checkpoint_path='model.pt',
    mcts_simulations=800,
    temp_threshold=30,       # Exploration enabled
    dirichlet_epsilon=0.1    # Some randomness
)
```

## Running Tournaments

### 1. Automated Tournament Script
```bash
# Run with default settings
python -m uttt.scripts.alphazero_tournament

# Custom tournament
python -m uttt.scripts.alphazero_tournament \
  --agents "AZ-30,AZ-20,Heuristic,Random" \
  --games 100 \
  --output tournaments/my_tournament.json
```

### 2. Manual Tournament Setup
```python
from uttt.eval.tournaments import RoundRobinTournament
from uttt.eval.alphazero_factory import create_alphazero_agent
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent

# Setup agents
agents = {
    'AZ-Epoch30': create_alphazero_agent('checkpoints/epoch_30.pt'),
    'AZ-Epoch24': create_alphazero_agent('checkpoints/epoch_24.pt'),
    'Heuristic': HeuristicAgent(),
    'Random': RandomAgent()
}

# Configure tournament
tournament = RoundRobinTournament(
    agents=agents,
    games_per_matchup=50,
    show_progress=True,
    save_games=True
)

# Run tournament
results = tournament.run()

# Save results
tournament.save_results('tournaments/comparison.json')
```

### 3. Progressive Evaluation
```python
# Test against increasingly strong opponents
opponents = [
    ('Random', RandomAgent()),
    ('Heuristic', HeuristicAgent()),
    ('AZ-Epoch10', create_alphazero_agent('epoch_10.pt')),
    ('AZ-Epoch20', create_alphazero_agent('epoch_20.pt'))
]

your_agent = create_alphazero_agent('epoch_30.pt')

for name, opponent in opponents:
    tournament = HeadToHeadTournament(
        agent1=('Your-Agent', your_agent),
        agent2=(name, opponent),
        games=100
    )
    
    results = tournament.run()
    print(f"vs {name}: {results['win_rate']:.1%}")
```

## Results Analysis

### Tournament Results Structure
```json
{
  "tournament_info": {
    "type": "round_robin",
    "timestamp": "2024-10-18T14:30:00",
    "games_per_matchup": 50,
    "total_games": 300
  },
  "agents": {
    "AZ-Epoch30": {"type": "alphazero", "checkpoint": "epoch_30.pt"},
    "AZ-Epoch24": {"type": "alphazero", "checkpoint": "epoch_24.pt"},
    "Heuristic": {"type": "heuristic"},
    "Random": {"type": "random"}
  },
  "matchups": [
    {
      "agent1": "AZ-Epoch30",
      "agent2": "AZ-Epoch24", 
      "games": 50,
      "wins_agent1": 32,
      "wins_agent2": 18,
      "draws": 0,
      "win_rate_agent1": 0.64
    }
  ],
  "summary": {
    "rankings": ["AZ-Epoch30", "AZ-Epoch24", "Heuristic", "Random"],
    "win_rates": [0.85, 0.72, 0.45, 0.12],
    "elo_ratings": [1650, 1520, 1380, 1200]
  }
}
```

### Viewing Results
```bash
# Interactive tournament viewer
python -m uttt.scripts.view_tournament

# Compute updated Elo ratings
python -m uttt.eval.compute_ratings

# Generate reports
python -c "
from uttt.eval.tournaments import load_tournament_results
results = load_tournament_results('tournaments/latest.json')
print(results.generate_report())
"
```

### Statistical Analysis
```python
import pandas as pd
from uttt.eval.tournaments import load_tournament_results

# Load results
results = load_tournament_results('tournaments/comparison.json')

# Create results matrix
df = results.to_dataframe()
print(df)

# Win rate matrix
win_matrix = results.get_win_rate_matrix()
print(win_matrix)

# Statistical significance
from scipy.stats import binomial_test

for matchup in results.matchups:
    p_value = binomial_test(
        matchup.wins_agent1, 
        matchup.total_games, 
        0.5
    )
    print(f"{matchup.agent1} vs {matchup.agent2}: p={p_value:.4f}")
```

## Advanced Evaluation

### 1. Opening Book Analysis
```python
from uttt.eval.openings import OpeningBook, analyze_opening_performance

# Test specific openings
openings = OpeningBook()
opening_positions = openings.get_standard_openings()

for opening_name, position in opening_positions.items():
    win_rate = analyze_opening_performance(
        agent=your_agent,
        opponent=baseline_agent,
        opening_position=position,
        games=20
    )
    print(f"{opening_name}: {win_rate:.1%}")
```

### 2. Time Control Analysis
```python
# Test under different time constraints
time_controls = [
    (0.1, "Blitz"),      # 100ms per move
    (1.0, "Rapid"),      # 1 second per move  
    (5.0, "Classical"),  # 5 seconds per move
    (float('inf'), "Unlimited")
]

for time_limit, name in time_controls:
    # Configure agent with time limit
    agent = create_alphazero_agent(
        'model.pt',
        time_limit=time_limit
    )
    
    # Run tournament
    results = quick_tournament(agent, baseline_agent, games=50)
    print(f"{name}: {results.win_rate:.1%}")
```

### 3. Position Complexity Analysis
```python
from uttt.eval.complexity import analyze_position_complexity

# Test performance by game phase
phases = {
    'Opening': lambda state: state.move_count < 10,
    'Midgame': lambda state: 10 <= state.move_count < 50, 
    'Endgame': lambda state: state.move_count >= 50
}

for phase_name, phase_filter in phases.items():
    win_rate = analyze_phase_performance(
        agent=your_agent,
        opponent=baseline_agent,
        phase_filter=phase_filter,
        games=100
    )
    print(f"{phase_name}: {win_rate:.1%}")
```

### 4. Error Analysis
```python
from uttt.eval.analysis import GameAnalyzer

# Analyze specific games for mistakes
analyzer = GameAnalyzer()

for game in tournament_games:
    analysis = analyzer.analyze_game(game)
    
    # Find critical mistakes
    mistakes = analysis.find_mistakes(threshold=0.1)  # 10% eval swing
    
    print(f"Game {game.id}: {len(mistakes)} mistakes")
    for mistake in mistakes:
        print(f"  Move {mistake.move}: {mistake.eval_before:.2f} -> {mistake.eval_after:.2f}")
```

### 5. Style Analysis
```python
from uttt.eval.style import StyleAnalyzer

# Compare playing styles
analyzer = StyleAnalyzer()

# Analyze decision patterns
style_profile = analyzer.analyze_agent_style(
    agent=your_agent,
    games=100,
    metrics=['aggression', 'risk_taking', 'positional_play']
)

print(f"Aggression: {style_profile.aggression:.2f}")
print(f"Risk Taking: {style_profile.risk_taking:.2f}")
print(f"Positional: {style_profile.positional:.2f}")
```

## Performance Benchmarks

### Hardware Requirements
- **Minimal**: 1 game/second (CPU only, 100 MCTS sims)
- **Standard**: 10 games/second (GPU, 400 MCTS sims)  
- **High-end**: 50+ games/second (Good GPU, 800+ MCTS sims)

### Tournament Scale Guidelines
- **Quick Test**: 10-20 games per matchup
- **Standard Evaluation**: 50-100 games per matchup
- **Professional**: 500+ games per matchup
- **Statistical Significance**: 1000+ games per matchup

### Time Estimates
```
Round Robin Tournament (4 agents, 100 games each):
- 6 matchups × 100 games = 600 total games
- At 10 games/second = 60 seconds = 1 minute
- At 1 game/second = 600 seconds = 10 minutes
```

## Troubleshooting

### Common Issues

#### 1. Slow Tournaments
**Solutions:**
- Reduce MCTS simulations
- Use GPU acceleration
- Parallelize game execution
- Reduce games per matchup

#### 2. Memory Issues
**Solutions:**
- Process games in batches
- Don't save detailed game logs
- Reduce model size
- Use memory profiling

#### 3. Inconsistent Results
**Solutions:**
- Increase games per matchup
- Check for non-deterministic behavior
- Control random seeds
- Validate agent implementations

#### 4. Agent Loading Errors
**Solutions:**
- Verify checkpoint paths
- Check configuration compatibility
- Validate model architecture
- Test agent creation separately

## Best Practices

### 1. Tournament Design
- Use appropriate sample sizes
- Control for randomness
- Test across multiple conditions
- Document exact configurations

### 2. Result Interpretation
- Check statistical significance
- Compare against multiple baselines
- Consider confidence intervals
- Account for style differences

### 3. Performance Tracking
- Maintain tournament history
- Track Elo rating evolution
- Monitor training correlation
- Save detailed game logs

### 4. Validation
- Cross-validate results
- Test on different hardware
- Verify deterministic behavior
- Compare with manual analysis

## Next Steps

After tournament evaluation:
1. **Identify Weaknesses**: Focus training on weak areas
2. **Optimize Performance**: Adjust MCTS parameters
3. **Scale Testing**: Run larger tournaments
4. **Compare Architectures**: Test different model designs
5. **Publish Results**: Share findings with community

For more information, see:
- [Training Guide](TRAINING_GUIDE.md) - Improve based on tournament results
- [Architecture Guide](ARCHITECTURE.md) - Understand model decisions
- [Examples](EXAMPLES.md) - See example tournament setups