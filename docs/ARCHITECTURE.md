# Architecture Guide

This guide explains the technical architecture of the Ultimate Tic-Tac-Toe AlphaZero implementation.

## Table of Contents
- [Overview](#overview)
- [Core Components](#core-components)
- [Neural Network Architecture](#neural-network-architecture)
- [MCTS Implementation](#mcts-implementation)
- [Training Pipeline](#training-pipeline)
- [Data Flow](#data-flow)
- [Extension Points](#extension-points)

## Overview

The system implements the AlphaZero algorithm for Ultimate Tic-Tac-Toe, combining:
- **Monte Carlo Tree Search (MCTS)** for move planning
- **Deep Neural Network** for position evaluation and move prediction
- **Self-play training** for learning through experience
- **Tournament system** for evaluation and comparison

### Key Design Principles
1. **Modularity**: Clear separation between components
2. **Configurability**: Extensive YAML-based configuration
3. **Extensibility**: Easy to add new agents and features
4. **Reproducibility**: Deterministic behavior with seed control
5. **Performance**: Optimized for GPU acceleration and parallel execution

## Core Components

### 1. Game Environment (`uttt/env/`)

#### `UTTTState`
The core game state representation.

```python
class UTTTState:
    """Ultimate Tic-Tac-Toe game state"""
    
    def __init__(self):
        self.big_board = np.zeros((3, 3), dtype=int)      # 3x3 meta-board
        self.small_boards = np.zeros((3, 3, 3, 3), dtype=int)  # 9 small 3x3 boards
        self.current_player = 1                            # 1 or -1
        self.active_board = None                          # Which small board is active
        self.move_count = 0
        self._cached_terminal = None
        self._cached_valid_actions = None
    
    def make_move(self, action: int) -> 'UTTTState':
        """Return new state after making a move"""
        
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices"""
        
    def is_terminal(self) -> bool:
        """Check if game is over"""
        
    def get_result(self) -> Optional[int]:
        """Get game result: 1, -1, or 0 for draw"""
```

**Key Features:**
- **Immutable States**: `make_move()` returns new state objects
- **Action Encoding**: 81 possible moves (9 boards × 9 positions)
- **Caching**: Expensive computations cached within state objects
- **Validation**: Comprehensive rule enforcement

#### Action Encoding
```
Action Index = big_board_index * 9 + small_board_index
where big_board_index, small_board_index ∈ [0, 8]

Example: Move to center of top-left board = 0 * 9 + 4 = 4
```

### 2. Agents (`uttt/agents/`)

#### Base Agent Interface
```python
class Agent(ABC):
    """Abstract base class for all agents"""
    
    @abstractmethod
    def get_action(self, state: UTTTState) -> int:
        """Return action index for given state"""
        
    @abstractmethod
    def reset(self):
        """Reset agent state between games"""
```

#### AlphaZero Agent
```python
class AlphaZeroAgent(Agent):
    def __init__(self, network, mcts_config):
        self.network = network
        self.mcts = MCTS(network, mcts_config)
    
    def get_action(self, state: UTTTState) -> int:
        # Run MCTS to get action probabilities
        action_probs = self.mcts.search(state)
        
        # Select action based on temperature
        if self.temperature > 0:
            action = np.random.choice(len(action_probs), p=action_probs)
        else:
            action = np.argmax(action_probs)
            
        return action
```

### 3. Neural Network (`uttt/agents/az/nn/`)

#### Network Architecture
```python
class AlphaZeroNetwork(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Input processing
        self.input_conv = nn.Conv2d(
            in_channels=config['input_channels'],
            out_channels=config['n_filters'],
            kernel_size=3,
            padding=1
        )
        
        # ResNet backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(config['n_filters']) 
            for _ in range(config['n_blocks'])
        ])
        
        # Policy head
        self.policy_head = PolicyHead(
            in_channels=config['n_filters'],
            action_space=81
        )
        
        # Value head  
        self.value_head = ValueHead(
            in_channels=config['n_filters']
        )
    
    def forward(self, x):
        # Input processing
        x = F.relu(self.input_conv(x))
        
        # ResNet blocks
        for block in self.res_blocks:
            x = block(x)
            
        # Output heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value
```

#### Input Representation
```python
def encode_state(state: UTTTState) -> torch.Tensor:
    """Encode game state as neural network input"""
    
    # 6 channels: [player1_big, player2_big, player1_small, player2_small, active_board, current_player]
    channels = torch.zeros(6, 9, 9)
    
    # Big board encoding (3x3 -> 9x9 with repetition)
    # Small boards encoding (flatten 3x3x3x3 -> 9x9)  
    # Active board mask
    # Current player channel
    
    return channels
```

### 4. MCTS Implementation (`uttt/mcts/`)

#### Core MCTS Class
```python
class MCTS:
    def __init__(self, network, config):
        self.network = network
        self.config = config
        self.tree = {}  # state_hash -> Node
        
    def search(self, state: UTTTState) -> np.ndarray:
        """Run MCTS simulations and return action probabilities"""
        
        for _ in range(self.config['n_simulations']):
            self._simulate(state)
            
        # Extract visit counts
        node = self.tree[state.hash()]
        visits = np.array([child.visit_count for child in node.children])
        
        # Convert to probabilities
        if self.config['temperature'] > 0:
            probs = visits ** (1 / self.config['temperature'])
            probs = probs / probs.sum()
        else:
            probs = np.zeros_like(visits)
            probs[np.argmax(visits)] = 1.0
            
        return probs
```

#### Node Structure
```python
class MCTSNode:
    def __init__(self, state: UTTTState, prior: float):
        self.state = state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False
    
    def value(self) -> float:
        """Average value from simulations"""
        return self.value_sum / max(1, self.visit_count)
    
    def ucb_score(self, c_puct: float, parent_visits: int) -> float:
        """Upper Confidence Bound for tree selection"""
        exploration = c_puct * self.prior * np.sqrt(parent_visits) / (1 + self.visit_count)
        return self.value() + exploration
```

#### Selection Strategy
```python
def _select_child(self, node: MCTSNode) -> MCTSNode:
    """Select child with highest UCB score"""
    
    best_score = float('-inf')
    best_child = None
    
    for child in node.children.values():
        score = child.ucb_score(self.config['c_puct'], node.visit_count)
        if score > best_score:
            best_score = score
            best_child = child
            
    return best_child
```

## Neural Network Architecture

### Input Encoding (6 channels, 9×9)

1. **Player 1 Big Board** (1 channel): Meta-game wins
2. **Player 2 Big Board** (1 channel): Meta-game wins  
3. **Player 1 Small Boards** (1 channel): Individual cell occupancy
4. **Player 2 Small Boards** (1 channel): Individual cell occupancy
5. **Active Board Mask** (1 channel): Which small board is active
6. **Current Player** (1 channel): Whose turn it is

### ResNet Backbone

```python
class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

### Output Heads

#### Policy Head
```python
class PolicyHead(nn.Module):
    def __init__(self, in_channels, action_space=81):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2, 1)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 9 * 9, action_space)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.fc(x), dim=1)
```

#### Value Head
```python
class ValueHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 1)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(9 * 9, 256)
        self.fc2 = nn.Linear(256, 1)
        
    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x))
```

## MCTS Implementation

### Four Phases of MCTS

#### 1. Selection
Navigate from root to leaf using UCB scores:
```python
def _select(self, node: MCTSNode) -> List[MCTSNode]:
    """Select path from root to leaf"""
    path = [node]
    
    while node.is_expanded and not node.state.is_terminal():
        node = self._select_child(node)
        path.append(node)
        
    return path
```

#### 2. Expansion  
Create child nodes for all valid actions:
```python
def _expand(self, node: MCTSNode):
    """Expand node by creating all children"""
    if node.state.is_terminal():
        return
        
    # Get network predictions
    policy, value = self.network(encode_state(node.state))
    policy = policy.softmax(dim=-1).squeeze().cpu().numpy()
    
    # Create child nodes
    for action in node.state.get_valid_actions():
        child_state = node.state.make_move(action)
        prior = policy[action]
        node.children[action] = MCTSNode(child_state, prior)
        
    node.is_expanded = True
```

#### 3. Simulation
Get value estimate from neural network:
```python
def _evaluate(self, node: MCTSNode) -> float:
    """Get value estimate for position"""
    if node.state.is_terminal():
        return node.state.get_result() * node.state.current_player
    
    _, value = self.network(encode_state(node.state))
    return value.item()
```

#### 4. Backpropagation
Update statistics along the path:
```python
def _backpropagate(self, path: List[MCTSNode], value: float):
    """Update node statistics along path"""
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += value
        value = -value  # Flip value for opponent
```

### Dirichlet Noise
Add exploration noise to root node:
```python
def _add_dirichlet_noise(self, node: MCTSNode):
    """Add Dirichlet noise to root for exploration"""
    if not node.children:
        return
        
    noise = np.random.dirichlet([self.config['dirichlet_alpha']] * len(node.children))
    epsilon = self.config['dirichlet_epsilon']
    
    for i, child in enumerate(node.children.values()):
        child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
```

## Training Pipeline

### Self-Play Data Generation

```python
class SelfPlayTrainer:
    def generate_training_data(self, n_games: int) -> List[TrainingExample]:
        """Generate training data through self-play"""
        
        examples = []
        
        for game_idx in range(n_games):
            game_examples = []
            state = UTTTState()
            
            while not state.is_terminal():
                # Run MCTS to get move probabilities
                action_probs = self.mcts.search(state)
                
                # Create training example
                example = TrainingExample(
                    state=state.copy(),
                    policy=action_probs.copy(),
                    value=None  # Will be filled after game
                )
                game_examples.append(example)
                
                # Make move
                action = self._sample_action(action_probs)
                state = state.make_move(action)
            
            # Get game result and assign values
            result = state.get_result()
            for i, example in enumerate(game_examples):
                player = example.state.current_player
                example.value = result * player
                
            examples.extend(game_examples)
            
        return examples
```

### Neural Network Training

```python
class NeuralNetworkTrainer:
    def train_step(self, examples: List[TrainingExample]) -> Dict[str, float]:
        """Single training step on batch of examples"""
        
        # Prepare batch
        states = torch.stack([encode_state(ex.state) for ex in examples])
        target_policies = torch.stack([torch.tensor(ex.policy) for ex in examples])
        target_values = torch.tensor([ex.value for ex in examples]).float()
        
        # Forward pass
        pred_policies, pred_values = self.network(states)
        
        # Compute losses
        policy_loss = F.nll_loss(pred_policies, target_policies)
        value_loss = F.mse_loss(pred_values.squeeze(), target_values)
        l2_loss = self._compute_l2_loss()
        
        total_loss = policy_loss + value_loss + self.config['l2_reg'] * l2_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'l2_loss': l2_loss.item()
        }
```

### Data Augmentation

```python
def augment_training_examples(examples: List[TrainingExample]) -> List[TrainingExample]:
    """Apply 8-fold rotational/reflection symmetry"""
    
    augmented = []
    
    for example in examples:
        for transform in SYMMETRY_TRANSFORMS:
            augmented_state = transform.apply_to_state(example.state)
            augmented_policy = transform.apply_to_policy(example.policy)
            
            augmented.append(TrainingExample(
                state=augmented_state,
                policy=augmented_policy,
                value=example.value
            ))
            
    return augmented
```

## Data Flow

### Training Loop Data Flow
```
1. Self-Play Games → Training Examples
2. Training Examples → Data Augmentation (8x)
3. Augmented Examples → Neural Network Training
4. Updated Network → MCTS Policy Update
5. New Network → Tournament Evaluation
6. Evaluation Results → Network Selection
```

### State Encoding Pipeline
```
UTTTState → encode_state() → Tensor(6, 9, 9) → Network → (Policy, Value)
```

### MCTS Search Pipeline  
```
Root State → Selection → Expansion → Evaluation → Backpropagation → Action Probabilities
```

## Extension Points

### 1. New Agent Types
```python
class CustomAgent(Agent):
    def get_action(self, state: UTTTState) -> int:
        # Implement custom logic
        return action
        
    def reset(self):
        # Reset any internal state
        pass
```

### 2. Alternative Network Architectures
```python
class TransformerNetwork(nn.Module):
    """Transformer-based network architecture"""
    
    def __init__(self, config):
        super().__init__()
        # Implement transformer layers
        
    def forward(self, x):
        # Custom forward pass
        return policy, value
```

### 3. Custom MCTS Variants
```python
class CustomMCTS(MCTS):
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        # Custom selection strategy
        pass
        
    def _evaluate(self, node: MCTSNode) -> float:
        # Custom evaluation function
        pass
```

### 4. Training Modifications
```python
class CustomTrainer(AlphaZeroTrainer):
    def training_step(self, examples: List[TrainingExample]):
        # Custom training logic
        # - Different loss functions
        # - Learning rate schedules
        # - Curriculum learning
        pass
```

### 5. Evaluation Metrics
```python
class CustomEvaluator:
    def evaluate_position(self, state: UTTTState, agent: Agent) -> Dict[str, float]:
        # Custom position evaluation
        return {
            'complexity': self._compute_complexity(state),
            'risk': self._compute_risk(state, agent),
            'style': self._analyze_style(state, agent)
        }
```

## Performance Considerations

### Memory Optimization
- **State Caching**: Cache expensive computations in state objects
- **Tree Pruning**: Remove old MCTS nodes to save memory
- **Batch Processing**: Process training examples in batches

### GPU Acceleration
- **Batch Inference**: Process multiple states simultaneously
- **Mixed Precision**: Use float16 for memory savings
- **Data Loading**: Asynchronous data preparation

### Parallelization
- **Self-Play**: Parallel game generation
- **MCTS**: Parallel tree simulation (virtual loss)
- **Training**: Distributed training across GPUs

## Debugging and Profiling

### State Validation
```python
def validate_state(state: UTTTState):
    """Comprehensive state validation"""
    assert state.current_player in [1, -1]
    assert 0 <= state.move_count <= 81
    # ... more checks
```

### MCTS Debugging
```python
def debug_mcts_tree(mcts: MCTS, state: UTTTState, depth: int = 3):
    """Print MCTS tree structure for debugging"""
    # Recursively print tree nodes
```

### Performance Profiling
```python
import cProfile

def profile_training():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run training step
    trainer.train_step(examples)
    
    profiler.disable()
    profiler.print_stats(sort='cumulative')
```

## Testing Framework

### Unit Tests
```python
class TestUTTTState(unittest.TestCase):
    def test_move_validation(self):
        state = UTTTState()
        valid_actions = state.get_valid_actions()
        self.assertEqual(len(valid_actions), 81)  # All moves valid initially
    
    def test_terminal_detection(self):
        # Test various terminal conditions
        pass
```

### Integration Tests  
```python
class TestTrainingPipeline(unittest.TestCase):
    def test_end_to_end_training(self):
        # Test complete training pipeline
        pass
```

### Performance Tests
```python
class TestPerformance(unittest.TestCase):
    def test_mcts_speed(self):
        # Benchmark MCTS search speed
        pass
```

## Next Steps

For implementation details, see:
- [Training Guide](TRAINING_GUIDE.md) - Training process details
- [Tournament Guide](TOURNAMENT_GUIDE.md) - Evaluation procedures  
- [Examples](EXAMPLES.md) - Code examples and tutorials

For advanced topics:
- Custom architectures and training methods
- Distributed training setups
- Performance optimization techniques
- Research extensions and experiments