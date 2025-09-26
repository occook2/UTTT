"""
Command-line test script for self-play functionality.
Run with: python uttt/tests/test_selfplay.py
"""
import numpy as np
import tempfile
import os
import traceback
from unittest.mock import Mock

from uttt.agents.az.agent import AlphaZeroAgent
from uttt.agents.az.self_play import SelfPlayTrainer, TrainingExample, run_self_play_session
from uttt.mcts.base import MCTSConfig
from uttt.env.state import UTTTEnv


def test_training_example_creation():
    """Test creating a training example."""
    print("Testing TrainingExample creation...")
    state = np.random.random((9, 9, 3))
    policy = np.random.random(81)
    value = 0.5
    
    example = TrainingExample(state=state, policy=policy, value=value)
    
    assert np.array_equal(example.state, state)
    assert np.array_equal(example.policy, policy)
    assert example.value == value
    print("✅ TrainingExample creation test passed")


def create_test_agent():
    """Create a test agent."""
    return AlphaZeroAgent()


def create_test_mcts_config():
    """Create a test MCTS config."""
    return MCTSConfig(
        n_simulations=5,  # Small number for fast tests
        c_puct=1.0,
        temperature=1.0,
        use_transposition_table=True
    )


def create_test_trainer():
    """Create a test trainer."""
    agent = create_test_agent()
    mcts_config = create_test_mcts_config()
    return SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=5,
        collect_data=True
    )


def test_trainer_initialization():
    """Test trainer initialization."""
    print("Testing trainer initialization...")
    agent = create_test_agent()
    mcts_config = create_test_mcts_config()
    
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=10,
        collect_data=True
    )
    
    assert trainer.agent == agent
    assert trainer.mcts_config == mcts_config
    assert trainer.temperature_threshold == 10
    assert trainer.collect_data is True
    assert len(trainer.training_examples) == 0
    print("✅ Trainer initialization test passed")


def test_play_game_completes():
    """Test that a single game completes successfully."""
    print("Testing game completion...")
    trainer = create_test_trainer()
    examples, stats = trainer.play_game()
    
    # Check that game completed
    assert isinstance(examples, list)
    assert isinstance(stats, dict)
    assert 'moves' in stats
    assert 'winner' in stats
    assert 'game_length' in stats
    assert stats['moves'] > 0
    assert stats['moves'] == stats['game_length']
    print("✅ Game completion test passed")


def test_play_game_generates_examples():
    """Test that playing a game generates training examples."""
    print("Testing example generation...")
    trainer = create_test_trainer()
    examples, stats = trainer.play_game()
    
    # Should have at least one example (games should have moves)
    assert len(examples) > 0
    
    # Check example structure
    for example in examples:
        assert isinstance(example, TrainingExample)
        assert example.state.shape == (7, 9, 9)  # UTTT state representation (7 planes)
        assert example.policy.shape == (81,)  # 9x9 grid
        assert isinstance(example.value, float)
        assert example.value in [-1.0, 0.0, 1.0]  # Valid outcome values
    print("✅ Example generation test passed")


def test_play_game_without_data_collection():
    """Test playing game without collecting training data."""
    print("Testing game without data collection...")
    agent = create_test_agent()
    mcts_config = create_test_mcts_config()
    
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        collect_data=False
    )
    
    examples, stats = trainer.play_game()
    
    assert len(examples) == 0  # No examples should be collected
    assert stats['moves'] > 0  # But game should still complete
    print("✅ No data collection test passed")


def test_policy_dict_to_array():
    """Test converting MCTS policy dict to array."""
    print("Testing policy dict to array conversion...")
    trainer = create_test_trainer()
    env = UTTTEnv()
    legal_actions = env.legal_actions()
    
    # Create a simple policy dict (simulate what might be passed)
    policy_dict = {legal_actions[0]: 10, legal_actions[1]: 5}
    
    policy_array = trainer._policy_dict_to_array(policy_dict, env)
    
    assert policy_array.shape == (81,)
    assert abs(np.sum(policy_array) - 1.0) < 1e-6  # Should be normalized
    assert abs(policy_array[legal_actions[0]] - 10/15) < 1e-6  # 10/(10+5)
    assert abs(policy_array[legal_actions[1]] - 5/15) < 1e-6   # 5/(10+5)
    print("✅ Policy conversion test passed")


def test_update_examples_with_outcome_player1_wins():
    """Test updating examples when player 1 wins."""
    print("Testing outcome updates (Player 1 wins)...")
    trainer = create_test_trainer()
    
    # Create mock examples (alternating players)
    examples = [
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0),  # Player 1 move
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0),  # Player 2 move
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0),  # Player 1 move
    ]
    
    # Player 1 wins (winner = 1)
    trainer._update_examples_with_outcome(examples, winner=1)
    
    assert examples[0].value == 1.0   # Player 1 wins
    assert examples[1].value == -1.0  # Player 2 loses
    assert examples[2].value == 1.0   # Player 1 wins
    print("✅ Player 1 wins outcome test passed")


def test_update_examples_with_outcome_draw():
    """Test updating examples when game is a draw."""
    print("Testing outcome updates (draw)...")
    trainer = create_test_trainer()
    
    examples = [
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0),
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0),
    ]
    
    # Draw (winner = 0)
    trainer._update_examples_with_outcome(examples, winner=0)
    
    assert examples[0].value == 0.0
    assert examples[1].value == 0.0
    print("✅ Draw outcome test passed")


def test_generate_training_data():
    """Test generating training data from multiple games."""
    print("Testing training data generation...")
    trainer = create_test_trainer()
    
    n_games = 3
    examples = trainer.generate_training_data(n_games)
    
    assert len(examples) > 0
    assert len(trainer.training_examples) == len(examples)
    
    # Check that examples are valid
    for example in examples:
        assert isinstance(example, TrainingExample)
        assert example.value in [-1.0, 0.0, 1.0]
    print("✅ Training data generation test passed")


def test_clear_training_examples():
    """Test clearing training examples."""
    print("Testing clear training examples...")
    trainer = create_test_trainer()
    
    # Add some examples first
    trainer.training_examples.append(
        TrainingExample(np.zeros((7, 9, 9)), np.zeros(81), 0.0)
    )
    
    assert len(trainer.training_examples) == 1
    
    trainer.clear_training_examples()
    
    assert len(trainer.training_examples) == 0
    print("✅ Clear examples test passed")


def test_save_and_load_training_data():
    """Test saving and loading training data."""
    print("Testing save and load training data...")
    trainer = create_test_trainer()
    
    # Generate some training data
    trainer.generate_training_data(2)
    original_count = len(trainer.training_examples)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        trainer.save_training_data(tmp_path)
        
        # Clear and reload
        trainer.clear_training_examples()
        assert len(trainer.training_examples) == 0
        
        trainer.load_training_data(tmp_path)
        assert len(trainer.training_examples) == original_count
        
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    print("✅ Save/load test passed")


def test_run_self_play_session():
    """Test running a self-play session."""
    print("Testing self-play session...")
    agent = create_test_agent()
    
    examples = run_self_play_session(
        agent=agent,
        n_games=2,
        n_simulations=5,  # Small number for fast test
        temperature_threshold=3
    )
    
    assert len(examples) > 0
    
    # Check example validity
    for example in examples:
        assert isinstance(example, TrainingExample)
        assert example.state.shape == (7, 9, 9)
        assert example.policy.shape == (81,)
        assert example.value in [-1.0, 0.0, 1.0]
    print("✅ Self-play session test passed")


def test_run_self_play_session_with_different_params():
    """Test running self-play with different parameters."""
    print("Testing self-play with different parameters...")
    agent = create_test_agent()
    
    examples = run_self_play_session(
        agent=agent,
        n_games=1,
        n_simulations=3,
        temperature_threshold=1
    )
    
    assert len(examples) > 0
    print("✅ Different parameters test passed")


def test_self_play_integration():
    """Test self-play with a real agent (integration test)."""
    print("Testing self-play integration...")
    # This test ensures the whole pipeline works together
    agent = create_test_agent()
    
    mcts_config = MCTSConfig(
        n_simulations=3,  # Very small for fast test
        c_puct=1.0,
        temperature=1.0,
        use_transposition_table=True
    )
    
    trainer = SelfPlayTrainer(
        agent=agent,
        mcts_config=mcts_config,
        temperature_threshold=2
    )
    
    # Play a single game
    examples, stats = trainer.play_game()
    
    # Verify everything worked
    assert len(examples) > 0
    assert stats['moves'] > 0
    assert stats['winner'] in [0, 1, -1]  # 0=draw, 1=player1, -1=player2
    
    # Verify examples are properly formatted
    for example in examples:
        assert not np.isnan(example.state).any()
        assert not np.isnan(example.policy).any()
        assert not np.isnan(example.value)
        assert abs(np.sum(example.policy) - 1.0) < 1e-6
    print("✅ Integration test passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 50)
    print("Running Self-Play Tests")
    print("=" * 50)
    
    tests = [
        test_training_example_creation,
        test_trainer_initialization,
        test_play_game_completes,
        test_play_game_generates_examples,
        test_play_game_without_data_collection,
        test_policy_dict_to_array,
        test_update_examples_with_outcome_player1_wins,
        test_update_examples_with_outcome_draw,
        test_generate_training_data,
        test_clear_training_examples,
        test_save_and_load_training_data,
        test_run_self_play_session,
        test_run_self_play_session_with_different_params,
        test_self_play_integration,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {str(e)}")
            traceback.print_exc()
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0



if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! Self-play functionality is working correctly.")
        
        # Run a quick demo
        print("\n" + "=" * 50)
        print("Running Quick Demo")
        print("=" * 50)
        
        agent = AlphaZeroAgent()
        examples = run_self_play_session(agent, n_games=1, n_simulations=3)
        
        print(f"✅ Generated {len(examples)} training examples")
        if examples:
            print(f"✅ Example state shape: {examples[0].state.shape}")
            print(f"✅ Example policy shape: {examples[0].policy.shape}")
            print(f"✅ Example value: {examples[0].value}")
        print("Demo completed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the output above.")
        exit(1)