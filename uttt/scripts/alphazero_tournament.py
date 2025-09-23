"""
Convenience script for running tournaments with AlphaZero agents.
"""
import argparse
import os
from typing import List, Tuple

from uttt.eval.tournaments import play_series_with_records, play_series_with_openings, save_series_json
from uttt.eval.alphazero_factory import (
    alphazero_agent_factory, 
    discover_alphazero_checkpoints,
    get_alphazero_agent_name
)
from uttt.eval.openings import DEFAULT_OPENING_BOOK, generate_opening_book
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent


def run_alphazero_vs_baseline(
    checkpoint_path: str,
    baseline_type: str = "random",
    n_games: int = 100,
    mcts_simulations: int = 100,
    output_dir: str = "runs",
    use_openings: bool = True,
    opening_moves: int = 1
) -> str:
    """
    Run a tournament between an AlphaZero checkpoint and a baseline agent.
    
    Args:
        checkpoint_path: Path to AlphaZero checkpoint
        baseline_type: "random" or "heuristic"
        n_games: Number of games to play
        mcts_simulations: MCTS simulations for AlphaZero
        output_dir: Directory to save results
        use_openings: Whether to use opening book
        opening_moves: Number of moves in each opening
        
    Returns:
        Path to saved results file
    """
    # Create AlphaZero agent factory
    az_factory = alphazero_agent_factory(
        checkpoint_path,
        mcts_simulations=mcts_simulations,
        temperature=0.0  # Deterministic for evaluation
    )
    
    # Create baseline agent
    if baseline_type == "random":
        baseline_agent = RandomAgent
        baseline_kwargs = {}
    elif baseline_type == "heuristic":
        baseline_agent = HeuristicAgent
        baseline_kwargs = {}
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    # Get readable names
    az_name = get_alphazero_agent_name(checkpoint_path, mcts_simulations=mcts_simulations)
    baseline_name = baseline_type.capitalize() + "Agent"
    
    print(f"Running tournament: {az_name} vs {baseline_name}")
    
    if use_openings:
        # Calculate number of openings needed for desired total games
        games_per_opening = 2  # Each opening plays 2 games (swap sides)
        n_openings = max(1, n_games // games_per_opening)
        
        if n_openings <= len(DEFAULT_OPENING_BOOK):
            opening_book = DEFAULT_OPENING_BOOK[:n_openings]
        else:
            opening_book = generate_opening_book(n_openings, opening_moves)
        
        print(f"Using {len(opening_book)} openings, {len(opening_book) * 2} total games")
        print(f"MCTS simulations: {mcts_simulations}")
        
        # Use the new opening-based tournament
        series = play_series_with_openings(
            AgentA=az_factory,
            AgentB=baseline_agent,
            opening_book=opening_book,
            instantiate_kwargs_A={},
            instantiate_kwargs_B=baseline_kwargs,
            agent_A_name=az_name,
            agent_B_name=baseline_name
        )
    else:
        print(f"Games: {n_games}, MCTS simulations: {mcts_simulations}")
        # Original tournament without openings
        series = play_series_with_records(
            AgentA=az_factory,
            AgentB=baseline_agent,
            n_games=n_games,
            instantiate_kwargs_A={},  # AlphaZero factory already configured
            instantiate_kwargs_B=baseline_kwargs,
            agent_A_name=az_name,
            agent_B_name=baseline_name
        )
    
    # Save results
    result_path = save_series_json(series, output_dir)
    
    # Print summary
    s = series.summary
    print(f"\nResults:")
    print(f"  {az_name} wins: {s.wins_A} ({s.wins_A/s.n_games*100:.1f}%)")
    print(f"  {baseline_name} wins: {s.wins_B} ({s.wins_B/s.n_games*100:.1f}%)")
    print(f"  Draws: {s.draws} ({s.draws/s.n_games*100:.1f}%)")
    print(f"  Average moves: {s.avg_moves:.1f}")
    print(f"  Results saved to: {result_path}")
    
    return result_path


def run_alphazero_vs_alphazero(
    checkpoint_a: str,
    checkpoint_b: str,
    n_games: int = 100,
    mcts_simulations: int = 100,
    output_dir: str = "runs",
    use_openings: bool = True,
    opening_moves: int = 1
) -> str:
    """
    Run a tournament between two AlphaZero checkpoints.
    """
    # Create agent factories
    az_factory_a = alphazero_agent_factory(
        checkpoint_a,
        mcts_simulations=mcts_simulations,
        temperature=0.0
    )
    
    az_factory_b = alphazero_agent_factory(
        checkpoint_b,
        mcts_simulations=mcts_simulations,
        temperature=0.0
    )
    
    # Get readable names
    name_a = get_alphazero_agent_name(checkpoint_a, mcts_simulations=mcts_simulations)
    name_b = get_alphazero_agent_name(checkpoint_b, mcts_simulations=mcts_simulations)
    
    print(f"Running tournament: {name_a} vs {name_b}")
    
    if use_openings:
        games_per_opening = 2
        n_openings = max(1, n_games // games_per_opening)
        
        if n_openings <= len(DEFAULT_OPENING_BOOK):
            opening_book = DEFAULT_OPENING_BOOK[:n_openings]
        else:
            opening_book = generate_opening_book(n_openings, opening_moves)
        
        print(f"Using {len(opening_book)} openings, {len(opening_book) * 2} total games")
        print(f"MCTS simulations: {mcts_simulations}")
        
        series = play_series_with_openings(
            AgentA=az_factory_a,
            AgentB=az_factory_b,
            opening_book=opening_book,
            instantiate_kwargs_A={},
            instantiate_kwargs_B={},
            agent_A_name=name_a,
            agent_B_name=name_b
        )
    else:
        print(f"Games: {n_games}, MCTS simulations: {mcts_simulations}")
        
        # Run the series
        series = play_series_with_records(
            AgentA=az_factory_a,
            AgentB=az_factory_b,
            n_games=n_games,
            instantiate_kwargs_A={},  # AlphaZero factories already configured
            instantiate_kwargs_B={},
            agent_A_name=name_a,
            agent_B_name=name_b
        )
    
    # Save results
    result_path = save_series_json(series, output_dir)
    
    # Print summary
    s = series.summary
    print(f"\nResults:")
    print(f"  {name_a} wins: {s.wins_A} ({s.wins_A/s.n_games*100:.1f}%)")
    print(f"  {name_b} wins: {s.wins_B} ({s.wins_B/s.n_games*100:.1f}%)")
    print(f"  Draws: {s.draws} ({s.draws/s.n_games*100:.1f}%)")
    print(f"  Average moves: {s.avg_moves:.1f}")
    print(f"  Results saved to: {result_path}")
    
    return result_path


def list_available_checkpoints(checkpoint_dir: str = "checkpoints"):
    """List all available AlphaZero checkpoints."""
    checkpoints = discover_alphazero_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"Available AlphaZero checkpoints in {checkpoint_dir}:")
    for name, path in sorted(checkpoints.items()):
        print(f"  {name}: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run AlphaZero tournaments")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List checkpoints command
    list_parser = subparsers.add_parser('list', help='List available checkpoints')
    list_parser.add_argument('--checkpoint-dir', default='checkpoints', 
                           help='Directory to search for checkpoints')
    
    # vs baseline command
    baseline_parser = subparsers.add_parser('vs-baseline', help='Run vs baseline agent')
    baseline_parser.add_argument('checkpoint', help='Path to AlphaZero checkpoint')
    baseline_parser.add_argument('--baseline', choices=['random', 'heuristic'], 
                                default='random', help='Baseline agent type')
    baseline_parser.add_argument('--games', type=int, default=100, help='Number of games')
    baseline_parser.add_argument('--simulations', type=int, default=100, 
                                help='MCTS simulations per move')
    baseline_parser.add_argument('--output-dir', default='runs', help='Output directory')
    baseline_parser.add_argument('--no-openings', action='store_true', 
                               help='Disable opening book (play from empty board)')
    baseline_parser.add_argument('--opening-moves', type=int, default=1,
                               help='Number of moves in each opening')
    
    # vs alphazero command
    az_parser = subparsers.add_parser('vs-alphazero', help='Run vs another AlphaZero')
    az_parser.add_argument('checkpoint_a', help='Path to first AlphaZero checkpoint')
    az_parser.add_argument('checkpoint_b', help='Path to second AlphaZero checkpoint')
    az_parser.add_argument('--games', type=int, default=100, help='Number of games')
    az_parser.add_argument('--simulations', type=int, default=100, 
                          help='MCTS simulations per move')
    az_parser.add_argument('--output-dir', default='runs', help='Output directory')
    az_parser.add_argument('--no-openings', action='store_true',
                          help='Disable opening book (play from empty board)')
    az_parser.add_argument('--opening-moves', type=int, default=1,
                          help='Number of moves in each opening')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_available_checkpoints(args.checkpoint_dir)
    elif args.command == 'vs-baseline':
        run_alphazero_vs_baseline(
            args.checkpoint,
            args.baseline,
            args.games,
            args.simulations,
            args.output_dir,
            use_openings=not args.no_openings,
            opening_moves=args.opening_moves
        )
    elif args.command == 'vs-alphazero':
        run_alphazero_vs_alphazero(
            args.checkpoint_a,
            args.checkpoint_b,
            args.games,
            args.simulations,
            args.output_dir,
            use_openings=not args.no_openings,
            opening_moves=args.opening_moves
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()