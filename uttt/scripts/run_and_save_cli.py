# uttt/scripts/run_and_save_cli.py
import argparse
import time

from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent
from uttt.agents.base import Agent
from uttt.agents.az.agent import AlphaZeroAgent
from uttt.mcts.base import MCTSConfig
from uttt.eval.tournaments import play_series_with_records, save_series_json

AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
    "az": lambda: AlphaZeroAgent(mcts_config=MCTSConfig(n_simulations=2, temperature=0.0)),
}

AGENT_NAMES = {
    "random": "RandomAgent",
    "heuristic": "HeuristicAgent", 
    "az": "AlphaZeroAgent",
}

def main():
    start = time.time()
    p = argparse.ArgumentParser()
    p.add_argument("--A", default="heuristic", choices=AGENTS.keys())
    p.add_argument("--B", default="random", choices=AGENTS.keys())
    p.add_argument("--n_games", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="runs")
    args = p.parse_args()

    pkg = play_series_with_records(
        AGENTS[args.A], 
        AGENTS[args.B], 
        n_games=args.n_games, 
        base_seed=args.seed,
        agent_A_name=AGENT_NAMES[args.A],
        agent_B_name=AGENT_NAMES[args.B]
    )
    end = time.time()
    print(
        f"Tournament Time: {end - start:.2f}s\n"
        f"{args.A} (A) vs {args.B} (B) over {pkg.summary.n_games} games:\n"
        f"  A wins: {pkg.summary.wins_A}\n"
        f"  B wins: {pkg.summary.wins_B}\n"
        f"  Draws : {pkg.summary.draws}\n"
        f"  Avg moves: {pkg.summary.avg_moves:.2f}"
    )
    path = save_series_json(pkg, out_dir=args.out_dir)
    print("Saved to:", path)

if __name__ == "__main__":
    main()
