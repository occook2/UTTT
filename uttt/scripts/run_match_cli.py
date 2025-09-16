import argparse

from uttt.env.state import UTTTEnv
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent
from uttt.agents.base import Agent
from uttt.eval.tournaments import play_series

AGENTS = {
    "random": RandomAgent,
    "heuristic": HeuristicAgent,
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--A", default="heuristic", choices=AGENTS.keys(),
                        help="Agent A (plays X on even-numbered games)")
    parser.add_argument("--B", default="random", choices=AGENTS.keys(),
                        help="Agent B (plays O on even-numbered games)")
    parser.add_argument("--n_games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = play_series(AGENTS[args.A], AGENTS[args.B], n_games=args.n_games, base_seed=args.seed)
    print(f"{args.A} (A) vs {args.B} (B) over {result.n_games} games:")
    print(f"  A wins: {result.wins_A}")
    print(f"  B wins: {result.wins_B}")
    print(f"  Draws : {result.draws}")
    print(f"  Avg moves: {result.avg_moves:.2f}")


if __name__ == "__main__":
    main()
