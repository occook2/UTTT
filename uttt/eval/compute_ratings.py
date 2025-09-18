# uttt/eval/compute_ratings.py
from __future__ import annotations
import argparse
import glob
import json
import os
from datetime import datetime
from typing import Dict, Any

from uttt.eval.elo import EloTable

def agent_id_from_meta(meta: Dict[str, Any], role: str) -> str:
    """
    role is 'A' or 'B'. We build a stable ID from class name + kwargs.
    Example: 'MCTSAgent(c_explore=1.4,n_simulations=200,rollout=heuristic)'
    For AlphaZero agents, include checkpoint information for distinction.
    """
    cls = meta[f"agent_{role}"]
    kwargs = meta.get(f"instantiate_kwargs_{role}", {}) or {}
    
    # Special handling for AlphaZero agents to include checkpoint info
    if "AlphaZero" in cls or "AZ-" in cls:
        # Use the full agent name that includes checkpoint info
        agent_id = meta.get(f"agent_{role}_id", cls)
        if agent_id != cls:
            return agent_id
        
        # Fallback to constructing from kwargs if available
        checkpoint = kwargs.get('checkpoint_path', 'Random')
        if checkpoint != 'Random':
            checkpoint_name = os.path.basename(checkpoint).replace('.pt', '')
            return f"AlphaZero({checkpoint_name})"
    
    # Standard handling for other agent types
    if not kwargs:
        return cls
    # canonicalize keys for stability
    parts = [f"{k}={kwargs[k]}" for k in sorted(kwargs.keys())]
    return f"{cls}(" + ",".join(parts) + ")"

def score_from_winner(winner: int, a_is_x: bool) -> float:
    """
    Convert game result to 'score for A' in [0, 0.5, 1].
    winner: +1 (X), -1 (O), 0 (draw)
    a_is_x: True if AgentA was X in this game
    """
    if winner == 0:
        return 0.5
    if winner == 1:
        return 1.0 if a_is_x else 0.0
    if winner == -1:
        return 0.0 if a_is_x else 1.0
    raise ValueError(f"Unexpected winner value: {winner}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs", help="Directory with series_*.json files")
    p.add_argument("--ratings_out", default="runs/elo_ratings.json", help="Path to write Elo table")
    p.add_argument("--ratings_in", default=None, help="Optional existing Elo JSON to start from")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.ratings_in and os.path.exists(args.ratings_in):
        book = EloTable.load(args.ratings_in)
    else:
        book = EloTable(default_rating=1500.0, k_base=24.0)

    paths = sorted(glob.glob(os.path.join(args.runs_dir, "series_*.json")))
    if not paths:
        print(f"No series files found under {args.runs_dir}")
        return

    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        meta = payload["meta"]
        games = payload["games"]

        idA = agent_id_from_meta(meta, "A")
        idB = agent_id_from_meta(meta, "B")
        timestamp = meta.get("timestamp", None)

        for g in games:
            a_is_x = bool(g["a_is_x"])
            winner = int(g["winner"])
            score_a = score_from_winner(winner, a_is_x)
            book.record(idA, idB, score_a, played_at=timestamp)

        if args.verbose:
            print(f"Processed {path}: {len(games)} games | {idA} vs {idB}")

    # Save
    os.makedirs(os.path.dirname(args.ratings_out), exist_ok=True)
    book.save(args.ratings_out)

    # Pretty print top table to stdout
    print("Top ratings:")
    for row in book.table()[:20]:
        print(f"{row['rating']:7.1f}  {row['games']:4d}  {row['id']}")

if __name__ == "__main__":
    main()
