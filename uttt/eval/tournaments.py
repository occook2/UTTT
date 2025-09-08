# eval/tournaments.py
from __future__ import annotations

import json
import os

from dataclasses import dataclass, asdict
import dataclasses as dc
from datetime import datetime
from typing import Type, Optional, Dict, Any, List
import numpy as np

from uttt.env.state import UTTTEnv
from uttt.agents.random import RandomAgent
from uttt.agents.heuristic import HeuristicAgent


@dataclass
class GameResult:
    winner: int        # +1 (X), -1 (O), 0 (draw)
    moves: int
    seed: Optional[int]


@dataclass
class SeriesResult:
    n_games: int
    wins_A: int
    wins_B: int
    draws: int
    avg_moves: float
    details: List[GameResult]

@dataclass
class GameRecord:
    seed: Optional[int]
    a_is_x: bool          # True if AgentA played X in this game
    winner: int           # +1 (X), -1 (O), 0 (draw)
    moves: List[int]      # list of 0..80 actions in order
    moves_len: int

@dataclass
class SeriesSummary:
    n_games: int
    wins_A: int
    wins_B: int
    draws: int
    avg_moves: float

@dataclass
class SeriesPackage:
    meta: Dict[str, Any]
    summary: SeriesSummary
    games: List[GameRecord]    

def play_game(agent_X: Any, agent_O: Any, seed: Optional[int] = None) -> GameResult:
    """
    Play a single game of UTTT with the given agents (X and O).
    Returns the winner and number of moves.
    """
    env = UTTTEnv(seed=seed)
    moves = 0

    while True:
        agent = agent_X if env.player == 1 else agent_O
        action = agent.select_action(env)
        step = env.step(action)
        moves += 1
        if step.terminated:
            winner = int(step.info.get("winner", 0))
            return GameResult(winner=winner, moves=moves, seed=seed)


def play_series(
    AgentA: Type, AgentB: Type,
    n_games: int = 200,
    base_seed: int = 0,
    instantiate_kwargs_A: Optional[Dict[str, Any]] = None,
    instantiate_kwargs_B: Optional[Dict[str, Any]] = None,
) -> SeriesResult:
    """
    Head-to-head series between AgentA and AgentB.
    Even-indexed games: A is X, B is O.
    Odd-indexed games:  B is X, A is O.

    Returns win/draw counts, avg moves, and per-game details.
    """
    instantiate_kwargs_A = instantiate_kwargs_A or {}
    instantiate_kwargs_B = instantiate_kwargs_B or {}

    wins_A = wins_B = draws = 0
    details: List[GameResult] = []
    total_moves = 0

    for g in range(n_games):
        if g % 20 == 0:
            print(f"Starting game {g+1}/{n_games}...")
        # Fresh agent instances each game (clean state)
        A = AgentA(**instantiate_kwargs_A)
        B = AgentB(**instantiate_kwargs_B)

        if g % 2 == 0:
            # A as X, B as O
            res = play_game(agent_X=A, agent_O=B, seed=base_seed + g)
            # Map winner to A/B
            if res.winner == 1:
                wins_A += 1
            elif res.winner == -1:
                wins_B += 1
            else:
                draws += 1
        else:
            # B as X, A as O
            res = play_game(agent_X=B, agent_O=A, seed=base_seed + g)
            if res.winner == 1:
                wins_B += 1
            elif res.winner == -1:
                wins_A += 1
            else:
                draws += 1

        details.append(res)
        total_moves += res.moves

    avg_moves = total_moves / max(1, n_games)
    return SeriesResult(
        n_games=n_games,
        wins_A=wins_A,
        wins_B=wins_B,
        draws=draws,
        avg_moves=avg_moves,
        details=details,
    )

def _play_game_record(agent_X: Any, agent_O: Any, seed: Optional[int]) -> GameRecord:
    env = UTTTEnv(seed=seed)
    moves: List[int] = []
    while True:
        agent = agent_X if env.player == 1 else agent_O
        action = agent.select_action(env)
        moves.append(int(action))
        step = env.step(action)
        if step.terminated:
            return GameRecord(
                seed=seed,
                a_is_x=False,  # filled by caller (we don't know here)
                winner=int(step.info.get("winner", 0)),
                moves=moves,
                moves_len=len(moves),
            )

def play_series_with_records(
    AgentA: Type, AgentB: Type,
    n_games: int = 200,
    base_seed: int = 0,
    instantiate_kwargs_A: Optional[Dict[str, Any]] = None,
    instantiate_kwargs_B: Optional[Dict[str, Any]] = None,
) -> SeriesPackage:
    """
    Like play_series, but returns full per-game move records suitable for saving & replay.
    Even games: A is X; odd games: B is X.
    """
    instantiate_kwargs_A = instantiate_kwargs_A or {}
    instantiate_kwargs_B = instantiate_kwargs_B or {}

    wins_A = wins_B = draws = 0
    total_moves = 0
    games: List[GameRecord] = []

    for g in range(n_games):
        A = AgentA(**instantiate_kwargs_A)
        B = AgentB(**instantiate_kwargs_B)
        seed = base_seed + g

        if g % 2 == 0:
            # A as X, B as O
            rec = _play_game_record(agent_X=A, agent_O=B, seed=seed)
            rec.a_is_x = True
            if rec.winner == 1:
                wins_A += 1
            elif rec.winner == -1:
                wins_B += 1
            else:
                draws += 1
        else:
            # B as X, A as O
            rec = _play_game_record(agent_X=B, agent_O=A, seed=seed)
            rec.a_is_x = False
            if rec.winner == 1:
                wins_B += 1
            elif rec.winner == -1:
                wins_A += 1
            else:
                draws += 1

        total_moves += rec.moves_len
        games.append(rec)

    summary = SeriesSummary(
        n_games=n_games,
        wins_A=wins_A,
        wins_B=wins_B,
        draws=draws,
        avg_moves=(total_moves / max(1, n_games)),
    )

    meta = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "agent_A": AgentA.__name__,
        "agent_B": AgentB.__name__,
        "base_seed": base_seed,
        "instantiate_kwargs_A": instantiate_kwargs_A,
        "instantiate_kwargs_B": instantiate_kwargs_B,
        "schema": {
            "game": ["seed", "a_is_x", "winner", "moves", "moves_len"],
            "winner_values": {"X": 1, "O": -1, "draw": 0},
        },
    }

    return SeriesPackage(meta=meta, summary=summary, games=games)

def _jsonify(obj):
    if dc.is_dataclass(obj):
        obj = asdict(obj)
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_jsonify(x) for x in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def save_series_json(pkg: SeriesPackage, out_dir: str = "runs") -> str:
    os.makedirs(out_dir, exist_ok=True)
    fname = f"series_{pkg.meta['agent_A']}_vs_{pkg.meta['agent_B']}_{pkg.meta['timestamp']}.json"
    path = os.path.join(out_dir, fname)

    payload = _jsonify({
        "meta": pkg.meta,
        "summary": asdict(pkg.summary),
        "games": [asdict(g) for g in pkg.games],
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path     

if __name__ == "__main__":
    # Quick smoke test: HeuristicAgent vs RandomAgent
    result = play_series(HeuristicAgent, RandomAgent, n_games=200, base_seed=42)
    print(
        f"Heuristic (A) vs Random (B) over {result.n_games} games:\n"
        f"  A wins: {result.wins_A}\n"
        f"  B wins: {result.wins_B}\n"
        f"  Draws : {result.draws}\n"
        f"  Avg moves: {result.avg_moves:.2f}"
    )
