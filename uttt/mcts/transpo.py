# transpo.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from uttt.env.state import UTTTEnv

def state_key(env: UTTTEnv) -> Tuple[bytes, bytes, Optional[Tuple[int, int]], int]:
    """
    Build a compact, hashable key capturing everything that matters for game logic:
    - full 9x9 board
    - 3x3 macro_wins (with 10 meaning 'closed via draw' in your env)
    - last_move (or None)
    - player to move (+1 or -1)
    """
    return (
        env.board.tobytes(),
        env.macro_wins.tobytes(),
        None if env.last_move is None else (int(env.last_move[0]), int(env.last_move[1])),
        int(env.player),
    )

def clone_env(src: UTTTEnv) -> UTTTEnv:
    """
    Fast, safe deep clone that mirrors the source env's public fields.
    Uses the class's constructor then overwrites attributes directly.
    """
    dst = UTTTEnv()
    # Core fields
    dst.player = int(src.player)
    dst.board = np.array(src.board, copy=True)
    dst.last_move = None if src.last_move is None else (int(src.last_move[0]), int(src.last_move[1]))
    dst.macro_wins = np.array(src.macro_wins, copy=True)
    dst.terminated = bool(src.terminated)
    # Copy RNG if present for deterministic tie-breakers
    if hasattr(src, "rng") and src.rng is not None:
        # numpy Generator isn't trivially clonable; reseed deterministically from bit generator state
        # to keep tie-break behavior stable but avoid shared state.
        st = src.rng.bit_generator.state
        seed_like = (st["state"]["state"] ^ st["state"]["inc"]) & 0xFFFFFFFF
        dst.rng = np.random.default_rng(int(seed_like))
    return dst
