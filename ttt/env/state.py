from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math
from typing import List, Tuple, Optional

ACTION_SPACE = 9  # 0..8 => (r, c)


def action_to_rc(a: int) -> Tuple[int, int]:
    return (int(math.floor(a/3)), int(a % 3))


def rc_to_action(r: int, c: int) -> int:
    return (r*3 + c)


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    info: dict


class TTTEnv:
    """
    Minimal Gym-like API for Tic-Tac-Toe.
    Planes: [current_player, X stones, O stones, legal mask, last_move]
    shape: (5, 3, 3)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.player = 1  # +1 = X, -1 = O
        self.board = np.zeros((3, 3), dtype=np.int8)  # -1,0,+1
        self.last_move: Optional[Tuple[int, int]] = None
        self.terminated = False
        return self._encode()

    def step(self, action: int) -> StepResult:
        assert not self.terminated, "Game over"
        r, c = action_to_rc(action)
        legal = self.legal_actions_mask()
        if legal[r, c] == 0:
            raise ValueError(f"Illegal move at {(r, c)}")

        self.board[r, c] = self.player
        self.last_move = (r, c)

        winner = self._macro_winner()
        if winner != 0 or not self.legal_actions_mask().any():
            self.terminated = True
            reward = float(winner)  # +1, -1, or 0 draw
        else:
            reward = 0.0
            self.player *= -1

        return StepResult(self._encode(), reward, self.terminated, {"winner": winner})

    def legal_actions(self) -> List[int]:
        mask = self.legal_actions_mask()
        return [rc_to_action(r, c) for r, c in zip(*np.where(mask == 1))]

    def legal_actions_mask(self) -> np.ndarray:
        return (self.board == 0).astype(np.int8)


    def _encode(self) -> np.ndarray:
        cur = np.full((3, 3), 1 if self.player == 1 else 0, dtype=np.int8)
        xs = (self.board == 1).astype(np.int8)
        os = (self.board == -1).astype(np.int8)
        legal = self.legal_actions_mask()
        last = np.zeros_like(xs)
        if self.last_move:
            r, c = self.last_move
            last[r, c] = 1
        planes = np.stack([cur, xs, os, legal, last], axis=0)
        return planes.astype(np.float32)


    def _macro_winner(self) -> int:
        lines = []
        lines.extend([self.board[i, :] for i in range(3)])
        lines.extend([self.board[:, j] for j in range(3)])
        lines.append(np.diag(self.board))
        lines.append(np.diag(np.fliplr(self.board)))
        for line in lines:
            s = int(line.sum())
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0