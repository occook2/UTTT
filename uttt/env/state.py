from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import List, Tuple, Optional

BOARD_N = 9
ACTION_SPACE = 81  # 0..80 => (r, c)


def action_to_rc(a: int) -> Tuple[int, int]:
    return divmod(a, BOARD_N)


def rc_to_action(r: int, c: int) -> int:
    return r * BOARD_N + c


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    terminated: bool
    info: dict


class UTTTEnv:
    """
    Minimal Gym-like API for Ultimate Tic-Tac-Toe.
    Planes: [current_player, X stones, O stones, legal mask,
             macro_X_wins, macro_O_wins, last_move]
    shape: (7, 9, 9)
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> np.ndarray:
        self.player = 1  # +1 = X, -1 = O
        self.board = np.zeros((9, 9), dtype=np.int8)  # -1,0,+1
        self.last_move: Optional[Tuple[int, int]] = None
        self.macro_wins = np.zeros((3, 3), dtype=np.int8)  # -1,0,+1
        self.terminated = False
        return self._encode()

    def step(self, action: int) -> StepResult:
        assert not self.terminated, "Game over"
        r, c = action_to_rc(action)
        legal = self.legal_actions_mask()
        if legal[r, c] == 0:
            raise ValueError(f"Illegal move at {(r, c)}")

        self.board[r, c] = self.player
        self._update_macro_wins(r, c)
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
        mask = (self.board == 0).astype(np.int8)
        if self.last_move is None:
            return mask
        mr, mc = self.last_move
        target_r, target_c = mr % 3, mc % 3  # target micro-board (3x3)
        r0, c0 = target_r * 3, target_c * 3
        micro = self.board[r0:r0+3, c0:c0+3]
        macro_cell = self.macro_wins[target_r, target_c]
        if macro_cell == 0 and (micro == 0).any():
            allowed = np.zeros_like(mask)
            allowed[r0:r0+3, c0:c0+3] = 1
            return (allowed & mask).astype(np.int8)
        return mask

    def _encode(self) -> np.ndarray:
        cur = np.full((9, 9), 1 if self.player == 1 else 0, dtype=np.int8)
        xs = (self.board == 1).astype(np.int8)
        os = (self.board == -1).astype(np.int8)
        legal = self.legal_actions_mask()
        mX = np.kron((self.macro_wins == 1).astype(np.int8), np.ones((3, 3), dtype=np.int8))
        mO = np.kron((self.macro_wins == -1).astype(np.int8), np.ones((3, 3), dtype=np.int8))
        last = np.zeros_like(xs)
        if self.last_move:
            r, c = self.last_move
            last[r, c] = 1
        planes = np.stack([cur, xs, os, legal, mX, mO, last], axis=0)
        return planes.astype(np.float32)

    def _update_macro_wins(self, r: int, c: int) -> None:
        mr, mc = r // 3, c // 3
        sub = self.board[mr*3:mr*3+3, mc*3:mc*3+3]
        w = self._micro_winner(sub)
        if w != 0:
            self.macro_wins[mr, mc] = w

    @staticmethod
    def _micro_winner(sub: np.ndarray) -> int:
        lines = []
        lines.extend([sub[i, :] for i in range(3)])
        lines.extend([sub[:, j] for j in range(3)])
        lines.append(np.diag(sub))
        lines.append(np.diag(np.fliplr(sub)))
        for line in lines:
            s = int(line.sum())
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def _macro_winner(self) -> int:
        return self._micro_winner(self.macro_wins)
