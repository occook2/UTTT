import random
from typing import List, Tuple

import numpy as np
from uttt.env.state import UTTTEnv, action_to_rc


class HeuristicAgent:
    """
    A simple, non-learning heuristic agent for Ultimate Tic-Tac-Toe that
    consistently beats a random agent by prioritizing:
      1) Immediate micro-board wins (and thus macro wins).
      2) Avoiding moves that send the opponent to a micro-board
         where they have an immediate win.
      3) Reasonable positional play (center > corners > edges).
      4) Light macro strategy: prefer moves that create/secure macro threats
         and that don't grant the opponent a 'free move anywhere'.

    This agent does NO simulation tree search and does NOT modify the env.
    It scores each legal move and picks the highest-scoring one (ties broken randomly).
    """

    # ---- Tunable weights (picked conservatively to beat random reliably) ----
    W_MACRO_WIN = 1_000_000
    W_MICRO_WIN = 10_000
    W_BLOCK_OPP_MACRO = 1_200      # if our micro-win also blocks opponent's macro triple
    W_CREATE_MACRO_THREAT = 500    # creating a 2-in-a-row on macro
    W_CREATE_MACRO_HALF = 60

    W_SEND_OPP_IMMEDIATE_WIN = -2_000  # per immediate win available to opponent in target micro
    W_SEND_FREE_MOVE = -120            # sending opponent to a closed micro (won/full) => free roam
    W_CONSTRAIN_OPP_SAFE = 30          # constraining opponent to an open micro with no immediate win

    W_PLAY_IN_WON_MICRO = -20          # playing inside a micro already won by either side is slightly bad

    W_CREATE_TWO_IN_ROW_LOCAL = 20     # after our move, local line with 2 of ours / 1 empty / 0 opp
    W_CREATE_ONE_IN_ROW_LOCAL = 6      # after our move, local line with 1 of ours / 2 empty / 0 opp

    POS_CENTER = 8
    POS_CORNER = 4
    POS_EDGE = 2

    def select_action(self, env: UTTTEnv) -> int:
        legal = env.legal_actions()
        if len(legal) == 1:
            return legal[0]

        best: List[int] = []
        best_score = -float("inf")
        for a in legal:
            s = self._score_action(env, a)
            if s > best_score + 1e-9:
                best_score = s
                best = [a]
            elif abs(s - best_score) <= 1e-9:
                best.append(a)

        # tie-breaker using env RNG if available for determinism, else Python's random
        if hasattr(env, "rng") and env.rng is not None:
            return int(env.rng.choice(np.array(best)))
        return random.choice(best)

    # ----------------------------- Scoring ---------------------------------

    def _score_action(self, env: UTTTEnv, action: int) -> float:
        r, c = action_to_rc(action)
        player = int(env.player)
        opp = -player

        score = 0.0

        mr, mc = r // 3, c // 3           # micro indices of the action
        lr, lc = r % 3, c % 3             # local cell inside the micro
        sub = env.board[mr * 3 : mr * 3 + 3, mc * 3 : mc * 3 + 3]
        macro = env.macro_wins

        # Positional preference in the local 3x3
        if lr == 1 and lc == 1:
            score += self.POS_CENTER
        elif (lr in (0, 2)) and (lc in (0, 2)):
            score += self.POS_CORNER
        else:
            score += self.POS_EDGE

        # Slight penalty for playing in an already-won micro (usually 'wasted' locally)
        if macro[mr, mc] != 0:
            score += self.W_PLAY_IN_WON_MICRO

        # If this move wins the micro, that’s very strong.
        micro_win = self._micro_win_after(sub, player, lr, lc)
        if micro_win:
            # Update macro hypothetically
            tm = macro.copy()
            if tm[mr, mc] == 0:
                tm[mr, mc] = player

            # Macro win?
            if self._micro_winner(tm) == player:
                score += self.W_MACRO_WIN
            else:
                score += self.W_MICRO_WIN
                # Creating macro threats by claiming this macro cell
                score += self._macro_threat_bonus(macro, (mr, mc), player)

                # If by taking this macro cell we also block opponent's imminent macro
                if self._blocks_opp_macro_now(macro, (mr, mc), player):
                    score += self.W_BLOCK_OPP_MACRO
        else:
            # Local line-building (only lines through (lr, lc))
            score += self._local_line_potential(sub, player, (lr, lc))

        # Where does this send the opponent next?
        target_r, target_c = r % 3, c % 3
        tr0, tc0 = target_r * 3, target_c * 3
        target_sub = env.board[tr0 : tr0 + 3, tc0 : tc0 + 3]
        target_macro_cell = macro[target_r, target_c]
        target_open = (target_macro_cell == 0) and (target_sub == 0).any()

        if target_open:
            # If opponent has immediate winning move there, that's BAD.
            opp_imm_wins = self._immediate_win_positions(target_sub, opp)
            if opp_imm_wins:
                score += self.W_SEND_OPP_IMMEDIATE_WIN * len(opp_imm_wins)
            else:
                score += self.W_CONSTRAIN_OPP_SAFE
        else:
            # Sending them to a closed micro gives them a 'free move anywhere'
            score += self.W_SEND_FREE_MOVE

        return score

    def best_of(self, env: UTTTEnv, legal: list[int]) -> int:
        if len(legal) == 1:
            return int(legal[0])

        best: list[int] = []
        best_score = -float("inf")
        for a in legal:
            s = self._score_action(env, a)
            if s > best_score + 1e-9:
                best_score = s
                best = [a]
            elif abs(s - best_score) <= 1e-9:
                best.append(a)

        if hasattr(env, "rng") and env.rng is not None:
            return int(env.rng.choice(np.array(best)))
        return random.choice(best)

    # ----------------------------- Helpers ---------------------------------

    @staticmethod
    def _micro_winner(sub: np.ndarray) -> int:
        """Same logic as env._micro_winner, kept locally to avoid importing private API."""
        lines = []
        lines.extend([sub[i, :] for i in range(3)])
        lines.extend([sub[:, j] for j in range(3)])
        lines.append(np.diag(sub))
        lines.append(np.diag(np.fliplr(sub)))
        for line in lines:
            s = int(np.sum(line))
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def _micro_win_after(self, sub: np.ndarray, player: int, lr: int, lc: int) -> bool:
        if sub[lr, lc] != 0:
            return False
        tmp = sub.copy()
        tmp[lr, lc] = player
        return self._micro_winner(tmp) == player

    def _immediate_win_positions(self, sub: np.ndarray, player: int) -> List[Tuple[int, int]]:
        wins: List[Tuple[int, int]] = []
        # Try all empties in this 3x3
        for rr in range(3):
            for cc in range(3):
                if sub[rr, cc] == 0:
                    if self._micro_win_after(sub, player, rr, cc):
                        wins.append((rr, cc))
        return wins

    def _local_line_potential(self, sub: np.ndarray, player: int, loc: Tuple[int, int]) -> float:
        """Reward creating clean lines (with no opponent marks) through the move cell."""
        lr, lc = loc
        if sub[lr, lc] != 0:
            return 0.0

        tmp = sub.copy()
        tmp[lr, lc] = player

        score = 0.0
        lines = []

        # Row and column through (lr, lc)
        lines.append(tmp[lr, :])
        lines.append(tmp[:, lc])

        # Diagonals if applicable
        if lr == lc:
            lines.append(np.diag(tmp))
        if lr + lc == 2:
            lines.append(np.diag(np.fliplr(tmp)))

        for line in lines:
            ours = int(np.sum(line == player))
            opps = int(np.sum(line == -player))
            empt = int(np.sum(line == 0))
            if opps == 0:
                if ours == 2 and empt == 1:
                    score += self.W_CREATE_TWO_IN_ROW_LOCAL
                elif ours == 1 and empt == 2:
                    score += self.W_CREATE_ONE_IN_ROW_LOCAL
        return score

    def _macro_threat_bonus(self, macro: np.ndarray, cell: Tuple[int, int], player: int) -> float:
        """After claiming macro[cell], award for creating 2-in-a-row threats."""
        mr, mc = cell
        if macro[mr, mc] != 0:
            return 0.0
        tmp = macro.copy()
        tmp[mr, mc] = player

        bonus = 0.0
        lines = []
        lines.extend([tmp[mr, :], tmp[:, mc]])
        if mr == mc:
            lines.append(np.diag(tmp))
        if mr + mc == 2:
            lines.append(np.diag(np.fliplr(tmp)))

        for line in lines:
            ours = int(np.sum(line == player))
            empt = int(np.sum(line == 0))
            opps = int(np.sum(line == -player))
            if opps == 0:
                if ours == 2 and empt == 1:
                    bonus += self.W_CREATE_MACRO_THREAT
                elif ours == 1 and empt == 2:
                    bonus += self.W_CREATE_MACRO_HALF
        return bonus

    def _blocks_opp_macro_now(self, macro: np.ndarray, cell: Tuple[int, int], player: int) -> bool:
        """
        True if (mr, mc) is currently the third cell of an opponent macro 3-in-a-row
        (i.e., opponent has two-in-a-row and this cell is empty). We only call this
        when we're also winning the micro to actually claim this macro cell now.
        """
        mr, mc = cell
        if macro[mr, mc] != 0:
            return False

        opp = -player

        def line_has_opp_two_and_empty(line: np.ndarray, idx: int) -> bool:
            # idx is the index of (mr, mc) within the given line
            return (int(np.sum(line == opp)) == 2) and (int(np.sum(line == 0)) == 1) and (line[idx] == 0)

        # Row
        if line_has_opp_two_and_empty(macro[mr, :], mc):
            return True
        # Column
        if line_has_opp_two_and_empty(macro[:, mc], mr):
            return True
        # Diagonals if applicable
        if mr == mc:
            diag = np.diag(macro)
            if line_has_opp_two_and_empty(diag, mr):
                return True
        if mr + mc == 2:
            ad = np.diag(np.fliplr(macro))
            if line_has_opp_two_and_empty(ad, mr):
                return True
        return False