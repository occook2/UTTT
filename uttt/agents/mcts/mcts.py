# mcts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
import time
from collections import defaultdict

from uttt.env.state import UTTTEnv
from uttt.agents.heuristic import HeuristicAgent
from .uct import uct_score
from .transpo import state_key, clone_env


Edge = Tuple[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], int]  # (state_key, action)


@dataclass
class MCTSConfig:
    n_simulations: int = 200
    c_explore: float = 0.8
    rollout: str = "eps_heuristic"  # "heuristic" | "random" | "eps_heuristic"
    epsilon: float = 0        # only used when rollout == "eps_heuristic"
    use_tt: bool = True         # enable transposition table (merge states)
    rng_seed: Optional[int] = None
    rollout_depth_limit: int = 4
    profile: bool = False


class MCTSAgent:
    """
    Drop-in agent: call `act(env)` to pick a legal move using Monte Carlo Tree Search.

    Conventions:
      - Values are stored from the perspective of the player to move at a state `s`.
      - Terminal outcome from env.step: +1 for X, -1 for O, 0 draw (absolute).
      - We convert terminal outcome to root-perspective, then to leaf-perspective,
        then flip by -1 per ply during backprop to keep node-local perspective.
    """

    def __init__(self, cfg: Optional[MCTSConfig] = None, heuristic: Optional[HeuristicAgent] = None):
        self.cfg = cfg or MCTSConfig()
        self.heuristic = heuristic or HeuristicAgent()

        # Edge statistics
        self.N: Dict[Edge, int] = {}
        self.W: Dict[Edge, float] = {}
        self.Q: Dict[Edge, float] = {}
        self.P: Dict[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], Dict[int, float]] = {}


        # Tree structure
        self.children: Dict[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], List[int]] = {}
        self.player_at: Dict[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], int] = {}

        # RNG for tie-breaks outside env.rng
        self._py_rng = random.Random(self.cfg.rng_seed)

        self._prof = defaultdict(float)
        self._prof_counts = defaultdict(int)

    # ------------------------- Public API -------------------------

    def select_action(self, env: UTTTEnv) -> int:
        """
        Choose a legal action using MCTS from the current env state.
        """
        if env.terminated:
            raise RuntimeError("MCTSAgent.act called on a terminal position")

        if self.cfg.profile:
            self._t_start()
        t_total0 = time.perf_counter()

        t0 = time.perf_counter()
        root = clone_env(env)
        self._t("clone", time.perf_counter() - t0 if self.cfg.profile else 0.0)

        t0 = time.perf_counter()
        root_key = state_key(root)
        self._t("sel_key", time.perf_counter() - t0 if self.cfg.profile else 0.0)

        root_player = int(root.player)

        # Ensure root is in the tree
        if root_key not in self.children:
            t0 = time.perf_counter()
            self.children[root_key] = self._legals(root)
            self._t("root_legals", time.perf_counter() - t0 if self.cfg.profile else 0.0)
            self.player_at[root_key] = int(root.player)
            self.P[root_key] = self._make_priors(root, root_key, self.children[root_key])


        # Run simulations
        for _ in range(self.cfg.n_simulations):
            self._simulate_once(root, root_key, root_player)
            if self.cfg.profile:
                self._c("sims")

        # Pick argmax by visit count
        legal = self.children[root_key]
        if not legal:
            # Should not happen in non-terminal root, but guard anyway
            raise RuntimeError("No legal actions found at non-terminal root")

        best_a = max(legal, key=lambda a: self.N.get((root_key, a), 0))

        if self.cfg.profile:
            self._t("total", time.perf_counter() - t_total0)
            self._t_report()
        
        return int(best_a)

    # ---------------------- Core MCTS loop ------------------------

    def _simulate_once(self, root: UTTTEnv, root_key, root_player: int) -> None:
        """
        One MCTS simulation: Selection → Expansion → Rollout → Backprop.
        """
        # Work on a fresh clone we can mutate
        t0 = time.perf_counter()
        env = clone_env(root)

        if self.cfg.profile:
            self._t("clone", time.perf_counter() - t0)

        # --- Selection ---
        path: List[Edge] = []
        s_key = root_key

        while True:
            # Expand on first visit
            if s_key not in self.children:
                t0 = time.perf_counter()
                if env.terminated:
                    self.children[s_key] = []
                    self.player_at[s_key] = int(env.player)
                    self.P[s_key] = {}
                    if self.cfg.profile: self._t("expand", time.perf_counter() - t0)
                    break

                self.children[s_key] = list(self._legals(env))
                self.player_at[s_key] = int(env.player)
                self.P[s_key] = self._make_priors(env, s_key, self.children[s_key])
                if self.cfg.profile: self._t("expand", time.perf_counter() - t0)
                break

            legal = self.children[s_key]
            if not legal or env.terminated:
                # terminal (or no moves) according to our tree state
                break

            if self.cfg.profile:
                self._c("select_steps")

            # Pick action with highest UCT
            t0 = time.perf_counter()
            priors = self.P.get(s_key)
            if priors is None:
                # Shouldn't happen, but be safe (fallback uniform)
                priors = {a: 1.0/len(legal) for a in legal}
                self.P[s_key] = priors

            n_parent = max(1, sum(self.N.get((s_key, a), 0) for a in legal))
            c_puct = self.cfg.c_explore  # reuse field; typical 0.8..1.8

            def score(a: int) -> float:
                q = self.Q.get((s_key, a), 0.0)
                n_sa = self.N.get((s_key, a), 0)
                u = c_puct * priors.get(a, 0.0) * (np.sqrt(n_parent) / (1 + n_sa))
                return q + u

            a_sel = max(legal, key=score)
            
            # Optional: virtual visit to stabilize early Q
            edge = (s_key, a_sel)
            if edge not in self.N:
                self.N[edge] = 1
                # bootstrap Q with prior value of the *child* node from current player's POV
                child = clone_env(env); child.step_fast(a_sel)
                boot = self._quick_value_from_pov(child, int(env.player))
                self.W[edge] = boot
                self.Q[edge] = boot

            if self.cfg.profile:
                self._t("sel_uct", time.perf_counter() - t0)
            
            path.append((s_key, a_sel))

            t0 = time.perf_counter()
            env.step_fast(a_sel)
            if self.cfg.profile:
                self._t("sel_step", time.perf_counter() - t0)
            
            t0 = time.perf_counter()
            s_key = state_key(env)
            if self.cfg.profile:
                self._t("sel_key", time.perf_counter() - t0)

        # --- Expansion (already ensured by selection loop) ---
        if (s_key not in self.children) and (not env.terminated):
            t0 = time.perf_counter()
            self.children[s_key] = list(self._legals(env))
            self.player_at[s_key] = int(env.player)
            self.P[s_key] = self._make_priors(env, s_key, self.children[s_key])
            if self.cfg.profile:
                self._t("expand", time.perf_counter() - t0)

        # --- Value at leaf, always from leaf player's POV ---
        if env.terminated:
            # At terminal, "player to move" is conceptually the loser; but we want POV of node s_key.
            # We stored player_at[s_key] when we created/visited it. Fall back to env.player if needed.
            leaf_player = self.player_at.get(s_key, int(env.player))
            v_leaf = self._terminal_value_from_pov(env, leaf_player)
        else:
            # Non-terminal: use rollout/cutoff but return value from leaf player's POV
            leaf_player = int(env.player)
            v_leaf = self._rollout_value_from_pov(env, leaf_player)


        # --- Backprop ---
        t0 = time.perf_counter()
        self._backprop(path, v_leaf)
        if self.cfg.profile:
            self._t("backprop", time.perf_counter() - t0)

    # ---------------------- Policy / Rollout ----------------------

    def _rollout_value_from_pov(self, env: UTTTEnv, pov_player: int) -> float:
        depth = 0
        limit = getattr(self.cfg, "rollout_depth_limit", None)
        mode = getattr(self.cfg, "rollout", "light")

        while not env.terminated:
            if limit is not None and depth >= limit:
                return self._quick_value_from_pov(env, pov_player)  # <- updated
            legal = self._legals(env)
            if not legal:
                break

            if self.cfg.profile:
                self._c("rollout_steps")

            if mode == "random":
                a = int(self._sample_random(env, legal))
            elif mode == "eps_heuristic":
                if self._py_rng.random() < self.cfg.epsilon:
                    a = int(self._sample_random(env, legal))
                else:
                    a = int(self._heuristic_pick(env, legal))
            elif mode == "heuristic":
                a = int(self._heuristic_pick(env, legal))
            else:  # "light"
                a = int(self._rollout_pick(env, legal))

            env.step_fast(a)
            depth += 1

        # terminal
        return self._terminal_value_from_pov(env, pov_player)



    def _quick_value_from_pov(self, env: UTTTEnv, pov_player: int) -> float:
        """
        Cheap heuristic for cutoff. Start with macro diff, then add small tactical terms
        in the next forced micro-board if any.
        Range-clamp to [-1, 1].
        """
        macro = env.macro_wins
        opp = -pov_player
        pov_w = int((macro == pov_player).sum())
        opp_w = int((macro == opp).sum())
        base = (pov_w - opp_w) / 3.0  # [-1,1] if spread is within 3

        # Tactical bonus: if a forced micro is open, check immediate win/block there.
        bonus = 0.0
        if env.last_move is not None:
            lr, lc = env.last_move
            ti, tj, r0, c0 = self._target_micro_indices(lr, lc)
            if env._micro_is_open(ti, tj):
                sub = env.board[r0:r0+3, c0:c0+3]
                if self._immediate_local_win(sub, pov_player) is not None:
                    bonus += 0.25
                if self._immediate_local_win(sub, opp) is not None:
                    bonus -= 0.25

        v = max(-1.0, min(1.0, base + bonus))
        return v


    def _heuristic_pick(self, env: UTTTEnv, legal: List[int]) -> int:
        # Prefer best_of to avoid recomputing legal actions
        if hasattr(self.heuristic, "best_of"):
            return int(self.heuristic.best_of(env, legal))
        return int(self.heuristic.select_action(env))

    def _sample_random(self, env: UTTTEnv, legal: List[int]) -> int:
        if hasattr(env, "rng") and env.rng is not None:
            return int(env.rng.choice(np.array(legal)))
        return int(self._py_rng.choice(legal))


    # -------------------------- Values ---------------------------

    @staticmethod
    def _terminal_value_from_root(env: UTTTEnv, root_player: int) -> float:
        """
        Convert env's terminal absolute winner (+1 for X, -1 for O, 0 draw)
        into a value from the root player's perspective.
        """
        # winner is stored in info during step, but we can recompute quickly:
        # env.step() set reward = winner when terminated. Safer to derive from board state:
        # However, recomputing here is unnecessary; reward isn't stored. Compute from macro_wins.
        # Reuse the same logic as env._macro_winner without accessing private method:
        macro = env.macro_wins
        winner = _macro_winner_public(macro)  # +1, -1, 0
        if winner == 0:
            return 0.0
        return 1.0 if winner == root_player else -1.0

    def _backprop(self, path: List[Edge], v_leaf: float) -> None:
        """
        Backpropagate the value up the path, flipping perspective each ply so that
        Q(s,a) is always from the point of view of the player to move at state s.
        """
        value = v_leaf
        for s_key, a in reversed(path):
            n = self.N.get((s_key, a), 0) + 1
            w = self.W.get((s_key, a), 0.0) + value
            self.N[(s_key, a)] = n
            self.W[(s_key, a)] = w
            self.Q[(s_key, a)] = w / n
            value = -value  # flip perspective one ply up

    def _legals(self, env: UTTTEnv) -> list[int]:
        if self.cfg.profile:
            self._c("legals_calls")
        if hasattr(env, "legal_actions_fast"):
            return env.legal_actions_fast()
        return env.legal_actions()

    def _target_micro_indices(self, r: int, c: int) -> tuple[int,int,int,int]:
        ti, tj = r % 3, c % 3
        return ti, tj, ti * 3, tj * 3

    def _immediate_local_win(self, sub: np.ndarray, player: int) -> Optional[tuple[int,int]]:
        # Try empties in the 3x3 and return first (rr,cc) that completes a line
        for rr in range(3):
            # micro-optim: local pointers to avoid repeated attribute lookups
            row0 = sub[rr, 0] + sub[rr, 1] + sub[rr, 2]
            for cc in range(3):
                if sub[rr, cc] != 0:
                    continue
                # place
                sub[rr, cc] = player
                win = (
                    (sub[rr, 0] + sub[rr, 1] + sub[rr, 2] == 3*player) or
                    (sub[0, cc] + sub[1, cc] + sub[2, cc] == 3*player) or
                    (rr == cc and (sub[0,0] + sub[1,1] + sub[2,2] == 3*player)) or
                    (rr + cc == 2 and (sub[0,2] + sub[1,1] + sub[2,0] == 3*player))
                )
                # undo
                sub[rr, cc] = 0
                if win:
                    return (rr, cc)
        return None

    def _rollout_pick(self, env: UTTTEnv, legal: list[int]) -> int:
        """
        Very fast rollout policy:
        1) take immediate local win in target micro
        2) else block opponent immediate local win
        3) else center > corners > edges in target micro
        4) else random among legals
        """
        if len(legal) == 1:
            return int(legal[0])

        player = int(env.player)
        opp = -player

        if env.last_move is not None:
            lr, lc = env.last_move
            ti, tj, r0, c0 = self._target_micro_indices(lr, lc)
            if env._micro_is_open(ti, tj):
                sub = env.board[r0:r0+3, c0:c0+3]
                # 1) win
                w = self._immediate_local_win(sub, player)
                if w is not None:
                    rr, cc = w
                    return int((r0 + rr) * 9 + (c0 + cc))
                # 2) block
                b = self._immediate_local_win(sub, opp)
                if b is not None:
                    rr, cc = b
                    return int((r0 + rr) * 9 + (c0 + cc))
                # 3) positional preference within target micro
                for rr, cc in [(1,1),(0,0),(0,2),(2,0),(2,2),(0,1),(1,0),(1,2),(2,1)]:
                    if sub[rr, cc] == 0:
                        return int((r0 + rr) * 9 + (c0 + cc))

        # 4) fallback random
        return self._sample_random(env, legal)

    # --- tiny helpers ---
    def _t_start(self):
        self._prof.clear()
        self._prof_counts.clear()

    def _t(self, key: str, dt: float):
        self._prof[key] += dt

    def _c(self, key: str, inc: int = 1):
        self._prof_counts[key] += inc

    def _t_report(self, label: str = "MCTS move profile"):
        # print one compact line
        p = self._prof; c = self._prof_counts
        total = p["total"]
        msg = (
            f"[{label}] sims={c['sims']}, select_steps={c['select_steps']}, rollout_steps={c['rollout_steps']}, "
            f"total={total:.3f}s | clone={p['clone']:.3f}  root_legals={p['root_legals']:.3f} "
            f"selection={{legals:{p['sel_legals']:.3f}, uct:{p['sel_uct']:.3f}, step:{p['sel_step']:.3f}, key:{p['sel_key']:.3f}}} "
            f"expansion={p['expand']:.3f} rollout={{legals:{p['roll_legals']:.3f}, policy:{p['roll_policy']:.3f}, step:{p['roll_step']:.3f}}} "
            f"backprop={p['backprop']:.3f}"
        )
        print(msg)

    def _terminal_value_from_pov(self, env: UTTTEnv, pov_player: int) -> float:
        """
        +1 if pov_player has a forced win in env (terminal) else -1 if loss, 0 draw.
        env is terminal.
        """
        macro = env.macro_wins
        winner = _macro_winner_public(macro)  # +1, -1, 0
        if winner == 0:
            return 0.0
        return 1.0 if winner == pov_player else -1.0

    def _make_priors(self, env: UTTTEnv, s_key, legal: list[int]) -> dict[int, float]:
        # If terminal or no moves, return empty (or uniform)
        if env.terminated or not legal:
            return {}  # or {a: 1.0/len(legal) for a in legal} if you prefer uniform

        player = int(env.player)
        vals = []
        for a in legal:
            child = clone_env(env)
            # EXTRA GUARD: never step if terminal
            if child.terminated:
                continue
            child.step_fast(a)
            v = self._quick_value_from_pov(child, player)
            vals.append((a, v))

        if not vals:
            # Fallback uniform if something odd happened
            return {a: 1.0/len(legal) for a in legal}

        tau = 0.75
        mx = max(v for _, v in vals)
        exped = [(a, np.exp((v - mx)/tau)) for a, v in vals]
        Z = sum(e for _, e in exped) or 1.0
        return {a: float(e / Z) for a, e in exped}


# ------------------ Tiny helper (public macro winner) ------------------

def _macro_winner_public(macro: np.ndarray) -> int:
    """
    Public copy of your micro-winner logic applied to the 3x3 macro board.
    Returns +1, -1, or 0.
    """
    lines = []
    lines.extend([macro[i, :] for i in range(3)])
    lines.extend([macro[:, j] for j in range(3)])
    lines.append(np.diag(macro))
    lines.append(np.diag(np.fliplr(macro)))
    for line in lines:
        s = int(np.sum(line == 1))
        if s == 3:
            return 1
        s = int(np.sum(line == -1))
        if s == 3:
            return -1
    return 0

