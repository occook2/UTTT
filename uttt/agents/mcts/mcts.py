# mcts.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import random
import numpy as np

from uttt.env.state import UTTTEnv
from uttt.agents.heuristic import HeuristicAgent
from .uct import uct_score
from .transpo import state_key, clone_env


Edge = Tuple[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], int]  # (state_key, action)


@dataclass
class MCTSConfig:
    n_simulations: int = 20
    c_explore: float = 1.4
    rollout: str = "heuristic"  # "heuristic" | "random" | "eps_heuristic"
    epsilon: float = 0.1        # only used when rollout == "eps_heuristic"
    use_tt: bool = True         # enable transposition table (merge states)
    rng_seed: Optional[int] = None


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

        # Tree structure
        self.children: Dict[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], List[int]] = {}
        self.player_at: Dict[Tuple[bytes, bytes, Optional[Tuple[int,int]], int], int] = {}

        # RNG for tie-breaks outside env.rng
        self._py_rng = random.Random(self.cfg.rng_seed)

    # ------------------------- Public API -------------------------

    def select_action(self, env: UTTTEnv) -> int:
        """
        Choose a legal action using MCTS from the current env state.
        """
        if env.terminated:
            raise RuntimeError("MCTSAgent.act called on a terminal position")

        root = clone_env(env)
        root_key = state_key(root)
        root_player = int(root.player)

        # Ensure root is in the tree
        if root_key not in self.children:
            self.children[root_key] = list(root.legal_actions())
            self.player_at[root_key] = int(root.player)

        # Run simulations
        for _ in range(self.cfg.n_simulations):
            self._simulate_once(root, root_key, root_player)

        # Pick argmax by visit count
        legal = self.children[root_key]
        if not legal:
            # Should not happen in non-terminal root, but guard anyway
            raise RuntimeError("No legal actions found at non-terminal root")

        best_a = max(legal, key=lambda a: self.N.get((root_key, a), 0))
        return int(best_a)

    # ---------------------- Core MCTS loop ------------------------

    def _simulate_once(self, root: UTTTEnv, root_key, root_player: int) -> None:
        """
        One MCTS simulation: Selection → Expansion → Rollout → Backprop.
        """
        # Work on a fresh clone we can mutate
        env = clone_env(root)

        # --- Selection ---
        path: List[Edge] = []
        s_key = root_key

        while True:
            # Expand on first visit
            if s_key not in self.children:
                self.children[s_key] = list(env.legal_actions())
                self.player_at[s_key] = int(env.player)
                break  # stop selection at newly expanded node

            legal = self.children[s_key]
            if not legal:
                # terminal (or no moves) according to our tree state
                break

            if env.terminated:
                break

            # Pick action with highest UCT
            n_parent = sum(self.N.get((s_key, a), 0) for a in legal)
            def score(a: int) -> float:
                q = self.Q.get((s_key, a), 0.0)
                n_sa = self.N.get((s_key, a), 0)
                return uct_score(q, n_parent, n_sa, self.cfg.c_explore)

            a_sel = max(legal, key=score)
            path.append((s_key, a_sel))
            env.step(a_sel)
            s_key = state_key(env)

        # --- Expansion (already ensured by selection loop) ---
        if (s_key not in self.children) and (not env.terminated):
            self.children[s_key] = list(env.legal_actions())
            self.player_at[s_key] = int(env.player)

        # --- Rollout ---
        if env.terminated:
            # Convert absolute winner to root-perspective value
            v_root = self._terminal_value_from_root(env, root_player)
        else:
            v_root = self._rollout_value_from_root(env, root_player)

        # Convert to leaf-node perspective before backprop
        leaf_player = int(env.player) if not env.terminated else self.player_at.get(s_key, int(env.player))
        v_leaf = v_root if leaf_player == root_player else -v_root

        # --- Backprop ---
        self._backprop(path, v_leaf)

    # ---------------------- Policy / Rollout ----------------------

    def _rollout_value_from_root(self, env: UTTTEnv, root_player: int) -> float:
        """
        Finish the game quickly from 'env' using the configured rollout policy.
        Return the terminal value from the root player's perspective in {-1,0,+1}.
        """
        # Simple loop until termination
        while not env.terminated:
            legal = env.legal_actions()
            if not legal:
                break

            if self.cfg.rollout == "random":
                a = int(self._sample_random(env, legal))
            elif self.cfg.rollout == "eps_heuristic":
                if self._py_rng.random() < self.cfg.epsilon:
                    a = int(self._sample_random(env, legal))
                else:
                    a = int(self._heuristic_pick(env, legal))
            else:  # "heuristic"
                a = int(self._heuristic_pick(env, legal))

            env.step(a)

        return self._terminal_value_from_root(env, root_player)

    def _heuristic_pick(self, env: UTTTEnv, legal: List[int]) -> int:
        # Your HeuristicAgent uses env.rng for deterministic tie-breaks if present.
        # We supply the legal list by leaving env.legal_actions() to it; it recomputes internally,
        # but that’s fine (few µs). If you like, you can add a `best_of(env, legal)` to your agent later.
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
