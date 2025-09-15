# uct.py
from __future__ import annotations
import math

def uct_score(q_sa: float, n_parent: int, n_sa: int, c_explore: float) -> float:
    """
    Classic UCT:
        UCT = Q(s,a) + c * sqrt( ln(N(s,*)) / (1 + N(s,a)) )
    All inputs are non-negative except q_sa which is in [-1, +1].
    """
    if n_sa <= 0:
        # Unvisited actions get infinite bonus in theory; use a large proxy.
        return float("inf")
    return q_sa + c_explore * math.sqrt(math.log(max(1, n_parent)) / (1.0 + n_sa))
