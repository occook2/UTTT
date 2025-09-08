# eval/elo.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


def expected_score(r_a: float, r_b: float) -> float:
    """Expected score for A vs B under Elo."""
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / 400.0))


def update_pair(
    r_a: float, r_b: float, score_a: float, k: float = 32.0
) -> Tuple[float, float]:
    """
    Update Elo for a single result.
    score_a: 1.0 (A win), 0.5 (draw), 0.0 (A loss)
    Returns (new_r_a, new_r_b).
    """
    ea = expected_score(r_a, r_b)
    eb = 1.0 - ea
    score_b = 1.0 - score_a
    new_a = r_a + k * (score_a - ea)
    new_b = r_b + k * (score_b - eb)
    return new_a, new_b


@dataclass
class EloTable:
    ratings: Dict[str, float] = field(default_factory=dict)
    default_rating: float = 1500.0
    k: float = 32.0

    def _get(self, name: str) -> float:
        return self.ratings.get(name, self.default_rating)

    def set(self, name: str, rating: float) -> None:
        self.ratings[name] = rating

    def record(self, name_a: str, name_b: str, score_a: float) -> None:
        """
        Record a single match result and update both players.
        score_a: 1.0 (A win), 0.5 (draw), 0.0 (A loss)
        """
        ra = self._get(name_a)
        rb = self._get(name_b)
        na, nb = update_pair(ra, rb, score_a, k=self.k)
        self.ratings[name_a] = na
        self.ratings[name_b] = nb
