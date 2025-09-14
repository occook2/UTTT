# uttt/eval/elo.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, List
import json
import math
from datetime import datetime

def expected_score(r_a: float, r_b: float, scale: float = 400.0) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / scale))

def update_pair(r_a: float, r_b: float, score_a: float, k: float = 32.0, scale: float = 400.0) -> Tuple[float, float]:
    ea = expected_score(r_a, r_b, scale)
    eb = 1.0 - ea
    score_b = 1.0 - score_a
    new_a = r_a + k * (score_a - ea)
    new_b = r_b + k * (score_b - eb)
    return new_a, new_b

@dataclass
class PlayerStats:
    rating: float = 1500.0
    games: int = 0
    last_played: str = ""  # ISO time

@dataclass
class EloTable:
    ratings: Dict[str, PlayerStats] = field(default_factory=dict)
    default_rating: float = 1500.0
    k_base: float = 24.0        # gentler default than 32
    scale: float = 400.0

    def _get(self, name: str) -> PlayerStats:
        if name not in self.ratings:
            self.ratings[name] = PlayerStats(rating=self.default_rating, games=0, last_played="")
        return self.ratings[name]

    def _k(self, name: str) -> float:
        # Dynamic K: larger when a player is new; taper over time
        n = self._get(name).games
        if n < 30: return self.k_base * 1.5
        if n < 100: return self.k_base
        return max(8.0, self.k_base * 0.6)

    def set(self, name: str, rating: float) -> None:
        ps = self._get(name)
        ps.rating = rating

    def record(self, name_a: str, name_b: str, score_a: float, played_at: str | None = None) -> None:
        """
        Record a single game. score_a: 1.0 (A win), 0.5 (draw), 0.0 (A loss)
        """
        pa = self._get(name_a)
        pb = self._get(name_b)

        kA = self._k(name_a)
        kB = self._k(name_b)

        new_a, new_b = update_pair(pa.rating, pb.rating, score_a, k=kA, scale=self.scale)
        pa.rating = new_a
        pb.rating = new_b
        pa.games += 1
        pb.games += 1
        ts = played_at or datetime.utcnow().isoformat()
        pa.last_played = ts
        pb.last_played = ts

    def table(self) -> List[dict]:
        return sorted(
            [{"id": name, "rating": ps.rating, "games": ps.games, "last_played": ps.last_played}
             for name, ps in self.ratings.items()],
            key=lambda d: d["rating"], reverse=True
        )

    # --- persistence ---
    def to_json(self) -> str:
        payload = {
            "default_rating": self.default_rating,
            "k_base": self.k_base,
            "scale": self.scale,
            "players": {
                name: {"rating": ps.rating, "games": ps.games, "last_played": ps.last_played}
                for name, ps in self.ratings.items()
            },
        }
        return json.dumps(payload, indent=2)

    @staticmethod
    def from_json(s: str) -> "EloTable":
        payload = json.loads(s)
        et = EloTable(
            default_rating=payload.get("default_rating", 1500.0),
            k_base=payload.get("k_base", 24.0),
            scale=payload.get("scale", 400.0),
        )
        for name, rec in payload.get("players", {}).items():
            et.ratings[name] = PlayerStats(
                rating=float(rec["rating"]),
                games=int(rec["games"]),
                last_played=str(rec.get("last_played", "")),
            )
        return et

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "EloTable":
        with open(path, "r", encoding="utf-8") as f:
            return EloTable.from_json(f.read())
