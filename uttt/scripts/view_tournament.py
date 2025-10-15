# uttt/scripts/view_tournament.py
from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Dict, Any, List, Optional

import numpy as np

from uttt.env.state import UTTTEnv, action_to_rc


# --- UI constants ---
CELL = 40       # px per small cell
PAD = 20        # canvas padding
LINE_W = 1
BOLD_W = 3
INSET = 8       # padding inside cell for drawing X/O
HILITE_W = 2
AUTO_MS = 300   # autoplay step interval (ms)


class ReplayApp:
    def __init__(self, root: tk.Tk, payload: Dict[str, Any]):
        self.root = root
        self.payload = payload
        self.games: List[Dict[str, Any]] = payload["games"]
        self.meta = payload["meta"]
        self.summary = payload["summary"]

        self.env = UTTTEnv(seed=None)
        self.game_idx = 0
        self.move_idx = 0
        self.moves: List[int] = []
        self.autoplay_job: Optional[str] = None

        # ---- UI layout ----
        self.root.title("UTTT Tournament Viewer")
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        # header
        top = ttk.Frame(outer)
        top.pack(side="top", fill="x")
        ttk.Label(
            top,
            text=f"{self.meta['agent_A']} (A) vs {self.meta['agent_B']} (B)  |  "
                 f"Games: {self.summary['n_games']}  A:{self.summary['wins_A']}  "
                 f"B:{self.summary['wins_B']}  D:{self.summary['draws']}"
        ).pack(side="left")

        # game selector + info
        mid = ttk.Frame(outer)
        mid.pack(side="top", fill="x", pady=(6, 6))
        ttk.Label(mid, text="Game:").pack(side="left")
        self.game_combo = ttk.Combobox(
            mid, state="readonly", width=12,
            values=[f"{i:03d}" for i in range(len(self.games))]
        )
        self.game_combo.current(0)
        self.game_combo.bind("<<ComboboxSelected>>", self._on_game_change)
        self.game_combo.pack(side="left", padx=4)

        self.info_label = ttk.Label(mid, text="", width=70)
        self.info_label.pack(side="left", padx=8)

        # controls
        btns = ttk.Frame(outer)
        btns.pack(side="top", fill="x", pady=(4, 2))
        ttk.Button(btns, text="⏮ Reset", command=self.reset_game).pack(side="left")
        ttk.Button(btns, text="◀ Prev", command=self.prev_move).pack(side="left", padx=4)
        ttk.Button(btns, text="Next ▶", command=self.next_move).pack(side="left", padx=4)
        self.play_btn = ttk.Button(btns, text="▶ Play", command=self.toggle_autoplay)
        self.play_btn.pack(side="left", padx=8)
        ttk.Button(btns, text="⏭ End", command=self.to_end).pack(side="left", padx=4)

        # scrubber
        self.move_scale = ttk.Scale(
            outer, from_=0, to=1, orient="horizontal", command=self._on_scale
        )
        self.move_scale.pack(side="top", fill="x", padx=4, pady=(2, 8))

        # canvas
        size = PAD * 2 + CELL * 9
        self.canvas = tk.Canvas(outer, width=size, height=size, bg="white")
        self.canvas.pack(side="top")

        # keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.prev_move())
        self.root.bind("<Right>", lambda e: self.next_move())
        self.root.bind("<space>", lambda e: self.toggle_autoplay())

        self._load_game(0)

    # ---------- game management ----------

    def _on_game_change(self, event=None):
        idx = int(self.game_combo.get())
        self._load_game(idx)

    def _on_scale(self, value: str):
        target = int(float(value))
        self.set_move_idx(target)

    def _load_game(self, idx: int):
        self.stop_autoplay()
        self.game_idx = idx
        g = self.games[idx]
        self.moves = list(g["moves"])
        self.winner = int(g["winner"])
        self.a_is_x = bool(g["a_is_x"])
        self.seed = g["seed"]
        self.move_idx = 0
        self._reset_env()
        self._update_info()
        self.move_scale.configure(from_=0, to=len(self.moves))
        self.move_scale.set(0)
        self._redraw()

    def reset_game(self):
        self.stop_autoplay()
        self.set_move_idx(0)

    def to_end(self):
        self.stop_autoplay()
        self.set_move_idx(len(self.moves))
        self.move_scale.set(self.move_idx)

    def set_move_idx(self, k: int):
        k = max(0, min(k, len(self.moves)))
        if k == self.move_idx:
            return
        self.move_idx = k
        self._reset_env()
        for i in range(k):
            self.env.step(self.moves[i])
        self._update_info()
        self._redraw()

    def next_move(self):
        if self.move_idx < len(self.moves):
            self.set_move_idx(self.move_idx + 1)
            self.move_scale.set(self.move_idx)
        else:
            self.stop_autoplay()

    def prev_move(self):
        if self.move_idx > 0:
            self.set_move_idx(self.move_idx - 1)
            self.move_scale.set(self.move_idx)

    def toggle_autoplay(self):
        if self.autoplay_job is None:
            self.play_btn.config(text="⏸ Pause")
            self._autoplay_step()
        else:
            self.stop_autoplay()

    def stop_autoplay(self):
        if self.autoplay_job is not None:
            try:
                self.root.after_cancel(self.autoplay_job)
            except Exception:
                pass
            self.autoplay_job = None
            self.play_btn.config(text="▶ Play")

    def _autoplay_step(self):
        if self.move_idx >= len(self.moves):
            self.stop_autoplay()
            return
        self.next_move()
        self.autoplay_job = self.root.after(AUTO_MS, self._autoplay_step)

    def _reset_env(self):
        self.env.reset()

    # ---------- drawing ----------

    def _redraw(self):
        self.canvas.delete("all")
        x0 = PAD
        y0 = PAD

        # 9x9 grid with bold 3x3 boundaries
        for i in range(10):
            w = BOLD_W if i % 3 == 0 else LINE_W
            self.canvas.create_line(x0 + i * CELL, y0, x0 + i * CELL, y0 + 9 * CELL, width=w)
            self.canvas.create_line(x0, y0 + i * CELL, x0 + 9 * CELL, y0 + i * CELL, width=w)

        # highlight macro-won boards
        mw = self.env.macro_wins
        for mr in range(3):
            for mc in range(3):
                owner = int(mw[mr, mc])
                if owner != 0:
                    rx0 = x0 + mc * 3 * CELL
                    ry0 = y0 + mr * 3 * CELL
                    rx1 = rx0 + 3 * CELL
                    ry1 = ry0 + 3 * CELL
                    if owner == 1:
                        color = "#e6f7ff"  # light blue for X
                    elif owner == -1:
                        color = "#ffe6e6"  # light red for O
                    else:
                        color = "#f0f0f0"  # light grey for draw
                    self.canvas.create_rectangle(rx0, ry0, rx1, ry1, fill=color, outline="")
                    self.canvas.create_rectangle(rx0, ry0, rx1, ry1, outline="black", width=BOLD_W)

        # draw stones
        b = self.env.board
        last = None
        if self.move_idx > 0:
            last = action_to_rc(self.moves[self.move_idx - 1])

        for r in range(9):
            for c in range(9):
                v = int(b[r, c])
                cx = x0 + c * CELL
                cy = y0 + r * CELL
                if v == 1:      # X
                    self.canvas.create_line(
                        cx + INSET, cy + INSET, cx + CELL - INSET, cy + CELL - INSET, width=2
                    )
                    self.canvas.create_line(
                        cx + CELL - INSET, cy + INSET, cx + INSET, cy + CELL - INSET, width=2
                    )
                elif v == -1:   # O
                    self.canvas.create_oval(
                        cx + INSET, cy + INSET, cx + CELL - INSET, cy + CELL - INSET, width=2
                    )

        # highlight last move
        if last is not None:
            lr, lc = last
            cx = x0 + lc * CELL
            cy = y0 + lr * CELL
            self.canvas.create_rectangle(cx + 2, cy + 2, cx + CELL - 2, cy + CELL - 2, outline="gray", width=HILITE_W)

        # hint: legal actions for NEXT move (small dots)
        mask = self.env.legal_actions_mask()
        for r, c in zip(*np.where(mask == 1)):
            cx = x0 + c * CELL
            cy = y0 + r * CELL
            d = 6
            self.canvas.create_oval(cx + CELL // 2 - d//2, cy + CELL // 2 - d//2,
                                    cx + CELL // 2 + d//2, cy + CELL // 2 + d//2,
                                    fill="#bbb", outline="")

    def _update_info(self):
        g = self.games[self.game_idx]
        w = g["winner"]
        outcome = "draw" if w == 0 else ("X wins" if w == 1 else "O wins")
        side = "A=X" if self.a_is_x else "A=O"
        self.info_label.config(
            text=f"Game {self.game_idx} ({side})  |  seed: {self.seed}  |  final: {outcome}  |  move {self.move_idx}/{len(self.moves)}"
        )


def main():
    # allow passing file on CLI, otherwise open a dialog
    path: Optional[str] = sys.argv[1] if len(sys.argv) >= 2 else None
    if not path:
        _root = tk.Tk()
        _root.withdraw()
        path = filedialog.askopenfilename(
            title="Select tournament JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.join(os.getcwd(), "tournaments") if os.path.isdir("tournaments") else os.getcwd(),
        )
        _root.destroy()
    if not path:
        print("No file selected.")
        return

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    root = tk.Tk()
    app = ReplayApp(root, payload)
    root.mainloop()


if __name__ == "__main__":
    main()
