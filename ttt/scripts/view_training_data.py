"""
Simple viewer for AlphaZero training data.
Shows games reconstructed from training examples.
"""
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import Dict, Any, List, Optional
import numpy as np
import math


def calculate_policy_entropy(policy: List[float]) -> float:
    """
    Calculate the entropy of a policy distribution.
    
    H(p) = -Σ p_i * log(p_i)
    
    Args:
        policy: List of probabilities (should sum to ~1.0)
        
    Returns:
        Entropy value (higher = more uncertain/exploratory)
    """
    entropy = 0.0
    for p in policy:
        if p > 1e-10:  # Avoid log(0)
            entropy -= p * math.log2(p)  # Using log2 for bits
    return entropy

# Constants for TTT visualization - larger cells since only 3x3 grid
CELL = 80
PAD = 30
LINE_W = 2
BOLD_W = 4
INSET = 15


class TrainingDataViewer:
    def __init__(self, root: tk.Tk, data: Dict[str, Any]):
        self.root = root
        self.data = data
        self.games: List[Dict[str, Any]] = data["games"]
        self.meta = data["meta"]
        
        if not self.games:
            messagebox.showerror("Error", "No games found in training data!")
            return
        
        self.game_idx = 0
        self.move_idx = 0
        
        # UI setup
        self.root.title("TTT Training Data Viewer")
        self.setup_ui()
        self.load_game_and_move(0, 0)

    def setup_ui(self):
        """Setup the user interface."""
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)

        # Header
        header = ttk.Label(
            outer,
            text=f"Training Data - Epoch {self.meta['epoch']} | "
                 f"Games: {len(self.games)} | "
                 f"Examples: {self.meta['total_examples']} | "
                 f"MCTS Sims: {self.meta['mcts_simulations']}"
        )
        header.pack(pady=(0, 10))

        # Game selection
        game_frame = ttk.Frame(outer)
        game_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Label(game_frame, text="Game:").pack(side="left")
        self.game_var = tk.StringVar()
        self.game_combo = ttk.Combobox(
            game_frame, textvariable=self.game_var, state="readonly", width=15
        )
        self.game_combo['values'] = [f"Game {i:03d}" for i in range(len(self.games))]
        self.game_combo.current(0)
        self.game_combo.bind('<<ComboboxSelected>>', self.on_game_change)
        self.game_combo.pack(side="left", padx=(5, 10))
        
        self.game_info = ttk.Label(game_frame, text="")
        self.game_info.pack(side="left")

        # Move selection
        move_frame = ttk.Frame(outer)
        move_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(move_frame, text="Move:").pack(side="left")
        self.move_var = tk.StringVar()
        self.move_combo = ttk.Combobox(
            move_frame, textvariable=self.move_var, state="readonly", width=15
        )
        self.move_combo.bind('<<ComboboxSelected>>', self.on_move_change)
        self.move_combo.pack(side="left", padx=(5, 10))
        
        self.move_info = ttk.Label(move_frame, text="")
        self.move_info.pack(side="left")

        # Navigation buttons
        nav_frame = ttk.Frame(outer)
        nav_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Button(nav_frame, text="◀ Prev Game", command=self.prev_game).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="Next Game ▶", command=self.next_game).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="◀ Prev Move", command=self.prev_move).pack(side="left", padx=(10, 2))
        ttk.Button(nav_frame, text="Next Move ▶", command=self.next_move).pack(side="left", padx=2)

        # Canvas container for board and policy
        canvas_container = ttk.Frame(outer)
        canvas_container.pack(fill="both", expand=True)
        
        # Board display
        board_frame = ttk.LabelFrame(canvas_container, text="Board State", padding=4)
        board_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        size = PAD * 2 + CELL * 3  # 3x3 grid for TTT
        self.board_canvas = tk.Canvas(board_frame, width=size, height=size, bg="white")
        self.board_canvas.pack()

        # Policy display
        policy_frame = ttk.LabelFrame(canvas_container, text="Policy Distribution", padding=4)
        policy_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        self.policy_canvas = tk.Canvas(policy_frame, width=size, height=size, bg="white")
        self.policy_canvas.pack()

        # Keyboard shortcuts info
        shortcuts_frame = ttk.Frame(outer)
        shortcuts_frame.pack(side="bottom", fill="x", pady=(10, 0))
        ttk.Label(shortcuts_frame, 
                 text="Shortcuts: ← → = moves, ↑ ↓ = games, Home/End = first/last game").pack()

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda e: self.prev_move())
        self.root.bind("<Right>", lambda e: self.next_move())
        self.root.bind("<Up>", lambda e: self.prev_game())
        self.root.bind("<Down>", lambda e: self.next_game())
        self.root.bind("<Home>", lambda e: self.load_game_and_move(0, 0))
        self.root.bind("<End>", lambda e: self.load_game_and_move(len(self.games) - 1, 0))
        self.root.focus_set()

    def load_game_and_move(self, game_idx: int, move_idx: int):
        """Load a specific game and move."""
        # Validate and set indices
        self.game_idx = max(0, min(game_idx, len(self.games) - 1))
        current_game = self.games[self.game_idx]
        self.move_idx = max(0, min(move_idx, len(current_game['moves']) - 1))
        
        # Update game combo
        self.game_combo.current(self.game_idx)
        
        # Update move combo
        moves = current_game['moves']
        move_values = [f"Move {i+1:02d}" for i in range(len(moves))]
        self.move_combo['values'] = move_values
        if move_values:
            self.move_combo.current(self.move_idx)
        
        # Update info labels
        self.update_info()
        
        # Redraw board and policy
        self.draw_board()
        self.draw_policy()

    def update_info(self):
        """Update information labels."""
        game = self.games[self.game_idx]
        move = game['moves'][self.move_idx]
        
        # Game info
        winner_text = "Draw" if game['winner'] == 0 else f"Player {game['winner']}"
        self.game_info.config(
            text=f"Length: {game['game_length']} moves | Winner: {winner_text} | Final Value: {game['final_value']:.3f}"
        )
        
        # Move info - show game outcome, agent evaluation, and policy entropy
        game_outcome = move.get('value', 0.0)
        agent_eval = move.get('agent_value', 0.0)
        policy = move.get('policy', [])
        policy_entropy = calculate_policy_entropy(policy) if policy else 0.0
        
        self.move_info.config(
            text=f"Player {move['player']} | Game Outcome: {game_outcome:.3f} | "
                 f"Agent Eval: {agent_eval:.3f} | Policy Max: {move['policy_max']:.3f} | "
                 f"Policy Entropy: {policy_entropy:.3f} bits"
        )

    def draw_board(self):
        """Draw the current board state."""
        self.board_canvas.delete("all")
        
        game = self.games[self.game_idx]
        move = game['moves'][self.move_idx]
        state = np.array(move['state'])  # Shape: (5, 3, 3) for TTT
        
        x0, y0 = PAD, PAD
        
        # Draw grid - 3x3 for TTT
        for i in range(4):  # 0, 1, 2, 3 for 3x3 grid
            # Vertical lines
            self.board_canvas.create_line(
                x0 + i * CELL, y0, x0 + i * CELL, y0 + 3 * CELL, width=LINE_W
            )
            # Horizontal lines
            self.board_canvas.create_line(
                x0, y0 + i * CELL, x0 + 3 * CELL, y0 + i * CELL, width=LINE_W
            )
        
        # Draw pieces
        x_pieces = state[1]  # Player 1 (X) - plane 1
        o_pieces = state[2]  # Player -1 (O) - plane 2
        
        for r in range(3):
            for c in range(3):
                x = x0 + c * CELL
                y = y0 + r * CELL
                
                if x_pieces[r][c] == 1:
                    # Draw X - larger for TTT
                    self.board_canvas.create_line(
                        x + INSET, y + INSET, x + CELL - INSET, y + CELL - INSET, width=4, fill='red'
                    )
                    self.board_canvas.create_line(
                        x + CELL - INSET, y + INSET, x + INSET, y + CELL - INSET, width=4, fill='red'
                    )
                elif o_pieces[r][c] == 1:
                    # Draw O - larger for TTT
                    self.board_canvas.create_oval(
                        x + INSET, y + INSET, x + CELL - INSET, y + CELL - INSET, 
                        width=4, outline='blue', fill=''
                    )

    def draw_policy(self):
        """Draw the policy distribution as a heatmap."""
        self.policy_canvas.delete("all")
        
        game = self.games[self.game_idx]
        move = game['moves'][self.move_idx]
        policy = np.array(move['policy'])  # Shape: (9,) for TTT
        
        x0, y0 = PAD, PAD
        
        # Draw grid - 3x3 for TTT
        for i in range(4):  # 0, 1, 2, 3 for 3x3 grid
            # Vertical lines
            self.policy_canvas.create_line(
                x0 + i * CELL, y0, x0 + i * CELL, y0 + 3 * CELL, width=LINE_W, fill='gray'
            )
            # Horizontal lines
            self.policy_canvas.create_line(
                x0, y0 + i * CELL, x0 + 3 * CELL, y0 + i * CELL, width=LINE_W, fill='gray'
            )
        
        # Draw policy heatmap
        policy_2d = policy.reshape(3, 3)  # Reshape 9 actions to 3x3 grid
        max_prob = np.max(policy) if np.max(policy) > 0 else 1.0
        
        for r in range(3):
            for c in range(3):
                x = x0 + c * CELL
                y = y0 + r * CELL
                prob = policy_2d[r, c]
                
                if prob > 0.001:  # Only show significant probabilities
                    # Color intensity based on probability (red = high, white = low)
                    intensity = min(1.0, prob / max_prob)
                    red_val = int(255 * intensity)
                    other_val = int(255 * (1 - intensity))
                    color = f"#{red_val:02x}{other_val:02x}{other_val:02x}"
                    
                    self.policy_canvas.create_rectangle(
                        x + 2, y + 2, x + CELL - 2, y + CELL - 2,
                        fill=color, outline=""
                    )
                    
                    # Show probability text - larger font for TTT
                    if prob > 0.01:
                        text_color = "white" if intensity > 0.5 else "black"
                        self.policy_canvas.create_text(
                            x + CELL//2, y + CELL//2,
                            text=f"{prob:.3f}", font=("Arial", 12, "bold"), fill=text_color
                        )

    # Navigation methods
    def on_game_change(self, event=None):
        self.load_game_and_move(self.game_combo.current(), 0)

    def on_move_change(self, event=None):
        self.load_game_and_move(self.game_idx, self.move_combo.current())

    def prev_game(self):
        if self.game_idx > 0:
            self.load_game_and_move(self.game_idx - 1, 0)

    def next_game(self):
        if self.game_idx < len(self.games) - 1:
            self.load_game_and_move(self.game_idx + 1, 0)

    def prev_move(self):
        if self.move_idx > 0:
            self.load_game_and_move(self.game_idx, self.move_idx - 1)

    def next_move(self):
        current_game = self.games[self.game_idx]
        if self.move_idx < len(current_game['moves']) - 1:
            self.load_game_and_move(self.game_idx, self.move_idx + 1)


def main():
    """Main function to launch the training data viewer."""
    # File selection
    path = sys.argv[1] if len(sys.argv) >= 2 else None
    if not path:
        root = tk.Tk()
        root.withdraw()
        
        default_dir = os.path.join(os.getcwd(), "ttt", "runs")
        if not os.path.isdir(default_dir):
            default_dir = os.path.join(os.getcwd(), "ttt")
        
        path = filedialog.askopenfilename(
            title="Select training data JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=default_dir,
        )
        root.destroy()
    
    if not path:
        print("No file selected.")
        return

    # Load data
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    # Launch viewer
    root = tk.Tk()
    app = TrainingDataViewer(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()