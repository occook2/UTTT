# ttt/scripts/play_vs_agent.py
"""
Interactive UI to play Tic-Tac-Toe against various agents.
"""
from __future__ import annotations

import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from typing import Dict, Any, List, Optional, Callable, Union
import threading
import time
import glob

import numpy as np

from ttt.env.state import TTTEnv, action_to_rc, rc_to_action
from ttt.agents.base import AgentLike
from ttt.agents.human import HumanAgent
from ttt.agents.random import RandomAgent
from ttt.eval.alphazero_factory import create_alphazero_agent, discover_alphazero_checkpoints


# --- UI constants ---
CELL = 80       # px per small cell (larger for TTT 3x3 grid)
PAD = 30        # canvas padding
LINE_W = 2
BOLD_W = 4
INSET = 15      # padding inside cell for drawing X/O (larger for TTT)
HILITE_W = 3
LEGAL_COLOR = "#90EE90"  # Light green for legal moves
LAST_COLOR = "#FFD700"   # Gold for last move


class PlayVsAgentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.env = TTTEnv(seed=None)
        self.human_agent = HumanAgent("Human")
        self.opponent_agent: Optional[AgentLike] = None
        self.human_is_x = True  # Human plays X (goes first) by default
        self.game_over = False
        self.last_move: Optional[tuple[int, int]] = None
        self.thinking = False  # True when AI is thinking
        
        # Agent move callback setup
        self.human_agent.set_move_callback(self._wait_for_human_move)
        
        # ---- UI layout ----
        self.root.title("TTT - Play vs Agent")
        outer = ttk.Frame(root, padding=8)
        outer.pack(fill="both", expand=True)

        # Agent selection frame
        self._setup_agent_selection(outer)
        
        # Game info frame
        info_frame = ttk.Frame(outer)
        info_frame.pack(side="top", fill="x", pady=(6, 6))
        self.info_label = ttk.Label(info_frame, text="Select an opponent to start playing!", width=60)
        self.info_label.pack(side="left")
        
        # Game controls
        controls = ttk.Frame(outer)
        controls.pack(side="top", fill="x", pady=(4, 2))
        ttk.Button(controls, text="🔄 New Game", command=self.new_game).pack(side="left")
        ttk.Button(controls, text="🔄 Swap Sides", command=self.swap_sides).pack(side="left", padx=4)
        self.thinking_label = ttk.Label(controls, text="", foreground="blue")
        self.thinking_label.pack(side="left", padx=20)

        # Canvas
        size = PAD * 2 + CELL * 3  # 3x3 grid for TTT
        self.canvas = tk.Canvas(outer, width=size, height=size, bg="white")
        self.canvas.pack(side="top", pady=10)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        
        # Status
        self.status_label = ttk.Label(outer, text="Welcome! Select an opponent and start playing.", foreground="green")
        self.status_label.pack(side="top", pady=5)

        # Initialize display
        self._redraw()
        self._update_info()

    def _setup_agent_selection(self, parent):
        """Setup the agent selection UI."""
        agent_frame = ttk.LabelFrame(parent, text="Select Opponent", padding=10)
        agent_frame.pack(side="top", fill="x", pady=(0, 10))
        
        # Agent type selection
        type_frame = ttk.Frame(agent_frame)
        type_frame.pack(side="top", fill="x")
        
        ttk.Label(type_frame, text="Agent Type:").pack(side="left")
        self.agent_type_var = tk.StringVar(value="random")
        
        agent_types = [
            ("Random Agent", "random"),
            ("AlphaZero Agent", "alphazero")
        ]
        
        for text, value in agent_types:
            ttk.Radiobutton(
                type_frame, text=text, variable=self.agent_type_var, 
                value=value, command=self._on_agent_type_change
            ).pack(side="left", padx=10)
        
        # AlphaZero checkpoint selection (initially hidden)
        self.az_frame = ttk.Frame(agent_frame)
        self.az_frame.pack(side="top", fill="x", pady=(10, 0))
        
        ttk.Label(self.az_frame, text="Checkpoint:").pack(side="left")
        self.checkpoint_var = tk.StringVar()
        self.checkpoint_combo = ttk.Combobox(
            self.az_frame, textvariable=self.checkpoint_var, 
            state="readonly", width=30
        )
        self.checkpoint_combo.pack(side="left", padx=10)
        
        ttk.Button(self.az_frame, text="📂 Browse", command=self._browse_checkpoint).pack(side="left", padx=5)
        ttk.Button(agent_frame, text="✅ Set Opponent", command=self._set_opponent).pack(side="top", pady=10)
        
        # Load checkpoints and update UI
        self._load_available_checkpoints()
        self._on_agent_type_change()  # Hide AZ frame initially

    def _load_available_checkpoints(self):
        """Load available AlphaZero checkpoints."""
        checkpoints = []
        
        # Search in runs directory
        runs_dir = "ttt/runs"
        if os.path.exists(runs_dir):
            for run_dir in os.listdir(runs_dir):
                checkpoint_dir = os.path.join(runs_dir, run_dir, "checkpoints")
                if os.path.exists(checkpoint_dir):
                    discovered = discover_alphazero_checkpoints(checkpoint_dir)
                    for name, path in discovered.items():
                        checkpoints.append((f"{run_dir}/{name}", path))
        
        # Search in top-level checkpoints directory
        if os.path.exists("ttt/checkpoints"):
            discovered = discover_alphazero_checkpoints("ttt/checkpoints")
            for name, path in discovered.items():
                checkpoints.append((name, path))
        
        self.available_checkpoints = dict(checkpoints)
        self.checkpoint_combo['values'] = list(self.available_checkpoints.keys())
        
        if checkpoints:
            self.checkpoint_combo.current(0)

    def _on_agent_type_change(self):
        """Handle agent type selection change."""
        agent_type = self.agent_type_var.get()
        if agent_type == "alphazero":
            self.az_frame.pack(side="top", fill="x", pady=(10, 0))
        else:
            self.az_frame.pack_forget()

    def _browse_checkpoint(self):
        """Browse for a checkpoint file."""
        path = filedialog.askopenfilename(
            title="Select AlphaZero Checkpoint",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")],
            initialdir="ttt/runs" if os.path.exists("ttt/runs") else "."
        )
        if path:
            # Add to available checkpoints
            name = f"Custom: {os.path.basename(path)}"
            self.available_checkpoints[name] = path
            
            # Update combo box
            values = list(self.available_checkpoints.keys())
            self.checkpoint_combo['values'] = values
            self.checkpoint_combo.set(name)

    def _set_opponent(self):
        """Create and set the selected opponent agent."""
        agent_type = self.agent_type_var.get()
        
        try:
            if agent_type == "random":
                self.opponent_agent = RandomAgent("Random Agent")
            elif agent_type == "alphazero":
                checkpoint_name = self.checkpoint_var.get()
                if not checkpoint_name:
                    messagebox.showerror("Error", "Please select an AlphaZero checkpoint.")
                    return
                
                checkpoint_path = self.available_checkpoints[checkpoint_name]
                if not os.path.exists(checkpoint_path):
                    messagebox.showerror("Error", f"Checkpoint file not found: {checkpoint_path}")
                    return
                
                # Create AlphaZero agent with moderate MCTS settings for interactive play
                self.opponent_agent = create_alphazero_agent(
                    checkpoint_path=checkpoint_path,
                    mcts_simulations=50,  # Reduced for faster play
                    temperature=0.0,
                    device="cpu"
                )
                self.opponent_agent.name = f"AlphaZero ({checkpoint_name})"
            
            self.status_label.config(
                text=f"Opponent set: {self.opponent_agent.name}. Click 'New Game' to start!",
                foreground="green"
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create agent: {str(e)}")
            self.opponent_agent = None

    def new_game(self):
        """Start a new game."""
        if self.opponent_agent is None:
            messagebox.showwarning("No Opponent", "Please select an opponent first.")
            return
        
        self.env.reset()
        self.game_over = False
        self.last_move = None
        self.thinking = False
        self._update_info()
        self._redraw()
        
        self.status_label.config(
            text=f"New game started! You are {'X' if self.human_is_x else 'O'}.",
            foreground="blue"
        )
        
        # If human is O (goes second), make AI move first
        if not self.human_is_x:
            self._make_ai_move()

    def swap_sides(self):
        """Swap who plays X and O."""
        self.human_is_x = not self.human_is_x
        self._update_info()
        
        # If there's an active game and it's not over, restart
        if self.opponent_agent is not None and not self.game_over:
            self.new_game()

    def _update_info(self):
        """Update the game info display."""
        if self.opponent_agent is None:
            self.info_label.config(text="Select an opponent to start playing!")
        else:
            human_symbol = "X" if self.human_is_x else "O"
            ai_symbol = "O" if self.human_is_x else "X"
            
            if self.game_over:
                winner = self.env._macro_winner()
                if winner == 0:
                    result = "Game Over - Draw!"
                elif (winner == 1 and self.human_is_x) or (winner == -1 and not self.human_is_x):
                    result = "You Win! 🎉"
                else:
                    result = "AI Wins! 🤖"
                self.info_label.config(text=result)
            else:
                turn = "Your turn" if self._is_human_turn() else f"{self.opponent_agent.name}'s turn"
                self.info_label.config(
                    text=f"Human ({human_symbol}) vs {self.opponent_agent.name} ({ai_symbol}) | {turn}"
                )

    def _is_human_turn(self) -> bool:
        """Check if it's currently the human's turn."""
        current_player = self.env.player
        return (current_player == 1 and self.human_is_x) or (current_player == -1 and not self.human_is_x)

    def _wait_for_human_move(self, env: TTTEnv) -> int:
        """
        Called by HumanAgent. This method blocks until the user clicks a legal move.
        """
        # This will be resolved by _on_canvas_click when user clicks
        self.human_agent._move_event.wait()  # Block until move is made
        return self.human_agent._selected_action

    def _on_canvas_click(self, event):
        """Handle mouse clicks on the canvas."""
        if self.opponent_agent is None or self.game_over or self.thinking:
            return
        
        if not self._is_human_turn():
            return
        
        # Convert click coordinates to grid position
        x = event.x - PAD
        y = event.y - PAD
        
        if x < 0 or y < 0 or x >= 3 * CELL or y >= 3 * CELL:  # 3x3 grid for TTT
            return
        
        col = x // CELL
        row = y // CELL
        
        # Check if this is a legal move
        action = rc_to_action(row, col)
        legal_mask = self.env.legal_actions_mask()
        if legal_mask[row, col] == 0:
            return
        
        # Make the human move
        self._make_human_move(action)

    def _make_human_move(self, action: int):
        """Execute a human move and trigger AI response if needed."""
        row, col = action_to_rc(action)
        self.last_move = (row, col)
        
        self.env.step(action)
        self._redraw()
        
        # Check if game is over
        if self.env.terminated:
            self.game_over = True
            self._update_info()
            return
        
        # If human agent is waiting, provide the action
        if self.human_agent._waiting_for_move:
            self.human_agent.provide_action(action)
        else:
            # Otherwise, make AI move next
            self._make_ai_move()

    def _make_ai_move(self):
        """Make the AI move in a separate thread to keep UI responsive."""
        if self.game_over or self._is_human_turn():
            return
        
        self.thinking = True
        self.thinking_label.config(text="🤔 AI is thinking...")
        self._update_info()
        
        # Run AI move in separate thread to keep UI responsive
        def ai_move_thread():
            try:
                action = self.opponent_agent.select_action(self.env)
                
                # Switch back to main thread for UI updates
                self.root.after(0, self._complete_ai_move, action)
            except Exception as e:
                self.root.after(0, self._ai_move_error, str(e))
        
        threading.Thread(target=ai_move_thread, daemon=True).start()

    def _complete_ai_move(self, action: int):
        """Complete the AI move (called in main thread)."""
        self.thinking = False
        self.thinking_label.config(text="")
        
        action = action[0]

        self.last_move = (action)
        
        self.env.step(action)
        self._redraw()
        
        # Check if game is over
        if self.env.terminated:
            self.game_over = True
        
        self._update_info()

    def _ai_move_error(self, error_msg: str):
        """Handle AI move error (called in main thread)."""
        self.thinking = False
        self.thinking_label.config(text="")
        messagebox.showerror("AI Error", f"AI agent error: {error_msg}")

    def _redraw(self):
        """Redraw the game board."""
        self.canvas.delete("all")
        x0 = PAD
        y0 = PAD

        # 3x3 grid for TTT
        for i in range(4):  # 0, 1, 2, 3 for 3x3 grid
            self.canvas.create_line(x0 + i * CELL, y0, x0 + i * CELL, y0 + 3 * CELL, width=LINE_W)
            self.canvas.create_line(x0, y0 + i * CELL, x0 + 3 * CELL, y0 + i * CELL, width=LINE_W)

        # Draw stones
        b = self.env.board
        for r in range(3):
            for c in range(3):
                v = int(b[r, c])
                cx = x0 + c * CELL
                cy = y0 + r * CELL
                if v == 1:      # X
                    self.canvas.create_line(
                        cx + INSET, cy + INSET, cx + CELL - INSET, cy + CELL - INSET, width=4, fill='red'
                    )
                    self.canvas.create_line(
                        cx + CELL - INSET, cy + INSET, cx + INSET, cy + CELL - INSET, width=4, fill='red'
                    )
                elif v == -1:   # O
                    self.canvas.create_oval(
                        cx + INSET, cy + INSET, cx + CELL - INSET, cy + CELL - INSET, width=4, outline='blue'
                    )

        # Highlight last move
        if self.last_move is not None:
            lr, lc = self.last_move
            cx = x0 + lc * CELL
            cy = y0 + lr * CELL
            self.canvas.create_rectangle(
                cx + 3, cy + 3, cx + CELL - 3, cy + CELL - 3, 
                outline=LAST_COLOR, width=HILITE_W
            )

        # Highlight legal moves for human turn (only when it's human's turn and game not over)
        if not self.game_over and self._is_human_turn() and not self.thinking:
            mask = self.env.legal_actions_mask()
            for r, c in zip(*np.where(mask == 1)):
                cx = x0 + c * CELL
                cy = y0 + r * CELL
                # Create a subtle green highlight
                self.canvas.create_rectangle(
                    cx + 4, cy + 4, cx + CELL - 4, cy + CELL - 4,
                    fill=LEGAL_COLOR, outline=LEGAL_COLOR, width=1
                )
                # Add a small dot to make it more visible
                d = 12  # larger dots for TTT
                self.canvas.create_oval(
                    cx + CELL // 2 - d//2, cy + CELL // 2 - d//2,
                    cx + CELL // 2 + d//2, cy + CELL // 2 + d//2,
                    fill="darkgreen", outline=""
                )


def main():
    """Main entry point."""
    root = tk.Tk()
    app = PlayVsAgentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()