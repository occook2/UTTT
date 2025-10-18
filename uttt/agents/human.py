# uttt/agents/human.py
"""
Human agent for interactive gameplay.
"""
from __future__ import annotations
from typing import Optional, Callable
import threading
import time

from uttt.agents.base import Agent
from uttt.env.state import UTTTEnv


class HumanAgent(Agent):
    """
    Human agent that waits for user input through a callback mechanism.
    """
    
    def __init__(self, name: str = "Human"):
        super().__init__(name)
        self.move_callback: Optional[Callable[[UTTTEnv], int]] = None
        self._selected_action: Optional[int] = None
        self._waiting_for_move = False
        self._move_event = threading.Event()
    
    def set_move_callback(self, callback: Callable[[UTTTEnv], int]):
        """Set the callback function that will provide the human's move."""
        self.move_callback = callback
    
    def select_action(self, env: UTTTEnv) -> int:
        """
        Wait for human input through the callback mechanism.
        This will block until a move is selected via the UI.
        """
        if self.move_callback is None:
            raise RuntimeError("No move callback set for HumanAgent. Use set_move_callback() first.")
        
        # Reset the event and start waiting
        self._move_event.clear()
        self._selected_action = None
        self._waiting_for_move = True
        
        # Call the callback to initiate the UI interaction
        action = self.move_callback(env)
        
        self._waiting_for_move = False
        return action
    
    def provide_action(self, action: int):
        """
        Called by the UI to provide the selected action.
        """
        self._selected_action = action
        self._move_event.set()
    
    def is_waiting_for_move(self) -> bool:
        """Check if the agent is currently waiting for a human move."""
        return self._waiting_for_move