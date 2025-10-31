# ttt/agents/base.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, Any
from ttt.env.state import TTTEnv

# Structural interface (duck-typed): anything with select_action(env)->int is an Agent.
@runtime_checkable
class AgentLike(Protocol):
    def select_action(self, env: TTTEnv) -> int: ...
    # Optional lifecycle hooks (no-ops by default if not implemented)

# (Optional) Nominal base class if you prefer inheritance + IDE help.
class Agent(AgentLike):
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def select_action(self, env: TTTEnv) -> int:  # pragma: no cover - abstract
        raise NotImplementedError("Agents must implement select_action(env) -> int")
