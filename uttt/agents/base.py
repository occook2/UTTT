# uttt/agents/base.py
from __future__ import annotations
from typing import Protocol, runtime_checkable, Optional, Any

# Structural interface (duck-typed): anything with select_action(env)->int is an Agent.
@runtime_checkable
class AgentLike(Protocol):
    def select_action(self, env: "UTTTEnv") -> int: ...
    # Optional lifecycle hooks (no-ops by default if not implemented)
    def reset(self) -> None: ...
    def seed(self, seed: Optional[int]) -> None: ...

# (Optional) Nominal base class if you prefer inheritance + IDE help.
class Agent(AgentLike):
    name: str

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__

    def reset(self) -> None:
        """Called between games/matches, override if the agent keeps internal state."""
        return None

    def seed(self, seed: Optional[int]) -> None:
        """Set internal RNG seed if the agent has one."""
        return None

    def select_action(self, env: "UTTTEnv") -> int:  # pragma: no cover - abstract
        raise NotImplementedError("Agents must implement select_action(env) -> int")
