"""Runtime buffering contract carried by every Execution Plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionContract:
    """Execution settings that affect both planning and backend allocation."""

    num_token_slots: int = 2

    def __post_init__(self) -> None:
        if self.num_token_slots <= 0:
            raise ValueError("num_token_slots must be > 0")
