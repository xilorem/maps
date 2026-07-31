
"""Physical memory contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class L1Memory:
    """Tile-local memory metadata."""

    size: int
    bandwidth: int

    def __post_init__(self) -> None:

        # check for invalid values
        if self.size <= 0:
            raise ValueError("L1 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L1 memory bandwidth must be > 0")


@dataclass(frozen=True)
class L2Memory:
    """Shared mesh-level memory metadata."""

    size: int
    bandwidth: int

    def __post_init__(self) -> None:

        # check for invalid values
        if self.size <= 0:
            raise ValueError("L2 memory size must be > 0")
        if self.bandwidth <= 0:
            raise ValueError("L2 memory bandwidth must be > 0")
