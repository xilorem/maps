"""Typed tile-execution policy contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from .device import WorkSignature


@dataclass(frozen=True)
class FixedDeviceAssignment:
    """Stable Device names selected for exact Work Signatures."""

    assignments: Mapping[WorkSignature, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assignments = dict(self.assignments)
        if any(not device_name for device_name in assignments.values()):
            raise ValueError("assigned device name must not be empty")
        object.__setattr__(self, "assignments", MappingProxyType(assignments))


@dataclass(frozen=True)
class PrecisionLoweringRecipe:
    """One chip-approved typed operation precision conversion."""

    source_signature: WorkSignature
    target_signature: WorkSignature
    device_name: str

    def __post_init__(self) -> None:
        if not self.device_name:
            raise ValueError("precision lowering device name must not be empty")
