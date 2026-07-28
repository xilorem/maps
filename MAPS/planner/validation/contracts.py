"""Configuration and result contracts for planner-side validation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlannerConstraints:
    """Hard legality constraints checked against a completed Pipeline.

    Capacity flags control whether L1 and L2 residency estimates are enforced.
    Cross-submesh policy controls transition layout legality.
    """

    max_stage_nodes: int = 0
    allow_cross_submesh_remap: bool = True
    enforce_l1_capacity: bool = True
    enforce_l2_capacity: bool = True

    def __post_init__(self) -> None:
        """Reject nonsensical stage-size limits at configuration time."""

        if self.max_stage_nodes < 0:
            raise ValueError("max_stage_nodes must be >= 0")


@dataclass(frozen=True)
class ConstraintViolation:
    """One categorized planner legality violation."""

    kind: str
    message: str


@dataclass(frozen=True)
class ConstraintReport:
    """Complete non-throwing result of planner constraint validation."""

    violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    @property
    def is_valid(self) -> bool:
        """Return whether validation found no violations."""

        return not self.violations


def append_violation(
    violations: list[ConstraintViolation],
    kind: str,
    message: str,
) -> None:
    """Append one consistently constructed violation to an accumulator."""

    violations.append(ConstraintViolation(kind=kind, message=message))
