"""Legacy bridge to Planning-owned Execution Plan validation."""

from maps.planning.validation import (
    require_valid_execution_plan,
    validate_execution_plan,
)

__all__ = ["require_valid_execution_plan", "validate_execution_plan"]
