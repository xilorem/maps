"""Legacy bridge to Planning-owned Execution Plan construction."""

from maps.planning.construction import (
    construct_execution_plan as lower_execution_plan,
)

__all__ = ["lower_execution_plan"]
