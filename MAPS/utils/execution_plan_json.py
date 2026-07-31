"""Legacy bridge to Deployment-owned Execution Plan serialization."""

from maps.deployment.serialization import (
    execution_plan_json_payload,
    write_execution_plan_json,
)

__all__ = ["execution_plan_json_payload", "write_execution_plan_json"]
