"""Deployment Bundles, serialization, applications, and artifact validation."""

from .bundle import (
    DeploymentBundle,
    build_deployment_bundle,
    validate_deployment_bundle,
    write_deployment_bundle,
)
from .serialization import write_execution_plan
from .application import (
    APPLICATION_SCHEMA_VERSION,
    DESCRIPTOR_ABI_VERSION,
    MAGIA_V2_TARGET,
    OPERATION_ABI_VERSION,
    application_build_summary,
    application_summary,
    build_application,
    normalize_application_name,
    validate_application,
)
from .weights import PackedInitializer, PackedWeights, pack_weights

__all__ = [
    "DeploymentBundle",
    "APPLICATION_SCHEMA_VERSION",
    "DESCRIPTOR_ABI_VERSION",
    "MAGIA_V2_TARGET",
    "OPERATION_ABI_VERSION",
    "PackedInitializer",
    "PackedWeights",
    "application_build_summary",
    "application_summary",
    "build_application",
    "build_deployment_bundle",
    "normalize_application_name",
    "pack_weights",
    "validate_deployment_bundle",
    "validate_application",
    "write_deployment_bundle",
    "write_execution_plan",
]
