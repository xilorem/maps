"""Deployment Bundles, serialization, packaging, and artifact validation."""

from .bundle import (
    DeploymentBundle,
    build_deployment_bundle,
    validate_deployment_bundle,
    validate_execution_plan_bundle_files,
    write_deployment_bundle,
    write_execution_plan_bundle,
)
from .serialization import write_execution_plan
from .application import (
    MAGIA_V2_TARGET,
    application_build_summary,
    build_application,
    normalize_application_name,
)
from .weights import PackedInitializer, PackedWeights, pack_weights
from .package import (
    package_summary,
    validate_deployment_package,
    write_deployment_package,
)

__all__ = [
    "DeploymentBundle",
    "MAGIA_V2_TARGET",
    "PackedInitializer",
    "PackedWeights",
    "application_build_summary",
    "build_application",
    "build_deployment_bundle",
    "normalize_application_name",
    "pack_weights",
    "package_summary",
    "validate_deployment_bundle",
    "validate_deployment_package",
    "validate_execution_plan_bundle_files",
    "write_deployment_bundle",
    "write_deployment_package",
    "write_execution_plan",
    "write_execution_plan_bundle",
]
