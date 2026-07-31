"""Migration bridge to the lowercase Graph-owned Imported Model."""

from maps.graph.model import ImportedModel, validate_imported_model

__all__ = ["ImportedModel", "validate_imported_model"]
