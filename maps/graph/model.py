
from dataclasses import dataclass

from maps.graph.constants import ConstantStore
from maps.graph.constants import validate_constants
from maps.graph.graph import Graph


@dataclass(frozen=True)
class ImportedModel:
    graph: Graph
    constants: ConstantStore

    def validate(self) -> None:
        """Validate that initializer metadata and immutable bytes agree."""

        validate_constants(self.graph, self.constants)


def validate_imported_model(model: ImportedModel) -> None:
    model.validate()
