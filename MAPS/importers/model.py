
from dataclasses import dataclass

from MAPS.core.constants import ConstantStore
from MAPS.core.graph import Graph

@dataclass(frozen=True)
class ImportedModel:
    graph: Graph
    constants: ConstantStore
