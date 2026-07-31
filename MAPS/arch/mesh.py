"""Mesh-level geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .memory import L2Memory
from .noc import NoC
from .tile import Tile
from .execution import PrecisionLoweringRecipe


@dataclass(frozen=True)
class Mesh:
    """Rectangular mesh of tiles stored in row-major order."""

    width: int
    height: int
    l2_memory: L2Memory
    noc: NoC
    tiles: tuple[Tile, ...]
    precision_lowering_recipes: tuple[PrecisionLoweringRecipe, ...] = ()
    required_graph_rewrites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # check for invalid sizes
        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.height <= 0:
            raise ValueError("height must be > 0")

        # check for valid tiles and noc descriptions
        self._validate_tiles(self.width, self.height, self.tiles)
        self._validate_noc(self.width, self.height, self.noc)
        self._validate_precision_lowering_recipes(self.precision_lowering_recipes)

    @staticmethod
    def _validate_precision_lowering_recipes(
        recipes: tuple[PrecisionLoweringRecipe, ...],
    ) -> None:
        sources = [recipe.source_signature for recipe in recipes]
        if len(set(sources)) != len(sources):
            raise ValueError("duplicate precision lowering recipe source signature")


    @staticmethod
    def _validate_tiles(width: int, height: int, tiles: tuple[Tile, ...]) -> None:
        if len(tiles) != width * height:
            raise ValueError("tiles length must match mesh area")

        for expected_id, tile in enumerate(tiles):
            if tile.tile_id != expected_id:
                raise ValueError("tiles must be ordered by row-major tile_id")
            expected_x = expected_id % width
            expected_y = expected_id // width
            if tile.x != expected_x or tile.y != expected_y:
                raise ValueError("tile coordinates must match row-major tile placement")

    @staticmethod
    def _validate_noc(width: int, height: int, noc: NoC) -> None:
        num_tiles = width * height
        for endpoint in noc.endpoints:
            if endpoint.tile_id is None:
                continue
            if not (0 <= endpoint.tile_id < num_tiles):
                raise ValueError(f"NoC endpoint tile_id out of bounds: {endpoint.tile_id}")

    @property
    def shape(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def num_tiles(self) -> int:
        return self.width * self.height

    def contains_coord(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def contains_tile_id(self, tile_id: int) -> bool:
        return 0 <= tile_id < self.num_tiles

    def tile_id(self, x: int, y: int) -> int:
        """Returns id value of a tile given xy coordinates"""

        if not self.contains_coord(x, y):
            raise ValueError(f"coordinates out of bounds: ({x}, {y})")
        return y * self.width + x

    def coords(self, tile_id: int) -> tuple[int, int]:
        """Returns xy coordinates of a tile given it's id value"""

        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return tile_id % self.width, tile_id // self.width

    def tile(self, x: int, y: int) -> Tile:
        """Returns tile object give it's xy coordinates"""

        return self.tile_by_id(self.tile_id(x, y))

    def tile_by_id(self, tile_id: int) -> Tile:
        if not self.contains_tile_id(tile_id):
            raise ValueError(f"tile_id out of bounds: {tile_id}")
        return self.tiles[tile_id]

    def row(self, y: int) -> tuple[Tile, ...]:
        if y < 0 or y >= self.height:
            raise ValueError(f"row out of bounds: {y}")
        start = y * self.width
        return self.tiles[start:start + self.width]

    def column(self, x: int) -> tuple[Tile, ...]:
        if x < 0 or x >= self.width:
            raise ValueError(f"column out of bounds: {x}")
        return tuple(self.tile(x, y) for y in range(self.height))

    def rectangle(
        self,
        x0: int,
        y0: int,
        width: int,
        height: int,
    ) -> tuple[Tile, ...]:
        """Return one rectangular placed region in row-major tile order."""

        if width <= 0 or height <= 0:
            raise ValueError("width and height must be > 0")
        if not self.contains_coord(x0, y0):
            raise ValueError(f"rectangle origin out of bounds: ({x0}, {y0})")
        if not self.contains_coord(x0 + width - 1, y0 + height - 1):
            raise ValueError("rectangle exceeds mesh bounds")

        return tuple(
            self.tile(x, y)
            for y in range(y0, y0 + height)
            for x in range(x0, x0 + width)
        )

    def all_rectangles(self) -> tuple[tuple[Tile, ...], ...]:
        """Enumerate every rectangular placement on this mesh."""

        rectangles = []
        for height in range(1, self.height + 1):
            for width in range(1, self.width + 1):
                for y0 in range(0, self.height - height + 1):
                    for x0 in range(0, self.width - width + 1):
                        rectangles.append(self.rectangle(x0, y0, width, height))
        return tuple(rectangles)

    @classmethod
    def manhattan_distance(cls, a: Tile, b: Tile) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)
