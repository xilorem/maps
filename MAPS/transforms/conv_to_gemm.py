"""Migration bridge to MAGIA-owned convolution specialization."""

from maps.target.magia.convolution import lower_convolutions

__all__ = ["lower_convolutions"]
