"""Migration-only lowercase entry point for the existing CLI."""

from MAPS.cli import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
