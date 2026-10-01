"""Target-owned software costs around runtime DMA copies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DMARuntimeCost:
    """Copy overheads, separate from payload bandwidth and NoC routing.

    Submission includes programming and observing one hardware descriptor.
    Setup and publication are charged once per logical transfer. Packed
    intermediates require both a producer copy and a consumer unpack copy.
    """

    submission_cycles: int = 0
    setup_cycles: int = 0
    publication_cycles: int = 0
    packed_intermediates: bool = False

    def __post_init__(self) -> None:
        if min(self.submission_cycles, self.setup_cycles, self.publication_cycles) < 0:
            raise ValueError("DMA runtime costs must be non-negative")

    def overhead(self, jobs: int, *, publish: bool = False) -> int:
        if jobs == 0:
            return 0
        return (
            self.setup_cycles
            + jobs * self.submission_cycles
            + (self.publication_cycles if publish else 0)
        )
