from dataclasses import dataclass

from .channel import LinkId


@dataclass(frozen=True)
class Ledger:
    """Everything the simulator did, in a form the independent checker can replay."""

    realization_id: int | None
    transfers: tuple[tuple[int, LinkId, str, float], ...]
    bandwidth: tuple[tuple[int, LinkId, float], ...]
    capacity_bits: tuple[tuple[int, LinkId, float], ...]
    drops: tuple[tuple[int, str, str, float], ...]
