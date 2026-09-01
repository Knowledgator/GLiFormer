"""Entity-first set-prediction relation extraction."""

from .decoder import OpenRelexDecoder
from .model import OpenRelexHead
from .processor import OpenRelexProcessor

__all__ = [
    "OpenRelexHead",
    "OpenRelexDecoder",
    "OpenRelexProcessor",
]
