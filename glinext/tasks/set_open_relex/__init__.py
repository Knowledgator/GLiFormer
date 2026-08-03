"""Independent entity-first set-prediction relation extraction."""

from .decoder import (
    SetOpenRelationExtractionDecoder,
    SetOpenRelexDecoder,
)
from .model import (
    SetOpenRelationExtractionHead,
    SetOpenRelexHead,
)
from .processor import SetOpenRelexProcessor

__all__ = [
    "SetOpenRelexHead",
    "SetOpenRelationExtractionHead",
    "SetOpenRelexDecoder",
    "SetOpenRelationExtractionDecoder",
    "SetOpenRelexProcessor",
]
