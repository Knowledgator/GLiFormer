"""Processor for entity-first set-prediction structuring."""

from ...processing.structuring_processor import StructuringProcessor as _BaseStructuringProcessor


class StructuringProcessor(_BaseStructuringProcessor):
    """Prepare mandatory entity spans and rectangular record assignments."""

    pad_dense_fixed_slots = False
    require_span_targets = True


__all__ = ["StructuringProcessor"]
