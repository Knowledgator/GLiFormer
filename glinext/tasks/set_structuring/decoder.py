"""Decoder for independent entity-first set structuring."""

from ..structuring.decoder import StructuringDecoder


class SetStructuringDecoder(StructuringDecoder):
    """Decode second-stage per-entity record/field logits."""

    config_attr = "set_structuring_config"
    token_logits_attr = "set_structuring_entity_logits"
    span_logits_attr = "set_structuring_logits"
    batch_origin_attr = "set_structuring_batch_origin"
    anchor_mask_attr = "set_structuring_anchor_mask"
    objectness_logits_attr = "set_structuring_objectness_logits"
    span_idx_attr = "set_structuring_span_idx"
    span_mask_attr = "set_structuring_span_mask"


__all__ = ["SetStructuringDecoder"]
