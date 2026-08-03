"""Processor for the independent set-structuring task."""

from ...processing.structuring_processor import (
    SET_MULTI_LEVEL_META_KEY,
    StructuringProcessor,
)


class SetStructuringProcessor(StructuringProcessor):
    """Task-specific view of the shared structuring tensor processor.

    Training corpora that still provide ``structuring`` are accepted through
    the fallback keys.  New multi-task batches use independent payload,
    mapping, metadata, and tensor namespaces so either task can enable the
    optional hierarchy steps without mutating the other.
    """

    task_name = "set_structuring"
    config_attr = "set_structuring_config"
    mapping_attr = "set_structuring_mapping"
    data_key = "set_structuring"
    schema_key = "set_structuring_schema"
    meta_key = SET_MULTI_LEVEL_META_KEY
    tensor_prefix = "set_structuring"
    fallback_data_key = "structuring"
    fallback_schema_key = "structuring_schema"
    resolved_flag = "_glinext_set_structuring_spans_resolved"
    pad_dense_fixed_slots = False
    require_span_targets = True

    @staticmethod
    def _select_structures(structures=None, set_structures=None):
        return set_structures if set_structures is not None else structures

    def contribute_inference_input(
        self,
        item,
        structures=None,
        set_structures=None,
        **kwargs,
    ):
        return super().contribute_inference_input(
            item,
            structures=self._select_structures(
                structures,
                set_structures,
            ),
            **kwargs,
        )

    def empty_inference_result(
        self,
        num_texts,
        structures=None,
        set_structures=None,
        **kwargs,
    ):
        return super().empty_inference_result(
            num_texts,
            structures=self._select_structures(
                structures,
                set_structures,
            ),
            **kwargs,
        )


__all__ = ["SetStructuringProcessor"]
