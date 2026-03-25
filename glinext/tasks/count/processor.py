"""Count task processor."""

import torch

from .. import TaskProcessor
from ...mappings import BatchClassesMapping


class CountProcessor(TaskProcessor):
    """Processor for count task."""

    def __init__(self, config, **kwargs):
        super().__init__(config)

    def get_classes_mapping(self, batch_list, **kwargs):
        return None

    def create_labels(self, batch_list, classes_mapping, **kwargs):
        total_cat = classes_mapping.total_cat_groups()
        total_ext = classes_mapping.total_extraction_groups()
        total_struct = classes_mapping.total_structuring_groups()
        total_parents = total_cat + total_ext + total_struct

        if total_parents == 0:
            return None

        count_targets = torch.zeros(total_parents, dtype=torch.float)
        count_batch_idx = torch.zeros(total_parents, dtype=torch.long)
        offset = 0

        for flat_idx, batch_idx, group_idx, _ in classes_mapping.flat_cat_iter():
            count_targets[offset] = 1.0
            count_batch_idx[offset] = batch_idx
            offset += 1

        for flat_idx, batch_idx, group_idx, _ in classes_mapping.flat_extraction_iter():
            count_targets[offset] = 1.0
            count_batch_idx[offset] = batch_idx
            offset += 1

        for flat_idx, batch_idx, group_idx, struct_item in classes_mapping.flat_structuring_iter():
            structuring_data = batch_list[batch_idx].get('structuring', {})
            schema_name = struct_item.name
            instances = structuring_data.get(schema_name, [])
            count_targets[offset] = float(len(instances))
            count_batch_idx[offset] = batch_idx
            offset += 1

        return {"count_targets": count_targets, "gold_count_val": count_targets.clone()}
