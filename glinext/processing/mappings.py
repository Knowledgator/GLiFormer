from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class BaseClassMapping:
    class_to_id: dict
    name: Optional[str] = None
    description: Optional[str] = None

    def get_reverse_mapping(self) -> Dict[int, str]:
        return {v: k for k, v in self.class_to_id.items()}


@dataclass
class CatClassMapping:
    cat_class_to_id: List[BaseClassMapping]


@dataclass
class ExtractionItemMapping:
    """Mapping for a single extraction item containing NER and optional relation classes."""
    ner_class_to_id: BaseClassMapping
    rel_class_to_id: Optional[BaseClassMapping] = None


@dataclass
class ExtractionClassMapping:
    """Per-example extraction mappings: list of items each with NER + optional REL."""
    items: List[ExtractionItemMapping] = field(default_factory=list)


@dataclass
class StructuringItemMapping:
    """Mapping for a single structuring schema (e.g. 'person' with fields 'name', 'age')."""
    field_class_to_id: BaseClassMapping  # field names → ids
    name: Optional[str] = None
    description: Optional[str] = None


@dataclass
class StructuringClassMapping:
    """Per-example structuring mappings: list of schemas each with field mappings."""
    items: List[StructuringItemMapping] = field(default_factory=list)


@dataclass
class OpenRelexItemMapping:
    """Mapping for a single open_relex group (relation types → ids)."""
    rel_class_to_id: BaseClassMapping
    name: Optional[str] = None


@dataclass
class OpenRelexClassMapping:
    """Per-example open_relex mappings: list of groups each with relation type mappings."""
    items: List[OpenRelexItemMapping] = field(default_factory=list)


@dataclass
class VisionItemMapping:
    """Mapping for a single image-level vision task group."""
    class_to_id: BaseClassMapping
    name: Optional[str] = None


@dataclass
class VisionClassMapping:
    """Per-example image task mappings: one or more label groups."""
    items: List[VisionItemMapping] = field(default_factory=list)


@dataclass
class BatchClassesMapping:
    PROMPT_TASK_ORDER = (
        "classification", "ner", "open_relex", "structuring",
        "image_classification", "object_detection", "segmentation",
        "audio_classification", "audio_segmentation",
    )

    cat_mapping: List[CatClassMapping]
    extraction_mapping: List[ExtractionClassMapping]
    structuring_mapping: List[StructuringClassMapping] = field(default_factory=list)
    open_relex_mapping: List[OpenRelexClassMapping] = field(default_factory=list)
    image_classification_mapping: List[VisionClassMapping] = field(default_factory=list)
    audio_classification_mapping: List[VisionClassMapping] = field(default_factory=list)
    object_detection_mapping: List[VisionClassMapping] = field(default_factory=list)
    segmentation_mapping: List[VisionClassMapping] = field(default_factory=list)
    audio_segmentation_mapping: List[VisionClassMapping] = field(default_factory=list)

    def get_item_mapping(self, index: int) -> Tuple[CatClassMapping, ExtractionClassMapping]:
        return self.cat_mapping[index], self.extraction_mapping[index]

    def total_cat_groups(self) -> int:
        """Total number of classification groups across the batch."""
        return sum(len(cm.cat_class_to_id) for cm in self.cat_mapping)

    def total_extraction_groups(self) -> int:
        """Total number of extraction groups across the batch."""
        return sum(len(em.items) for em in self.extraction_mapping)

    def total_structuring_groups(self) -> int:
        """Total number of structuring groups (schemas) across the batch."""
        return sum(len(sm.items) for sm in self.structuring_mapping)

    def flat_cat_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, mapping) over all cat groups."""
        flat_idx = 0
        for batch_idx, cm in enumerate(self.cat_mapping):
            for group_idx, mapping in enumerate(cm.cat_class_to_id):
                yield flat_idx, batch_idx, group_idx, mapping
                flat_idx += 1

    def flat_extraction_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, item_mapping) over all extraction groups."""
        flat_idx = 0
        for batch_idx, em in enumerate(self.extraction_mapping):
            for group_idx, item_mapping in enumerate(em.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    def total_open_relex_groups(self) -> int:
        """Total number of open_relex groups across the batch."""
        return sum(len(om.items) for om in self.open_relex_mapping)

    def flat_open_relex_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, item_mapping) over all open_relex groups."""
        flat_idx = 0
        for batch_idx, om in enumerate(self.open_relex_mapping):
            for group_idx, item_mapping in enumerate(om.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    def flat_structuring_iter(self):
        """Iterate (flat_idx, batch_idx, group_idx, item_mapping) over all structuring groups."""
        flat_idx = 0
        for batch_idx, sm in enumerate(self.structuring_mapping):
            for group_idx, item_mapping in enumerate(sm.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    @staticmethod
    def _flat_vision_iter(mapping_list):
        flat_idx = 0
        for batch_idx, vm in enumerate(mapping_list):
            for group_idx, item_mapping in enumerate(vm.items):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    def total_image_classification_groups(self) -> int:
        return sum(len(vm.items) for vm in self.image_classification_mapping)

    def total_object_detection_groups(self) -> int:
        return sum(len(vm.items) for vm in self.object_detection_mapping)

    def total_segmentation_groups(self) -> int:
        return sum(len(vm.items) for vm in self.segmentation_mapping)

    def total_audio_classification_groups(self) -> int:
        return sum(len(vm.items) for vm in self.audio_classification_mapping)

    def total_audio_segmentation_groups(self) -> int:
        return sum(len(vm.items) for vm in self.audio_segmentation_mapping)

    def flat_image_classification_iter(self):
        yield from self._flat_vision_iter(self.image_classification_mapping)

    def flat_audio_classification_iter(self):
        yield from self._flat_vision_iter(self.audio_classification_mapping)

    def flat_object_detection_iter(self):
        yield from self._flat_vision_iter(self.object_detection_mapping)

    def flat_segmentation_iter(self):
        yield from self._flat_vision_iter(self.segmentation_mapping)

    def flat_audio_segmentation_iter(self):
        yield from self._flat_vision_iter(self.audio_segmentation_mapping)

    def group_count(self, task_name: str, batch_idx: int) -> int:
        """Number of groups for a task in one batch item."""
        if task_name == "classification":
            return len(self.cat_mapping[batch_idx].cat_class_to_id)
        if task_name in ("ner", "joint_relex"):
            return len(self.extraction_mapping[batch_idx].items)

        mapping_attr = f"{task_name}_mapping"
        mapping_list = getattr(self, mapping_attr, None)
        if mapping_list is None or batch_idx >= len(mapping_list):
            return 0
        return len(mapping_list[batch_idx].items)

    def group_counts(self, task_name: str, batch_size: int) -> List[int]:
        """Number of groups for a task for each batch item."""
        return [self.group_count(task_name, batch_idx) for batch_idx in range(batch_size)]

    def parent_offset_for_item(self, task_name: str, batch_idx: int) -> int:
        """First parent-token position for a task within one batch item."""
        effective_task = "ner" if task_name == "joint_relex" else task_name
        offset = 0
        for current_task in self.PROMPT_TASK_ORDER:
            if current_task == effective_task:
                break
            offset += self.group_count(current_task, batch_idx)
        return offset

    def child_size(self, task_name: str, batch_idx: int, group_idx: int) -> int:
        """Number of child labels for a task group."""
        if task_name in ("ner", "joint_relex"):
            return len(self.extraction_mapping[batch_idx].items[group_idx].ner_class_to_id.class_to_id)
        if task_name == "classification":
            return len(self.cat_mapping[batch_idx].cat_class_to_id[group_idx].class_to_id)
        if task_name == "structuring":
            return len(self.structuring_mapping[batch_idx].items[group_idx].field_class_to_id.class_to_id)
        if task_name == "open_relex":
            return len(self.open_relex_mapping[batch_idx].items[group_idx].rel_class_to_id.class_to_id)
        if task_name in (
            "image_classification", "audio_classification", "object_detection",
            "segmentation", "audio_segmentation",
        ):
            mapping_list = getattr(self, f"{task_name}_mapping")
            return len(mapping_list[batch_idx].items[group_idx].class_to_id.class_to_id)
        return 0

    def flat_iter(self, task_name: str):
        """Return the appropriate flat iterator for a task."""
        iters = {
            "ner": self.flat_extraction_iter,
            "joint_relex": self.flat_extraction_iter,
            "classification": self.flat_cat_iter,
            "structuring": self.flat_structuring_iter,
            "open_relex": self.flat_open_relex_iter,
            "image_classification": self.flat_image_classification_iter,
            "audio_classification": self.flat_audio_classification_iter,
            "object_detection": self.flat_object_detection_iter,
            "segmentation": self.flat_segmentation_iter,
            "audio_segmentation": self.flat_audio_segmentation_iter,
        }
        return iters[task_name]()
