"""Shared tensor machinery for entity-first structuring."""

import json
import random
from copy import deepcopy

import torch

from ..tasks.span_processor import SpanProcessor
from ._structuring_hierarchy import (
    INTERNAL_INSTANCE_KEYS,
    MULTI_LEVEL_META_KEY,
    MULTI_LEVEL_ROOT_KEY,
    NODE_ID_KEY,
    NODE_KEEP_KEY,
    STRUCTURING_TYPES_KEY,
    TYPED_ANCHORS_KEY,
    TYPED_UNRESOLVED_KEY,
    TYPED_VALUES_KEY,
    StructuringProcessorComponent,
    is_internal_instance_key,
    resolve_structuring_processor,
    typed_field_layout,
)
from .label_augmentation import AugmentableLabelGroup
from .mappings import (
    BaseClassMapping,
    StructuringClassMapping,
    StructuringItemMapping,
)
from .typed_fields import (
    QUANTITY_KIND,
    TypedFieldSpec,
    build_quantity_target,
    find_anchor,
    is_typed_descriptor,
    lookup_group_types,
)

STRUCTURING_EVIDENCE_KEY = "structuring_evidence"
_TYPED_PREPARED_FLAG = "_gliformer_structuring_typed_prepared"


class StructuringProcessor(SpanProcessor):
    """Processor for structuring task."""

    task_name = "structuring"
    config_attr = "structuring_config"
    mapping_attr = "structuring_mapping"
    data_key = "structuring"
    schema_key = "structuring_schema"
    meta_key = MULTI_LEVEL_META_KEY
    tensor_prefix = "structuring"
    resolved_flag = "_gliformer_structuring_spans_resolved"
    # Entity-first structuring matches predicted record slots against the
    # compact gold-record axis and always needs explicit entity-span targets.
    pad_dense_fixed_slots = False
    require_span_targets = True

    def __init__(self, config, tokenizer=None, words_splitter=None, **kwargs):
        super().__init__(config, tokenizer, words_splitter,
                         parent_token=getattr(config, 'struct_parent_token', None), **kwargs)
        self.child_token = config.child_token
        self.task_config = getattr(config, self.config_attr, None)
        if self.task_config is None:
            raise ValueError(f"{self.config_attr} is required for {self.task_name}")
        self.typed_enabled = bool(getattr(self.task_config, "typed_fields", False))
        self.option_token = getattr(config, "option_token", "[OPTION]")
        self.typed_num_bins = int(getattr(self.task_config, "typed_num_bins", 256))
        self._typed_spec_cache = {}

        # Resolve the processing strategy independently of the prediction
        # slot width. Gold records remain a compact rectangular target axis;
        # Hungarian matching accounts for unmatched prediction slots.
        mode = (
            self.task_config.effective_structure_mode()
            if hasattr(self.task_config, "effective_structure_mode")
            else None
        )
        configured_multi_level = (
            bool(mode.is_multi_level)
            if mode is not None
            else bool(getattr(self.task_config, "multi_level", False))
        )
        self.component = resolve_structuring_processor(
            (
                mode.processor_spec()
                if mode is not None and hasattr(mode, "processor_spec")
                else getattr(mode, "processor", None)
            ),
            multi_level=configured_multi_level,
            child_token=config.structuring_child_token,
            end_token=config.structuring_end_token,
            data_key=self.data_key,
            schema_key=self.schema_key,
            meta_key=self.meta_key,
            typed_fields=self.typed_enabled,
        )
        self.multi_level = bool(
            getattr(self.component, "multi_level", configured_multi_level)
        )
        # Entity spans are mandatory targets for the second stage rather than
        # an optional auxiliary representation.
        self._span_config = (
            self.task_config
            if self.require_span_targets
            or getattr(self.task_config, "represent_spans", False)
            else None
        )
        # Entity-first structuring uses a rectangular assignment (predicted
        # slots versus gold records), so its gold axis must not be padded to
        # the fixed prediction width.
        fixed_slot_configs = (
            [self.task_config] if self.pad_dense_fixed_slots else []
        )
        self._fixed_slot_pad = max(
            (
                struct_cfg.effective_anchor_num_slots()
                if struct_cfg.effective_anchor_mode()
                in (
                    'fixed',
                    'fixed_rnn',
                    'fixed_transformer',
                    'position_buckets',
                    'topk_norm',
                    'topk_distinct',
                    'topk_parent',
                    'topk_density_distinct',
                )
                else 0
            )
            for struct_cfg in fixed_slot_configs
        ) if fixed_slot_configs else 0

    def _ensure_task_payload(self, item):
        """Return the canonical structuring payload unchanged."""
        return item

    def _mapping_list(self, classes_mapping):
        return getattr(classes_mapping, self.mapping_attr, [])

    def _flat_structuring_iter(self, classes_mapping):
        flat_idx = 0
        for batch_idx, structuring_mapping in enumerate(
            self._mapping_list(classes_mapping)
        ):
            for group_idx, item_mapping in enumerate(
                structuring_mapping.items
            ):
                yield flat_idx, batch_idx, group_idx, item_mapping
                flat_idx += 1

    def _total_structuring_groups(self, classes_mapping):
        return sum(
            len(mapping.items) for mapping in self._mapping_list(classes_mapping)
        )

    def _tensor_key(self, suffix):
        return f"{self.tensor_prefix}_{suffix}"

    @staticmethod
    def _instance_sort_key(instance):
        """Return the earliest span start position across all fields in an instance."""
        min_start = float('inf')
        for key, value in instance.items():
            if is_internal_instance_key(key):
                continue
            if isinstance(value, dict):
                st = value.get('start', -1)
                if st >= 0:
                    min_start = min(min_start, st)
            elif isinstance(value, list):
                for v in value:
                    if isinstance(v, dict):
                        st = v.get('start', -1)
                        if st >= 0:
                            min_start = min(min_start, st)
        return min_start

    @staticmethod
    def _instance_has_valid_span(
        instance,
        max_seq_len=0,
        *,
        enforce_limit=False,
    ):
        if instance.get(NODE_KEEP_KEY, False) and not enforce_limit:
            return True
        if instance.get(TYPED_VALUES_KEY):
            # Class/quantity supervision does not depend on a visible span.
            return True
        for key, value in instance.items():
            if is_internal_instance_key(key):
                continue
            values = value if isinstance(value, list) else [value]
            for field_value in values:
                if not isinstance(field_value, dict):
                    continue
                st = field_value.get('start', -1)
                ed = field_value.get('end', -1)
                if st < 0 or ed < st:
                    continue
                if (
                    (enforce_limit or max_seq_len > 0)
                    and (st >= max_seq_len or ed >= max_seq_len)
                ):
                    continue
                return True
        return False

    @staticmethod
    def _normalize_sequence_lengths(batch_list, sequence_lengths, max_seq_len):
        """Validate optional per-item source lengths retained by tokenization."""

        if sequence_lengths is None:
            return None, int(max_seq_len)
        if torch.is_tensor(sequence_lengths):
            values = sequence_lengths.detach().reshape(-1).cpu().tolist()
        else:
            values = list(sequence_lengths)
        if len(values) != len(batch_list):
            raise ValueError(
                "sequence_lengths must contain one value per batch item"
            )
        normalized = []
        for value in values:
            if isinstance(value, bool) or int(value) != value or value < 0:
                raise ValueError(
                    "sequence_lengths values must be non-negative integers"
                )
            normalized.append(int(value))
        return normalized, max(normalized, default=0)

    def _filter_instances(
        self,
        item,
        data_key,
        instances,
        max_seq_len,
        *,
        enforce_limit=False,
    ):
        """Keep visible records and, for hierarchies, their ancestors."""

        visible = [
            instance
            for instance in instances
            if self._instance_has_valid_span(
                instance,
                max_seq_len,
                enforce_limit=enforce_limit,
            )
        ]
        if not enforce_limit or not self.multi_level:
            return visible

        multi_meta = item.get(self.meta_key) or {}
        group_meta = next(
            (
                group
                for group in multi_meta.get('groups', [])
                if group.get('data_key') == data_key
            ),
            {},
        )
        retained_node_ids = {
            instance.get(NODE_ID_KEY)
            for instance in visible
            if instance.get(NODE_ID_KEY) is not None
        }
        changed = True
        relations = group_meta.get('relations', [])
        while changed:
            changed = False
            for parent_node, child_node in relations:
                if (
                    child_node in retained_node_ids
                    and parent_node not in retained_node_ids
                ):
                    retained_node_ids.add(parent_node)
                    changed = True

        visible_object_ids = {id(instance) for instance in visible}
        return [
            instance
            for instance in instances
            if id(instance) in visible_object_ids
            or instance.get(NODE_ID_KEY) in retained_node_ids
        ]

    @staticmethod
    def _iter_field_values(value):
        """Flatten nested value arrays into groundable scalar occurrences."""

        if isinstance(value, list):
            for child in value:
                yield from StructuringProcessor._iter_field_values(child)
        else:
            yield value

    def get_classes_mapping(self, batch_list, shuffle_labels=False, **kwargs):
        structuring_mapping = []
        for item in batch_list:
            self._ensure_task_payload(item)
            self.component.normalize_item(item)
            self._prepare_typed_instances(item)
            structuring_data = item.get(self.data_key, {})
            structuring_schema = item.get(self.schema_key) or {}
            multi_meta = item.get(self.meta_key) or {}
            meta_groups = {
                group.get('data_key'): group
                for group in multi_meta.get('groups', [])
            }
            prompt = item.get('prompt')
            item_mappings = []
            for data_key, instances in structuring_data.items():
                group_meta = meta_groups.get(data_key, {})
                schema_name = group_meta.get('name', data_key)
                field_names = []
                seen = set()
                # Inference stubs and explicitly supplied schemas carry the
                # complete field list separately from their (empty) values.
                # Prefer that list so invalid training annotations can be
                # pruned without making inference depend on placeholder
                # strings surviving span resolution.
                schema_field_names = self._structure_fields(
                    structuring_schema.get(data_key, [])
                )
                if not schema_field_names and not self.multi_level:
                    # Declared typed fields describe the schema: keep their
                    # order so training prompts match inference prompts.
                    schema_field_names = list(
                        self._raw_group_types(item, data_key, schema_name)
                    )
                for field_name in schema_field_names:
                    if field_name not in seen:
                        field_names.append(field_name)
                        seen.add(field_name)
                for instance in instances:
                    for field_name in instance:
                        if is_internal_instance_key(field_name):
                            continue
                        if field_name not in seen:
                            field_names.append(field_name)
                            seen.add(field_name)
                raw_types = self._raw_group_types(item, data_key, schema_name)
                typed_names = [
                    *raw_types,
                    *(
                        label
                        for instance in instances
                        for label in (instance.get(TYPED_VALUES_KEY) or {})
                    ),
                ]
                for field_name in typed_names:
                    if field_name not in seen and (
                        field_name in raw_types
                    ):
                        field_names.append(field_name)
                        seen.add(field_name)
                field_types = self._mapping_field_types(
                    raw_types,
                    training=shuffle_labels,
                )
                if shuffle_labels:
                    random.shuffle(field_names)
                field_class_to_id = {name: idx for idx, name in enumerate(field_names)}
                if isinstance(prompt, dict):
                    description = prompt.get(schema_name)
                elif isinstance(prompt, str):
                    description = prompt
                else:
                    description = None
                if not isinstance(description, str) or not description.strip():
                    description = None
                else:
                    description = description.strip()
                item_mappings.append(StructuringItemMapping(
                    field_class_to_id=BaseClassMapping(
                        class_to_id=field_class_to_id,
                        name=schema_name,
                        description=description,
                    ),
                    name=schema_name,
                    description=description,
                    data_key=data_key,
                    hierarchy=deepcopy(group_meta.get('hierarchy') or []),
                    multi_level=bool(group_meta),
                    field_types=field_types,
                ))
            reference = item.get("reference_date")
            structuring_mapping.append(StructuringClassMapping(
                items=item_mappings,
                output_mode=multi_meta.get('output_mode', 'schemas'),
                multi_level=bool(multi_meta),
                reference_date=str(reference) if reference is not None else None,
            ))
        return structuring_mapping

    # -- typed fields -------------------------------------------------------

    def _spec(self, raw):
        """Cached :class:`TypedFieldSpec` for a descriptor dictionary."""

        if isinstance(raw, TypedFieldSpec):
            return raw
        key = json.dumps(raw, sort_keys=True, default=str)
        spec = self._typed_spec_cache.get(key)
        if spec is None:
            spec = TypedFieldSpec.from_value(raw)
            if len(self._typed_spec_cache) > 4096:
                self._typed_spec_cache.clear()
            self._typed_spec_cache[key] = spec
        return spec

    def _raw_group_types(self, item, data_key, schema_name=None):
        """Typed descriptors (label -> dict) declared for one schema group."""

        if not self.typed_enabled:
            return {}
        multi_meta = item.get(self.meta_key) or {}
        for group in multi_meta.get('groups', []):
            if group.get('data_key') == data_key:
                return {
                    label: (spec.to_dict() if isinstance(spec, TypedFieldSpec) else spec)
                    for label, spec in (group.get('field_types') or {}).items()
                }
        raw = lookup_group_types(
            item.get(STRUCTURING_TYPES_KEY),
            names=tuple(
                name for name in (schema_name, data_key, MULTI_LEVEL_ROOT_KEY)
                if name is not None
            ),
        )
        return {
            str(label): self._spec(spec).to_dict()
            for label, spec in raw.items()
        }

    def _mapping_field_types(self, raw_types, *, training):
        """Copy typed descriptors, applying training-time prompt augmentation."""

        result = {}
        dropout = float(getattr(self.task_config, "typed_description_dropout", 0.0))
        shuffle_options = bool(getattr(self.task_config, "typed_shuffle_options", False))
        for label, raw in raw_types.items():
            spec = dict(raw)
            if training:
                if spec.get("$description") and random.random() < dropout:
                    spec.pop("$description", None)
                if shuffle_options and spec.get("$options"):
                    options = list(spec["$options"])
                    random.shuffle(options)
                    spec["$options"] = options
            result[label] = spec
        return result

    def _prepare_typed_instances(self, item):
        """Move class/quantity values of flat records beside their spans.

        Multi-level normalization already stores typed values under
        ``TYPED_VALUES_KEY``; flat records still carry them as ordinary keys.
        """

        if not self.typed_enabled or self.multi_level or item.get(_TYPED_PREPARED_FLAG):
            return
        structuring = item.get(self.data_key)
        if not isinstance(structuring, dict):
            return
        for data_key, instances in structuring.items():
            if not isinstance(instances, list):
                continue
            typed_labels = {
                label
                for label, raw in self._raw_group_types(item, data_key, data_key).items()
                if self._spec(raw).kind != "span"
            }
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                for label in typed_labels:
                    # Label items are rebuilt after span resolution; by then a
                    # quantity key holds its anchor span, not the gold value.
                    if label not in instance or label in (
                        instance.get(TYPED_VALUES_KEY) or {}
                    ):
                        continue
                    value = instance.pop(label)
                    if not is_typed_descriptor(value):
                        instance.setdefault(TYPED_VALUES_KEY, {})[label] = value
        item[_TYPED_PREPARED_FLAG] = True

    def _group_evidence(self, item, data_key, schema_name):
        evidence = item.get(STRUCTURING_EVIDENCE_KEY)
        if not isinstance(evidence, dict):
            return {}
        for name in (schema_name, data_key, MULTI_LEVEL_ROOT_KEY):
            group = evidence.get(name)
            if isinstance(group, dict):
                return group
        return evidence

    def _attach_quantity_anchors(self, item, text):
        """Tag the span a quantity was read from as that field's span target.

        A value copied from the text becomes an anchor with zero offset; the
        quantity head then only has to learn the offset from it.
        """

        structuring = item.get(self.data_key, {})
        multi_meta = item.get(self.meta_key) or {}
        names = {
            group.get('data_key'): group.get('name')
            for group in multi_meta.get('groups', [])
        }
        for data_key, instances in structuring.items():
            schema_name = names.get(data_key, data_key)
            specs = {
                label: self._spec(raw)
                for label, raw in self._raw_group_types(item, data_key, schema_name).items()
            }
            if not specs:
                continue
            evidence = self._group_evidence(item, data_key, schema_name)
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                for label, gold in (instance.get(TYPED_VALUES_KEY) or {}).items():
                    spec = specs.get(label)
                    if (
                        spec is None
                        or spec.kind != QUANTITY_KIND
                        or spec.bounded
                        or spec.horizon > 1
                    ):
                        continue
                    anchor = None
                    explicit = gold.get("anchor") if isinstance(gold, dict) else None
                    point = (
                        gold.get("value", gold.get("median"))
                        if isinstance(gold, dict)
                        else gold
                    )
                    target_x = spec.encode(point) if not isinstance(point, list) else None
                    if isinstance(explicit, dict) and "start" in explicit and "end" in explicit:
                        start, end = int(explicit["start"]), int(explicit["end"])
                        anchor_text = text[start:end]
                        anchor_x = spec.parse_text(anchor_text)
                        if anchor_x is not None:
                            anchor = {
                                "text": anchor_text,
                                "start": start,
                                "end": end,
                                "x": anchor_x,
                            }
                    elif target_x is not None:
                        snippets = evidence.get(label, ()) if isinstance(evidence, dict) else ()
                        anchor = find_anchor(
                            text,
                            spec,
                            target_x,
                            snippets=snippets if isinstance(snippets, list) else (),
                        )
                    if anchor is not None:
                        instance[label] = {
                            "text": anchor["text"],
                            "start": anchor["start"],
                            "end": anchor["end"],
                        }
                        instance.setdefault(TYPED_ANCHORS_KEY, {})[label] = float(anchor["x"])
                    elif target_x is not None:
                        unresolved = instance.setdefault(TYPED_UNRESOLVED_KEY, [])
                        if label not in unresolved:
                            unresolved.append(label)

    def _attach_list_item_spans(self, item, text):
        """Ground number/date items of typed list fields on their source spans.

        Lists such as ``{"$type": "list", "$items": "date"}`` hold normalized
        values ("2024-03-03") that rarely occur verbatim. Each item is mapped
        to the span it was read from; items without one mark the field so its
        span loss is not trained as "nothing here".
        """

        structuring = item.get(self.data_key, {})
        multi_meta = item.get(self.meta_key) or {}
        names = {
            group.get('data_key'): group.get('name')
            for group in multi_meta.get('groups', [])
        }
        for data_key, instances in structuring.items():
            schema_name = names.get(data_key, data_key)
            list_specs = {}
            for label, raw in self._raw_group_types(item, data_key, schema_name).items():
                spec = self._spec(raw)
                if spec.type == "list" and spec.items in {
                    "number", "date", "time", "datetime", "duration",
                }:
                    list_specs[label] = TypedFieldSpec(
                        type=spec.items, unit=spec.unit, integer=spec.integer,
                        decimals=spec.decimals,
                    )
            if not list_specs:
                continue
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                for label, item_spec in list_specs.items():
                    values = instance.get(label)
                    if not isinstance(values, list):
                        continue
                    grounded = []
                    missing = False
                    cursor = 0
                    for value in values:
                        if isinstance(value, dict) or (
                            isinstance(value, str) and value and value in text
                        ):
                            grounded.append(value)
                            continue
                        target_x = item_spec.encode(value)
                        anchor = (
                            find_anchor(text[cursor:], item_spec, target_x, allow_offset=False)
                            if target_x is not None
                            else None
                        )
                        if anchor is None:
                            missing = True
                            continue
                        grounded.append({
                            "text": anchor["text"],
                            "start": cursor + anchor["start"],
                            "end": cursor + anchor["end"],
                        })
                        cursor += anchor["end"]
                    instance[label] = grounded
                    if missing:
                        unresolved = instance.setdefault(TYPED_UNRESOLVED_KEY, [])
                        if label not in unresolved:
                            unresolved.append(label)

    def typed_layout(self, struct_item):
        """Class and quantity fields of one group in prompt order.

        Returns ``(class_fields, quantity_fields)``; each entry is
        ``(label, field_id, TypedFieldSpec)``.
        """

        if not self.typed_enabled:
            return [], []
        return typed_field_layout(struct_item, spec_factory=self._spec)

    def prepare_typed_inputs(self, classes_mapping, use_labels_encoder=False):
        """Label-free tensors the typed heads need at training and inference."""

        if not self.typed_enabled:
            return {}
        mapping_list = self._mapping_list(classes_mapping)
        layouts = [
            (batch_idx, *self.typed_layout(struct_item))
            for _flat, batch_idx, _group, struct_item in self._flat_structuring_iter(
                classes_mapping
            )
        ]
        if not any(class_fields or quantity_fields for _, class_fields, quantity_fields in layouts):
            return {}
        total_groups = len(layouts)
        batch_size = len(mapping_list)
        max_class = max((len(c) for _, c, _ in layouts), default=0)
        max_options = max(
            (len(spec.options) for _, c, _ in layouts for _, _, spec in c),
            default=0,
        )
        max_quantity = max((len(q) for _, _, q in layouts), default=0)
        tensors = {}
        if max_class > 0 and max_options > 0:
            class_field_idx = torch.zeros(total_groups, max_class, dtype=torch.long)
            class_field_mask = torch.zeros(total_groups, max_class, dtype=torch.bool)
            class_multi = torch.zeros(total_groups, max_class, dtype=torch.bool)
            option_index = torch.zeros(total_groups, max_class, max_options, dtype=torch.long)
            option_mask = torch.zeros(total_groups, max_class, max_options, dtype=torch.bool)
            option_count = torch.zeros(batch_size, dtype=torch.long)
            for group_idx, (batch_idx, class_fields, _) in enumerate(layouts):
                for slot, (_label, field_id, spec) in enumerate(class_fields):
                    class_field_idx[group_idx, slot] = field_id
                    class_field_mask[group_idx, slot] = not use_labels_encoder
                    class_multi[group_idx, slot] = spec.multi
                    for option_slot in range(len(spec.options)):
                        option_index[group_idx, slot, option_slot] = option_count[batch_idx]
                        option_mask[group_idx, slot, option_slot] = not use_labels_encoder
                        option_count[batch_idx] += 1
            if use_labels_encoder:
                option_count.zero_()
            tensors.update({
                "structuring_typed_class_field_idx": class_field_idx,
                "structuring_typed_class_field_mask": class_field_mask,
                "structuring_typed_class_multi": class_multi,
                "structuring_typed_option_index": option_index,
                "structuring_typed_option_mask": option_mask,
                "structuring_typed_option_count": option_count,
            })
        if max_quantity > 0:
            quantity_field_idx = torch.zeros(total_groups, max_quantity, dtype=torch.long)
            quantity_field_mask = torch.zeros(total_groups, max_quantity, dtype=torch.bool)
            quantity_horizon = torch.ones(total_groups, max_quantity, dtype=torch.long)
            quantity_bounded = torch.zeros(total_groups, max_quantity, dtype=torch.bool)
            quantity_reference = torch.zeros(total_groups, max_quantity, dtype=torch.bool)
            for group_idx, (batch_idx, _, quantity_fields) in enumerate(layouts):
                reference = getattr(mapping_list[batch_idx], "reference_date", None)
                for slot, (_label, field_id, spec) in enumerate(quantity_fields):
                    quantity_field_idx[group_idx, slot] = field_id
                    quantity_field_mask[group_idx, slot] = True
                    quantity_horizon[group_idx, slot] = spec.horizon
                    quantity_bounded[group_idx, slot] = spec.bounded
                    quantity_reference[group_idx, slot] = (
                        not spec.bounded
                        and spec.reference_origin(reference) is not None
                    )
            tensors.update({
                "structuring_typed_quantity_field_idx": quantity_field_idx,
                "structuring_typed_quantity_field_mask": quantity_field_mask,
                "structuring_typed_quantity_horizon": quantity_horizon,
                "structuring_typed_quantity_bounded": quantity_bounded,
                "structuring_typed_quantity_reference": quantity_reference,
            })
        return tensors

    @staticmethod
    def _hierarchy_field_label(field):
        return field.get("label") if isinstance(field, dict) else field

    def _apply_augmented_field_labels(self, struct_item, labels):
        """Apply fields while preserving the inline hierarchy prompt contract."""

        labels = list(dict.fromkeys(labels))
        mapping = struct_item.field_class_to_id
        hierarchy = list(getattr(struct_item, "hierarchy", None) or [])
        if not struct_item.multi_level or not hierarchy:
            mapping.class_to_id = {
                label: index for index, label in enumerate(labels)
            }
            return

        label_order = {label: index for index, label in enumerate(labels)}
        represented = set()
        root = None
        for node in hierarchy:
            if tuple(node.get("path") or ()) == ():
                root = node
            retained = []
            for field in node.get("fields") or []:
                label = self._hierarchy_field_label(field)
                if label in label_order:
                    retained.append(field)
                    represented.add(label)
            retained.sort(
                key=lambda field: label_order[
                    self._hierarchy_field_label(field)
                ]
            )
            node["fields"] = retained

        if root is None:
            root = {
                "path": [],
                "parent_path": None,
                "parent_field_path": [],
                "fields": [],
                "containers": [],
            }
            hierarchy.insert(0, root)
            struct_item.hierarchy = hierarchy

        for label in labels:
            if label in represented:
                continue
            root["fields"].append({
                "label": label,
                "path": [label],
                "local_path": [label],
                "shape": {"kind": "scalar"},
            })
        root["fields"].sort(
            key=lambda field: label_order[
                self._hierarchy_field_label(field)
            ]
        )

        by_path = {
            tuple(node.get("path") or ()): node for node in hierarchy
        }
        children = {}
        for node in hierarchy:
            parent_path = node.get("parent_path")
            if parent_path is not None:
                children.setdefault(tuple(parent_path), []).append(node)

        rendered = []
        visited = set()

        def visit(path):
            if path in visited:
                return
            visited.add(path)
            node = by_path.get(path)
            if node is None:
                return
            rendered.extend(
                self._hierarchy_field_label(field)
                for field in node.get("fields") or []
            )
            for child in children.get(path, []):
                visit(tuple(child.get("path") or ()))

        visit(())
        mapping.class_to_id = {
            label: index
            for index, label in enumerate(dict.fromkeys(rendered))
        }

    def get_augmentable_label_groups(self, batch_list, classes_mapping):
        groups = []
        for batch_idx, item_mapping in enumerate(
            self._mapping_list(classes_mapping)
        ):
            structuring_data = batch_list[batch_idx].get(self.data_key, {})
            for group_idx, struct_item in enumerate(item_mapping.items):
                mapping = struct_item.field_class_to_id
                if not mapping.class_to_id:
                    continue
                data_key = struct_item.data_key or struct_item.name
                positives = list(dict.fromkeys(
                    [
                        field_name
                        for instance in structuring_data.get(data_key, [])
                        for field_name in instance
                        if not is_internal_instance_key(field_name)
                    ]
                    + [
                        field_name
                        for instance in structuring_data.get(data_key, [])
                        for field_name in (instance.get(TYPED_VALUES_KEY) or {})
                    ]
                ))

                def apply_labels(labels, struct_item=struct_item):
                    self._apply_augmented_field_labels(
                        struct_item,
                        labels,
                    )

                groups.append(AugmentableLabelGroup(
                    task=self.task_name,
                    batch_idx=batch_idx,
                    group_idx=group_idx,
                    mapping=mapping,
                    positive_labels=positives,
                    parent_name=mapping.name,
                    apply_labels=apply_labels,
                ))
        return groups

    def contribute_prompt(self, classes_mapping, batch_idx, use_labels_encoder=False):
        mapping_list = self._mapping_list(classes_mapping)
        if batch_idx >= len(mapping_list):
            return []
        prompt = []
        for struct_item in mapping_list[batch_idx].items:
            prompt.extend(self.component.contribute_prompt(
                struct_item,
                parent_token=self.parent_token,
                field_token=self.child_token,
                sep_token=self.sep_token,
                use_labels_encoder=use_labels_encoder,
                option_token=self.option_token if self.typed_enabled else None,
            ))
        return prompt

    @staticmethod
    def _structure_fields(fields):
        if isinstance(fields, list):
            return fields
        if isinstance(fields, dict):
            return fields.get("fields", [])
        return []

    def contribute_inference_input(self, item, structures=None, **kwargs):
        if structures is None:
            return
        self.component.contribute_inference_input(item, structures)

    def empty_inference_result(self, num_texts: int, structures=None, **kwargs):
        if structures is None:
            return None
        root_spec = (
            structures.get(MULTI_LEVEL_ROOT_KEY)
            if isinstance(structures, dict)
            and set(structures) == {MULTI_LEVEL_ROOT_KEY}
            else None
        )
        is_root_list = self.multi_level and (
            isinstance(structures, list) or isinstance(root_spec, list)
        )

        def empty_values():
            return [([] if is_root_list else {}) for _ in range(num_texts)]

        return {self.task_name: empty_values()}

    def _normalize_field_value(
        self,
        text,
        tokens_with_spans,
        field_name,
        value,
        first_only=True,
        used_spans=None,
    ):
        value_text = value.get('text') if isinstance(value, dict) else value
        if value_text is None and isinstance(value, dict) and 'start' in value and 'end' in value:
            try:
                value_text = text[int(value['start']):int(value['end'])]
            except (TypeError, ValueError):
                value_text = ''

        spans = self._resolve_labeled_span(
            text,
            tokens_with_spans,
            value,
            label=field_name,
            # Resolve all candidates here so repeated scalar values can be
            # assigned to successive records instead of copied onto every
            # record containing that value.
            first_only=False,
        )
        if used_spans is not None:
            spans = [
                span for span in spans
                if (span[0], span[1]) not in used_spans
            ]
        if first_only:
            spans = spans[:1]
        if not spans:
            return []

        if used_spans is not None:
            used_spans.update((start, end) for start, end, _ in spans)

        return [
            {
                'text': str(value_text),
                'start': start,
                'end': end,
            }
            for start, end, _ in spans
        ]

    def resolve_spans(self, item):
        self._ensure_task_payload(item)
        self.component.normalize_item(item)
        if item.get(self.resolved_flag):
            return
        structuring = item.get(self.data_key, {})
        if not structuring:
            return

        text = item.get('text', '')
        tokens_with_spans, _ = self._tokenize_text(item)
        if tokens_with_spans is None:
            return
        if self.typed_enabled:
            self._prepare_typed_instances(item)
            if not text and item.get("tokenized_text"):
                text = " ".join(item["tokenized_text"])
            self._attach_quantity_anchors(item, text)
            self._attach_list_item_spans(item, text)

        for schema_name, instances in list(structuring.items()):
            used_spans_by_field = {}
            resolved_instances = []
            for instance in instances:
                if not isinstance(instance, dict):
                    continue
                for field_name, value in list(instance.items()):
                    if is_internal_instance_key(field_name):
                        continue
                    used_spans = used_spans_by_field.setdefault(field_name, set())
                    if isinstance(value, list):
                        resolved_list = []
                        for v in self._iter_field_values(value):
                            resolved_list.extend(self._normalize_field_value(
                                text,
                                tokens_with_spans,
                                field_name,
                                v,
                                used_spans=used_spans,
                            ))
                        if resolved_list:
                            instance[field_name] = resolved_list
                        else:
                            # An annotated value that cannot be grounded must
                            # not become an all-zero (false-negative) target.
                            instance.pop(field_name, None)
                    else:
                        normalized = self._normalize_field_value(
                            text,
                            tokens_with_spans,
                            field_name,
                            value,
                            used_spans=used_spans,
                        )
                        if normalized:
                            instance[field_name] = normalized[0]
                        else:
                            instance.pop(field_name, None)
                anchors = instance.get(TYPED_ANCHORS_KEY)
                if anchors:
                    for label in list(anchors):
                        if label not in instance:
                            # The anchor text could not be mapped to tokens.
                            anchors.pop(label)
                            unresolved = instance.setdefault(TYPED_UNRESOLVED_KEY, [])
                            if label not in unresolved:
                                unresolved.append(label)
                if any(
                    not is_internal_instance_key(key)
                    for key in instance
                ) or instance.get(NODE_KEEP_KEY, False) or instance.get(
                    TYPED_VALUES_KEY
                ):
                    resolved_instances.append(instance)
            structuring[schema_name] = resolved_instances
        item[self.resolved_flag] = True

    def create_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        sequence_lengths, max_seq_len = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("sequence_lengths"),
            max_seq_len,
        )
        source_sequence_lengths, _ = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("source_sequence_lengths"),
            max_seq_len,
        )
        for item in batch_list:
            self.resolve_spans(item)

        total_groups = self._total_structuring_groups(classes_mapping)
        if total_groups == 0:
            return None

        max_instances = 0
        max_fields = 0
        has_schema = False

        for _flat_idx, batch_idx, _group_idx, struct_item in self._flat_structuring_iter(classes_mapping):
            structuring_data = batch_list[batch_idx].get(self.data_key, {})
            data_key = struct_item.data_key or struct_item.name
            if data_key not in structuring_data:
                continue
            instances = structuring_data[data_key]
            if instances:
                has_schema = True
                item_limit = (
                    sequence_lengths[batch_idx]
                    if sequence_lengths is not None
                    else max_seq_len
                )
                enforce_limit = sequence_lengths is not None and (
                    source_sequence_lengths is None
                    or item_limit < source_sequence_lengths[batch_idx]
                )
                valid_instances = self._filter_instances(
                    batch_list[batch_idx],
                    data_key,
                    instances,
                    item_limit,
                    enforce_limit=enforce_limit,
                )
                max_instances = max(max_instances, len(valid_instances))
                max_fields = max(max_fields, len(struct_item.field_class_to_id.class_to_id))

        if not has_schema or max_fields == 0:
            return None
        if max_instances == 0:
            max_instances = 1

        if self._fixed_slot_pad:
            max_instances = max(max_instances, self._fixed_slot_pad)

        structuring_labels = torch.zeros(
            total_groups, max_instances, max_seq_len, max_fields, 3,
            dtype=torch.float,
        )
        structuring_mask = torch.zeros(total_groups, dtype=torch.bool)
        structuring_batch_idx = torch.zeros(total_groups, dtype=torch.long)
        structuring_count = torch.zeros(total_groups, dtype=torch.long)
        structuring_relation_labels = (
            torch.zeros(
                total_groups, max_instances, max_instances, dtype=torch.float,
            )
            if self.multi_level
            else None
        )
        structuring_relation_group_mask = (
            torch.zeros(total_groups, dtype=torch.bool)
            if self.multi_level
            else None
        )

        group_instances = {}
        for flat_idx, batch_idx, _group_idx, struct_item in self._flat_structuring_iter(classes_mapping):
            structuring_batch_idx[flat_idx] = batch_idx
            structuring_data = batch_list[batch_idx].get(self.data_key, {})
            data_key = struct_item.data_key or struct_item.name
            if data_key not in structuring_data:
                continue

            item_limit = (
                sequence_lengths[batch_idx]
                if sequence_lengths is not None
                else max_seq_len
            )
            enforce_limit = sequence_lengths is not None and (
                source_sequence_lengths is None
                or item_limit < source_sequence_lengths[batch_idx]
            )
            instances = self._filter_instances(
                batch_list[batch_idx],
                data_key,
                structuring_data[data_key],
                item_limit,
                enforce_limit=enforce_limit,
            )
            instances = sorted(instances, key=self._instance_sort_key)
            group_instances[flat_idx] = (instances[:max_instances], item_limit)
            field_to_id = struct_item.field_class_to_id.class_to_id
            structuring_mask[flat_idx] = True
            structuring_count[flat_idx] = len(instances)
            if structuring_relation_group_mask is not None:
                # A root-only multi-level group has no meaningful directed
                # hierarchy to supervise. In particular, repeated flat records
                # must not turn every fixed-slot pair into an all-negative
                # relation target merely because they share the multi-level
                # processor with genuinely nested examples.
                hierarchy = list(getattr(struct_item, 'hierarchy', None) or [])
                structuring_relation_group_mask[flat_idx] = bool(
                    struct_item.multi_level
                    and any(
                        node.get('parent_path') is not None
                        for node in hierarchy
                        if isinstance(node, dict)
                    )
                )

            if struct_item.multi_level:
                multi_meta = batch_list[batch_idx].get(self.meta_key) or {}
                group_meta = next(
                    (
                        group for group in multi_meta.get('groups', [])
                        if group.get('data_key') == data_key
                    ),
                    {},
                )
                node_to_instance = {
                    instance.get(NODE_ID_KEY): inst_idx
                    for inst_idx, instance in enumerate(instances[:max_instances])
                    if instance.get(NODE_ID_KEY) is not None
                }
                for parent_node, child_node in group_meta.get('relations', []):
                    parent_idx = node_to_instance.get(parent_node)
                    child_idx = node_to_instance.get(child_node)
                    if parent_idx is not None and child_idx is not None:
                        structuring_relation_labels[
                            flat_idx, parent_idx, child_idx
                        ] = 1.0

            for inst_idx, instance in enumerate(instances):
                if inst_idx >= max_instances:
                    break
                for field_name, value in instance.items():
                    if is_internal_instance_key(field_name):
                        continue
                    if field_name not in field_to_id:
                        continue
                    field_id = field_to_id[field_name]
                    if field_id >= max_fields:
                        continue
                    values = value if isinstance(value, list) else [value]
                    for field_value in values:
                        if not isinstance(field_value, dict):
                            continue
                        st = field_value.get('start', -1)
                        ed = field_value.get('end', -1)
                        if (
                            st < 0
                            or ed < st
                            or st >= item_limit
                            or ed >= item_limit
                        ):
                            continue
                        structuring_labels[flat_idx, inst_idx, st, field_id, 0] = 1.0
                        structuring_labels[flat_idx, inst_idx, ed, field_id, 1] = 1.0
                        structuring_labels[flat_idx, inst_idx, st:ed + 1, field_id, 2] = 1.0

        result = {
            self._tensor_key("labels"): structuring_labels,
            self._tensor_key("mask"): structuring_mask,
            self._tensor_key("batch_idx"): structuring_batch_idx,
            self._tensor_key("count"): structuring_count,
        }
        if structuring_relation_labels is not None:
            result[self._tensor_key("relation_labels")] = structuring_relation_labels
            result[self._tensor_key("relation_group_mask")] = (
                structuring_relation_group_mask
            )
        if self.typed_enabled:
            result.update(self._create_typed_labels(
                classes_mapping,
                group_instances,
                total_groups=total_groups,
                max_instances=max_instances,
                max_fields=max_fields,
            ))
        return result

    @staticmethod
    def _class_gold_labels(gold):
        if gold is None:
            return []
        if isinstance(gold, (list, tuple)):
            return [str(value) for value in gold if value is not None]
        return [str(gold)]

    def _create_typed_labels(
        self,
        classes_mapping,
        group_instances,
        *,
        total_groups,
        max_instances,
        max_fields,
    ):
        """Targets for class and quantity fields on the gold-record axis.

        Shapes (G groups, R gold records, Cc/Cq typed fields, Kc options,
        H horizon steps, P quantile points):

        * ``typed_class_targets``   (G, R, Cc, Kc) multi-hot option targets
        * ``typed_class_unit_mask`` (G, R, Cc) record/field pairs with a label
        * ``typed_quantity_levels`` / ``typed_quantity_values``
          (G, R, Cq, H, P) quantile levels and their unit-space values
        * ``typed_quantity_point``  (G, R, Cq, H) unit-space point value
        * ``typed_quantity_presence`` (G, R, Cq) 1 present, 0 null
        * ``typed_quantity_origin`` (G, R, Cq) absolute/reference/anchor mode
        * ``typed_bio_field_mask``  (G, C) false where a value may sit in the
          text in a form the codec could not find (do not train "no span")
        """

        mapping_list = self._mapping_list(classes_mapping)
        layouts = {}
        field_maps = {}
        for flat_idx, batch_idx, _group, struct_item in self._flat_structuring_iter(
            classes_mapping
        ):
            layouts[flat_idx] = (batch_idx, *self.typed_layout(struct_item))
            field_maps[flat_idx] = struct_item.field_class_to_id.class_to_id
        max_class = max((len(c) for _, c, _ in layouts.values()), default=0)
        max_options = max(
            (len(spec.options) for _, c, _ in layouts.values() for _, _, spec in c),
            default=0,
        )
        max_quantity = max((len(q) for _, _, q in layouts.values()), default=0)
        max_horizon = max(
            (spec.horizon for _, _, q in layouts.values() for _, _, spec in q),
            default=1,
        )

        bio_field_mask = torch.ones(total_groups, max(max_fields, 1), dtype=torch.bool)
        class_targets = torch.zeros(
            total_groups, max_instances, max(max_class, 1), max(max_options, 1)
        )
        class_unit_mask = torch.zeros(
            total_groups, max_instances, max(max_class, 1), dtype=torch.bool
        )
        quantity_entries = []
        max_points = 1

        for flat_idx, (instances, item_limit) in group_instances.items():
            batch_idx, class_fields, quantity_fields = layouts[flat_idx]
            reference = getattr(mapping_list[batch_idx], "reference_date", None)
            for inst_idx, instance in enumerate(instances):
                typed_values = instance.get(TYPED_VALUES_KEY) or {}
                for label in instance.get(TYPED_UNRESOLVED_KEY) or ():
                    field_id = field_maps[flat_idx].get(label)
                    if field_id is not None and field_id < bio_field_mask.shape[1]:
                        bio_field_mask[flat_idx, field_id] = False
                for slot, (label, _field_id, spec) in enumerate(class_fields):
                    if label not in typed_values:
                        continue
                    gold = self._class_gold_labels(typed_values[label])
                    option_ids = {option: index for index, option in enumerate(spec.options)}
                    known = [option_ids[value] for value in gold if value in option_ids]
                    if gold and not known:
                        continue
                    class_unit_mask[flat_idx, inst_idx, slot] = True
                    for option_id in known:
                        class_targets[flat_idx, inst_idx, slot, option_id] = 1.0
                for slot, (label, _field_id, spec) in enumerate(quantity_fields):
                    if label not in typed_values:
                        continue
                    gold = typed_values[label]
                    anchor_x = None
                    anchor = instance.get(label)
                    if (
                        label in (instance.get(TYPED_ANCHORS_KEY) or {})
                        and isinstance(anchor, dict)
                        and 0 <= anchor.get("start", -1) <= anchor.get("end", -1) < item_limit
                    ):
                        anchor_x = instance[TYPED_ANCHORS_KEY][label]
                    mode, origin = spec.resolve_origin(anchor_x, reference)
                    target = build_quantity_target(
                        gold,
                        spec,
                        origin_mode=mode,
                        origin=origin,
                    )
                    if target is None:
                        continue
                    max_points = max(
                        max_points,
                        max((len(points) for points in target.points), default=1),
                    )
                    quantity_entries.append((flat_idx, inst_idx, slot, target))

        result = {
            "structuring_typed_bio_field_mask": bio_field_mask,
        }
        if max_class > 0 and max_options > 0:
            result["structuring_typed_class_targets"] = class_targets
            result["structuring_typed_class_unit_mask"] = class_unit_mask
        if max_quantity > 0:
            shape = (total_groups, max_instances, max_quantity)
            levels = torch.zeros(*shape, max_horizon, max_points)
            values = torch.zeros(*shape, max_horizon, max_points)
            point_mask = torch.zeros(*shape, max_horizon, max_points, dtype=torch.bool)
            point_value = torch.zeros(*shape, max_horizon)
            point_value_mask = torch.zeros(*shape, max_horizon, dtype=torch.bool)
            presence = torch.zeros(*shape)
            presence_mask = torch.zeros(*shape, dtype=torch.bool)
            origin_mode = torch.zeros(*shape, dtype=torch.long)
            for flat_idx, inst_idx, slot, target in quantity_entries:
                presence_mask[flat_idx, inst_idx, slot] = True
                origin_mode[flat_idx, inst_idx, slot] = target.origin_mode
                if not target.present:
                    continue
                presence[flat_idx, inst_idx, slot] = 1.0
                for step, points in enumerate(target.points[:max_horizon]):
                    for point_idx, (level, u) in enumerate(points[:max_points]):
                        levels[flat_idx, inst_idx, slot, step, point_idx] = level
                        values[flat_idx, inst_idx, slot, step, point_idx] = u
                        point_mask[flat_idx, inst_idx, slot, step, point_idx] = True
                    point_u = target.point_u[step]
                    if point_u is not None:
                        point_value[flat_idx, inst_idx, slot, step] = point_u
                        point_value_mask[flat_idx, inst_idx, slot, step] = True
            result.update({
                "structuring_typed_quantity_levels": levels,
                "structuring_typed_quantity_values": values,
                "structuring_typed_quantity_point_mask": point_mask,
                "structuring_typed_quantity_point": point_value,
                "structuring_typed_quantity_point_value_mask": point_value_mask,
                "structuring_typed_quantity_presence": presence,
                "structuring_typed_quantity_presence_mask": presence_mask,
                "structuring_typed_quantity_origin": origin_mode,
            })
        return result

    def create_span_labels(self, batch_list, classes_mapping, max_seq_len=0, **kwargs):
        """Create span-level labels for structuring when represent_spans is enabled.

        Returns dict with:
            structuring_span_idx: (total_groups, max_spans, 2)
            structuring_span_labels: (total_groups, max_spans, max_instances, max_fields)
            structuring_span_mask: (total_groups, max_spans)
            structuring_span_batch_idx: (total_groups,)
        """
        struct_cfg = self._span_config
        if struct_cfg is None:
            return None

        sequence_lengths, max_seq_len = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("sequence_lengths"),
            max_seq_len,
        )
        source_sequence_lengths, _ = self._normalize_sequence_lengths(
            batch_list,
            kwargs.get("source_sequence_lengths"),
            max_seq_len,
        )

        for item in batch_list:
            self.resolve_spans(item)

        total_groups = self._total_structuring_groups(classes_mapping)
        if total_groups == 0:
            return None

        neg_ratio = getattr(struct_cfg, 'neg_spans_ratio', 1.0)
        max_instances = 0
        max_fields = 0
        has_any = False

        all_group_spans = []
        batch_indices = []

        for _flat_idx, batch_idx, _group_idx, struct_item in self._flat_structuring_iter(classes_mapping):
            structuring_data = batch_list[batch_idx].get(self.data_key, {})
            data_key = struct_item.data_key or struct_item.name
            field_to_id = struct_item.field_class_to_id.class_to_id
            max_fields = max(max_fields, len(field_to_id))
            batch_indices.append(batch_idx)

            # One pooled candidate per unique entity boundary. The value is a
            # set of (record, field) targets so multi-field/multi-record
            # annotations remain expressible without sending duplicate span
            # representations into the second stage.
            span_targets = {}
            item_limit = (
                sequence_lengths[batch_idx]
                if sequence_lengths is not None
                else max_seq_len
            )
            enforce_limit = sequence_lengths is not None and (
                source_sequence_lengths is None
                or item_limit < source_sequence_lengths[batch_idx]
            )

            if data_key in structuring_data:
                instances = self._filter_instances(
                    batch_list[batch_idx],
                    data_key,
                    structuring_data[data_key],
                    item_limit,
                    enforce_limit=enforce_limit,
                )
                instances = sorted(instances, key=self._instance_sort_key)
                max_instances = max(max_instances, len(instances))

                for inst_idx, instance in enumerate(instances):
                    for field_name, value in instance.items():
                        if is_internal_instance_key(field_name):
                            continue
                        if field_name not in field_to_id:
                            continue
                        field_id = field_to_id[field_name]
                        values = value if isinstance(value, list) else [value]
                        for field_value in values:
                            if not isinstance(field_value, dict):
                                continue
                            st = field_value.get('start', -1)
                            ed = field_value.get('end', -1)
                            if 0 <= st <= ed < item_limit:
                                span_targets.setdefault((st, ed), set()).add(
                                    (inst_idx, field_id)
                                )
                                has_any = True

            group_spans = [
                (start, end, targets)
                for (start, end), targets in span_targets.items()
            ]
            neg_count = int(len(group_spans) * neg_ratio)
            if neg_count > 0 and item_limit > 0:
                negatives = self._generate_negative_spans(
                    set(span_targets),
                    item_limit,
                    neg_count,
                )
                for st, ed in negatives:
                    group_spans.append((st, ed, set()))

            all_group_spans.append(group_spans)

        if not has_any or max_instances == 0 or max_fields == 0:
            return None

        if self._fixed_slot_pad:
            max_instances = max(max_instances, self._fixed_slot_pad)

        max_spans = max((len(s) for s in all_group_spans), default=0)
        if max_spans == 0:
            return None

        span_idx = torch.zeros(total_groups, max_spans, 2, dtype=torch.long)
        span_labels = torch.zeros(total_groups, max_spans, max_instances, max_fields, dtype=torch.float)
        span_mask = torch.zeros(total_groups, max_spans, dtype=torch.bool)
        span_batch_idx = torch.tensor(batch_indices, dtype=torch.long)

        for g, group_spans in enumerate(all_group_spans):
            for s, (st, ed, targets) in enumerate(group_spans):
                span_idx[g, s, 0] = st
                span_idx[g, s, 1] = ed
                span_mask[g, s] = True
                for inst_idx, field_id in targets:
                    span_labels[g, s, inst_idx, field_id] = 1.0

        return {
            self._tensor_key("span_idx"): span_idx,
            self._tensor_key("span_labels"): span_labels,
            self._tensor_key("span_mask"): span_mask,
            self._tensor_key("span_batch_idx"): span_batch_idx,
        }

    def prepare_label_encoder_inputs(self, classes_mapping, labels_tokenizer):
        if labels_tokenizer is None:
            return None

        all_label_strings = []
        group_sizes = []
        has_any = False

        total_groups = self._total_structuring_groups(classes_mapping)
        if total_groups == 0:
            return None

        for _, _, _, struct_item in self._flat_structuring_iter(classes_mapping):
            labels = list(struct_item.field_class_to_id.class_to_id.keys())
            if labels:
                has_any = True
            group_sizes.append(len(labels))
            all_label_strings.extend(labels)

        if not has_any or not all_label_strings:
            return None

        tokenized = labels_tokenizer(
            all_label_strings, return_tensors="pt", truncation=True,
            padding="longest", add_special_tokens=True
        )
        label_prefix = "child_labels"
        return {
            f"{label_prefix}_input_ids": tokenized["input_ids"],
            f"{label_prefix}_attention_mask": tokenized["attention_mask"],
            f"{label_prefix}_group_size": torch.LongTensor(group_sizes),
        }


__all__ = [
    "INTERNAL_INSTANCE_KEYS",
    "MULTI_LEVEL_META_KEY",
    "MULTI_LEVEL_ROOT_KEY",
    "NODE_ID_KEY",
    "NODE_KEEP_KEY",
    "StructuringProcessor",
    "StructuringProcessorComponent",
    "is_internal_instance_key",
    "resolve_structuring_processor",
]
