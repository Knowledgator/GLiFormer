"""Joint NER + Relation Extraction head (GLiNER-relex style).

Inherits NER scoring from NERHead and adds relation scoring against [RELATION]
type embeddings. When enabled, the adjacency layer filters candidate pairs;
otherwise the head falls back to scoring all directed entity pairs, matching
the original GLiNER behavior.
"""


import warnings

import torch
from gliner.modeling.layers import create_projection_layer
from gliner.modeling.multitask.relations_layers import RelationsRepLayer
from gliner.modeling.multitask.triples_layers import TriplesScoreLayer
from gliner.modeling.span_rep import SpanRepLayer
from gliner.modeling.utils import (
    build_all_entity_pairs,
    build_entity_pairs,
    extract_prompt_features,
    extract_spans_from_tokens,
)
from torch.nn import functional as F

from ...layers import AnchorModeling, AnchorPairRelationsLayer
from .. import TaskHeadOutput
from ..losses import binary_focal_or_bce
from ..matcher import batched_masked_assignment
from ..ner.model import NERHead
from ..span_decoder import SpanDecoder


class JointRelexHead(NERHead):
    """Joint NER + relation extraction with optional adjacency pair selection.

    Inherits NER forward pass from NERHead, then:
    1. Selects entity spans from NER scores
    2. Optionally builds an adjacency matrix between entities
    3. Scores candidate entity pairs against [RELATION] type embeddings
    """

    name = "joint_relex"
    # When a standalone NER head is enabled, the orchestrator supplies its
    # already-computed output. Without one, Joint Relex runs its owned NER
    # pipeline and remains a self-contained head.
    dependencies = ["ner"]

    def __init__(
        self,
        config,
        hidden_size,
        dropout,
        shared_layers=None,
        ner_head=None,
    ):
        rel_cfg = config.joint_relex_config
        owns_ner_head = ner_head is None
        if owns_ner_head:
            # Preserve the historical state-dict layout for self-contained
            # Joint Relex checkpoints by initializing NER directly on this
            # module. The Joint config supplies fallback NER settings when a
            # standalone ner_config was explicitly disabled.
            NERHead.__init__(
                self,
                config,
                hidden_size,
                dropout,
                shared_layers=shared_layers,
                task_config=config.ner_config or rel_cfg,
            )
        else:
            # Do not register the same module below both heads.ner and
            # heads.joint_relex: that would duplicate checkpoint paths and
            # optimizer traversal. The execution dependency normally supplies
            # its cached output; this reference is only a direct-call fallback.
            torch.nn.Module.__init__(self)
            self.config = config
            self.__dict__["_reused_ner_head"] = ner_head
        self._owns_ner_head = owns_ner_head
        shared_layers = shared_layers or {}
        ner_cfg = config.ner_config or rel_cfg
        self.ner_loss_coef = getattr(ner_cfg, "loss_coef", 1.0)
        self.relation_loss_coef = rel_cfg.relation_loss_coef
        self.relation_loss_reduction = rel_cfg.relation_loss_reduction
        self.relation_focal_loss_alpha = rel_cfg.relation_focal_loss_alpha
        self.relation_focal_loss_gamma = rel_cfg.relation_focal_loss_gamma
        self.relation_focal_loss_prob_margin = (
            rel_cfg.relation_focal_loss_prob_margin
        )
        self.adjacency_loss_coef = rel_cfg.adjacency_loss_coef
        self.rel_token_index = rel_cfg.rel_token_index
        self.embed_rel_token = rel_cfg.embed_rel_token
        self.max_relation_span_width = rel_cfg.max_relation_span_width
        self.relation_span_nms = rel_cfg.relation_span_nms
        self.max_relation_entities = rel_cfg.max_relation_entities
        self.relation_top_k_neighbors = rel_cfg.relation_top_k_neighbors
        self.relation_neighbor_chunk_size = rel_cfg.relation_neighbor_chunk_size
        self._anchor_pair_capacity_warning_emitted = False
        # Owned mode combines NER and relation losses internally; reused mode
        # contributes relation loss only because NER is already a separate task.
        self.loss_coef = 1.0
        self.rel_span_rep_layer = SpanRepLayer(
            span_mode="token_level",
            hidden_size=hidden_size,
            max_width=getattr(config, "max_width", 12),
            dropout=dropout,
        )

        # Like GLiNER, the adjacency representation layer is optional.  The
        # joint head itself remains active and scores all directed pairs when it
        # is absent.
        if rel_cfg.relations_layer == "anchor_modeling":
            relation_anchor_layer = self._build_anchor_layer(
                rel_cfg,
                hidden_size,
                dropout,
            )
            relation_anchor_normalizer = self._build_anchor_normalizer(
                rel_cfg,
                hidden_size,
            )
            relation_anchor_modeling = shared_layers.get(
                "anchor_modeling"
            ) or AnchorModeling.from_config(
                rel_cfg.anchor_modeling,
                hidden_size,
                dropout=dropout,
            )
            relation_anchor_refinement = self._build_anchor_refinement(
                rel_cfg,
                hidden_size,
                dropout,
                shared_layers,
            )
            refinement_positions = None
            self_attention_bias = None
            cross_attention_bias = None
            if relation_anchor_refinement is not None:
                refinement_positions = self._build_refinement_positions(
                    rel_cfg,
                    hidden_size,
                )
                self_attention_bias, cross_attention_bias = (
                    self._build_refinement_biases(
                        rel_cfg,
                        num_heads=relation_anchor_refinement.num_heads,
                    )
                )
            groups_layer = getattr(
                relation_anchor_layer,
                "groups_layer",
                None,
            )
            max_anchors = int(
                getattr(
                    relation_anchor_layer,
                    "num_slots",
                    getattr(groups_layer, "max_count", rel_cfg.max_count),
                )
            )
            self.anchor_relations_layer = AnchorPairRelationsLayer(
                hidden_size=hidden_size,
                anchor_layer=relation_anchor_layer,
                anchor_normalizer=relation_anchor_normalizer,
                anchor_modeling=relation_anchor_modeling,
                max_anchors=max_anchors,
                anchor_refinement=relation_anchor_refinement,
                refinement_positions=refinement_positions,
                self_attention_bias=self_attention_bias,
                cross_attention_bias=cross_attention_bias,
            )
        elif rel_cfg.relations_layer is not None:
            self.relations_rep_layer = RelationsRepLayer(
                in_dim=hidden_size, relation_mode=rel_cfg.relations_layer,
            )
        if rel_cfg.triples_layer is not None:
            self.triples_score_layer = TriplesScoreLayer(rel_cfg.triples_layer)
        else:
            # GLiNER's default when ``triples_layer`` is null: concatenate head
            # and tail, then apply a two-layer projection MLP.
            self.pair_rep_layer = create_projection_layer(
                hidden_size * 2, dropout, hidden_size,
            )

    @classmethod
    def from_config(cls, config, shared_layers=None, ner_head=None, **kwargs):
        if config.joint_relex_config is None:
            return None
        return cls(
            config,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            shared_layers=shared_layers,
            ner_head=ner_head,
        )

    def _forward_ner(
        self,
        shared,
        dependency_outputs,
        flat_inputs,
        base_loss_fn,
        batch,
    ):
        """Return the shared NER output or execute the owned fallback head."""
        dependency_output = dependency_outputs.get("ner")
        if (
            dependency_output is not None
            and dependency_output.logits is not None
        ):
            return dependency_output
        if self._owns_ner_head:
            return NERHead.forward(
                self,
                shared,
                dependency_outputs={},
                flat_inputs=flat_inputs,
                base_loss_fn=base_loss_fn,
                **batch,
            )
        reused_ner_head = self.__dict__.get("_reused_ner_head")
        if reused_ner_head is None:
            raise RuntimeError("the reused NER head is no longer available")
        return reused_ner_head(
            shared,
            dependency_outputs={},
            flat_inputs=flat_inputs,
            base_loss_fn=base_loss_fn,
            **batch,
        )

    def _pool_entity_spans(self, words_embedding, span_idx, span_mask):
        """Represent entity spans with GLiNER-relex's token-level SpanRepLayer."""
        span_idx = span_idx * span_mask.unsqueeze(-1).long()
        span_rep = self.rel_span_rep_layer(words_embedding, span_idx)
        span_rep = span_rep * span_mask.unsqueeze(-1).to(span_rep.dtype)
        return span_rep, span_mask.to(torch.long)

    def _decode_relation_entity_spans(
        self,
        ner_scores,
        flat_inputs,
        threshold,
        flat_ner,
        multi_label,
    ):
        """Decode the exact entity list used later by ``NERDecoder``.

        The historical relation path used GLiNER's raw span proposal helper,
        while the public NER decoder applied confidence ranking and overlap
        removal. Relation pair indices then referred to a different entity
        list. Decode once with the shared BIO implementation and retain the
        group-local class id for unambiguous endpoint mapping.
        """
        batch_size, _, class_count, _ = ner_scores.shape
        if flat_inputs is not None and flat_inputs.child_mask is not None:
            if flat_inputs.child_mask.shape[0] != batch_size:
                raise ValueError(
                    "Joint Relex NER class masks and logits must share the "
                    "flattened group axis"
                )
            id_to_classes = [
                {
                    class_idx: class_idx
                    for class_idx in torch.where(
                        flat_inputs.child_mask[batch_idx].bool()
                    )[0].tolist()
                    if class_idx < class_count
                }
                for batch_idx in range(batch_size)
            ]
        else:
            id_to_classes = [
                {class_idx: class_idx for class_idx in range(class_count)}
                for _ in range(batch_size)
            ]

        decoder = SpanDecoder(self.config)
        decoded = decoder.decode_bio_spans_batch(
            logits=ner_scores,
            id_to_classes=id_to_classes,
            batch_size=batch_size,
            threshold=threshold,
            flat_ner=flat_ner,
            multi_label=multi_label,
        )
        max_entities = max((len(spans) for spans in decoded), default=0)
        # Pair builders expect a concrete entity axis even for an empty group.
        max_entities = max(max_entities, 1)
        span_idx = torch.zeros(
            batch_size, max_entities, 2,
            dtype=torch.long, device=ner_scores.device,
        )
        span_mask = torch.zeros(
            batch_size, max_entities,
            dtype=torch.bool, device=ner_scores.device,
        )
        span_class_idx = torch.full(
            (batch_size, max_entities), -1,
            dtype=torch.long, device=ner_scores.device,
        )
        for batch_idx, spans in enumerate(decoded):
            for entity_idx, span in enumerate(spans):
                span_idx[batch_idx, entity_idx] = torch.tensor(
                    (span.start, span.end),
                    dtype=torch.long,
                    device=ner_scores.device,
                )
                span_mask[batch_idx, entity_idx] = True
                span_class_idx[batch_idx, entity_idx] = int(span.entity_type)
        return span_idx, span_mask, span_class_idx

    @staticmethod
    def _remap_relation_entity_classes(
        span_class_idx,
        source_indices,
        span_mask,
    ):
        if span_class_idx is None:
            return None
        if span_class_idx.shape != span_mask.shape and source_indices is None:
            raise ValueError(
                "rel_span_class_idx and rel_span_mask must have the same shape"
            )
        if source_indices is None:
            return span_class_idx.masked_fill(~span_mask.bool(), -1)
        if span_class_idx.shape[0] != source_indices.shape[0]:
            raise ValueError(
                "rel_span_class_idx and selected spans must share the group axis"
            )
        if span_class_idx.shape[1] == 0:
            return torch.full_like(source_indices, -1)
        safe_indices = source_indices.clamp(
            min=0, max=span_class_idx.shape[1] - 1,
        )
        selected = torch.gather(span_class_idx, 1, safe_indices)
        return selected.masked_fill(~span_mask.bool(), -1)

    @staticmethod
    def _validate_relation_alignment(
        flat_inputs,
        rel_prompts_mask,
        rel_labels,
        rel_pair_mask,
        rel_group_mask,
        rel_batch_idx,
        rel_span_idx,
        rel_span_mask,
        rel_span_class_idx,
    ):
        """Fail fast when processor axes cannot index the relation scorer."""
        if rel_prompts_mask.dim() != 2:
            raise ValueError("relation prompt mask must have shape (BN, C_rel)")
        group_count, relation_class_count = rel_prompts_mask.shape
        if flat_inputs is not None:
            if flat_inputs.words_embedding.shape[0] != group_count:
                raise ValueError(
                    "Joint Relex word features and relation prompts must share "
                    "the flattened group axis"
                )
            if rel_batch_idx is not None:
                actual_origin = rel_batch_idx.reshape(-1).to(
                    flat_inputs.batch_origin.device
                )
                expected_origin = flat_inputs.batch_origin.reshape(-1)
                if (
                    actual_origin.shape != expected_origin.shape
                    or not torch.equal(actual_origin, expected_origin)
                ):
                    raise ValueError(
                        "rel_batch_idx does not match the Joint Relex flattened "
                        "group order"
                    )

        prompted_groups = rel_prompts_mask.bool().any(dim=-1)
        if rel_group_mask is not None:
            rel_group_mask = rel_group_mask.reshape(-1).bool().to(
                prompted_groups.device
            )
            if (
                rel_group_mask.shape != prompted_groups.shape
                or not torch.equal(rel_group_mask, prompted_groups)
            ):
                raise ValueError(
                    "rel_mask does not match groups containing relation prompts"
                )

        span_tensors_present = (
            rel_span_idx is not None,
            rel_span_mask is not None,
        )
        if any(span_tensors_present) and not all(span_tensors_present):
            raise ValueError(
                "rel_span_idx and rel_span_mask must be provided together"
            )
        if rel_labels is not None and not all(span_tensors_present):
            raise ValueError(
                "relation labels require aligned rel_span_idx and rel_span_mask"
            )
        if rel_span_idx is None:
            return
        if rel_span_idx.dim() != 3 or rel_span_idx.shape[-1] != 2:
            raise ValueError("rel_span_idx must have shape (BN, E, 2)")
        if rel_span_mask.shape != rel_span_idx.shape[:2]:
            raise ValueError(
                "rel_span_mask must match rel_span_idx's (BN, E) axes"
            )
        if rel_span_idx.shape[0] != group_count:
            raise ValueError(
                "relation spans and prompts must share the flattened group axis"
            )

        valid_spans = rel_span_mask.bool()
        starts = rel_span_idx[..., 0]
        ends = rel_span_idx[..., 1]
        invalid_bounds = valid_spans & ((starts < 0) | (ends < starts))
        if flat_inputs is not None:
            word_lengths = flat_inputs.mask.bool().sum(dim=1).unsqueeze(1)
            invalid_bounds = invalid_bounds | (
                valid_spans & (ends >= word_lengths)
            )
        if invalid_bounds.any():
            bad_group, bad_entity = torch.nonzero(
                invalid_bounds, as_tuple=False,
            )[0].tolist()
            raise ValueError(
                "invalid relation span indices at flattened group "
                f"{bad_group}, entity {bad_entity}"
            )

        if rel_span_class_idx is not None:
            if rel_span_class_idx.shape != valid_spans.shape:
                raise ValueError(
                    "rel_span_class_idx must match rel_span_mask's (BN, E) axes"
                )
            invalid_classes = valid_spans & (rel_span_class_idx < 0)
            if flat_inputs is not None:
                class_count = flat_inputs.child_mask.shape[1]
                invalid_classes = invalid_classes | (
                    valid_spans & (rel_span_class_idx >= class_count)
                )
                safe_classes = rel_span_class_idx.clamp(
                    min=0, max=max(class_count - 1, 0),
                )
                if class_count == 0:
                    invalid_classes = invalid_classes | valid_spans
                else:
                    class_is_prompted = torch.gather(
                        flat_inputs.child_mask.bool(), 1, safe_classes,
                    )
                    invalid_classes = invalid_classes | (
                        valid_spans & ~class_is_prompted
                    )
            if invalid_classes.any():
                raise ValueError(
                    "rel_span_class_idx contains a class that is not prompted "
                    "for its extraction group"
                )
            if (rel_span_class_idx[~valid_spans] != -1).any():
                raise ValueError(
                    "padded relation spans must use class index -1"
                )

        if rel_labels is None:
            return
        entity_count = rel_span_idx.shape[1]
        expected_label_shape = (
            group_count, entity_count, entity_count, relation_class_count,
        )
        if tuple(rel_labels.shape) != expected_label_shape:
            raise ValueError(
                "rel_labels must align exactly with scorer axes "
                f"(BN, E, E, C_rel)={expected_label_shape}; got "
                f"{tuple(rel_labels.shape)}"
            )
        valid_pairs = valid_spans.unsqueeze(2) & valid_spans.unsqueeze(1)
        if (rel_labels.bool() & ~valid_pairs.unsqueeze(-1)).any():
            raise ValueError("rel_labels contains an endpoint in a padded span slot")
        valid_classes = rel_prompts_mask.bool().unsqueeze(1).unsqueeze(1)
        if (rel_labels.bool() & ~valid_classes).any():
            raise ValueError("rel_labels contains an unprompted relation class")
        diagonal = torch.diagonal(rel_labels, dim1=1, dim2=2)
        if diagonal.bool().any():
            raise ValueError(
                "self-relation labels are unsupported because relation pair "
                "builders exclude identical endpoints"
            )

        if rel_pair_mask is not None:
            if tuple(rel_pair_mask.shape) != (
                group_count, entity_count, entity_count,
            ):
                raise ValueError(
                    "rel_pair_mask must have shape (BN, E, E) matching rel_labels"
                )
            candidate_pairs = rel_pair_mask.bool()
            if (candidate_pairs & ~valid_pairs).any():
                raise ValueError(
                    "rel_pair_mask contains an endpoint in a padded span slot"
                )
            positive_pairs = rel_labels.bool().any(dim=-1)
            if (positive_pairs & ~candidate_pairs).any():
                raise ValueError(
                    "rel_pair_mask must include every positive relation pair"
                )

    def _relation_span_safeguards_enabled(self):
        return (
            self.max_relation_span_width is not None
            or self.relation_span_nms
            or self.max_relation_entities is not None
        )

    @staticmethod
    def _score_relation_spans(ner_scores, spans, span_ids, batch_idx):
        """Score spans by their strongest consistent BIO entity class."""
        probabilities = torch.sigmoid(ner_scores[batch_idx]).detach()
        confidences = probabilities.new_zeros(span_ids.numel())
        for output_idx, span_id in enumerate(span_ids.tolist()):
            start = int(spans[span_id, 0].item())
            end = int(spans[span_id, 1].item())
            if start < 0 or end < start or end >= probabilities.shape[0]:
                continue
            start_scores = probabilities[start, :, 0]
            end_scores = probabilities[end, :, 1]
            inside_scores = probabilities[start:end + 1, :, 2].amin(dim=0)
            confidences[output_idx] = torch.minimum(
                torch.minimum(start_scores, end_scores), inside_scores,
            ).amax()
        return confidences

    def _select_relation_spans(self, ner_scores, span_idx, span_mask):
        """Apply optional width, flat-NMS, and entity-count safeguards.

        Returns compacted spans plus source indices into the original entity
        axis. Source indices let training relation targets be remapped without
        relying on the filtered spans retaining their original positions.
        """
        if not self._relation_span_safeguards_enabled():
            return span_idx, span_mask, None

        selected_per_batch = []
        needs_ranking = self.relation_span_nms or self.max_relation_entities is not None
        for batch_idx in range(span_idx.shape[0]):
            source_ids = torch.where(span_mask[batch_idx].bool())[0]
            if self.max_relation_span_width is not None and source_ids.numel() > 0:
                spans = span_idx[batch_idx, source_ids]
                widths = spans[:, 1] - spans[:, 0] + 1
                source_ids = source_ids[widths <= self.max_relation_span_width]

            if needs_ranking and source_ids.numel() > 0:
                confidences = self._score_relation_spans(
                    ner_scores, span_idx[batch_idx], source_ids, batch_idx,
                )
                order = torch.argsort(confidences, descending=True, stable=True)
                source_ids = source_ids[order]

            if self.relation_span_nms and source_ids.numel() > 0:
                kept = []
                for source_id in source_ids.tolist():
                    start = int(span_idx[batch_idx, source_id, 0].item())
                    end = int(span_idx[batch_idx, source_id, 1].item())
                    overlaps = any(
                        not (
                            end < int(span_idx[batch_idx, kept_id, 0].item())
                            or start > int(span_idx[batch_idx, kept_id, 1].item())
                        )
                        for kept_id in kept
                    )
                    if not overlaps:
                        kept.append(source_id)
                source_ids = torch.tensor(
                    kept, dtype=torch.long, device=span_idx.device,
                )

            if self.max_relation_entities is not None:
                source_ids = source_ids[:self.max_relation_entities]
            selected_per_batch.append(source_ids)

        max_entities = max((ids.numel() for ids in selected_per_batch), default=0)
        max_entities = max(max_entities, 1)
        selected_spans = span_idx.new_zeros(span_idx.shape[0], max_entities, 2)
        selected_mask = span_mask.new_zeros(span_idx.shape[0], max_entities)
        source_indices = torch.full(
            (span_idx.shape[0], max_entities), -1,
            dtype=torch.long, device=span_idx.device,
        )
        for batch_idx, source_ids in enumerate(selected_per_batch):
            count = source_ids.numel()
            if count == 0:
                continue
            selected_spans[batch_idx, :count] = span_idx[batch_idx, source_ids]
            selected_mask[batch_idx, :count] = True
            source_indices[batch_idx, :count] = source_ids
        return selected_spans, selected_mask, source_indices

    @staticmethod
    def _remap_relation_targets(rel_labels, rel_pair_mask, source_indices, span_mask):
        """Project dense entity-axis targets onto a filtered entity set."""
        if source_indices is None:
            return rel_labels, rel_pair_mask

        batch_size, entity_count = source_indices.shape
        valid_pairs = span_mask.bool().unsqueeze(2) & span_mask.bool().unsqueeze(1)
        batch_indices = torch.arange(
            batch_size, device=source_indices.device,
        ).view(batch_size, 1, 1)

        def remap(tensor, has_class_axis):
            if tensor is None:
                return None
            if tensor.shape[1] == 0 or tensor.shape[2] == 0:
                output_shape = (batch_size, entity_count, entity_count)
                if has_class_axis:
                    output_shape += (tensor.shape[-1],)
                return tensor.new_zeros(output_shape)
            safe_indices = source_indices.clamp(min=0, max=tensor.shape[1] - 1)
            heads = safe_indices.unsqueeze(2).expand(batch_size, entity_count, entity_count)
            tails = safe_indices.unsqueeze(1).expand(batch_size, entity_count, entity_count)
            result = tensor[batch_indices, heads, tails]
            mask = valid_pairs.unsqueeze(-1) if has_class_axis else valid_pairs
            return result * mask.to(result.dtype)

        return remap(rel_labels, True), remap(rel_pair_mask, False)

    @staticmethod
    def _empty_entity_pairs(span_rep):
        batch_size, _, hidden_size = span_rep.shape
        pair_idx = torch.full(
            (batch_size, 1, 2), -1, dtype=torch.long, device=span_rep.device,
        )
        pair_mask = torch.zeros(
            (batch_size, 1), dtype=torch.bool, device=span_rep.device,
        )
        empty_rep = span_rep.new_zeros(batch_size, 1, hidden_size)
        return pair_idx, pair_mask, empty_rep, empty_rep

    @staticmethod
    def _gather_pair_representations(span_rep, pair_idx, pair_mask):
        """Gather hard source/target entity representations for relation slots."""
        batch_size, entity_count, hidden_size = span_rep.shape
        if entity_count == 0:
            empty_rep = span_rep.new_zeros(
                batch_size, pair_idx.shape[1], hidden_size,
            )
            return empty_rep, empty_rep

        safe_pair_idx = pair_idx.clamp(min=0, max=entity_count - 1)
        batch_indices = torch.arange(
            batch_size, device=span_rep.device,
        ).unsqueeze(1)
        head_rep = span_rep[batch_indices, safe_pair_idx[..., 0]]
        tail_rep = span_rep[batch_indices, safe_pair_idx[..., 1]]
        mask = pair_mask.unsqueeze(-1).to(span_rep.dtype)
        return head_rep * mask, tail_rep * mask

    @staticmethod
    def _pack_anchor_pair_targets(candidate_matrix, entity_mask):
        """Pack a dense pair mask into padded directed-pair targets."""
        if candidate_matrix.dim() != 3:
            raise ValueError("relation pair targets must have shape (B, E, E)")
        if candidate_matrix.shape[:2] != entity_mask.shape:
            raise ValueError(
                "relation pair targets and entity mask must share (B, E)"
            )
        if candidate_matrix.shape[2] != entity_mask.shape[1]:
            raise ValueError("relation pair targets must be square on entity axes")

        batch_size, entity_count = entity_mask.shape
        valid_entities = entity_mask.bool()
        target_mask = candidate_matrix > 0
        target_mask = (
            target_mask
            & valid_entities.unsqueeze(2)
            & valid_entities.unsqueeze(1)
        )
        if entity_count > 0:
            diagonal = torch.eye(
                entity_count,
                dtype=torch.bool,
                device=candidate_matrix.device,
            ).unsqueeze(0)
            target_mask = target_mask & ~diagonal

        targets = [
            torch.nonzero(target_mask[batch_idx], as_tuple=False)
            for batch_idx in range(batch_size)
        ]
        max_targets = max((target.shape[0] for target in targets), default=0)
        # Preserve a concrete pair axis even when the batch has no candidates.
        max_targets = max(max_targets, 1)
        pair_idx = torch.full(
            (batch_size, max_targets, 2),
            -1,
            dtype=torch.long,
            device=candidate_matrix.device,
        )
        pair_mask = torch.zeros(
            batch_size,
            max_targets,
            dtype=torch.bool,
            device=candidate_matrix.device,
        )
        for batch_idx, target in enumerate(targets):
            count = target.shape[0]
            if count == 0:
                continue
            pair_idx[batch_idx, :count] = target
            pair_mask[batch_idx, :count] = True
        return pair_idx, pair_mask

    @staticmethod
    def _match_anchor_pairs(
        assignment_logits,
        anchor_mask,
        target_pair_idx,
        target_pair_mask,
    ):
        """Hungarian-match active anchors to gold pairs by endpoint NLL."""
        log_probabilities = F.log_softmax(
            assignment_logits.float(), dim=2,
        )

        def pair_cost(batch_idx, prediction_ids, target_ids):
            prediction = log_probabilities[batch_idx, prediction_ids]
            targets = target_pair_idx[batch_idx, target_ids]
            source_cost = -prediction[:, targets[:, 0], 0]
            target_cost = -prediction[:, targets[:, 1], 1]
            return source_cost + target_cost

        return batched_masked_assignment(
            anchor_mask.bool(),
            target_pair_mask.bool(),
            pair_cost,
        )

    @staticmethod
    def _matched_anchor_pair_indices(
        target_pair_idx,
        anchor_mask,
        matches,
    ):
        """Scatter matched gold pairs onto their assigned anchor slots."""
        batch_size, anchor_count = anchor_mask.shape
        pair_idx = torch.full(
            (batch_size, anchor_count, 2),
            -1,
            dtype=torch.long,
            device=target_pair_idx.device,
        )
        pair_mask = torch.zeros_like(anchor_mask, dtype=torch.bool)
        for batch_idx, batch_matches in enumerate(matches):
            for anchor_idx, target_idx in batch_matches:
                if anchor_mask[batch_idx, anchor_idx]:
                    pair_idx[batch_idx, anchor_idx] = target_pair_idx[
                        batch_idx, target_idx
                    ]
                    pair_mask[batch_idx, anchor_idx] = True
        return pair_idx, pair_mask

    @classmethod
    def _training_anchor_pairs(
        cls,
        predicted_pair_idx,
        predicted_pair_mask,
        target_pair_idx,
        anchor_mask,
        matches,
        rel_labels,
    ):
        """Combine matched positives with predicted background pairs.

        Hungarian matching owns only positive relation pairs. Active unmatched
        anchors still need relation-level background supervision, so their
        valid predicted pairs are gathered with an all-zero relation target.
        Predictions that coincide with a positive pair are excluded to avoid
        teaching an unmatched duplicate that the positive relation is absent.
        """
        matched_pair_idx, matched_anchor_mask = (
            cls._matched_anchor_pair_indices(
                target_pair_idx,
                anchor_mask,
                matches,
            )
        )
        pair_idx = predicted_pair_idx.clone()
        predicted_pair_mask = (
            predicted_pair_mask.bool() & anchor_mask.bool()
        )
        safe_pair_idx = pair_idx.clamp_min(0)
        batch_indices = torch.arange(
            pair_idx.shape[0], device=pair_idx.device,
        ).unsqueeze(1)
        predicted_positive = (
            rel_labels[
                batch_indices,
                safe_pair_idx[..., 0],
                safe_pair_idx[..., 1],
            ].sum(dim=-1)
            > 0
        )
        background_mask = (
            predicted_pair_mask
            & ~matched_anchor_mask
            & ~predicted_positive
        )
        pair_mask = matched_anchor_mask | background_mask
        pair_idx[matched_anchor_mask] = matched_pair_idx[matched_anchor_mask]
        pair_idx = pair_idx.masked_fill(~pair_mask.unsqueeze(-1), -1)
        return pair_idx, pair_mask, matched_anchor_mask, background_mask

    @staticmethod
    def _anchor_assignment_loss(
        assignment_logits,
        target_pair_idx,
        anchor_mask,
        matches,
    ):
        """Endpoint-selection loss over Hungarian-matched active anchors."""
        log_probabilities = F.log_softmax(
            assignment_logits.float(), dim=2,
        )
        loss = assignment_logits.new_zeros((), dtype=torch.float32)
        for batch_idx, batch_matches in enumerate(matches):
            for anchor_idx, target_idx in batch_matches:
                if not anchor_mask[batch_idx, anchor_idx]:
                    continue
                source_idx, tail_idx = target_pair_idx[
                    batch_idx, target_idx
                ]
                loss = loss - log_probabilities[
                    batch_idx, anchor_idx, source_idx, 0
                ]
                loss = loss - log_probabilities[
                    batch_idx, anchor_idx, tail_idx, 1
                ]
        return loss

    def _build_top_k_dot_pairs(self, span_rep, span_mask, threshold):
        """Select at most k dot-adjacency neighbors without a dense E x E tensor."""
        batch_size, entity_count, _ = span_rep.shape
        neighbor_count = min(
            self.relation_top_k_neighbors, max(entity_count - 1, 0),
        )
        if neighbor_count == 0:
            return self._empty_entity_pairs(span_rep)

        tail_indices = torch.zeros(
            batch_size, entity_count, neighbor_count,
            dtype=torch.long, device=span_rep.device,
        )
        selected_mask = torch.zeros(
            batch_size, entity_count, neighbor_count,
            dtype=torch.bool, device=span_rep.device,
        )
        key_mask = span_mask.bool()
        chunk_size = self.relation_neighbor_chunk_size or max(entity_count, 1)
        for start in range(0, entity_count, chunk_size):
            end = min(start + chunk_size, entity_count)
            scores = torch.sigmoid(torch.bmm(
                span_rep[:, start:end], span_rep.transpose(1, 2),
            ))
            valid = (
                key_mask[:, None, :]
                & key_mask[:, start:end, None]
            )
            query_ids = torch.arange(start, end, device=span_rep.device)
            valid[:, torch.arange(end - start, device=span_rep.device), query_ids] = False
            scores = scores.masked_fill(~valid, float("-inf"))
            top_scores, top_indices = scores.topk(neighbor_count, dim=-1)
            gathered_key_mask = torch.gather(
                key_mask, 1, top_indices.reshape(batch_size, -1),
            ).reshape_as(top_indices)
            current_mask = (
                key_mask[:, start:end, None]
                & gathered_key_mask
                & torch.isfinite(top_scores)
                & (top_scores > threshold)
            )
            tail_indices[:, start:end] = top_indices
            selected_mask[:, start:end] = current_mask

        head_indices = torch.arange(
            entity_count, device=span_rep.device,
        ).view(1, entity_count, 1).expand_as(tail_indices)
        pair_idx = torch.stack((head_indices, tail_indices), dim=-1).reshape(
            batch_size, entity_count * neighbor_count, 2,
        )
        pair_mask = selected_mask.reshape(batch_size, entity_count * neighbor_count)
        pair_idx = pair_idx.masked_fill(~pair_mask.unsqueeze(-1), -1)
        batch_indices = torch.arange(batch_size, device=span_rep.device).unsqueeze(1)
        safe_pair_idx = pair_idx.clamp_min(0)
        head_rep = span_rep[batch_indices, safe_pair_idx[..., 0]]
        tail_rep = span_rep[batch_indices, safe_pair_idx[..., 1]]
        head_rep = head_rep * pair_mask.unsqueeze(-1).to(head_rep.dtype)
        tail_rep = tail_rep * pair_mask.unsqueeze(-1).to(tail_rep.dtype)
        return pair_idx, pair_mask, head_rep, tail_rep

    def _chunked_dot_adjacency_loss(
        self, span_rep, span_mask, adjacency_labels, base_loss_fn, loss_kwargs,
    ):
        """Compute the dense dot-adjacency objective without storing its matrix."""
        total_loss = span_rep.new_zeros(())
        entity_count = span_rep.shape[1]
        chunk_size = self.relation_neighbor_chunk_size or max(entity_count, 1)
        for start in range(0, entity_count, chunk_size):
            end = min(start + chunk_size, entity_count)
            probabilities = torch.sigmoid(torch.bmm(
                span_rep[:, start:end], span_rep.transpose(1, 2),
            ))
            labels = adjacency_labels[:, start:end]
            valid = (
                span_mask[:, start:end].bool().unsqueeze(2)
                & span_mask.bool().unsqueeze(1)
            )
            batch_size = span_rep.shape[0]
            probabilities = probabilities.reshape(batch_size, -1, 1)
            labels = labels.reshape(batch_size, -1, 1)
            valid = valid.reshape(batch_size, -1, 1)
            losses = self._call_elementwise_loss(
                base_loss_fn, probabilities, labels,
                normalize_prob=False, **loss_kwargs,
            )
            total_loss = total_loss + (losses * valid.to(losses.dtype)).sum()
        return total_loss

    def _get_rel_prompts(self, shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask):
        """Return per-group (BN, C_rel, D) [RELATION] embeddings.

        Prefers processor/model-provided flat tensors (already split per
        extraction group). Falls back to batch-level extraction when those
        aren't available (e.g. unit tests calling the head directly).
        """
        if flat_rel_prompts is not None:
            if flat_rel_prompts_mask is None:
                flat_rel_prompts_mask = torch.ones(
                    flat_rel_prompts.shape[:-1],
                    dtype=shared.attention_mask.dtype,
                    device=flat_rel_prompts.device,
                )
            return flat_rel_prompts, flat_rel_prompts_mask
        if rel_label_embeds is not None:
            rel_prompts = rel_label_embeds
            rel_mask = torch.ones(
                rel_label_embeds.shape[:-1], dtype=shared.attention_mask.dtype,
                device=shared.attention_mask.device,
            )
            return rel_prompts, rel_mask
        batch_size = shared.token_embeds.shape[0]
        embed_dim = shared.token_embeds.shape[2]
        rel_prompts, rel_mask = extract_prompt_features(
            self.rel_token_index, shared.token_embeds, shared.input_ids,
            shared.attention_mask, batch_size, embed_dim, self.embed_rel_token,
        )
        return rel_prompts, rel_mask

    def _forward_relations(self, shared, target_span_rep, target_span_mask,
                           rel_labels, adjacency_threshold, rel_label_embeds,
                           flat_rel_prompts=None, flat_rel_prompts_mask=None,
                           rel_pair_mask=None, base_loss_fn=None, loss_kwargs=None,
                           relation_context=None, anchor_threshold=0.5):
        """Relation scoring with optional adjacency-based pair selection."""
        B, E_ent, D = target_span_rep.shape
        use_anchor_pairs = hasattr(self, "anchor_relations_layer")
        use_adjacency = hasattr(self, "relations_rep_layer")
        use_sparse_neighbors = (
            use_adjacency and self.relation_top_k_neighbors is not None
        )
        pred_adj_matrix = None
        if use_adjacency and not use_sparse_neighbors:
            pred_adj_matrix = self.relations_rep_layer(target_span_rep, target_span_mask)

        rel_prompts, rel_prompts_mask = self._get_rel_prompts(
            shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask,
        )
        C_rel = rel_prompts.size(1)

        if rel_labels is not None:
            positive_adj_matrix = (rel_labels.sum(dim=-1) > 0).float()
            if rel_pair_mask is not None:
                adj_matrix = rel_pair_mask.float()
            else:
                adj_matrix = positive_adj_matrix
        else:
            adj_matrix = None

        anchor_output = None
        anchor_matches = None
        assignment_loss = None
        if use_anchor_pairs:
            if relation_context is None:
                raise ValueError(
                    "anchor-modeling relation selection requires a parent "
                    "context embedding"
                )
            anchor_output = self.anchor_relations_layer(
                relation_context,
                target_span_rep,
                target_span_mask,
                threshold=anchor_threshold,
                # During training Hungarian matching decides which gold pair
                # belongs to each anchor. At inference repeated predictions are
                # collapsed before relation decoding.
                deduplicate=rel_labels is None,
            )
            if adj_matrix is not None:
                # Set prediction matches relation-bearing pairs only. Sampled
                # no-relation candidates are relation-classification
                # background; letting them compete in this assignment can
                # crowd true relations out of a bounded anchor set.
                target_pair_idx, target_pair_mask = self._pack_anchor_pair_targets(
                    positive_adj_matrix,
                    target_span_mask,
                )
                gold_counts = target_pair_mask.sum(dim=1)
                active_anchor_counts = anchor_output.anchor_mask.sum(dim=1)
                overflow = gold_counts > active_anchor_counts
                if (
                    overflow.any()
                    and not self._anchor_pair_capacity_warning_emitted
                ):
                    overflow_idx = (gold_counts - active_anchor_counts).argmax()
                    gold_count = int(gold_counts[overflow_idx].item())
                    active_count = int(
                        active_anchor_counts[overflow_idx].item()
                    )
                    warnings.warn(
                        "Joint relation gold-pair count exceeds the active "
                        f"anchor capacity ({gold_count} gold pairs, "
                        f"{active_count} active anchors). Hungarian matching "
                        "will supervise at most the active anchor count; "
                        "unmatched gold pairs are ignored. Increase the "
                        "configured relation anchor capacity.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._anchor_pair_capacity_warning_emitted = True
                anchor_matches = self._match_anchor_pairs(
                    anchor_output.assignment_logits,
                    anchor_output.anchor_mask,
                    target_pair_idx,
                    target_pair_mask,
                )
                pair_idx, pair_mask, _, _ = self._training_anchor_pairs(
                    anchor_output.pair_idx,
                    anchor_output.pair_mask,
                    target_pair_idx,
                    anchor_output.anchor_mask,
                    anchor_matches,
                    rel_labels,
                )
                assignment_loss = self._anchor_assignment_loss(
                    anchor_output.assignment_logits,
                    target_pair_idx,
                    anchor_output.anchor_mask,
                    anchor_matches,
                )
            else:
                pair_idx = anchor_output.pair_idx
                pair_mask = anchor_output.pair_mask
            head_rep, tail_rep = self._gather_pair_representations(
                target_span_rep,
                pair_idx,
                pair_mask,
            )
        elif use_adjacency:
            if adj_matrix is not None:
                pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
                    adj_matrix, target_span_rep, threshold=adjacency_threshold,
                )
            elif use_sparse_neighbors:
                pair_idx, pair_mask, head_rep, tail_rep = self._build_top_k_dot_pairs(
                    target_span_rep, target_span_mask, adjacency_threshold,
                )
            else:
                pair_idx, pair_mask, head_rep, tail_rep = build_entity_pairs(
                    pred_adj_matrix, target_span_rep, threshold=adjacency_threshold,
                )
        else:
            pair_idx, pair_mask, head_rep, tail_rep = build_all_entity_pairs(
                target_span_rep, target_span_mask,
            )
        N = head_rep.size(1)
        pair_scores = None

        if hasattr(self, "pair_rep_layer"):
            pair_rep = self.pair_rep_layer(
                torch.cat((head_rep, tail_rep), dim=-1),
            )
            pair_scores = torch.einsum("BND,BCD->BNC", pair_rep, rel_prompts)
        elif hasattr(self, "triples_score_layer"):
            h = head_rep.unsqueeze(2).expand(B, N, C_rel, D)
            t = tail_rep.unsqueeze(2).expand(B, N, C_rel, D)
            r = rel_prompts.unsqueeze(1).expand(B, N, C_rel, D)
            pair_scores = self.triples_score_layer(
                h.reshape(-1, D), r.reshape(-1, D), t.reshape(-1, D),
            ).view(B, N, C_rel)

        loss = None
        if rel_labels is not None and pair_scores is not None:
            if rel_labels.size(-1) != C_rel:
                raise ValueError(
                    "rel_labels and relation prompts must have the same "
                    f"class axis; got {rel_labels.size(-1)} and {C_rel}"
                )
            head_indices = pair_idx[..., 0].clamp(min=0)
            tail_indices = pair_idx[..., 1].clamp(min=0)
            batch_idx_t = torch.arange(B, device=rel_labels.device).unsqueeze(1)
            rel_matrix = rel_labels[batch_idx_t, head_indices, tail_indices]

            rel_mask_expanded = pair_mask.unsqueeze(-1).expand(B, N, C_rel)
            class_mask = rel_prompts_mask.unsqueeze(1).expand(B, N, C_rel)
            combined = rel_mask_expanded * class_mask
            if loss_kwargs is None:
                loss_kwargs = {}
            rel_losses = self._call_elementwise_loss(
                base_loss_fn, pair_scores, rel_matrix, normalize_prob=True, **loss_kwargs,
            )
            valid_relation_cells = combined.to(rel_losses.dtype)
            rel_loss = (rel_losses * valid_relation_cells).sum()
            if self.relation_loss_reduction == "mean":
                # Normalize against every possible directed pair of active
                # entities and every active relation class. For group b this
                # contributes e_b * (e_b - 1) * C_b; summing groups is the
                # mask-aware form of BN * e * (e - 1) * C.
                entity_counts = target_span_mask.to(rel_losses.dtype).sum(dim=1)
                relation_counts = rel_prompts_mask.to(rel_losses.dtype).sum(dim=1)
                relation_normalizer = (
                    entity_counts
                    * (entity_counts - 1.0).clamp(min=0.0)
                    * relation_counts
                ).sum()
                rel_loss = rel_loss / relation_normalizer.clamp(min=1.0)

            if use_anchor_pairs:
                loss = (
                    assignment_loss * self.adjacency_loss_coef
                    + rel_loss * self.relation_loss_coef
                )
            elif use_adjacency and adj_matrix is not None:
                if use_sparse_neighbors:
                    adj_loss = self._chunked_dot_adjacency_loss(
                        target_span_rep, target_span_mask, adj_matrix,
                        base_loss_fn, loss_kwargs,
                    )
                else:
                    adj_mask = target_span_mask.float().unsqueeze(1) * target_span_mask.float().unsqueeze(2)
                    adj_logits = pred_adj_matrix.unsqueeze(-1).view(B, -1, 1)
                    adj_labels = adj_matrix.unsqueeze(-1).view(B, -1, 1)
                    adj_losses = self._call_elementwise_loss(
                        base_loss_fn, adj_logits, adj_labels, normalize_prob=False, **loss_kwargs,
                    )
                    adj_loss = (adj_losses * adj_mask.unsqueeze(-1).view(B, -1, 1)).sum()
                loss = (
                    adj_loss * self.adjacency_loss_coef
                    + rel_loss * self.relation_loss_coef
                )
            else:
                loss = rel_loss * self.relation_loss_coef

        extra = {"rel_idx": pair_idx, "rel_mask": pair_mask}
        if anchor_output is not None:
            extra.update({
                "rel_assignment_logits": anchor_output.assignment_logits,
                "rel_anchors": anchor_output.anchors,
                "rel_anchor_mask": anchor_output.anchor_mask,
                "rel_anchor_matches": anchor_matches,
                "rel_assignment_loss": (
                    assignment_loss.detach()
                    if assignment_loss is not None else None
                ),
            })
        return TaskHeadOutput(
            loss=loss,
            logits=pair_scores,
            extra=extra,
        )

    def _relation_loss_kwargs(self, batch):
        """Resolve relation-specific focal controls and runtime fallbacks."""
        kwargs = {}
        for out_key, config_name, sources in (
            (
                "focal_loss_alpha",
                "relation_focal_loss_alpha",
                ("rel_focal_loss_alpha", "focal_loss_alpha"),
            ),
            (
                "focal_loss_gamma",
                "relation_focal_loss_gamma",
                ("rel_focal_loss_gamma", "focal_loss_gamma"),
            ),
            (
                "focal_loss_prob_margin",
                "relation_focal_loss_prob_margin",
                ("rel_focal_loss_prob_margin", "focal_loss_prob_margin"),
            ),
        ):
            configured_value = getattr(self, config_name)
            if configured_value is not None:
                kwargs[out_key] = configured_value
                continue
            for source in sources:
                value = batch.get(source)
                if value is not None:
                    kwargs[out_key] = value
                    break
        for out_key, sources in (
            ("label_smoothing", ("rel_label_smoothing", "label_smoothing")),
            ("negatives", ("rel_negatives", "negatives")),
            ("masking", ("rel_masking", "masking")),
        ):
            for source in sources:
                value = batch.get(source)
                if value is not None:
                    kwargs[out_key] = value
                    break
        return kwargs

    @staticmethod
    def _call_elementwise_loss(loss_fn, logits, labels, normalize_prob=True, **kwargs):
        """Call BaseModel._loss when available, falling back to focal loss.

        Unit tests may pass another compatible loss directly; production uses
        GLiNExT's stable elementwise binary-loss primitive.
        """
        if loss_fn is not None:
            try:
                return loss_fn(logits, labels, normalize_prob=normalize_prob, **kwargs)
            except TypeError:
                supported = {
                    "focal_loss_alpha",
                    "focal_loss_gamma",
                    "focal_loss_prob_margin",
                    "label_smoothing",
                }
                fallback_kwargs = {k: v for k, v in kwargs.items() if k in supported}
                return loss_fn(logits, labels, normalize_prob=normalize_prob, **fallback_kwargs)
        supported = {
            "focal_loss_alpha",
            "focal_loss_gamma",
            "focal_loss_prob_margin",
            "label_smoothing",
        }
        fallback_kwargs = {k: v for k, v in kwargs.items() if k in supported}
        return binary_focal_or_bce(
            logits, labels, normalize_prob=normalize_prob, **fallback_kwargs,
        )

    def forward(self, shared, dependency_outputs, flat_inputs=None, base_loss_fn=None,
                rel_label_embeds=None, flat_rel_prompts=None, flat_rel_prompts_mask=None,
                **batch):
        # 1. Run NER forward (inherited) — passes flat_inputs through
        ner_output = self._forward_ner(
            shared,
            dependency_outputs,
            flat_inputs,
            base_loss_fn,
            batch,
        )

        # 2. Select entity spans.
        #    Training: use processor-provided per-entity spans so target_span_rep
        #    is aligned with rel_labels' entity_id axis.
        #    Inference: fall back to NER-extracted spans.
        ner_scores = ner_output.logits
        words_embedding = ner_output.extra.get("words_embedding", shared.words_embedding)

        # A multi-task checkpoint can contain this head even when the current
        # request is NER-only.  In that case there are no relation prompts, so
        # constructing every possible predicted entity pair is both useless
        # and potentially quadratic in a large number of noisy NER spans.
        rel_prompts, rel_prompts_mask = self._get_rel_prompts(
            shared, rel_label_embeds, flat_rel_prompts, flat_rel_prompts_mask,
        )
        self._validate_relation_alignment(
            flat_inputs,
            rel_prompts_mask,
            batch.get("rel_labels"),
            batch.get("rel_pair_mask"),
            batch.get("rel_mask"),
            batch.get("rel_batch_idx"),
            batch.get("rel_span_idx"),
            batch.get("rel_span_mask"),
            batch.get("rel_span_class_idx"),
        )
        if rel_prompts.size(1) == 0:
            return TaskHeadOutput(
                loss=(
                    ner_output.loss * self.ner_loss_coef
                    if self._owns_ner_head and ner_output.loss is not None
                    else None
                ),
                logits=ner_output.logits,
                extra={
                    **ner_output.extra,
                    "rel_logits": None,
                    "rel_idx": None,
                    "rel_mask": None,
                    "rel_entity_spans": None,
                    "rel_entity_class_idx": None,
                    "rel_assignment_logits": None,
                    "rel_anchors": None,
                    "rel_anchor_mask": None,
                    "rel_anchor_matches": None,
                    "rel_assignment_loss": None,
                },
            )

        rel_span_idx = batch.get("rel_span_idx")
        rel_span_mask = batch.get("rel_span_mask")
        span_class_idx = batch.get("rel_span_class_idx")

        if rel_span_idx is not None and rel_span_mask is not None:
            span_idx, span_mask = rel_span_idx, rel_span_mask
        else:
            span_idx, span_mask, span_class_idx = (
                self._decode_relation_entity_spans(
                    ner_scores,
                    flat_inputs,
                    threshold=batch.get("threshold", 0.5),
                    flat_ner=batch.get("relation_flat_ner", True),
                    multi_label=batch.get("relation_multi_label", False),
                )
            )

        span_idx, span_mask, source_indices = self._select_relation_spans(
            ner_scores, span_idx, span_mask,
        )
        span_class_idx = self._remap_relation_entity_classes(
            span_class_idx, source_indices, span_mask,
        )
        target_span_rep, target_span_mask = self._pool_entity_spans(
            words_embedding, span_idx, span_mask,
        )
        rel_labels, rel_pair_mask = self._remap_relation_targets(
            batch.get("rel_labels"), batch.get("rel_pair_mask"),
            source_indices, target_span_mask,
        )

        # 3. Build candidate pairs and score relation types
        rel_output = self._forward_relations(
            shared, target_span_rep, target_span_mask,
            rel_labels, batch.get("adjacency_threshold", 0.5),
            rel_label_embeds,
            flat_rel_prompts=rel_prompts,
            flat_rel_prompts_mask=rel_prompts_mask,
            rel_pair_mask=rel_pair_mask,
            base_loss_fn=base_loss_fn,
            loss_kwargs=self._relation_loss_kwargs(batch),
            relation_context=(
                flat_inputs.parent_embedding
                if flat_inputs is not None else None
            ),
            anchor_threshold=batch.get(
                "anchor_threshold", batch.get("threshold", 0.5),
            ),
        )

        # 4. Combine losses
        combined_loss = None
        owned_ner_loss = (
            ner_output.loss if self._owns_ner_head else None
        )
        if owned_ner_loss is not None or rel_output.loss is not None:
            ner_loss = (
                owned_ner_loss * self.ner_loss_coef
                if owned_ner_loss is not None else 0.0
            )
            rel_loss = rel_output.loss if rel_output.loss is not None else 0.0
            combined_loss = ner_loss + rel_loss

        # 5. Return merged output
        return TaskHeadOutput(
            loss=combined_loss,
            logits=ner_output.logits,
            extra={
                **ner_output.extra,
                "rel_logits": rel_output.logits,
                "rel_idx": rel_output.extra.get("rel_idx"),
                "rel_mask": rel_output.extra.get("rel_mask"),
                "rel_entity_spans": span_idx,
                "rel_entity_class_idx": span_class_idx,
                "rel_assignment_logits": rel_output.extra.get(
                    "rel_assignment_logits"
                ),
                "rel_anchors": rel_output.extra.get("rel_anchors"),
                "rel_anchor_mask": rel_output.extra.get(
                    "rel_anchor_mask"
                ),
                "rel_anchor_matches": rel_output.extra.get(
                    "rel_anchor_matches"
                ),
                "rel_assignment_loss": rel_output.extra.get(
                    "rel_assignment_loss"
                ),
            },
        )
