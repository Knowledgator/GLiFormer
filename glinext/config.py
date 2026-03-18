from typing import Optional

from transformers.models.auto import CONFIG_MAPPING

from gliner.config import BaseGLiNERConfig


class GLiNextConfig(BaseGLiNERConfig):
    model_type = "glinext"

    def __init__(
        self,
        # Layer selection (None = disabled)
        relations_layer: Optional[str] = None,
        classifier_layer: Optional[str] = "dot",
        groups_layer: Optional[str] = None,
        count_layer: Optional[str] = None,
        # Relations config
        rel_mode: str = "adjacency",  # "adjacency" (pair-based) or "prompt" (prompt-guided source/target)
        pair_rep_type: str = "concat_proj",  # for adjacency mode: "concat_proj", "bilinear", "additive", "mlp"
        triples_layer: Optional[str] = None,
        embed_rel_token: bool = True,
        rel_token_index: int = -1,
        # Labels encoder (bi-encoder style)
        labels_encoder: Optional[str] = None,
        labels_encoder_config: Optional[dict] = None,
        # Decoder model
        decoder_model: Optional[str] = None,
        decoder_config: Optional[dict] = None,
        full_decoder_context: bool = True,
        # Loss coefficients
        ner_loss_coef: float = 1.0,
        cat_loss_coef: float = 1.0,
        rel_loss_coef: float = 1.0,
        adjacency_loss_coef: float = 1.0,
        decoder_loss_coef: float = 0.5,
        count_loss_coef: float = 1.0,
        groups_loss_coef: float = 1.0,
        embedding_loss_coef: float = 1.0,
        # Count layer config
        count_mode: str = "regression",
        max_count: int = 20,
        # Groups layer config
        groups_num_heads: int = 4,
        groups_num_layers: int = 2,
        # Span representation
        represent_spans: bool = False,
        neg_spans_ratio: float = 1.0,
        span_loss_coef: float = 1.0,
        # Structuring config
        structuring_loss_coef: float = 1.0,
        child_token_index: int = -1,
        embed_child_token: bool = True,
        # Classifier config
        cat_token_index: int = -1,
        embed_cat_token: bool = True,
        # Special tokens
        seq_token: str = "[SEQ]",
        cat_token: str = "[CAT]",
        rel_token: str = "[REL]",
        parent_token: str = "[PARENT]",
        child_token: str = "[CHILD]",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Decoder config
        if isinstance(decoder_config, dict):
            decoder_config["model_type"] = decoder_config.get("model_type", "gpt2")
            decoder_config = CONFIG_MAPPING[decoder_config["model_type"]](**decoder_config)
        self.decoder_model = decoder_model
        self.decoder_config = decoder_config
        self.full_decoder_context = full_decoder_context

        # Labels encoder config
        if isinstance(labels_encoder_config, dict):
            labels_encoder_config["model_type"] = labels_encoder_config.get("model_type", "deberta-v2")
            labels_encoder_config = CONFIG_MAPPING[labels_encoder_config["model_type"]](**labels_encoder_config)
        self.labels_encoder = labels_encoder
        self.labels_encoder_config = labels_encoder_config

        # Layer configs
        self.relations_layer = relations_layer
        self.rel_mode = rel_mode
        self.pair_rep_type = pair_rep_type
        self.triples_layer = triples_layer
        self.embed_rel_token = embed_rel_token
        self.rel_token_index = rel_token_index

        self.classifier_layer = classifier_layer
        self.cat_token_index = cat_token_index
        self.embed_cat_token = embed_cat_token

        self.groups_layer = groups_layer
        self.groups_num_heads = groups_num_heads
        self.groups_num_layers = groups_num_layers

        self.represent_spans = represent_spans
        self.neg_spans_ratio = neg_spans_ratio
        self.span_loss_coef = span_loss_coef

        self.structuring_loss_coef = structuring_loss_coef
        self.child_token_index = child_token_index
        self.embed_child_token = embed_child_token

        self.count_layer = count_layer
        self.count_mode = count_mode
        self.max_count = max_count

        # Loss coefficients
        self.ner_loss_coef = ner_loss_coef
        self.cat_loss_coef = cat_loss_coef
        self.rel_loss_coef = rel_loss_coef
        self.adjacency_loss_coef = adjacency_loss_coef
        self.decoder_loss_coef = decoder_loss_coef
        self.count_loss_coef = count_loss_coef
        self.groups_loss_coef = groups_loss_coef
        self.embedding_loss_coef = embedding_loss_coef

        # Special tokens
        self.seq_token = seq_token
        self.cat_token = cat_token
        self.rel_token = rel_token
        self.parent_token = parent_token
        self.child_token = child_token
