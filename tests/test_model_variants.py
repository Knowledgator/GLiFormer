from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import glinext.model as model_module

from glinext.config import (
    GLiNextAudioConfig,
    GLiNextConfig,
    GLiNextLayoutConfig,
    GLiNextOmniConfig,
    GLiNextTextConfig,
    GLiNextVisionConfig,
    ObjectDetectionHeadConfig,
)
from glinext.model import (
    GLiNExTAudioModel,
    GLiNExTAudioOutput,
    GLiNExTLayoutModel,
    GLiNExTLayoutOutput,
    GLiNExTOmniModel,
    GLiNExTOmniOutput,
    GLiNExTTextModel,
    GLiNExTTextOutput,
    GLiNExTVisionModel,
    GLiNExTVisionOutput,
    resolve_glinext_model_class,
)
from glinext.glinext import (
    BaseGLiNeXT,
    GLiNExT,
    GLiNExTAudio,
    GLiNExTLayout,
    GLiNExTOmni,
    GLiNExTText,
    GLiNExTVision,
)
from glinext.encoders.audio import AudioBiEncoder
from glinext.encoders.base import Transformer
from glinext.encoders.omni import LayoutBiEncoder, LayoutEncoder
from glinext.encoders.vision import VisionBiEncoder
from glinext.backbones.deberta_2d import (
    LayoutDebertaConfig,
    LayoutDebertaEmbeddings,
    LayoutDebertaModel,
)
from glinext.processing.mappings import (
    BaseClassMapping,
    BatchClassesMapping,
    CatClassMapping,
    ExtractionClassMapping,
    ExtractionItemMapping,
    OpenRelexClassMapping,
    OpenRelexItemMapping,
    StructuringClassMapping,
    StructuringItemMapping,
    VisionClassMapping,
    VisionItemMapping,
)
from glinext.tasks import TASK_REGISTRY, TaskFlatInputs, TaskHeadOutput


def make_config(**overrides):
    defaults = {
        "model_name": "unused",
        "hidden_size": 16,
        "vocab_size": 64,
        "ner_config": None,
    }
    defaults.update(overrides)
    return GLiNextConfig(**defaults)


def test_named_label_batches_share_one_padded_encoder_pass():
    class RecordingLabelsEncoder:
        def __init__(self):
            self.calls = []

        def encode_labels(self, input_ids, attention_mask):
            self.calls.append((input_ids.clone(), attention_mask.clone()))
            return input_ids.float()

    labels_encoder = RecordingLabelsEncoder()
    owner = SimpleNamespace(token_rep_layer=labels_encoder)
    encoded = model_module.BaseGLiNextModel._encode_label_inputs_batched(
        owner,
        {
            "classification": (
                torch.tensor([[1, 2], [3, 4]]),
                torch.ones(2, 2, dtype=torch.long),
            ),
            "joint_relex": (None, None),
            "structuring": (
                torch.tensor([[5, 6, 7]]),
                torch.ones(1, 3, dtype=torch.long),
            ),
        },
    )

    assert len(labels_encoder.calls) == 1
    assert torch.equal(
        labels_encoder.calls[0][0],
        torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 7]]),
    )
    assert encoded["classification"].shape == (2, 3)
    assert encoded["joint_relex"] is None
    assert torch.equal(encoded["structuring"], torch.tensor([[5.0, 6.0, 7.0]]))


def test_word_rnn_skips_empty_rows_in_mixed_batches():
    class RecordingRNN:
        def __init__(self):
            self.calls = []

        def __call__(self, embeddings, mask):
            self.calls.append((embeddings.shape, mask.clone()))
            assert mask.bool().any(dim=1).all()
            return embeddings + 1.0

    rnn = RecordingRNN()
    owner = SimpleNamespace(rnn=rnn)
    words = torch.zeros(3, 4, 2)
    mask = torch.tensor([
        [True, True, False, False],
        [False, False, False, False],
        [True, False, False, False],
    ])

    encoded = model_module.BaseGLiNextModel._apply_word_rnn(
        owner,
        words,
        mask,
    )

    assert rnn.calls[0][0] == (2, 4, 2)
    assert torch.equal(encoded[0], torch.ones(4, 2))
    assert not encoded[1].any()
    assert torch.equal(encoded[2], torch.ones(4, 2))


def test_word_rnn_accepts_a_completely_empty_word_axis():
    class UnexpectedRNN:
        def __call__(self, *args, **kwargs):
            raise AssertionError("RNN was called for an empty word tensor")

    owner = SimpleNamespace(rnn=UnexpectedRNN())
    words = torch.empty(2, 0, 3)
    mask = torch.empty(2, 0, dtype=torch.bool)

    encoded = model_module.BaseGLiNextModel._apply_word_rnn(
        owner,
        words,
        mask,
    )

    assert encoded is words


def test_resolves_explicit_single_modality_models():
    assert resolve_glinext_model_class(make_config(model_variant="text")) is GLiNExTTextModel
    assert resolve_glinext_model_class(make_config(model_variant="layout")) is GLiNExTLayoutModel
    assert resolve_glinext_model_class(make_config(model_variant="vision")) is GLiNExTVisionModel
    assert resolve_glinext_model_class(make_config(model_variant="audio")) is GLiNExTAudioModel
    assert resolve_glinext_model_class(make_config(model_variant="omni")) is GLiNExTOmniModel


@pytest.mark.parametrize(
    "variant",
    [
        "text-only",
        "textonly",
        "text-vision",
        "text-audio",
        "text-vision-audio",
        "vision-only",
        "image",
        "image-only",
        "vision-bi-encoder",
        "image-bi-encoder",
        "audio-only",
        "audio-bi-encoder",
        "text-vision-layout",
        "vision-layout",
    ],
)
def test_model_variant_aliases_are_rejected(variant):
    with pytest.raises(ValueError, match="model_variant"):
        make_config(model_variant=variant)


def test_intermediate_text_media_model_classes_are_removed():
    assert not hasattr(model_module, "GLiNExTTextVisionBiEncoderModel")
    assert not hasattr(model_module, "GLiNExTTextVisionUniEncoderModel")
    assert not hasattr(model_module, "GLiNExTTextAudioModel")
    assert not hasattr(model_module, "GLiNExTTextAudioUniEncoderModel")


def test_omni_model_infers_active_modalities():
    assert GLiNExTOmniModel._infer_omni_modalities(make_config(model_variant="omni")) == ("text", "vision", "audio")


def test_text_variant_does_not_infer_omni_from_encoder_fields():
    config = make_config(model_variant="text", vision_encoder_type="patch", audio_encoder_type="conv")
    assert resolve_glinext_model_class(config) is GLiNExTTextModel


def test_omni_model_rejects_unknown_variant_for_modalities():
    with pytest.raises(ValueError, match="requires model_variant"):
        GLiNExTOmniModel._infer_omni_modalities(make_config(model_variant="text"))


def test_user_facing_factory_resolves_concrete_wrappers():
    assert GLiNExT._get_glinext_class(make_config(model_variant="text")) is GLiNExTText
    assert GLiNExT._get_glinext_class(make_config(model_variant="layout")) is GLiNExTLayout
    assert GLiNExT._get_glinext_class(make_config(model_variant="vision")) is GLiNExTVision
    assert GLiNExT._get_glinext_class(make_config(model_variant="audio")) is GLiNExTAudio
    assert GLiNExT._get_glinext_class(make_config(model_variant="omni")) is GLiNExTOmni


def test_user_facing_wrappers_pin_internal_model_classes():
    assert issubclass(GLiNExTText, BaseGLiNeXT)
    assert GLiNExTText.model_class is GLiNExTTextModel
    assert GLiNExTLayout.model_class is GLiNExTLayoutModel
    assert GLiNExTVision.model_class is GLiNExTVisionModel
    assert GLiNExTAudio.model_class is GLiNExTAudioModel
    assert GLiNExTOmni.model_class is GLiNExTOmniModel


def test_structure_forwards_an_explicit_objectness_threshold():
    class Probe:
        _normalize_texts = staticmethod(BaseGLiNeXT._normalize_texts)
        _single_or_batch = staticmethod(BaseGLiNeXT._single_or_batch)

        def inference(self, texts, **kwargs):
            self.inference_call = (texts, kwargs)
            return {"structuring": [{"record": []}]}

    probe = Probe()

    result = BaseGLiNeXT.structure(
        probe,
        "document",
        structures={"record": ["field"]},
        threshold=0.4,
        objectness_threshold=0.7,
    )

    assert result == {"record": []}
    assert probe.inference_call[1]["threshold"] == 0.4
    assert probe.inference_call[1]["objectness_threshold"] == 0.7


def test_structure_formats_direct_typed_templates():
    class Probe:
        _normalize_texts = staticmethod(BaseGLiNeXT._normalize_texts)
        _single_or_batch = staticmethod(BaseGLiNeXT._single_or_batch)

        def inference(self, texts, **kwargs):
            return {
                "structuring": [
                    {"record": [{"count": "3", "active": "yes"}]}
                ]
            }

    result = BaseGLiNeXT.structure(
        Probe(),
        "document",
        structures={
            "record": {"count": "integer", "active": "boolean"}
        },
    )

    assert result == {"record": [{"count": 3, "active": True}]}


def test_structure_returns_opt_in_anchor_diagnostics():
    diagnostics = {
        "summary": {
            "activated_anchor_count": 2,
            "selected_connection_count": 1,
        },
        "groups": [],
    }

    class Probe:
        _normalize_texts = staticmethod(BaseGLiNeXT._normalize_texts)
        _single_or_batch = staticmethod(BaseGLiNeXT._single_or_batch)

        def inference(self, texts, **kwargs):
            assert kwargs["return_anchor_diagnostics"] is True
            return {
                "structuring": [{"record": [{"field": "value"}]}],
                "structuring_anchor_diagnostics": [diagnostics],
            }

    result = BaseGLiNeXT.structure(
        Probe(),
        "document",
        structures={"record": ["field"]},
        return_anchor_diagnostics=True,
    )

    assert result == (
        {"record": [{"field": "value"}]},
        diagnostics,
    )


def test_task_registry_exposes_head_order():
    assert TASK_REGISTRY.execution_order == (
        "ner",
        "classification",
        "count",
        "joint_relex",
        "open_relex",
        "structuring",
        "image_classification",
        "object_detection",
        "segmentation",
        "audio_classification",
        "audio_segmentation",
        "embedding",
    )
    assert TASK_REGISTRY.text_tasks == (
        "ner",
        "classification",
        "count",
        "joint_relex",
        "open_relex",
        "structuring",
        "embedding",
    )
    assert TASK_REGISTRY.vision_tasks == (
        "image_classification",
        "object_detection",
        "segmentation",
    )
    assert TASK_REGISTRY.audio_tasks == (
        "audio_classification",
        "audio_segmentation",
    )
    assert [head.name for head in TASK_REGISTRY.head_classes()] == list(TASK_REGISTRY.execution_order)


def test_predict_relations_uses_canonical_entity_first_inference_argument():
    triple = {
        "head": {"text": "Alice"},
        "tail": {"text": "Acme"},
        "relation": "works_at",
    }

    class Probe:
        _normalize_texts = staticmethod(BaseGLiNeXT._normalize_texts)
        _single_or_batch = staticmethod(BaseGLiNeXT._single_or_batch)
        _label_group_count = staticmethod(BaseGLiNeXT._label_group_count)
        _collapse_single_group_results = staticmethod(
            BaseGLiNeXT._collapse_single_group_results
        )

        def inference(self, texts, **kwargs):
            assert texts == ["Alice Acme"]
            assert kwargs["relations"] == ["works_at"]
            return {"open_relex": [[[triple]]]}

    result = BaseGLiNeXT.predict_relations(
        Probe(),
        "Alice Acme",
        ["works_at"],
    )

    assert result == [triple]


def test_single_media_models_use_dedicated_bi_encoders():
    assert GLiNExTVisionModel.bi_encoder_cls is VisionBiEncoder
    assert GLiNExTAudioModel.bi_encoder_cls is AudioBiEncoder
    assert GLiNExTLayoutModel.layout_encoder_cls is LayoutEncoder
    assert GLiNExTLayoutModel.layout_bi_encoder_cls is LayoutBiEncoder


def test_media_parent_embedding_source_rejects_unknown_values():
    with pytest.raises(ValueError, match="media_parent_embedding_source"):
        make_config(media_parent_embedding_source="pooled")


def test_media_parent_embedding_source_fixed_uses_task_parameter():
    dummy = SimpleNamespace(
        config=make_config(media_parent_embedding_source="fixed"),
        media_parent_embeddings={"object_detection": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    media_tokens = torch.zeros(2, 3, 4)
    media_mask = torch.ones(2, 3)

    parent = GLiNExTVisionModel._media_parent_embedding_for_task(
        dummy,
        "object_detection",
        media_tokens,
        media_mask,
        device=media_tokens.device,
        dtype=media_tokens.dtype,
    )

    assert parent.tolist() == [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]


def test_media_parent_embedding_source_mean_uses_masked_media_tokens():
    dummy = SimpleNamespace(config=make_config(media_parent_embedding_source="average"))
    media_tokens = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]],
        [[5.0, 6.0], [100.0, 100.0], [100.0, 100.0]],
    ])
    media_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    parent = GLiNExTVisionModel._media_parent_embedding_for_task(
        dummy,
        "object_detection",
        media_tokens,
        media_mask,
        device=media_tokens.device,
        dtype=media_tokens.dtype,
    )

    assert parent.tolist() == [[2.0, 3.0], [5.0, 6.0]]


def test_media_parent_embedding_source_first_uses_first_valid_media_token():
    dummy = SimpleNamespace(config=make_config(media_parent_embedding_source="first"))
    media_tokens = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
    ])
    media_mask = torch.tensor([[0, 1], [1, 1]])

    parent = GLiNExTVisionModel._media_parent_embedding_for_task(
        dummy,
        "object_detection",
        media_tokens,
        media_mask,
        device=media_tokens.device,
        dtype=media_tokens.dtype,
    )

    assert parent.tolist() == [[3.0, 4.0], [5.0, 6.0]]


def test_layout_encoder_forwards_layout_inputs_by_name():
    captured = {}

    class FakeLayoutBackbone:
        model = SimpleNamespace(
            embeddings=SimpleNamespace(page_embeddings=object()),
            supports_page_input_mask=True,
        )

        def __call__(self, **kwargs):
            captured.update(kwargs)
            batch, seq_len = kwargs["input_ids"].shape
            return torch.zeros(batch, seq_len + 2, 8)

    encoder = object.__new__(LayoutEncoder)
    encoder.bert_layer = FakeLayoutBackbone()
    input_ids = torch.ones(1, 4, dtype=torch.long)
    attention_mask = torch.ones(1, 4, dtype=torch.long)
    bbox = torch.zeros(1, 4, 4, dtype=torch.long)
    layout_input_mask = torch.ones(1, dtype=torch.bool)
    page_token_ids = torch.zeros(1, 4, dtype=torch.long)
    page_input_mask = torch.ones(1, dtype=torch.bool)
    pixel_values = torch.zeros(1, 3, 16, 16)

    output = LayoutEncoder.forward(
        encoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        bbox=bbox,
        layout_input_mask=layout_input_mask,
        page_token_ids=page_token_ids,
        page_input_mask=page_input_mask,
        pixel_values=pixel_values,
    )

    assert captured["input_ids"] is input_ids
    assert captured["attention_mask"] is attention_mask
    assert captured["bbox"] is bbox
    assert captured["layout_input_mask"] is layout_input_mask
    assert captured["page_token_ids"] is page_token_ids
    assert captured["page_input_mask"] is page_input_mask
    assert captured["pixel_values"] is pixel_values
    assert output.shape == (1, 4, 8)


def test_layout_deberta_adds_page_token_embeddings():
    config = LayoutDebertaConfig(
        vocab_size=16,
        hidden_size=8,
        embedding_size=8,
        max_position_embeddings=16,
        type_vocab_size=0,
        max_page_embeddings=4,
        hidden_dropout_prob=0.0,
        layout_embedding_type="none",
    )
    embeddings = LayoutDebertaEmbeddings(config)
    embeddings.eval()
    input_ids = torch.ones(1, 3, dtype=torch.long)
    page_zero = torch.zeros(1, 3, dtype=torch.long)
    page_one = torch.ones(1, 3, dtype=torch.long)

    output_missing = embeddings(input_ids=input_ids)
    output_masked = embeddings(
        input_ids=input_ids,
        page_token_ids=page_zero,
        page_input_mask=torch.tensor([False]),
    )
    output_zero = embeddings(input_ids=input_ids, page_token_ids=page_zero)
    output_one = embeddings(input_ids=input_ids, page_token_ids=page_one)

    assert output_missing.shape == output_masked.shape == output_zero.shape == output_one.shape == (1, 3, 8)
    assert torch.allclose(output_missing, output_masked)
    assert not torch.allclose(output_missing, output_zero)
    assert not torch.allclose(output_zero, output_one)


def test_layout_deberta_page_gate_supports_mixed_batches():
    config = LayoutDebertaConfig(
        vocab_size=16,
        hidden_size=8,
        embedding_size=8,
        max_position_embeddings=16,
        type_vocab_size=0,
        max_page_embeddings=4,
        hidden_dropout_prob=0.0,
        layout_embedding_type="none",
    )
    embeddings = LayoutDebertaEmbeddings(config).eval()
    input_ids = torch.ones(2, 3, dtype=torch.long)
    page_token_ids = torch.ones_like(input_ids)

    mixed = embeddings(
        input_ids=input_ids,
        page_token_ids=page_token_ids,
        page_input_mask=torch.tensor([True, False]),
    )
    text_only = embeddings(input_ids=input_ids[1:])

    assert not torch.allclose(mixed[:1], text_only)
    assert torch.allclose(mixed[1:], text_only)


def test_layout_deberta_rejects_page_mask_without_page_ids():
    config = LayoutDebertaConfig(
        vocab_size=16,
        hidden_size=8,
        embedding_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        type_vocab_size=0,
        max_page_embeddings=4,
        layout_embedding_type="none",
    )
    model = LayoutDebertaModel(config)

    with pytest.raises(ValueError, match="requires page_token_ids"):
        model(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            page_input_mask=torch.tensor([False]),
        )


def test_max_page_embeddings_is_wired_to_deberta_2d_backbone():
    encoder_config = LayoutDebertaConfig(
        vocab_size=16,
        hidden_size=8,
        embedding_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        type_vocab_size=0,
        max_page_embeddings=1024,
        layout_embedding_type="none",
    )
    config = make_config(
        backbone_type="deberta_2d",
        encoder_config=encoder_config,
        max_page_embeddings=7,
    )

    transformer = Transformer("unused", config)

    assert transformer.model.config.max_page_embeddings == 7
    assert transformer.model.embeddings.page_embeddings.num_embeddings == 7


def test_layout_deberta_masked_row_matches_missing_bbox():
    config = LayoutDebertaConfig(
        vocab_size=32,
        hidden_size=12,
        embedding_size=12,
        num_hidden_layers=1,
        num_attention_heads=3,
        intermediate_size=24,
        max_position_embeddings=16,
        type_vocab_size=0,
        max_page_embeddings=0,
        coordinate_size=2,
        shape_size=2,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        relative_attention=True,
        position_buckets=8,
        max_relative_positions=16,
        layout_embedding_type="both",
    )
    model = LayoutDebertaModel(config).eval()
    input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    attention_mask = torch.ones_like(input_ids)
    bbox = torch.tensor(
        [
            [[10, 20, 30, 40], [40, 20, 60, 40], [10, 50, 30, 70], [40, 50, 60, 70]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )

    with torch.no_grad():
        mixed = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            layout_input_mask=torch.tensor([True, False]),
        ).last_hidden_state
        text_only = model(
            input_ids=input_ids[1:],
            attention_mask=attention_mask[1:],
            bbox=None,
        ).last_hidden_state
        unmasked = model(
            input_ids=input_ids[1:],
            attention_mask=attention_mask[1:],
            bbox=bbox[1:],
        ).last_hidden_state
        layout_bias = model.encoder.get_layout_attention_bias(
            bbox,
            torch.tensor([True, False]),
        )

    assert torch.allclose(mixed[1:], text_only, atol=1e-6)
    assert not torch.allclose(unmasked, text_only)
    assert torch.count_nonzero(layout_bias[1]).item() == 0


def test_layout_deberta_rejects_layout_mask_without_bbox():
    config = LayoutDebertaConfig(
        vocab_size=16,
        hidden_size=8,
        embedding_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=8,
        type_vocab_size=0,
        max_page_embeddings=0,
        layout_embedding_type="none",
    )
    model = LayoutDebertaModel(config)

    with pytest.raises(ValueError, match="requires bbox"):
        model(
            input_ids=torch.ones(1, 3, dtype=torch.long),
            layout_input_mask=torch.tensor([False]),
        )


def test_layout_kwargs_only_accepts_canonical_input_names():
    bbox = torch.zeros(1, 4, 4, dtype=torch.long)
    pixel_values = torch.zeros(1, 3, 16, 16)
    result = GLiNExTLayoutModel._layout_kwargs({"bbox": bbox, "pixel_values": pixel_values})

    assert set(result) == {"bbox", "pixel_values"}
    assert result["bbox"] is bbox
    assert result["pixel_values"] is pixel_values

    layout_input_mask = torch.tensor([True])
    result = GLiNExTLayoutModel._layout_kwargs(
        {"bbox": bbox, "layout_input_mask": layout_input_mask}
    )
    assert result["layout_input_mask"] is layout_input_mask

    page_token_ids = torch.zeros(1, 4, dtype=torch.long)
    page_input_mask = torch.tensor([True])
    result = GLiNExTLayoutModel._layout_kwargs(
        {"page_token_ids": page_token_ids, "page_input_mask": page_input_mask}
    )
    assert result["page_token_ids"] is page_token_ids
    assert result["page_input_mask"] is page_input_mask

    with pytest.raises(ValueError, match="canonical input names"):
        GLiNExTLayoutModel._layout_kwargs({"word_bboxes": bbox})

    with pytest.raises(ValueError, match="canonical input names"):
        GLiNExTLayoutModel._layout_kwargs({"layout_bbox": bbox})


def test_omni_kwargs_only_accepts_canonical_input_names():
    pixel_values = torch.zeros(1, 3, 16, 16)
    audio_values = torch.zeros(1, 16000)
    result = GLiNExTOmniModel._omni_kwargs({"pixel_values": pixel_values, "audio_values": audio_values})

    assert set(result) == {"pixel_values", "audio_values"}
    assert result["pixel_values"] is pixel_values
    assert result["audio_values"] is audio_values

    with pytest.raises(ValueError, match="canonical input names"):
        GLiNExTOmniModel._omni_kwargs({"input_values": audio_values})

    with pytest.raises(ValueError, match="canonical input names"):
        GLiNExTOmniModel._omni_kwargs({"vision_pixel_values": pixel_values})


def test_text_and_layout_forward_use_text_task_path(monkeypatch):
    calls = []

    def fake_text_forward(self, *args, **kwargs):
        calls.append(self.__class__.__name__)
        return self.output_cls()

    def fail_joint_forward(self, *args, **kwargs):
        raise AssertionError("text/layout forward should not call a generic or omni path")

    monkeypatch.setattr(GLiNExTTextModel, "_forward_text_task_heads", fake_text_forward)
    monkeypatch.setattr(GLiNExTLayoutModel, "_forward_text_task_heads", fake_text_forward)
    monkeypatch.setattr(GLiNExTTextModel, "_forward_task_heads", fail_joint_forward)
    monkeypatch.setattr(GLiNExTLayoutModel, "_forward_task_heads", fail_joint_forward)
    monkeypatch.setattr(GLiNExTTextModel, "_forward_all_tasks", fail_joint_forward)
    monkeypatch.setattr(GLiNExTLayoutModel, "_forward_all_tasks", fail_joint_forward)
    monkeypatch.setattr(GLiNExTTextModel, "_forward_omni_task_heads", fail_joint_forward)
    monkeypatch.setattr(GLiNExTLayoutModel, "_forward_omni_task_heads", fail_joint_forward)

    text_model = object.__new__(GLiNExTTextModel)
    layout_model = object.__new__(GLiNExTLayoutModel)

    assert isinstance(GLiNExTTextModel.forward(text_model), GLiNExTTextOutput)
    assert isinstance(GLiNExTLayoutModel.forward(layout_model), GLiNExTLayoutOutput)
    assert calls == ["GLiNExTTextModel", "GLiNExTLayoutModel"]


def test_layout_forward_accepts_pixel_values_and_rejects_audio(monkeypatch):
    captured = {}

    def fake_text_forward(self, *args, **kwargs):
        captured.update(kwargs)
        return self.output_cls()

    monkeypatch.setattr(GLiNExTLayoutModel, "_forward_text_task_heads", fake_text_forward)

    layout_model = object.__new__(GLiNExTLayoutModel)
    pixel_values = torch.zeros(1, 3, 16, 16)
    bbox = torch.zeros(1, 4, 4, dtype=torch.long)

    assert isinstance(
        GLiNExTLayoutModel.forward(layout_model, pixel_values=pixel_values, bbox=bbox),
        GLiNExTLayoutOutput,
    )
    assert captured["pixel_values"] is pixel_values
    assert captured["bbox"] is bbox

    with pytest.raises(ValueError, match="unsupported media arguments"):
        GLiNExTLayoutModel.forward(layout_model, audio_values=torch.zeros(1, 16000))


def test_omni_forward_uses_omni_task_path(monkeypatch):
    def fake_omni_forward(self, *args, **kwargs):
        return self.output_cls()

    def fail_text_forward(self, *args, **kwargs):
        raise AssertionError("omni forward should not call the text-only path")

    monkeypatch.setattr(GLiNExTOmniModel, "_forward_omni_task_heads", fake_omni_forward)
    monkeypatch.setattr(GLiNExTOmniModel, "_forward_text_task_heads", fail_text_forward)

    omni_model = object.__new__(GLiNExTOmniModel)

    assert isinstance(GLiNExTOmniModel.forward(omni_model), GLiNExTOmniOutput)


def test_media_task_features_do_not_fallback_to_text_embeddings():
    words = torch.ones(1, 2, 4)
    word_mask = torch.ones(1, 2, dtype=torch.long)
    vision = torch.full((1, 3, 4), 2.0)
    vision_mask = torch.ones(1, 3, dtype=torch.long)
    audio = torch.full((1, 5, 4), 3.0)
    audio_mask = torch.ones(1, 5, dtype=torch.long)

    image_features, image_mask = GLiNExTOmniModel._features_for_task(
        task_name="image_classification",
        words_embedding=words,
        word_mask=word_mask,
        vision_embedding=vision,
        vision_mask=vision_mask,
        audio_embedding=audio,
        audio_mask=audio_mask,
    )
    audio_features, returned_audio_mask = GLiNExTOmniModel._features_for_task(
        task_name="audio_classification",
        words_embedding=words,
        word_mask=word_mask,
        vision_embedding=vision,
        vision_mask=vision_mask,
        audio_embedding=audio,
        audio_mask=audio_mask,
    )

    assert image_features is vision
    assert image_mask is vision_mask
    assert audio_features is audio
    assert returned_audio_mask is audio_mask


def test_media_task_features_raise_when_modality_missing():
    words = torch.ones(1, 2, 4)
    word_mask = torch.ones(1, 2, dtype=torch.long)

    with pytest.raises(ValueError, match="requires vision embeddings"):
        GLiNExTOmniModel._features_for_task(
            task_name="object_detection",
            words_embedding=words,
            word_mask=word_mask,
            vision_embedding=None,
            vision_mask=None,
            audio_embedding=None,
            audio_mask=None,
        )

    with pytest.raises(ValueError, match="requires audio embeddings"):
        GLiNExTOmniModel._features_for_task(
            task_name="audio_segmentation",
            words_embedding=words,
            word_mask=word_mask,
            vision_embedding=None,
            vision_mask=None,
            audio_embedding=None,
            audio_mask=None,
        )


def test_text_task_path_does_not_call_generic_forward(monkeypatch):
    captured_batch_kwargs = {}

    def fail_generic_forward(self, *args, **kwargs):
        raise AssertionError("text task path should not delegate to _forward_task_heads")

    def fake_get_representations(self, *args, **kwargs):
        token_embeds = torch.zeros(1, 3, 4)
        prompts_embedding = torch.zeros(1, 1, 4)
        prompts_embedding_mask = torch.ones(1, 1, dtype=torch.long)
        words_embedding = torch.zeros(1, 2, 4)
        mask = torch.ones(1, 2, dtype=torch.long)
        return token_embeds, prompts_embedding, prompts_embedding_mask, words_embedding, mask

    monkeypatch.setattr(GLiNExTTextModel, "_forward_task_heads", fail_generic_forward)
    monkeypatch.setattr(GLiNExTTextModel, "get_representations", fake_get_representations)
    monkeypatch.setattr(
        GLiNExTTextModel,
        "_encode_all_labels_batched",
        lambda self, *args: (None, None, None, None),
    )
    monkeypatch.setattr(
        GLiNExTTextModel,
        "_build_forward_flat_inputs",
        lambda self, **kwargs: ({}, None, None),
    )
    monkeypatch.setattr(
        GLiNExTTextModel,
        "_encode_embedding_pair_inputs",
        lambda self, *args: (None, None),
    )
    def capture_head_inputs(self, **kwargs):
        del self
        captured_batch_kwargs.update(kwargs["batch_kwargs"])
        return None, {}

    monkeypatch.setattr(
        GLiNExTTextModel,
        "_execute_forward_heads",
        capture_head_inputs,
    )
    monkeypatch.setattr(
        GLiNExTTextModel,
        "_collect_forward_output",
        lambda self, **kwargs: self.output_cls(),
    )

    text_model = object.__new__(GLiNExTTextModel)
    structuring_span_idx = torch.tensor([[[0, 0]]])
    structuring_span_mask = torch.ones(1, 1, dtype=torch.bool)
    structuring_span_labels = torch.ones(1, 1, 1, 1)

    assert isinstance(
        GLiNExTTextModel._forward_text_task_heads(
            text_model,
            structuring_span_idx=structuring_span_idx,
            structuring_span_mask=structuring_span_mask,
            structuring_span_labels=structuring_span_labels,
        ),
        GLiNExTTextOutput,
    )
    assert captured_batch_kwargs["structuring_span_idx"] is structuring_span_idx
    assert captured_batch_kwargs["structuring_span_mask"] is structuring_span_mask
    assert (
        captured_batch_kwargs["structuring_span_labels"]
        is structuring_span_labels
    )


def test_text_model_forward_returns_exact_text_output(monkeypatch):
    def fake_text_forward(self, *args, **kwargs):
        return GLiNExTOmniOutput(
            ner_logits=torch.zeros(1, 1, 1),
            image_classification_logits=torch.zeros(1, 1),
            vision_embedding=torch.zeros(1, 1, 1),
        )

    monkeypatch.setattr(GLiNExTTextModel, "_forward_text_task_heads", fake_text_forward)

    text_model = object.__new__(GLiNExTTextModel)
    output = GLiNExTTextModel.forward(text_model)

    assert type(output) is GLiNExTTextOutput
    assert output.ner_logits is not None
    assert not hasattr(output, "image_classification_logits")
    assert not hasattr(output, "vision_embedding")


def test_omni_task_path_forwards_structuring_span_targets(monkeypatch):
    captured_batch_kwargs = {}
    representations = {
        "token_embeds": torch.zeros(1, 3, 4),
        "prompts_embedding": torch.zeros(1, 1, 4),
        "prompts_embedding_mask": torch.ones(1, 1, dtype=torch.long),
        "words_embedding": torch.zeros(1, 2, 4),
        "mask": torch.ones(1, 2, dtype=torch.long),
        "vision_embedding": None,
        "vision_mask": None,
        "audio_embedding": None,
        "audio_mask": None,
        "vision_spatial_shape": None,
        "vision_prefix_tokens": None,
    }
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_encode_forward_representations",
        lambda self, **kwargs: representations,
    )
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_encode_all_labels_batched",
        lambda self, *args: (None, None, None, None),
    )
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_encode_media_labels_batched",
        lambda self, kwargs: None,
    )
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_build_forward_flat_inputs",
        lambda self, **kwargs: ({}, None, None),
    )
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_encode_embedding_pair_inputs",
        lambda self, *args: (None, None),
    )

    def capture_head_inputs(self, **kwargs):
        del self
        captured_batch_kwargs.update(kwargs["batch_kwargs"])
        return None, {}

    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_execute_forward_heads",
        capture_head_inputs,
    )
    monkeypatch.setattr(
        GLiNExTOmniModel,
        "_collect_forward_output",
        lambda self, **kwargs: self.output_cls(),
    )

    model = object.__new__(GLiNExTOmniModel)
    structuring_span_idx = torch.tensor([[[0, 0]]])
    structuring_span_mask = torch.ones(1, 1, dtype=torch.bool)
    structuring_span_labels = torch.ones(1, 1, 1, 1)

    output = GLiNExTOmniModel._forward_task_heads(
        model,
        structuring_span_idx=structuring_span_idx,
        structuring_span_mask=structuring_span_mask,
        structuring_span_labels=structuring_span_labels,
    )

    assert isinstance(output, GLiNExTOmniOutput)
    assert captured_batch_kwargs["structuring_span_idx"] is structuring_span_idx
    assert captured_batch_kwargs["structuring_span_mask"] is structuring_span_mask
    assert (
        captured_batch_kwargs["structuring_span_labels"]
        is structuring_span_labels
    )


def test_collect_output_preserves_both_structuring_stages():
    model = object.__new__(GLiNExTTextModel)
    entity_logits = torch.zeros(1, 4, 2, 3)
    field_logits = torch.zeros(1, 3, 2)
    membership_logits = torch.zeros(1, 4, 3)
    assignment_logits = membership_logits.transpose(1, 2)
    span_idx = torch.zeros(1, 3, 2, dtype=torch.long)
    span_mask = torch.ones(1, 3, dtype=torch.bool)
    anchor_mask = torch.ones(1, 4, dtype=torch.bool)
    origin = torch.tensor([0])
    features = torch.zeros(1, 4, 8)
    feature_mask = torch.ones(1, 4, dtype=torch.long)
    prompts = torch.zeros(1, 2, 8)
    prompt_mask = torch.ones(1, 2, dtype=torch.long)

    output = GLiNExTTextModel._collect_forward_output(
        model,
        final_loss=None,
        head_outputs={
            "structuring": TaskHeadOutput(
                logits=entity_logits,
                extra={
                    "entity_field_logits": field_logits,
                    "structuring_logits": membership_logits,
                    "entity_assignment_logits": assignment_logits,
                    "span_idx": span_idx,
                    "span_mask": span_mask,
                    "anchor_mask": anchor_mask,
                },
            )
        },
        flat_inputs_map={
            "structuring": SimpleNamespace(batch_origin=origin)
        },
        batch_size=1,
        words_embedding=features,
        mask=feature_mask,
        prompts_embedding=prompts,
        prompts_embedding_mask=prompt_mask,
        vision_embedding=None,
        vision_mask=None,
        audio_embedding=None,
        audio_mask=None,
    )

    assert output.structuring_entity_logits is entity_logits
    assert output.structuring_field_logits is field_logits
    assert output.structuring_logits is membership_logits
    assert output.structuring_assignment_logits is assignment_logits
    assert output.structuring_span_idx is span_idx
    assert output.structuring_span_mask is span_mask
    assert output.structuring_anchor_mask is anchor_mask


def test_modality_outputs_preserve_flat_attribute_api():
    assert hasattr(GLiNExTTextOutput(), "ner_logits")
    assert hasattr(GLiNExTLayoutOutput(), "structuring_logits")
    assert hasattr(GLiNExTTextOutput(), "structuring_field_logits")
    assert hasattr(GLiNExTTextOutput(), "structuring_assignment_logits")
    assert not hasattr(GLiNExTTextOutput(), "set_structuring_field_logits")
    assert hasattr(GLiNExTVisionOutput(), "object_detection_logits")
    assert hasattr(GLiNExTAudioOutput(), "audio_segmentation_logits")
    assert hasattr(GLiNExTOmniOutput(), "ner_logits")
    assert hasattr(GLiNExTOmniOutput(), "image_classification_logits")
    assert hasattr(GLiNExTOmniOutput(), "audio_classification_logits")


def test_specialized_config_variants_and_defaults():
    text = GLiNextTextConfig(model_name="unused")
    assert text.model_variant == "text"
    assert text.model_type == "glinext-text"
    layout = GLiNextLayoutConfig(model_name="unused")
    assert layout.model_variant == "layout"
    assert layout.model_type == "glinext-layout"
    assert layout.use_layout is True
    vision = GLiNextVisionConfig(model_name="unused")
    assert vision.model_variant == "vision"
    assert vision.model_type == "glinext-vision"
    audio = GLiNextAudioConfig(model_name="unused")
    assert audio.model_variant == "audio"
    assert audio.model_type == "glinext-audio"
    omni = GLiNextOmniConfig(model_name="unused")
    assert omni.model_variant == "omni"
    assert omni.model_type == "glinext-omni"


def test_specialized_configs_reject_wrong_variant():
    with pytest.raises(ValueError, match="requires model_variant='text'"):
        GLiNextTextConfig(model_name="unused", model_variant="vision")
    with pytest.raises(ValueError, match="requires model_variant='layout'"):
        GLiNextLayoutConfig(model_name="unused", model_variant="text")


def test_single_media_configs_do_not_default_enable_ner():
    assert GLiNextVisionConfig(model_name="unused").ner_config is None
    assert GLiNextAudioConfig(model_name="unused").ner_config is None
    assert GLiNextTextConfig(model_name="unused").ner_config is not None


def test_specialized_configs_reject_cross_modality_task_configs():
    with pytest.raises(ValueError, match="text tasks only"):
        GLiNextTextConfig(model_name="unused", image_classification_config={})
    with pytest.raises(ValueError, match="vision tasks only"):
        GLiNextVisionConfig(model_name="unused", classification_config={})
    with pytest.raises(ValueError, match="audio tasks only"):
        GLiNextAudioConfig(model_name="unused", segmentation_config={})


def test_factory_coerces_dicts_to_specialized_config_classes():
    assert isinstance(
        GLiNExT._coerce_config({
            "model_name": "unused",
            "model_type": "glinext-layout",
        }),
        GLiNextLayoutConfig,
    )
    assert isinstance(
        GLiNExT._coerce_config({
            "model_name": "unused",
            "model_variant": "vision",
            "image_classification_config": {},
        }),
        GLiNextVisionConfig,
    )
    assert isinstance(
        GLiNExT._coerce_config({
            "model_name": "unused",
            "model_variant": "audio",
            "audio_classification_config": {},
        }),
        GLiNextAudioConfig,
    )


def test_factory_model_type_takes_precedence_over_model_variant():
    config = GLiNExT._coerce_config({
        "model_name": "unused",
        "model_type": "glinext-layout",
        "model_variant": "vision",
    })
    assert isinstance(config, GLiNextLayoutConfig)
    assert config.model_variant == "layout"


def test_specialized_config_serialization_is_variant_specific():
    layout = GLiNextLayoutConfig(
        model_name="unused",
        structuring_config={},
        layout_image_tokens=False,
        max_page_embeddings=77,
    ).to_dict()
    assert layout["model_type"] == "glinext-layout"
    assert layout["model_variant"] == "layout"
    assert "structuring_config" in layout
    assert "audio_model_name" not in layout
    assert "audio_classification_config" not in layout
    assert "vision_encoder_config" not in layout
    assert layout["layout_image_tokens"] is False
    assert layout["max_page_embeddings"] == 77

    vision = GLiNextVisionConfig(model_name="unused", image_classification_config={}).to_dict()
    assert vision["model_type"] == "glinext-vision"
    assert "image_classification_config" in vision
    assert "ner_config" not in vision
    assert "audio_encoder_config" not in vision
    assert "represent_spans" not in vision
    assert "neg_spans_ratio" not in vision
    assert "span_loss_coef" not in vision

    omni = GLiNextOmniConfig(
        model_name="unused",
        layout_image_tokens=False,
        max_page_embeddings=91,
    ).to_dict()
    assert omni["layout_image_tokens"] is False
    assert omni["max_page_embeddings"] == 91


def test_dense_vision_config_rejects_untracked_geometric_processors():
    detection = asdict(ObjectDetectionHeadConfig())
    with pytest.raises(ValueError, match="center-cropped"):
        GLiNextVisionConfig(
            model_name="unused",
            object_detection_config=detection,
            vision_center_crop_size=128,
        )
    with pytest.raises(ValueError, match="vision_processor_type='custom'"):
        GLiNextVisionConfig(
            model_name="unused",
            object_detection_config=detection,
            vision_processor_type="auto",
        )


def test_local_learned_positions_default_to_patch_capacity():
    config = GLiNextVisionConfig(
        model_name="unused",
        vision_encoder_type="patch",
        image_size=32,
        vision_patch_size=8,
        vision_position_embedding_type="learned",
    )

    assert config.vision_position_embedding_kwargs["num_embeddings"] == 16


def test_legacy_detector_migration_is_non_mutating_and_preserves_semantics():
    legacy_detection = {
        "reference_points": True,
        "dense_coord_features": True,
        "pos_emb_scale": 1.0,
        # v3 serialized this value although the detector never consumed it.
        "anchor_modeling": "linear",
    }
    original = dict(legacy_detection)

    config = GLiNextConfig(
        model_name="unused",
        ner_config=None,
        object_detection_config=legacy_detection,
        segmentation_config=dict(legacy_detection),
    )

    assert legacy_detection == original
    assert config.object_detection_config.anchor_modeling == "identity"
    assert config.object_detection_config.scorer_type == "dot"
    assert config.object_detection_config.class_probability == "softmax"
    assert config.object_detection_config.multi_label is False
    assert config.object_detection_config.memory_position_in_values is True
    assert config.segmentation_config.reuse_detection_head is False
    assert GLiNextConfig(
        model_name="unused",
        ner_config=None,
        segmentation_config={},
    ).segmentation_config.reuse_detection_head is True


def test_modern_set_prediction_uses_scaled_dot_scoring():
    config = GLiNextConfig(
        model_name="unused",
        ner_config=None,
        object_detection_config={},
    )

    assert config.object_detection_config.scorer_type == "scaled-dot"


def test_registry_position_checkpoint_preserves_legacy_keys_only_values():
    config = GLiNextConfig(
        model_name="unused",
        ner_config=None,
        object_detection_config={
            "memory_position_embedding_type": "sine2d",
            "query_position_embedding_type": "sine2d",
        },
    )

    assert config.object_detection_config.memory_position_in_values is False


def test_factory_coercion_enforces_variant_specific_validation():
    with pytest.raises(ValueError, match="vision tasks only"):
        GLiNExT._coerce_config({
            "model_name": "unused",
            "model_variant": "vision",
            "classification_config": {},
        })

    permissive_base = GLiNextConfig(
        model_name="unused",
        model_variant="audio",
        segmentation_config={},
    )
    with pytest.raises(ValueError, match="audio tasks only"):
        GLiNExT._coerce_config(permissive_base)


def test_config_resolves_task_config_by_name():
    config = GLiNextOmniConfig(
        model_name="unused",
        classification_config={},
        image_classification_config={},
        audio_classification_config={},
        count_config={},
    )
    assert config.get_task_config("ner") is config.ner_config
    assert config.get_task_config("classification") is config.classification_config
    assert config.get_task_config("image_classification") is config.image_classification_config
    assert config.get_task_config("audio_classification") is config.audio_classification_config
    assert config.get_task_config("count") is config.count_config
    assert config.get_task_config("unknown") is None


def test_batch_classes_mapping_exposes_task_shape_helpers():
    one = BaseClassMapping({"a": 0})
    two = BaseClassMapping({"a": 0, "b": 1})
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([one, two])],
        extraction_mapping=[ExtractionClassMapping([ExtractionItemMapping(two)])],
        open_relex_mapping=[OpenRelexClassMapping([OpenRelexItemMapping(two)])],
        structuring_mapping=[StructuringClassMapping([StructuringItemMapping(two)])],
        image_classification_mapping=[VisionClassMapping([VisionItemMapping(two)])],
    )

    assert mapping.group_count("classification", 0) == 2
    assert mapping.group_count("ner", 0) == 1
    assert mapping.group_count("joint_relex", 0) == 1
    assert mapping.group_count("audio_classification", 0) == 0
    assert mapping.group_counts("classification", 1) == [2]
    assert mapping.parent_offset_for_item("open_relex", 0) == 3
    assert mapping.child_size("classification", 0, 1) == 2
    assert mapping.child_size("image_classification", 0, 0) == 2
    assert mapping.label_size("joint_relex", 0, 0, "relation") == 0
    assert len(list(mapping.flat_iter("joint_relex"))) == 1


def test_ner_only_flat_inputs_do_not_activate_joint_relex():
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([])],
        extraction_mapping=[ExtractionClassMapping([
            ExtractionItemMapping(BaseClassMapping({"person": 0})),
        ])],
    )
    model = object.__new__(GLiNExTTextModel)
    torch.nn.Module.__init__(model)
    model.heads = {"ner": None, "joint_relex": None}
    model.config = SimpleNamespace(
        uses_per_task_parents=False,
        parent_token_index=1,
        embed_parent_token=True,
    )

    token_embeds = torch.zeros(1, 3, 4)
    input_ids = torch.tensor([[1, 0, 0]])
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    words_embedding = torch.zeros(1, 2, 4)
    word_mask = torch.ones(1, 2, dtype=torch.long)
    child_embedding = torch.zeros(1, 1, 4)
    child_mask = torch.ones(1, 1, dtype=torch.long)

    flat_inputs, rel_prompts, rel_prompt_mask = model._build_forward_flat_inputs(
        classes_mapping=mapping,
        token_embeds=token_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        words_embedding=words_embedding,
        mask=word_mask,
        prompts_embedding=child_embedding,
        prompts_embedding_mask=child_mask,
        batch_size=1,
        embed_dim=4,
        cat_label_embeds=None,
        rel_label_embeds=None,
        child_label_embeds=None,
        open_rel_label_embeds=None,
        media_label_embeds=None,
        vision_embedding=None,
        vision_mask=None,
        audio_embedding=None,
        audio_mask=None,
        include_media=False,
        kwargs={},
    )

    assert set(flat_inputs) == {"ner"}
    assert rel_prompts is None
    assert rel_prompt_mask is None


def test_structuring_prompt_markers_are_partitioned_by_schema(monkeypatch):
    fields = BaseClassMapping({
        "root_name": 0,
        "children.child_name": 1,
        "children.child_code": 2,
    })
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping([
            StructuringItemMapping(fields),
        ])],
    )
    model = object.__new__(GLiNExTTextModel)
    torch.nn.Module.__init__(model)
    model.heads = {"structuring": None}
    model.config = SimpleNamespace(
        uses_per_task_parents=False,
        parent_token_index=1,
        embed_parent_token=True,
        structuring_config=SimpleNamespace(
            child_token_index=2,
            embed_child_token=True,
        ),
    )

    parent_markers = torch.tensor([[[100.0]]])
    field_markers = torch.tensor([[[11.0], [12.0], [13.0]]])

    def fake_extract_prompt_features(marker_index, *args, **kwargs):
        if marker_index == 1:
            return parent_markers, torch.ones(1, 1, dtype=torch.long)
        if marker_index == 2:
            return field_markers, torch.ones(1, 3, dtype=torch.long)
        raise AssertionError(f"unexpected prompt marker index {marker_index}")

    monkeypatch.setattr(
        model_module,
        "extract_prompt_features",
        fake_extract_prompt_features,
    )

    flat_inputs, _, _ = model._build_forward_flat_inputs(
        classes_mapping=mapping,
        token_embeds=torch.zeros(1, 1, 1),
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        words_embedding=torch.zeros(1, 1, 1),
        mask=torch.ones(1, 1, dtype=torch.long),
        prompts_embedding=torch.zeros(1, 0, 1),
        prompts_embedding_mask=torch.zeros(1, 0, dtype=torch.long),
        batch_size=1,
        embed_dim=1,
        cat_label_embeds=None,
        rel_label_embeds=None,
        child_label_embeds=None,
        open_rel_label_embeds=None,
        media_label_embeds=None,
        vision_embedding=None,
        vision_mask=None,
        audio_embedding=None,
        audio_mask=None,
        include_media=False,
        kwargs={},
    )

    assert set(flat_inputs) == {"structuring"}
    assert flat_inputs["structuring"].child_embedding[:, :, 0].tolist() == [
        [11.0, 12.0, 13.0],
    ]
    assert flat_inputs["structuring"].child_mask.tolist() == [
        [1.0, 1.0, 1.0],
    ]


def test_structuring_flat_inputs_accept_flat_label_encoder_namespace():
    fields = BaseClassMapping({"name": 0, "age": 1})
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([])],
        extraction_mapping=[ExtractionClassMapping()],
        structuring_mapping=[StructuringClassMapping([
            StructuringItemMapping(fields),
        ])],
    )
    model = object.__new__(GLiNExTTextModel)
    torch.nn.Module.__init__(model)
    model.heads = {"structuring": None}
    model.config = SimpleNamespace(
        uses_per_task_parents=False,
        parent_token_index=1,
        embed_parent_token=True,
        structuring_config=SimpleNamespace(
            child_token_index=2,
            embed_child_token=True,
        ),
    )

    # Label encoders return one flat row per label; the legacy group-size
    # tensor is what repacks those rows into batch/schema groups.
    legacy_labels = torch.tensor([[11.0], [12.0]])
    flat_inputs, _, _ = model._build_forward_flat_inputs(
        classes_mapping=mapping,
        token_embeds=torch.zeros(1, 1, 1),
        input_ids=torch.zeros(1, 1, dtype=torch.long),
        attention_mask=torch.ones(1, 1, dtype=torch.long),
        words_embedding=torch.zeros(1, 1, 1),
        mask=torch.ones(1, 1, dtype=torch.long),
        prompts_embedding=torch.zeros(1, 0, 1),
        prompts_embedding_mask=torch.zeros(1, 0, dtype=torch.long),
        batch_size=1,
        embed_dim=1,
        cat_label_embeds=None,
        rel_label_embeds=None,
        child_label_embeds=legacy_labels,
        open_rel_label_embeds=None,
        media_label_embeds=None,
        vision_embedding=None,
        vision_mask=None,
        audio_embedding=None,
        audio_mask=None,
        include_media=False,
        kwargs={"child_labels_group_size": torch.tensor([2])},
    )

    assert set(flat_inputs) == {"structuring"}
    assert flat_inputs["structuring"].child_embedding[:, :, 0].tolist() == [
        [11.0, 12.0]
    ]


def test_flat_inputs_pack_flat_biencoder_labels_by_group_size():
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([]), CatClassMapping([])],
        extraction_mapping=[ExtractionClassMapping(), ExtractionClassMapping()],
        image_classification_mapping=[
            VisionClassMapping([
                VisionItemMapping(BaseClassMapping({"a": 0, "b": 1})),
                VisionItemMapping(BaseClassMapping({"c": 0})),
            ]),
            VisionClassMapping([
                VisionItemMapping(BaseClassMapping({"d": 0, "e": 1, "f": 2})),
            ]),
        ],
    )
    model = object.__new__(GLiNExTVisionModel)
    words = torch.zeros(2, 4, 3)
    word_mask = torch.ones(2, 4, dtype=torch.long)
    parents = torch.ones(2, 2, 3)
    child_embeds = torch.arange(18, dtype=torch.float).view(6, 3)

    flat = GLiNExTVisionModel._build_flat_inputs(
        model,
        parents,
        child_embeds,
        None,
        words,
        word_mask,
        mapping,
        "image_classification",
        label_group_sizes=torch.tensor([2, 1, 3]),
        per_task_parents=True,
    )

    assert flat.batch_origin.tolist() == [0, 0, 1]
    assert flat.child_embedding.shape == (3, 3, 3)
    assert flat.child_mask.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    assert torch.equal(flat.child_embedding[0, :2], child_embeds[:2])
    assert torch.equal(flat.child_embedding[1, :1], child_embeds[2:3])
    assert torch.equal(flat.child_embedding[2, :3], child_embeds[3:6])


def test_relation_prompts_reuse_group_layout_for_per_item_packing():
    entity_mapping = BaseClassMapping({"entity": 0})
    relation_mappings = [
        BaseClassMapping({"r1": 0, "r2": 1}),
        BaseClassMapping({"r3": 0}),
        BaseClassMapping({"r4": 0, "r5": 1, "r6": 2}),
    ]
    mapping = BatchClassesMapping(
        cat_mapping=[CatClassMapping([]), CatClassMapping([])],
        extraction_mapping=[
            ExtractionClassMapping([
                ExtractionItemMapping(entity_mapping, relation_mappings[0]),
                ExtractionItemMapping(entity_mapping, relation_mappings[1]),
            ]),
            ExtractionClassMapping([
                ExtractionItemMapping(entity_mapping, relation_mappings[2]),
            ]),
        ],
    )
    relation_prompts = torch.tensor(
        [
            [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]],
            [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0]],
        ]
    )
    relation_mask = torch.ones(2, 3, dtype=torch.long)
    model = object.__new__(GLiNExTTextModel)

    flat_prompts, flat_mask = model._build_flat_rel_prompts(
        relation_prompts,
        mapping,
        rel_prompts_mask=relation_mask,
    )

    assert flat_prompts.shape == (3, 3, 2)
    assert torch.equal(flat_prompts[0, :2], relation_prompts[0, :2])
    assert torch.equal(flat_prompts[1, :1], relation_prompts[0, 2:3])
    assert torch.equal(flat_prompts[2], relation_prompts[1])
    assert flat_mask.tolist() == [
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1],
    ]


@pytest.mark.parametrize(
    ("task_name", "item_counts", "expected"),
    [
        ("object_detection", [2, 1], [2, 2, 1]),
        ("segmentation", [3, 4], [3, 3, 4]),
        ("audio_segmentation", [5, 2], [5, 5, 2]),
    ],
)
def test_set_prediction_counts_expand_from_items_to_flat_label_groups(
    task_name,
    item_counts,
    expected,
):
    flat_inputs = TaskFlatInputs(
        words_embedding=torch.zeros(3, 1, 2),
        mask=torch.ones(3, 1),
        parent_embedding=torch.zeros(3, 2),
        child_embedding=torch.zeros(3, 1, 2),
        child_mask=torch.ones(3, 1),
        batch_origin=torch.tensor([0, 0, 1]),
    )

    expanded = GLiNExTVisionModel._flatten_set_prediction_count(
        torch.tensor(item_counts),
        flat_inputs,
        task_name,
    )

    assert expanded.tolist() == expected
