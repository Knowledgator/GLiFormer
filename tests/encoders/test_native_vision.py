"""Images through a backbone's own vision tower (``vision_encoder_type='native'``)."""

import copy

import pytest
import torch

from gliformer.backbones import get_backbone, qwen3_5
from gliformer.config import GLiFormerOmniConfig
from gliformer.encoders.native_vision import (
    NativeVisionSpec,
    build_native_vision_batch,
    gather_text_tokens,
    gather_vision_tokens,
    native_vision_spec,
)
from gliformer.processing.processor import GLiFormerOmniProcessor, NativeImageProcessor

SPEC = NativeVisionSpec(image_token_id=90, start_token_id=91, end_token_id=92, merge_size=2, pad_token_id=0)
PATCH_DIM = 5


def _batch(grids, text_lengths, max_text=6):
    """Rows of text ids 1..n and one image per non-zero grid, patches filled with the row number."""
    input_ids = torch.zeros(len(grids), max_text, dtype=torch.long)
    attention_mask = torch.zeros(len(grids), max_text, dtype=torch.long)
    for row, length in enumerate(text_lengths):
        input_ids[row, :length] = torch.arange(1, length + 1)
        attention_mask[row, :length] = 1
    image_grid_thw = torch.tensor(grids, dtype=torch.long)
    pixel_values = torch.zeros(len(grids), int(image_grid_thw.prod(-1).max()), PATCH_DIM)
    for row, grid in enumerate(grids):
        pixel_values[row, : grid[0] * grid[1] * grid[2]] = row + 1
    return input_ids, attention_mask, pixel_values, image_grid_thw


class TestNativeVisionBatch:
    def test_image_block_sits_in_front_of_each_rows_text(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4), (0, 0, 0), (1, 2, 4)], [3, 6, 2])
        batch = build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values, grids)

        assert batch.input_ids[0].tolist()[:9] == [91, 90, 90, 90, 90, 92, 1, 2, 3]
        assert batch.input_ids[1].tolist()[:6] == [1, 2, 3, 4, 5, 6]
        assert batch.input_ids[2].tolist()[:6] == [91, 90, 90, 92, 1, 2]
        assert batch.attention_mask.sum(1).tolist() == [9, 6, 6]
        assert batch.mm_token_type_ids.sum(1).tolist() == [4, 0, 2]
        assert torch.equal(batch.mm_token_type_ids.bool(), batch.input_ids == SPEC.image_token_id)

    def test_padding_is_on_the_right_only(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4), (0, 0, 0), (1, 2, 4)], [3, 6, 2])
        batch = build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values, grids)
        assert (batch.attention_mask.diff(dim=1) <= 0).all()

    def test_patches_are_flattened_in_row_order_without_padding(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4), (0, 0, 0), (1, 2, 4)], [3, 6, 2])
        batch = build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values, grids)
        assert batch.image_grid_thw.tolist() == [[1, 4, 4], [1, 2, 4]]
        assert batch.pixel_values.shape == (16 + 8, PATCH_DIM)
        assert batch.pixel_values[:, 0].tolist() == [1.0] * 16 + [3.0] * 8

    def test_gathers_return_text_and_vision_states_per_row(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4), (0, 0, 0), (1, 2, 4)], [3, 6, 2])
        batch = build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values, grids)
        encoded = batch.input_ids.unsqueeze(-1).float()

        text = gather_text_tokens(encoded, batch).squeeze(-1).long()
        assert torch.equal(text * attention_mask, input_ids)
        vision = gather_vision_tokens(encoded, batch).squeeze(-1)
        assert batch.vision_mask.sum(1).tolist() == [4, 0, 2]
        assert vision.tolist() == [[90.0] * 4, [0.0] * 4, [90.0, 90.0, 0.0, 0.0]]
        assert batch.spatial_shape.tolist() == [[2, 2], [0, 0], [1, 2]]

    def test_vision_input_mask_drops_a_rows_image(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4), (1, 2, 4)], [3, 2])
        batch = build_native_vision_batch(
            SPEC, input_ids, attention_mask, pixel_values, grids, vision_input_mask=torch.tensor([0, 1])
        )
        assert batch.image_grid_thw.tolist() == [[1, 2, 4]]
        assert batch.input_ids[0].tolist()[:3] == [1, 2, 3]
        assert batch.pixel_values[:, 0].tolist() == [2.0] * 8

    def test_no_images_means_no_native_batch(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(0, 0, 0), (0, 0, 0)], [3, 2])
        assert build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values[:, :1], grids) is None

    def test_backbones_without_marker_tokens(self):
        spec = NativeVisionSpec(image_token_id=90, start_token_id=None, end_token_id=None, merge_size=1, pad_token_id=0)
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 1, 2)], [2])
        batch = build_native_vision_batch(spec, input_ids, attention_mask, pixel_values, grids)
        assert batch.input_ids[0].tolist()[:4] == [90, 90, 1, 2]
        assert batch.vision_index.tolist() == [[0, 1]]

    def test_rejects_image_batched_pixel_values(self):
        input_ids, attention_mask, _, grids = _batch([(1, 4, 4)], [3])
        with pytest.raises(ValueError, match="batch, patches, patch_dim"):
            build_native_vision_batch(SPEC, input_ids, attention_mask, torch.zeros(1, 3, 32, 32), grids)

    def test_rejects_grid_larger_than_the_patches(self):
        input_ids, attention_mask, pixel_values, grids = _batch([(1, 4, 4)], [3])
        with pytest.raises(ValueError, match="patches"):
            build_native_vision_batch(SPEC, input_ids, attention_mask, pixel_values[:, :8], grids)


def _native_config(**overrides):
    kwargs = {"model_name": "Qwen/Qwen3.5-0.8B", "backbone_type": "qwen3_5", "vision_encoder_type": "native"}
    kwargs.update(overrides)
    return GLiFormerOmniConfig(**kwargs)


@pytest.mark.skipif(not qwen3_5.QWEN3_5_AVAILABLE, reason="needs transformers with Qwen3.5")
class TestNativeVisionConfig:
    def test_registry_marks_backbones_with_a_vision_tower(self):
        assert get_backbone("qwen3_5").native_modalities == ("vision",)
        assert get_backbone("qwen3_5_text").native_modalities == ()
        assert get_backbone("qwen3").native_modalities == ()

    def test_native_encoder_selects_the_native_processor(self):
        config = _native_config()
        assert config.vision_processor_type == "native"
        assert config.freeze_native_vision is True

    @pytest.mark.parametrize("backbone_type", ["qwen3_5_text", "qwen3", "auto"])
    def test_requires_a_backbone_with_a_vision_tower(self, backbone_type):
        with pytest.raises(ValueError, match="ships a vision tower"):
            _native_config(backbone_type=backbone_type)

    def test_processor_and_encoder_must_agree(self):
        with pytest.raises(ValueError, match="go together"):
            GLiFormerOmniConfig(backbone_type="qwen3_5", vision_processor_type="native")
        with pytest.raises(ValueError, match="go together"):
            _native_config(vision_processor_type="auto")

    def test_rejects_a_second_vision_model(self):
        with pytest.raises(ValueError, match="vision_model_name"):
            _native_config(vision_model_name="google/vit-base-patch16-224")

    def test_dense_vision_heads_are_not_supported_yet(self):
        with pytest.raises(ValueError, match="not supported"):
            _native_config(object_detection_config={})

    def test_fields_survive_serialization(self):
        config = _native_config(vision_min_pixels=4096, vision_max_pixels=65536, freeze_native_vision=False)
        restored = GLiFormerOmniConfig(**{k: v for k, v in config.to_dict().items() if k != "model_type"})
        assert restored.vision_encoder_type == "native"
        assert restored.vision_processor_type == "native"
        assert (restored.vision_min_pixels, restored.vision_max_pixels) == (4096, 65536)
        assert restored.freeze_native_vision is False


class _FakeSize:
    shortest_edge = 64


class _FakePatchProcessor:
    """Stands in for a Qwen-style image processor: one 2x2-pixel patch per grid cell."""

    size = _FakeSize()

    def __init__(self):
        self.sizes = []

    def __call__(self, images, return_tensors=None, size=None):
        self.sizes.append(size)
        height, width = images.height // 2, images.width // 2
        return {
            "pixel_values": torch.full((height * width, PATCH_DIM), float(images.width)),
            "image_grid_thw": torch.tensor([[1, height, width]]),
        }


class _Config:
    vision_processor_type = "native"


def _processor():
    from PIL import Image

    processor = object.__new__(GLiFormerOmniProcessor)
    processor.config = _Config()
    processor.task_processors = {}
    processor.vision_input_processor = NativeImageProcessor(_FakePatchProcessor(), max_pixels=1024)
    processor._vision_processor_initialized = True
    return processor, Image


class TestNativeImageProcessing:
    def test_pixel_budget_is_passed_to_the_backbone_processor(self):
        fake = _FakePatchProcessor()
        from PIL import Image

        NativeImageProcessor(fake, min_pixels=16, max_pixels=1024)(images=Image.new("RGB", (8, 4)))
        assert fake.sizes == [{"shortest_edge": 16, "longest_edge": 1024}]

    def test_budget_defaults_are_bounded(self):
        processor = NativeImageProcessor(_FakePatchProcessor())
        assert processor.max_pixels == NativeImageProcessor.DEFAULT_MAX_PIXELS
        assert processor.min_pixels == 64
        assert NativeImageProcessor(_FakePatchProcessor(), max_pixels=32).min_pixels == 32

    def test_images_are_loaded_without_a_vision_head(self):
        processor, _ = _processor()
        assert processor._has_vision_inputs()

    def test_rows_are_padded_over_patches_and_text_rows_get_an_empty_grid(self):
        processor, Image = _processor()
        items = [
            {"image": Image.new("RGB", (8, 4))},
            {"_skip_vision_tasks": True},
            {"image": Image.new("RGB", (4, 4))},
        ]
        batch = processor._prepare_optional_vision_batch(items)

        assert batch["pixel_values"].shape == (3, 8, PATCH_DIM)
        assert batch["image_grid_thw"].tolist() == [[1, 2, 4], [0, 0, 0], [1, 2, 2]]
        assert batch["vision_input_mask"].tolist() == [1, 0, 1]
        assert batch["image_sizes"].tolist() == [[4, 8], [0, 0], [4, 4]]
        assert batch["pixel_values"][2, :4, 0].tolist() == [4.0] * 4
        assert batch["pixel_values"][2, 4:].abs().sum() == 0

    def test_text_only_batches_carry_no_vision_inputs(self):
        processor, _ = _processor()
        assert processor._prepare_optional_vision_batch([{"_skip_vision_tasks": True}]) == {}

    def test_precomputed_patches_need_their_grid(self):
        processor, _ = _processor()
        with pytest.raises(ValueError, match="image_grid_thw"):
            processor.load_native_image({"pixel_values": torch.zeros(4, PATCH_DIM)})
        patches, grid, _ = processor.load_native_image(
            {"pixel_values": torch.zeros(4, PATCH_DIM), "image_grid_thw": [1, 2, 2]}
        )
        assert patches.shape == (4, PATCH_DIM) and grid.tolist() == [1, 2, 2]


needs_cuda = pytest.mark.skipif(
    not (qwen3_5.QWEN3_5_AVAILABLE and torch.cuda.is_available()),
    reason="Qwen3.5 linear attention needs transformers support and CUDA",
)


def _tiny_qwen3_5(**options):
    text_config = qwen3_5.Qwen3_5TextConfig(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        layer_types=["linear_attention", "full_attention"] * 2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        pad_token_id=0,
    )
    config = qwen3_5.Qwen3_5Config(
        text_config=text_config.to_dict(),
        vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64},
        image_token_id=90,
        vision_start_token_id=91,
        vision_end_token_id=92,
    )
    for name, value in options.items():
        setattr(config, name, value)
    return config


def _native_inputs(model):
    spec = native_vision_spec(model)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(1, 80, (3, 7), generator=generator)
    attention_mask = torch.ones(3, 7, dtype=torch.long)
    attention_mask[1, 4:] = 0
    grids = torch.tensor([[1, 4, 6], [1, 2, 2], [0, 0, 0]])
    patch_dim = 3 * 2 * 16 * 16
    pixel_values = torch.randn(3, 24, patch_dim, generator=generator)
    pixel_values[1, 4:] = 0
    pixel_values[2] = 0
    tensors = [t.to(model.device) for t in (input_ids, attention_mask, pixel_values, grids)]
    return spec, tensors


def _encode(model, batch):
    with torch.no_grad():
        return model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            pixel_values=batch.pixel_values,
            image_grid_thw=batch.image_grid_thw,
            mm_token_type_ids=batch.mm_token_type_ids,
        ).last_hidden_state


@needs_cuda
class TestNativeVisionBackbone:
    def test_spec_is_read_from_the_backbone(self):
        model = qwen3_5.GLiFormerQwen3_5Model(_tiny_qwen3_5())
        assert native_vision_spec(model) == NativeVisionSpec(90, 91, 92, 2, 0)
        assert native_vision_spec(model.language_model) is None

    @pytest.mark.parametrize("bidirectional", [False, True])
    def test_batched_rows_match_their_unbatched_encoding(self, bidirectional):
        model = qwen3_5.GLiFormerQwen3_5Model(_tiny_qwen3_5(bidirectional_attention=bidirectional)).eval().cuda()
        spec, (input_ids, attention_mask, pixel_values, grids) = _native_inputs(model)
        batch = build_native_vision_batch(spec, input_ids, attention_mask, pixel_values, grids)
        encoded = _encode(model, batch)

        for row in range(3):
            length = int(attention_mask[row].sum())
            single = build_native_vision_batch(
                spec,
                input_ids[row : row + 1, :length],
                attention_mask[row : row + 1, :length],
                pixel_values[row : row + 1],
                grids[row : row + 1],
            )
            if single is None:
                with torch.no_grad():
                    reference = model(input_ids=input_ids[row : row + 1, :length]).last_hidden_state[0]
            else:
                reference = _encode(model, single)[0]
            assert torch.allclose(encoded[row, : reference.shape[0]], reference, atol=1e-4)

    def test_causal_mode_is_the_stock_multimodal_model(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

        model = qwen3_5.GLiFormerQwen3_5Model(_tiny_qwen3_5()).eval().cuda()
        stock = Qwen3_5Model(copy.deepcopy(model.config)).eval().cuda()
        stock.load_state_dict(model.state_dict())
        spec, (input_ids, attention_mask, pixel_values, grids) = _native_inputs(model)
        batch = build_native_vision_batch(spec, input_ids, attention_mask, pixel_values, grids)
        valid = batch.attention_mask.bool()
        assert torch.allclose(_encode(model, batch)[valid], _encode(stock, batch)[valid], atol=1e-5)

    def test_image_changes_the_text_states(self):
        model = qwen3_5.GLiFormerQwen3_5Model(_tiny_qwen3_5(bidirectional_attention=True)).eval().cuda()
        spec, (input_ids, attention_mask, pixel_values, grids) = _native_inputs(model)
        batch = build_native_vision_batch(spec, input_ids, attention_mask, pixel_values, grids)
        with_image = gather_text_tokens(_encode(model, batch), batch)
        with torch.no_grad():
            text_only = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        assert not torch.allclose(with_image[0], text_only[0], atol=1e-3)

    def test_position_deltas_do_not_leak_into_the_next_call(self):
        model = qwen3_5.GLiFormerQwen3_5Model(_tiny_qwen3_5()).eval().cuda()
        spec, (input_ids, attention_mask, pixel_values, grids) = _native_inputs(model)
        _encode(model, build_native_vision_batch(spec, input_ids, attention_mask, pixel_values, grids))
        assert model.rope_deltas is None
