import pytest
import torch

from glinext.layers import AnchorNormalizer


@pytest.mark.parametrize("normalization", ["layer_norm", "rms_norm", "l2"])
def test_anchor_normalizers_mask_invalid_slots_and_backpropagate(normalization):
    anchors = torch.randn(2, 4, 8, requires_grad=True)
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.bool)
    normalizer = AnchorNormalizer.from_config(normalization, hidden_size=8)

    output = normalizer(anchors, mask)

    assert output.shape == anchors.shape
    assert torch.equal(output[~mask], torch.zeros_like(output[~mask]))
    output.sum().backward()
    assert anchors.grad is not None


def test_l2_anchor_normalization_produces_unit_valid_anchors():
    anchors = torch.randn(2, 3, 8)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)

    output = AnchorNormalizer.from_config("l2", 8)(anchors, mask)

    torch.testing.assert_close(
        output[mask].norm(dim=-1),
        torch.ones(mask.sum()),
    )


def test_center_rms_uses_masked_source_mean_and_keeps_single_anchor():
    source = torch.tensor(
        [[[10.0, 2.0], [12.0, 4.0], [0.0, 0.0]]]
    )
    source_mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    anchors = torch.tensor([[[10.0, 2.0], [12.0, 4.0], [9.0, 9.0]]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    normalizer = AnchorNormalizer.from_config("center_rms", 2)

    output = normalizer(
        anchors,
        mask,
        source_embeddings=source,
        source_mask=source_mask,
    )

    torch.testing.assert_close(output[:, 0], -output[:, 1])
    assert torch.equal(output[:, 2], torch.zeros_like(output[:, 2]))

    single = normalizer(anchors[:, :1], mask[:, :1])
    assert single.abs().sum() > 0


def test_anchor_normalization_mapping_is_strict():
    normalizer = AnchorNormalizer.from_config(
        {"type": "rms_norm", "params": {"eps": 1e-5}},
        hidden_size=8,
    )
    assert normalizer.eps == pytest.approx(1e-5)

    with pytest.raises(ValueError, match="Unsupported"):
        AnchorNormalizer.from_config(
            {"type": "l2", "params": {"epsilon_typo": 1e-5}},
            hidden_size=8,
        )
