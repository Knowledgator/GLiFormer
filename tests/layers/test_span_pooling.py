import pytest
import torch
from gliner.modeling.span_rep import SpanRepLayer
from torch import nn

from gliformer.config import NERHeadConfig, StructuringHeadConfig
from gliformer.layers.span_pooling import (
    MarkerSpanPooler,
    MeanSpanPooler,
    SpanTokenSource,
    TokensSpanPooler,
    build_span_pooler,
    config_needs_token_map,
    gather_span_tokens,
)

D = 16


def _source():
    """2 texts; 4 words; word 2 of text 0 has three subwords.

    Text 0 subwords: [CLS] w1 w2a w2b w2c w3 w4 [SEP]
    Text 1 subwords: [CLS] w1 w2 w3 w4 [SEP] pad pad
    """

    units = torch.tensor([
        [0, 1, 2, 2, 2, 3, 4, 0],
        [0, 1, 2, 3, 4, 0, 0, 0],
    ])
    tokens = torch.randn(2, 8, D)
    return SpanTokenSource(tokens, units), tokens


def test_marker_matches_span_rep_layer_and_its_checkpoint_keys():
    torch.manual_seed(0)
    reference = SpanRepLayer(span_mode="token_level", hidden_size=D, max_width=12, dropout=0.0)
    pooler = MarkerSpanPooler(D, 0.0)
    assert pooler.state_dict().keys() == reference.state_dict().keys()
    pooler.load_state_dict(reference.state_dict())
    words = torch.randn(2, 4, D)
    spans = torch.tensor([[[0, 1], [2, 3]], [[1, 1], [0, 3]]])
    assert torch.allclose(pooler(words, spans), reference(words, spans))
    # It ignores subword sources.
    source, _ = _source()
    assert torch.allclose(pooler(words, spans, token_source=source), reference(words, spans))


def test_gather_keeps_every_subword_of_the_span_in_order():
    source, tokens = _source()
    words = torch.randn(2, 4, D)
    spans = torch.tensor([[[1, 1], [0, 2]], [[1, 2], [3, 3]]])
    gathered, mask = gather_span_tokens(words, spans, 8, source)
    assert gathered.shape == (2, 2, 8, D)
    assert mask.sum(-1).tolist() == [[3, 5], [2, 1]]
    assert torch.equal(gathered[0, 0, :3], tokens[0, 2:5])  # w2a w2b w2c
    assert torch.equal(gathered[0, 1, :5], tokens[0, 1:6])  # w1 .. w3
    assert torch.equal(gathered[1, 0, :2], tokens[1, 2:4])


def test_gather_truncates_to_max_tokens_and_keeps_one_slot_for_empty_spans():
    source, tokens = _source()
    words = torch.randn(2, 4, D)
    spans = torch.tensor([[[0, 3]], [[3, 3]]])
    source.token_units[1] = 0  # text 1 lost all its subwords
    gathered, mask = gather_span_tokens(words, spans, 2, source)
    assert gathered.shape == (2, 1, 2, D)
    assert torch.equal(gathered[0, 0], tokens[0, 1:3])
    assert mask[0, 0].all()
    assert mask[1, 0].tolist() == [True, False]


def test_gather_maps_head_rows_to_encoder_rows():
    source, tokens = _source()
    rows_source = SpanTokenSource(source.tokens, source.token_units, rows=torch.tensor([1, 0, 0]))
    words = torch.randn(3, 4, D)
    spans = torch.tensor([[[1, 1]], [[1, 1]], [[0, 0]]])
    gathered, mask = gather_span_tokens(words, spans, 4, rows_source)
    assert mask.sum(-1).squeeze(-1).tolist() == [1, 3, 1]
    assert torch.equal(gathered[0, 0, 0], tokens[1, 2])
    assert torch.equal(gathered[1, 0, :3], tokens[0, 2:5])


def test_gather_without_source_reads_unit_states():
    words = torch.randn(1, 5, D)
    gathered, mask = gather_span_tokens(words, torch.tensor([[[1, 3]]]), 8)
    assert mask.sum().item() == 3
    assert torch.equal(gathered[0, 0, :3], words[0, 1:4])


def test_mean_pools_all_subwords():
    source, tokens = _source()
    pooler = MeanSpanPooler(D, 0.0)
    pooler.project = nn.Identity()
    pooler.out_project = nn.Identity()
    words = torch.randn(2, 4, D)
    spans = torch.tensor([[[1, 2]], [[0, 0]]])
    out = pooler(words, spans, token_source=source)
    assert torch.allclose(out[0, 0], tokens[0, 2:6].mean(0).relu(), atol=1e-6)
    assert torch.allclose(out[1, 0], tokens[1, 1].relu(), atol=1e-6)
    # Without subwords it averages unit states.
    assert torch.allclose(pooler(words, spans)[0, 0], words[0, 1:3].mean(0).relu(), atol=1e-6)


def test_tokens_pooler_returns_span_tokens_and_reads_inner_subwords():
    torch.manual_seed(0)
    source, tokens = _source()
    pooler = TokensSpanPooler(D, 0.0, max_tokens=4, queries=2, heads=4).eval()
    words = torch.randn(2, 4, D)
    spans = torch.tensor([[[1, 1], [2, 3]], [[0, 1], [3, 3]]])
    out = pooler.encode(words, spans, token_source=source)
    assert out.vectors.shape == (2, 2, D)
    assert out.tokens.shape == (2, 2, 4, D)
    assert out.token_mask.sum(-1).tolist() == [[3, 2], [2, 1]]
    assert torch.allclose(pooler(words, spans, token_source=source), out.vectors)
    # A fresh pooler is the marker: the token part is gated off.
    assert torch.allclose(out.vectors, pooler.marker(words, spans))

    with torch.no_grad():
        pooler.token_gate.fill_(1.0)
    out = pooler.encode(words, spans, token_source=source)
    # Changing a middle subword (w2b) changes the span vector; the marker
    # alone never sees it.
    changed = source.tokens.clone()
    changed[0, 3] += 1.0
    again = pooler(words, spans, token_source=SpanTokenSource(changed, source.token_units))
    assert not torch.allclose(again[0, 0], out.vectors[0, 0])
    assert torch.allclose(again[1], out.vectors[1])

    (out.vectors * torch.randn_like(out.vectors)).sum().backward()
    assert pooler.queries.grad is not None and pooler.queries.grad.abs().sum() > 0


def test_poolers_handle_no_spans():
    words = torch.randn(2, 4, D)
    spans = torch.zeros(2, 0, 2, dtype=torch.long)
    for pooler in (MarkerSpanPooler(D, 0.0), MeanSpanPooler(D, 0.0), TokensSpanPooler(D, 0.0, heads=4)):
        assert pooler(words, spans).shape == (2, 0, D)


def test_config_selects_and_validates_the_pooler():
    assert NERHeadConfig().span_pooler == "marker"
    for name, cls in (("marker", MarkerSpanPooler), ("mean", MeanSpanPooler), ("tokens", TokensSpanPooler)):
        cfg = StructuringHeadConfig(span_pooler=name.upper(), span_max_tokens=7)
        pooler = build_span_pooler(cfg, D, 0.0)
        assert isinstance(pooler, cls)
    assert build_span_pooler(StructuringHeadConfig(span_pooler="tokens", span_max_tokens=7), D, 0.0).max_tokens == 7
    with pytest.raises(ValueError, match="span_pooler"):
        NERHeadConfig(span_pooler="max")
    with pytest.raises(ValueError, match="span_max_tokens"):
        NERHeadConfig(span_max_tokens=0)


def test_config_needs_token_map():
    class Cfg:
        ner_config = NERHeadConfig()
        structuring_config = None
        normalization_config = None

    assert not config_needs_token_map(Cfg)
    Cfg.structuring_config = StructuringHeadConfig(span_pooler="tokens")
    assert config_needs_token_map(Cfg)
    Cfg.structuring_config = StructuringHeadConfig(span_pooler="mean")
    assert config_needs_token_map(Cfg)
