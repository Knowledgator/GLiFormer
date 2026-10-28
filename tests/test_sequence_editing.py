"""Sequence editing head: alignment, labels, train + infer, save/load."""

import json
from pathlib import Path

import pytest
import torch

from gliformer.tasks.sequence_editing.alignment import (
    APPEND,
    DELETE,
    KEEP,
    REPLACE,
    apply_edits,
    char_edits,
    normalize_space,
    plan_edits,
)
from gliformer.tasks.sequence_editing.processor import apply_char_edits, normalize_groups, target_of

BACKBONE = "microsoft/deberta-v3-small"
DATA = Path(__file__).resolve().parents[1] / "data" / "grammar_correction" / "validation.jsonl"


def _cached(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    weights = [try_to_load_from_cache(repo_id, name) for name in ("pytorch_model.bin", "model.safetensors")]
    config = try_to_load_from_cache(repo_id, "config.json")
    return isinstance(config, str) and any(isinstance(path, str) for path in weights)


needs_backbone = pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")


def _words(text):
    from gliner.data_processing.tokenizer import WordsSplitter

    return [tuple(unit) for unit in WordsSplitter("whitespace")(text)]


def _roundtrip(source, target, split=_words):
    units = split(source)
    plan = plan_edits(source, units, target, split(target))
    out = apply_edits(source, units, plan.ops, plan.contents)
    assert normalize_space(out) == normalize_space(target), (source, target, out)
    return plan


# -- alignment ---------------------------------------------------------------


@pytest.mark.parametrize(
    "source,target",
    [
        ("She go to school yesterday .", "She went to school yesterday."),
        ("Thanks you for your lovely comment Sue.", "Thank you for your lovely comment, Sue."),
        ("go home now", "I go home now"),
        ("um the cat sat", "the cat sat"),
        ("the cat sat", "the cat sat"),
        ("a b c d", "x y"),
        ("a b", "a x y z b"),
        ("Hello world", ""),
        ("word", "two words here"),
        ("In's turn it is", "In turn, it is"),
        ("  leading and trailing  ", "leading, and trailing"),
        ("line one\n\nline two", "line one\n\nline 2"),
    ],
)
def test_plan_roundtrip(source, target):
    _roundtrip(source, target)


def test_plan_ops_and_spacing():
    plan = _roundtrip("She go to school yesterday .", "She went to school yesterday.")
    # slot 0 = BOS; "go" replaced; "." re-spaced (replace, no space before).
    assert plan.ops == [KEEP, KEEP, REPLACE, KEEP, KEEP, KEEP, REPLACE]
    assert plan.contents[2].text == "went" and plan.contents[2].space
    assert plan.contents[6].text == "." and not plan.contents[6].space


def test_plan_insert_at_start_and_delete():
    plan = _roundtrip("go home now", "I go home now")
    assert plan.ops[0] == APPEND and plan.contents[0].text == "I"
    plan = _roundtrip("um the cat", "the cat")
    assert plan.ops == [KEEP, DELETE, KEEP, KEEP]


def test_identity_has_no_edits():
    plan = _roundtrip("Nothing to fix here.", "Nothing to fix here.")
    assert plan.is_identity()


def test_kept_whitespace_is_copied():
    source = "line one\n\nline two"
    units = _words(source)
    plan = plan_edits(source, units, "line one\n\nline 2", _words("line one\n\nline 2"))
    assert apply_edits(source, units, plan.ops, plan.contents) == "line one\n\nline 2"


def test_char_edits_offsets():
    source = "She go to school ."
    units = _words(source)
    plan = plan_edits(source, units, "She went to school.", _words("She went to school."))
    edits = char_edits(source, units, plan.ops, plan.contents)
    assert edits[0] == {"op": "replace", "start": 4, "end": 6, "source": "go", "replacement": "went", "space": True}
    assert edits[1]["source"] == "." and edits[1]["start"] == 17


def test_char_edit_targets():
    text = "Call John at 555-1234."
    edits = [{"start": 13, "end": 21, "replacement": "[PHONE]"}, {"start": 5, "end": 9, "replacement": "[NAME]"}]
    assert apply_char_edits(text, edits) == "Call [NAME] at [PHONE]."
    assert apply_char_edits(text, [{"start": 0, "end": 6}, {"start": 3, "end": 8}]) is None
    group = normalize_groups({"instruction": "redact", "edits": edits})[0]
    assert target_of(text, group) == "Call [NAME] at [PHONE]."
    assert normalize_groups("fix grammar") == [{"instruction": "fix grammar"}]


@pytest.mark.skipif(not DATA.exists(), reason="grammar_correction data not downloaded")
def test_plan_roundtrip_on_dataset_rows():
    with DATA.open() as handle:
        rows = [json.loads(next(handle)) for _ in range(500)]
    for row in rows:
        _roundtrip(row["input"], row["output"])


# -- model ---------------------------------------------------------------------

ROWS = [
    {"text": "She go to school yesterday .",
     "sequence_editing": {"instruction": "fix grammar", "target": "She went to school yesterday."}},
    {"text": "Thanks you for your lovely comment Sue.",
     "sequence_editing": {"instruction": "fix grammar", "target": "Thank you for your lovely comment, Sue."}},
    {"text": "go home now", "sequence_editing": {"instruction": "fix grammar", "target": "I go home now"}},
    {"text": "Call John at noon.",
     "sequence_editing": {"instruction": "redact names", "edits": [{"start": 5, "end": 9, "replacement": "[NAME]"}]}},
]


def _model(text_unit="word", **editing):
    from gliformer import GLiFormer

    torch.manual_seed(0)
    return GLiFormer.load_from_config(
        {
            "model_name": BACKBONE,
            "name": "glinext",
            "hidden_size": 128,
            "dropout": 0.0,
            "max_len": 128,
            "max_types": 30,
            "max_width": 30,
            "text_unit": text_unit,
            "ner_config": {},
            "sequence_editing_config": {"max_content_len": 4, **editing},
        }
    ).float()


def _batch(model, rows):
    from gliformer.processing.collator import resolve_gliformer_collator_class

    collator = resolve_gliformer_collator_class(model.config)(
        model.config, data_processor=model.data_processor, prepare_labels=True,
    )
    return collator([json.loads(json.dumps(row)) for row in rows])


def _train(model, rows, steps=80, lr=3e-4):
    batch = _batch(model, rows)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        loss = model.model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(float(loss.detach()))
    return losses


@needs_backbone
def test_labels_shapes_and_prompt():
    model = _model()
    batch = _batch(model, ROWS)
    edit_index = model.config.sequence_editing_config.edit_token_index
    assert int((batch["input_ids"] == edit_index).sum()) == len(ROWS)
    assert batch["seq_edit_group_row"].tolist() == [0, 1, 2, 3]
    ops = batch["seq_edit_op"]
    assert ops.shape[0] == 4
    assert ops[0, :7].tolist() == [KEEP, KEEP, REPLACE, KEEP, KEEP, KEEP, REPLACE]
    assert ops[2, 0].item() == APPEND
    num_edits = int(((ops == REPLACE) | (ops == APPEND)).sum())
    assert batch["seq_edit_pos"].shape == (num_edits, 2)
    assert batch["seq_edit_tokens"].shape == (num_edits, 4)
    # Only the BOS edit has no space target.
    assert int((batch["seq_edit_space"] == -100).sum()) == 1


@needs_backbone
def test_truncated_units_are_ignored():
    model = _model()
    model.config.max_len = 24
    model.data_processor.config.max_len = 24
    long_text = " ".join(["word"] * 40) + " end ."
    target = " ".join(["word"] * 40) + " end."
    batch = _batch(model, [{"text": long_text, "sequence_editing": {"instruction": "fix", "target": target}}])
    kept = int(batch["text_lengths"][0, 0])
    assert kept < 42
    assert (batch["seq_edit_op"][0, kept + 1:] == -100).all()
    model.train()
    assert torch.isfinite(model.model(**batch).loss)


@needs_backbone
@pytest.mark.parametrize("text_unit", ["word", "token"])
def test_overfit_and_edit_text(text_unit):
    model = _model(text_unit)
    losses = _train(model, ROWS, steps=200)
    assert losses[-1] < 0.2 * losses[0]
    results = model.edit_text([row["text"] for row in ROWS], [row["sequence_editing"]["instruction"] for row in ROWS])
    expected = [
        "She went to school yesterday.",
        "Thank you for your lovely comment, Sue.",
        "I go home now",
        "Call [NAME] at noon.",
    ]
    assert [result["text"] for result in results] == expected
    first = results[0]["edits"][0]
    assert (first["op"], first["start"], first["end"], first["replacement"]) == ("replace", 4, 6, "went")
    assert 0.0 <= first["score"] <= 1.0
    # A huge KEEP bias turns every edit off.
    unchanged = model.edit_text(ROWS[0]["text"], "fix grammar", keep_bias=50.0)
    assert unchanged["text"] == ROWS[0]["text"] and unchanged["edits"] == []


@needs_backbone
def test_save_load_roundtrip(tmp_path):
    from gliformer import GLiFormer

    model = _model()
    _train(model, ROWS, steps=40)
    before = model.edit_text([row["text"] for row in ROWS], "fix grammar")
    model.save_pretrained(tmp_path)
    loaded = GLiFormer.from_pretrained(str(tmp_path)).float()
    assert "sequence_editing" in loaded.model.heads
    assert loaded.config.sequence_editing_config.edit_token_index == model.config.sequence_editing_config.edit_token_index
    after = loaded.edit_text([row["text"] for row in ROWS], "fix grammar")
    assert [r["text"] for r in after] == [r["text"] for r in before]


@needs_backbone
def test_hot_add_head_to_existing_model():
    from gliformer import GLiFormer

    torch.manual_seed(0)
    model = GLiFormer.load_from_config(
        {"model_name": BACKBONE, "name": "glinext", "hidden_size": 128, "dropout": 0.0,
         "max_len": 128, "max_types": 30, "max_width": 30, "ner_config": {}}
    ).float()
    vocab_before = model.config.vocab_size
    assert model.enable_sequence_editing({"max_content_len": 4})
    assert model.config.vocab_size == vocab_before + 1
    assert not model.enable_sequence_editing({"keep_bias": 1.5})
    assert model.config.sequence_editing_config.keep_bias == 1.5
    losses = _train(model, ROWS, steps=40)
    assert losses[-1] < losses[0]
    assert model.edit_text("go home now", "fix grammar")["text"]


@needs_backbone
def test_local_context_covers_edited_unit_subwords():
    from types import SimpleNamespace

    model = _model(local_window=1)
    head = model.model.heads["sequence_editing"]
    text = "an unbelievably tricky sentence"
    batch = _batch(model, [{"text": text, "sequence_editing": {"instruction": "x", "target": text}}])
    tokens = batch["input_ids"][0].tolist()
    words_mask = batch["words_mask"]
    embeds = torch.arange(len(tokens), dtype=torch.float)[None, :, None].expand(1, -1, head.hidden_size)
    shared = SimpleNamespace(token_embeds=embeds, attention_mask=batch["attention_mask"])
    with torch.no_grad():
        head.local_type.weight.zero_()
        local, pad = head._local_context(
            shared, {"span_token_words": words_mask}, torch.tensor([0]), torch.tensor([2]),
        )
    picked = sorted(int(v) for v in local[0, ~pad[0], 0].tolist())
    first = words_mask[0].tolist()
    start = first.index(1)  # unit 1 = "an"
    end = first.index(4)  # unit 4 = "sentence" (exclusive: window is units 1..3)
    assert picked == list(range(start, end))


# -- tags content mode -------------------------------------------------------


def _row_pairs(rows):
    return [(row["text"], target_of(row["text"], row["sequence_editing"])) for row in rows]


def _tag_vocab(rows=None, drop=()):
    from gliformer.tasks.sequence_editing.tags import build_tag_vocab

    vocab = build_tag_vocab(_row_pairs(rows or ROWS), _words, size=100, min_count=1)
    return [tag for tag in vocab if tag not in drop] + ["<filler>"]


def test_tag_keys_carry_the_space_bit():
    from gliformer.tasks.sequence_editing.tags import tag_content, tag_key
    from gliformer.tasks.sequence_editing.alignment import Content

    assert tag_key(Content("went", True), 2) == " went"
    assert tag_key(Content(",", False), 5) == ","
    # The space bit does not matter at BOS.
    assert tag_key(Content("I", False), 0) == " I"
    assert tag_content(" went") == Content("went", True)
    assert tag_content(".") == Content(".", False)


def test_tags_mode_needs_a_vocab():
    from gliformer.config import SequenceEditingHeadConfig

    with pytest.raises(ValueError):
        SequenceEditingHeadConfig(content_mode="tags")


@needs_backbone
def test_tags_labels_and_oov_edits():
    vocab = _tag_vocab(drop=(" went",))
    model = _model(content_mode="tags", tag_vocab=vocab)
    head = model.model.heads["sequence_editing"]
    assert head.tag_mode and not hasattr(head, "decoder")
    batch = _batch(model, ROWS)
    ops = batch["seq_edit_op"]
    # "go" -> "went" is out of the vocabulary: its op is not supervised.
    assert ops[0, :7].tolist() == [KEEP, KEEP, -100, KEEP, KEEP, KEEP, REPLACE]
    num_edits = int(((ops == REPLACE) | (ops == APPEND)).sum())
    assert batch["seq_edit_pos"].shape == (num_edits, 2)
    assert batch["seq_edit_tag"].shape == (num_edits,)
    assert "seq_edit_tokens" not in batch
    tags = [vocab[i] for i in batch["seq_edit_tag"].tolist()]
    assert "." in tags and " went" not in tags


@needs_backbone
def test_tags_overfit_edit_text_and_save_load(tmp_path):
    from gliformer import GLiFormer

    model = _model(content_mode="tags", tag_vocab=_tag_vocab())
    losses = _train(model, ROWS, steps=200)
    assert losses[-1] < 0.2 * losses[0]
    texts = [row["text"] for row in ROWS]
    results = model.edit_text(texts, [row["sequence_editing"]["instruction"] for row in ROWS])
    assert [normalize_space(r["text"]) for r in results] == [normalize_space(t) for _, t in _row_pairs(ROWS)]
    first = results[0]["edits"][0]
    assert (first["op"], first["start"], first["end"], first["replacement"]) == ("replace", 4, 6, "went")
    model.save_pretrained(tmp_path)
    loaded = GLiFormer.from_pretrained(str(tmp_path)).float()
    assert loaded.config.sequence_editing_config.tag_vocab == model.config.sequence_editing_config.tag_vocab
    after = loaded.edit_text(texts, [row["sequence_editing"]["instruction"] for row in ROWS])
    assert [r["text"] for r in after] == [r["text"] for r in results]


@needs_backbone
def test_tags_hot_add_from_vocab_file(tmp_path):
    from gliformer import GLiFormer

    path = tmp_path / "tags.json"
    path.write_text(json.dumps(_tag_vocab()))
    model = GLiFormer.load_from_config(
        {"model_name": BACKBONE, "name": "glinext", "hidden_size": 128, "dropout": 0.0,
         "max_len": 128, "max_types": 30, "max_width": 30, "ner_config": {}}
    ).float()
    assert model.enable_sequence_editing({"content_mode": "tags", "tag_vocab_path": str(path)})
    assert model.config.sequence_editing_config.tag_vocab == _tag_vocab()
    losses = _train(model, ROWS, steps=40)
    assert losses[-1] < losses[0]


def test_text_token_states_drop_appended_image_states():
    """Layout backbones append image patch states after the text tokens."""
    from types import SimpleNamespace

    from gliformer.tasks import text_token_states

    tokens = torch.randn(2, 5 + 3, 4)
    shared = SimpleNamespace(token_embeds=tokens, input_ids=torch.zeros(2, 5, dtype=torch.long))
    assert torch.equal(text_token_states(shared), tokens[:, :5])
    plain = SimpleNamespace(token_embeds=tokens, input_ids=torch.zeros(2, 8, dtype=torch.long))
    assert text_token_states(plain) is tokens


def test_word_rnn_keeps_input_width():
    """Masks with holes (image states after padded words) must not shrink the output."""
    from gliformer.layers.rnn import RnnSeq2SeqEncoder

    rnn = RnnSeq2SeqEncoder(4, 2, bidirectional=True)
    x = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.long)
    mask[0, :6] = 1          # 6 words, no image
    mask[1, :2] = 1          # 2 words ...
    mask[1, 8:] = 1          # ... and 2 appended image states
    assert rnn(x, mask).shape[:2] == (2, 10)


def test_apply_word_rnn_keeps_mask_width():
    from types import SimpleNamespace

    from gliner.modeling.layers import LstmSeq2SeqEncoder

    from gliformer.model import BaseGLiFormerModel

    holder = SimpleNamespace(rnn=LstmSeq2SeqEncoder(SimpleNamespace(hidden_size=4), num_layers=1))
    holder._pad_rnn_output = BaseGLiFormerModel._pad_rnn_output
    x = torch.randn(2, 10, 4)
    mask = torch.zeros(2, 10, dtype=torch.long)
    mask[0, :6] = 1
    mask[1, :2] = 1
    mask[1, 8:] = 1
    assert BaseGLiFormerModel._apply_word_rnn(holder, x, mask).shape[:2] == (2, 10)
