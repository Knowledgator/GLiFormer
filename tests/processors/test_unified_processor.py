"""Tests for the unified GLiFormerProcessor orchestrator."""

import pytest
import torch
from unittest.mock import MagicMock
from dataclasses import asdict

from gliformer.processing.processor import GLiFormerProcessor, GLiFormerTextProcessor
from gliformer.config import (
    NERHeadConfig, ClassificationHeadConfig,
    JointRelexHeadConfig, OpenRelexHeadConfig, StructuringHeadConfig,
    CountHeadConfig, EmbeddingHeadConfig,
)
from gliformer.processing.mappings import BatchClassesMapping
from tests.conftest import make_config, FakeWordsSplitter


# ── Fake tokenizer ─────────────────────────────────────────────────────

class FakeTokenizer:
    """Minimal tokenizer mock for testing prompt construction."""

    def __init__(self):
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.last_max_length = None

    def __call__(self, texts, is_split_into_words=False, return_tensors=None,
                 truncation=False, max_length=None, padding=False,
                 add_special_tokens=True):
        self.last_max_length = max_length
        if is_split_into_words:
            encoded_lengths = [len(t) for t in texts]
            if truncation and max_length is not None:
                encoded_lengths = [min(length, max_length) for length in encoded_lengths]
            padded_len = max(encoded_lengths)
            input_ids = []
            attention_mask = []
            word_ids = []
            for t, encoded_len in zip(texts, encoded_lengths):
                ids = list(range(1, encoded_len + 1))
                mask = [1] * encoded_len
                item_word_ids = list(range(encoded_len))
                # Pad
                ids += [0] * (padded_len - encoded_len)
                mask += [0] * (padded_len - encoded_len)
                item_word_ids += [None] * (padded_len - encoded_len)
                input_ids.append(ids)
                attention_mask.append(mask)
                word_ids.append(item_word_ids)
            result = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }
            # Mock word_ids method
            result_obj = MagicMock()
            result_obj.__getitem__ = lambda self_, k: result[k]
            result_obj.__setitem__ = lambda self_, k, v: result.__setitem__(k, v)
            result_obj.__contains__ = lambda self_, k: k in result
            result_obj.update = result.update
            result_obj.word_ids = lambda batch_index=0: word_ids[batch_index]
            return result_obj
        else:
            # Label tokenization
            max_len = max(len(s.split()) for s in texts) if texts else 1
            input_ids = []
            attention_mask = []
            for s in texts:
                tokens = s.split()
                ids = list(range(1, len(tokens) + 1))
                mask = [1] * len(tokens)
                ids += [0] * (max_len - len(tokens))
                mask += [0] * (max_len - len(tokens))
                input_ids.append(ids)
                attention_mask.append(mask)
            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            }


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def words_splitter():
    return FakeWordsSplitter()


# ── Processor fixtures ─────────────────────────────────────────────────

@pytest.fixture
def ner_processor(fake_tokenizer, words_splitter):
    config = make_config()
    return GLiFormerProcessor(config, fake_tokenizer, words_splitter)


@pytest.fixture
def all_tasks_processor(fake_tokenizer, words_splitter):
    config = make_config(
        ner_config=asdict(NERHeadConfig()),
        classification_config=asdict(ClassificationHeadConfig()),
        joint_relex_config=asdict(JointRelexHeadConfig()),
        open_relex_config=asdict(OpenRelexHeadConfig()),
        structuring_config=asdict(StructuringHeadConfig()),
        count_config=asdict(CountHeadConfig()),
        embedding_config=asdict(EmbeddingHeadConfig()),
    )
    return GLiFormerProcessor(config, fake_tokenizer, words_splitter)


@pytest.fixture
def ner_cls_processor(fake_tokenizer, words_splitter):
    config = make_config(
        ner_config=asdict(NERHeadConfig()),
        classification_config=asdict(ClassificationHeadConfig()),
    )
    return GLiFormerProcessor(config, fake_tokenizer, words_splitter)


# ── Tests ──────────────────────────────────────────────────────────────

class TestTaskProcessorRegistration:
    def test_ner_only(self, ner_processor):
        assert "ner" in ner_processor.task_processors
        assert "classification" not in ner_processor.task_processors

    def test_all_tasks(self, all_tasks_processor):
        expected = {"ner", "classification", "joint_relex", "open_relex",
                    "structuring", "count", "embedding"}
        assert set(all_tasks_processor.task_processors.keys()) == expected

    def test_disabled_tasks(self, fake_tokenizer, words_splitter):
        config = make_config(ner_config=None)
        proc = GLiFormerProcessor(config, fake_tokenizer, words_splitter)
        # NER is None but default creates it... let's check with explicit None
        # Actually, make_config sets ner_config={} by default (from GLiFormerConfig defaults)
        # Need to check: when ner_config is set to None explicitly
        assert "classification" not in proc.task_processors

    def test_joint_relex_owns_extraction_processing_without_standalone_ner(
        self,
        fake_tokenizer,
        words_splitter,
        joint_relex_item,
    ):
        config = make_config(
            default_ner_config=False,
            ner_config=None,
            joint_relex_config=asdict(JointRelexHeadConfig()),
        )
        processor = GLiFormerTextProcessor(
            config,
            fake_tokenizer,
            words_splitter,
            labels_tokenizer=fake_tokenizer,
        )

        raw_batch = processor.collate_raw_batch([joint_relex_item])
        model_inputs = processor.tokenize_and_prepare_labels(
            raw_batch,
            prepare_labels=True,
        )

        assert "ner" not in processor.task_processors
        assert "joint_relex" in processor.task_processors
        assert raw_batch["classes_mapping"].total_extraction_groups() == 1
        assert "ner_labels" in model_inputs
        assert "rel_labels" in model_inputs
        assert "ner_labels_input_ids" in model_inputs
        assert "rel_labels_input_ids" in model_inputs

        all_labels = processor.create_all_labels(
            [joint_relex_item],
            raw_batch["classes_mapping"],
            max_seq_len=5,
        )
        assert "ner_labels" in all_labels
        assert "rel_labels" in all_labels


class TestBatchGenerateClassMappings:
    def test_ner_mapping(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item])
        assert isinstance(mapping, BatchClassesMapping)
        assert mapping.total_extraction_groups() == 1
        assert mapping.total_cat_groups() == 0

    def test_classification_mapping(self, ner_cls_processor, classification_item):
        mapping = ner_cls_processor.batch_generate_class_mappings([classification_item])
        assert mapping.total_cat_groups() == 1

    def test_multi_task_mapping(self, all_tasks_processor, multi_task_item):
        mapping = all_tasks_processor.batch_generate_class_mappings([multi_task_item])
        assert mapping.total_extraction_groups() == 1
        assert mapping.total_cat_groups() == 1
        assert mapping.total_structuring_groups() == 1
        assert mapping.total_open_relex_groups() == 1

    def test_empty_batch(self, ner_processor):
        item = {"text": "hello world"}
        mapping = ner_processor.batch_generate_class_mappings([item])
        assert mapping.total_extraction_groups() == 0
        assert mapping.total_cat_groups() == 0

    def test_multi_item_batch(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item, ner_item])
        assert mapping.total_extraction_groups() == 2
        assert len(mapping.extraction_mapping) == 2


class TestPrepareInputs:
    def test_ner_prompt(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item])
        texts = [ner_item["text"].split()]
        input_texts, prompt_lengths = ner_processor.prepare_inputs(texts, mapping)

        assert len(input_texts) == 1
        assert len(prompt_lengths) == 1
        # Prompt should start with [SEQ]
        assert input_texts[0][0] == "[SEQ]"
        # Should end with text tokens
        assert "John" in input_texts[0]
        # Prompt length should be > 0
        assert prompt_lengths[0] > 0

    def test_classification_prompt(self, ner_cls_processor, classification_item):
        mapping = ner_cls_processor.batch_generate_class_mappings([classification_item])
        texts = [classification_item["text"].split()]
        input_texts, prompt_lengths = ner_cls_processor.prepare_inputs(texts, mapping)
        # Should have classification tokens before NER tokens
        flat = " ".join(input_texts[0])
        cat_pos = flat.find("[CAT]")
        assert cat_pos >= 0

    def test_prompt_ordering(self, all_tasks_processor, multi_task_item):
        mapping = all_tasks_processor.batch_generate_class_mappings([multi_task_item])
        texts = [multi_task_item["text"].split()]
        input_texts, prompt_lengths = all_tasks_processor.prepare_inputs(texts, mapping)

        flat = input_texts[0]
        # Find token type positions
        cat_pos = next((i for i, t in enumerate(flat) if "[CAT]" in t), -1)
        ent_pos = next((i for i, t in enumerate(flat) if "[ENT]" in t), -1)
        rel_pos = next((i for i, t in enumerate(flat) if "[REL]" in t), -1)
        child_pos = next((i for i, t in enumerate(flat) if "[CHILD]" in t), -1)

        # Order: classification < NER < open_relex < structuring
        if cat_pos >= 0 and ent_pos >= 0:
            assert cat_pos < ent_pos
        if ent_pos >= 0 and rel_pos >= 0:
            assert ent_pos < rel_pos
        if rel_pos >= 0 and child_pos >= 0:
            assert rel_pos < child_pos


class TestCreateAllLabels:
    def test_ner_labels(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item])
        result = ner_processor.create_all_labels([ner_item], mapping, max_seq_len=10)
        assert "ner_labels" in result
        assert "ner_batch_idx" in result

    def test_multi_task_labels(self, all_tasks_processor, multi_task_item):
        mapping = all_tasks_processor.batch_generate_class_mappings([multi_task_item])
        result = all_tasks_processor.create_all_labels([multi_task_item], mapping, max_seq_len=10)
        # Should have labels from multiple tasks
        assert "ner_labels" in result
        assert "cat_labels" in result

    def test_empty_returns_empty_dict(self, ner_processor):
        item = {"text": "hello world"}
        mapping = ner_processor.batch_generate_class_mappings([item])
        result = ner_processor.create_all_labels([item], mapping, max_seq_len=10)
        assert result == {}


class TestLegacyLabelMethods:
    def test_create_cat_labels(self, ner_cls_processor, classification_item):
        mapping = ner_cls_processor.batch_generate_class_mappings([classification_item])
        result = ner_cls_processor.create_cat_labels([classification_item], mapping)
        assert result is not None
        cat_labels, cat_batch_idx = result
        assert cat_labels.shape[0] == 1

    def test_create_ner_labels(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item])
        result = ner_processor.create_ner_labels([ner_item], mapping, max_seq_len=10)
        assert result is not None
        ner_labels, ner_batch_idx = result
        assert ner_labels.shape[1] == 10

    def test_create_cat_labels_none_when_disabled(self, ner_processor, classification_item):
        mapping = ner_processor.batch_generate_class_mappings([classification_item])
        result = ner_processor.create_cat_labels([classification_item], mapping)
        assert result is None


class TestSpanResolution:
    def test_resolve_extraction_spans(self, ner_processor, ner_item_text_spans):
        ner_processor.resolve_extraction_spans(ner_item_text_spans)
        ner = ner_item_text_spans["extraction"][0]["ner"]
        assert all(isinstance(e[0], int) for e in ner)

    def test_resolve_structuring_spans(self, all_tasks_processor):
        item = {
            "text": "John lives in New York",
            "structuring": {"person": [{"name": "John"}]},
        }
        all_tasks_processor.resolve_structuring_spans(item)
        inst = item["structuring"]["person"][0]
        assert isinstance(inst["name"], dict)
        assert inst["name"]["start"] == 0

    def test_resolve_open_relex_spans(self, all_tasks_processor):
        item = {
            "text": "John lives in New York",
            "open_relex": [
                {"relations": [{"relation": "lives_in", "head": "John", "tail": "New York"}]}
            ],
        }
        all_tasks_processor.resolve_open_relex_spans(item)
        rel = item["open_relex"][0]["relations"][0]
        assert isinstance(rel["head"], dict)
        assert rel["head"]["start"] == 0


class TestCollateRawBatch:
    def test_basic_ner(self, ner_processor, ner_item_text_spans):
        batch = ner_processor.collate_raw_batch([ner_item_text_spans])
        assert "tokens" in batch
        assert "classes_mapping" in batch
        assert "seq_length" in batch
        assert isinstance(batch["classes_mapping"], BatchClassesMapping)

    def test_resolves_spans(self, ner_processor, ner_item_text_spans):
        ner_processor.collate_raw_batch([ner_item_text_spans])
        # After collation, spans should be resolved
        # The original item should have tokenized_text
        assert "tokenized_text" in ner_item_text_spans

    def test_structuring_retokenizes_generated_words_with_inference_splitter(
        self, all_tasks_processor,
    ):
        item = {
            "text": "Claim CLM-789456 approved",
            # Simulate the old generator, which split hyphenated identifiers.
            "tokenized_text": ["Claim", "CLM", "-", "789456", "approved"],
            "structuring": {
                "claim": [{"claim_number": "CLM-789456"}],
            },
        }

        batch = all_tasks_processor.collate_raw_batch([item])

        assert item["tokenized_text"] == ["Claim", "CLM-789456", "approved"]
        assert batch["tokens"][0] == ["Claim", "CLM-789456", "approved"]
        assert item["structuring"]["claim"][0]["claim_number"]["start"] == 1
        assert item["structuring"]["claim"][0]["claim_number"]["end"] == 1

    def test_multi_item(self, ner_processor, ner_item_text_spans):
        batch = ner_processor.collate_raw_batch([ner_item_text_spans, ner_item_text_spans])
        assert len(batch["tokens"]) == 2
        assert batch["seq_length"].shape[0] == 2

    def test_represent_spans_adds_span_batch_fields(self, fake_tokenizer, words_splitter, ner_item_text_spans):
        config = make_config(ner_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = GLiFormerProcessor(config, fake_tokenizer, words_splitter)
        batch = proc.collate_raw_batch([ner_item_text_spans])
        assert "span_idx" in batch
        assert "span_label" in batch
        assert "span_mask" in batch
        assert batch["span_idx"].shape[0] == 1


class TestTokenizeAndPrepareLabels:
    def test_encoder_subtokens_are_hard_capped_by_max_len(
        self, fake_tokenizer, words_splitter,
    ):
        config = make_config(max_len=8)
        processor = GLiFormerProcessor(config, fake_tokenizer, words_splitter)
        item = {"text": " ".join(f"word-{i}" for i in range(20))}

        raw_batch = processor.collate_raw_batch([item])
        model_input = processor.tokenize_and_prepare_labels(
            raw_batch,
            prepare_labels=False,
        )

        assert fake_tokenizer.last_max_length == 8
        assert model_input["input_ids"].shape[1] == 8

    def test_represent_spans_uses_model_span_keys(self, fake_tokenizer, words_splitter, ner_item_text_spans):
        config = make_config(ner_config={"represent_spans": True, "neg_spans_ratio": 0.0})
        proc = GLiFormerProcessor(config, fake_tokenizer, words_splitter)
        raw_batch = proc.collate_raw_batch([ner_item_text_spans])
        model_input = proc.tokenize_and_prepare_labels(raw_batch, prepare_labels=True)

        assert "span_idx" in model_input
        assert "span_mask" in model_input
        assert "span_labels" in model_input
        assert "ner_span_idx" not in model_input
        assert "ner_span_mask" not in model_input
        assert "ner_span_labels" not in model_input

    def test_multi_level_relation_sidecar_survives_collation(
        self, fake_tokenizer, words_splitter,
    ):
        config = make_config(
            default_ner_config=False,
            structuring_config={"multi_level": True},
        )
        processor = GLiFormerProcessor(
            config,
            fake_tokenizer,
            words_splitter,
        )
        item = {
            "text": "Root Child",
            "structuring": {
                "tree": [{
                    "name": "Root",
                    "children": [{"name": "Child"}],
                }],
            },
        }

        raw_batch = processor.collate_raw_batch([item])
        model_input = processor.tokenize_and_prepare_labels(
            raw_batch,
            prepare_labels=True,
        )

        assert model_input["structuring_count"].tolist() == [2]
        assert model_input["structuring_relation_group_mask"].tolist() == [True]
        assert model_input["structuring_relation_labels"][0, 0, 1] == 1.0

class TestPrepareAllLabelEncoderInputs:
    def test_without_labels_tokenizer(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item])
        result = ner_processor.prepare_all_label_encoder_inputs(mapping)
        assert result == {}

    def test_with_labels_tokenizer(self, fake_tokenizer, words_splitter, ner_item):
        config = make_config()
        proc = GLiFormerProcessor(config, fake_tokenizer, words_splitter, labels_tokenizer=fake_tokenizer)
        mapping = proc.batch_generate_class_mappings([ner_item])
        result = proc.prepare_all_label_encoder_inputs(mapping)
        assert "ner_labels_input_ids" in result
        assert "ner_labels_attention_mask" in result
        assert "ner_labels_group_size" in result


class TestMappingsCounts:
    """Test that BatchClassesMapping flat iteration works correctly with processor output."""

    def test_flat_extraction_iter(self, ner_processor, ner_item):
        mapping = ner_processor.batch_generate_class_mappings([ner_item, ner_item])
        items = list(mapping.flat_extraction_iter())
        assert len(items) == 2
        assert items[0][0] == 0  # flat_idx
        assert items[0][1] == 0  # batch_idx
        assert items[1][0] == 1
        assert items[1][1] == 1

    def test_flat_cat_iter(self, ner_cls_processor, classification_item):
        mapping = ner_cls_processor.batch_generate_class_mappings(
            [classification_item, classification_item]
        )
        items = list(mapping.flat_cat_iter())
        assert len(items) == 2

    def test_multiple_groups_per_item(self, ner_processor):
        item = {
            "text": "test",
            "extraction": [
                {"name": "g1", "ner": [[0, 0, "A"]]},
                {"name": "g2", "ner": [[0, 0, "B"]]},
            ],
        }
        mapping = ner_processor.batch_generate_class_mappings([item])
        assert mapping.total_extraction_groups() == 2
        items = list(mapping.flat_extraction_iter())
        assert len(items) == 2
        # Both should map to batch_idx 0
        assert items[0][1] == 0
        assert items[1][1] == 0
