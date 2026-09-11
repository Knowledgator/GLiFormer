import torch

from gliformer.processing.collator import GLiFormerTextDataCollator
from gliformer.processing.processor import GLiFormerTextProcessor
from tests.conftest import FakeWordsSplitter, make_config
from tests.processors.test_unified_processor import FakeTokenizer


def test_root_candidate_embedding_rows_keep_anchor_text_during_collation():
    config = make_config(
        default_ner_config=False,
        ner_config=None,
        embedding_config={},
    )
    processor = GLiFormerTextProcessor(
        config,
        FakeTokenizer(),
        FakeWordsSplitter(),
    )
    collator = GLiFormerTextDataCollator(config, processor)

    batch = collator(
        [
            {
                "text": "Two men are fastening a sign.",
                "embedding": [
                    ["two men are working", 1.0],
                    ["the men are asleep", -1.0],
                ],
            }
        ]
    )

    assert torch.equal(batch["embedding_labels"], torch.tensor([1.0, -1.0]))
    assert batch["embedding_pair_idx"].tolist() == [[0, 1], [2, 3]]
    assert batch["embedding_input_ids"].shape[0] == 4
