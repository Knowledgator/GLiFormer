"""Training-time sequence packing gives the same token states as padding."""

import pytest
import torch

BACKBONE = "microsoft/deberta-v3-small"


def _cached(repo_id):
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    weights = [try_to_load_from_cache(repo_id, name) for name in ("pytorch_model.bin", "model.safetensors")]
    return isinstance(try_to_load_from_cache(repo_id, "config.json"), str) or any(isinstance(w, str) for w in weights)


@pytest.mark.skipif(not _cached(BACKBONE), reason=f"{BACKBONE} not cached")
def test_packed_training_matches_padded():
    from gliformer import GLiFormer

    torch.manual_seed(0)
    model = GLiFormer.load_from_config(
        {"model_name": BACKBONE, "name": "glinext", "hidden_size": 128, "dropout": 0.0,
         "max_len": 128, "max_types": 30, "max_width": 30, "ner_config": {}}
    ).float()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    encoder = model.model.token_rep_layer
    tokenizer = model.data_processor.transformer_tokenizer
    batch = tokenizer(
        ["short text", "a much longer piece of text that sets the padded width", "mid length one"],
        return_tensors="pt", padding="longest",
    )
    encoder.train()
    model.config.train_sequence_packing = False
    padded = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    model.config.train_sequence_packing = True
    assert encoder._packs_in_training(batch["input_ids"], batch["attention_mask"], (), {})
    packed = encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    valid = batch["attention_mask"].bool()
    assert packed.shape == padded.shape
    assert torch.allclose(packed[valid], padded[valid], atol=1e-4)
    packed.sum().backward()
    assert any(p.grad is not None for p in encoder.parameters())
