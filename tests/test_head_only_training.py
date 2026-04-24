from torch import nn

from glinext.glinext import GLiNExT


def test_train_head_only_parameters_keeps_shared_layers_frozen():
    wrapper = object.__new__(GLiNExT)

    model = nn.Module()
    model.encoder = nn.Linear(4, 4)
    model.shared_anchor_modeling = nn.Linear(4, 4)

    head = nn.Module()
    head.private = nn.Linear(4, 2)
    head.shared = model.shared_anchor_modeling
    model.heads = nn.ModuleDict({"dummy": head})

    wrapper.__dict__["model"] = model

    stats = GLiNExT.train_head_only_parameters(wrapper)

    assert stats["trainable_params"] == sum(param.numel() for param in head.private.parameters())
    assert all(not param.requires_grad for param in model.encoder.parameters())
    assert all(not param.requires_grad for param in model.shared_anchor_modeling.parameters())
    assert all(param.requires_grad for param in head.private.parameters())
