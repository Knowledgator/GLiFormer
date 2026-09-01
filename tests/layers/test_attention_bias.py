import pytest
import torch

from glinext.layers import AttentionBias, GaussianDistanceAttentionBias


def _bias(module, queries, keys, *, heads=1):
    return module(
        query_coordinates=queries,
        key_coordinates=keys,
        query_length=queries.shape[-2],
        key_length=keys.shape[-2],
        dtype=torch.float32,
        device=queries.device,
    )


def test_gaussian_distance_bias_prefers_nearby_coordinates():
    queries = torch.tensor([[[0.0], [1.0]]])
    keys = torch.tensor([[[0.0], [0.5], [1.0]]])
    module = AttentionBias.from_config(
        {"type": "gaussian_distance", "sigma": 0.25},
        num_heads=2,
    )

    bias = _bias(module, queries, keys)

    assert bias.shape == (1, 2, 3)
    assert bias[0, 0].argmax().item() == 0
    assert bias[0, 1].argmax().item() == 2


def test_gaussian_parameters_are_fixed_and_stateless_by_default():
    module = AttentionBias.from_config(
        {"type": "gaussian_distance", "sigma": 0.25, "weight": 1.5},
        num_heads=4,
    )

    assert isinstance(module, GaussianDistanceAttentionBias)
    assert module.sigma == 0.25
    assert module.weight == 1.5
    assert not tuple(module.parameters())
    assert not module.state_dict()


@pytest.mark.parametrize(
    ("learnable_sigma", "learnable_weight", "parameter_name"),
    [
        (True, False, "raw_sigma"),
        (False, True, "raw_weight"),
    ],
)
def test_gaussian_sigma_and_weight_can_be_learned_independently(
    learnable_sigma,
    learnable_weight,
    parameter_name,
):
    queries = torch.tensor([[[0.0], [0.7]]])
    keys = torch.tensor([[[0.1], [0.4], [1.0]]])
    fixed = AttentionBias.from_config(
        {"type": "gaussian_distance", "sigma": 0.4, "weight": 1.3},
        num_heads=3,
    )
    learned = AttentionBias.from_config(
        {
            "type": "gaussian_distance",
            "params": {
                "sigma": 0.4,
                "weight": 1.3,
                "learnable_sigma": learnable_sigma,
                "learnable_weight": learnable_weight,
            },
        },
        num_heads=3,
    )

    fixed_bias = _bias(fixed, queries, keys)
    learned_bias = _bias(learned, queries, keys)
    torch.testing.assert_close(learned_bias, fixed_bias)

    learned_bias.square().mean().backward()
    parameters = dict(learned.named_parameters())
    assert set(parameters) == {parameter_name}
    assert parameters[parameter_name].grad is not None
    assert parameters[parameter_name].grad.abs().sum() > 0


def test_gaussian_parameters_can_be_learned_per_head():
    queries = torch.tensor(
        [
            [[0.0], [0.6]],
            [[0.2], [0.9]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.1], [0.5], [1.0]],
            [[0.0], [0.4], [0.8]],
        ]
    )
    module = AttentionBias.from_config(
        {
            "type": "gaussian_distance",
            "params": {
                "sigma": 0.3,
                "weight": 1.0,
                "learnable_sigma": True,
                "learnable_weight": True,
                "per_head": True,
            },
        },
        num_heads=4,
    )

    bias = _bias(module, queries, keys)
    bias.square().mean().backward()

    assert bias.shape == (2, 4, 2, 3)
    assert module.raw_sigma.shape == (4,)
    assert module.raw_weight.shape == (4,)
    assert module.raw_sigma.grad.abs().sum() > 0
    assert module.raw_weight.grad.abs().sum() > 0


def test_learnable_gaussian_parameters_round_trip_through_state_dict():
    specification = {
        "type": "gaussian_distance",
        "params": {
            "sigma": 0.3,
            "weight": 1.0,
            "learnable_sigma": True,
            "learnable_weight": True,
            "per_head": True,
        },
    }
    source = AttentionBias.from_config(specification, num_heads=3)
    with torch.no_grad():
        source.raw_sigma.add_(torch.tensor([-0.2, 0.0, 0.2]))
        source.raw_weight.add_(torch.tensor([0.3, 0.0, -0.3]))

    destination = AttentionBias.from_config(specification, num_heads=3)
    result = destination.load_state_dict(source.state_dict(), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    torch.testing.assert_close(destination.sigma, source.sigma)
    torch.testing.assert_close(destination.weight, source.weight)


def test_relative_mlp_bias_is_per_head_and_differentiable():
    queries = torch.rand(2, 3, 2)
    keys = torch.rand(2, 5, 2)
    module = AttentionBias.from_config(
        {
            "type": "relative_mlp",
            "params": {"coordinate_dimensions": 2, "hidden_size": 7},
        },
        num_heads=4,
    )

    bias = _bias(module, queries, keys)
    bias.sum().backward()

    assert bias.shape == (2, 4, 3, 5)
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_attention_bias_list_composes_strategies():
    coordinates = torch.tensor([[[0.0], [0.5], [1.0]]])
    gaussian = AttentionBias.from_config(
        {"type": "gaussian", "sigma": 1.0},
        num_heads=1,
    )
    causal = AttentionBias.from_config("causal", num_heads=1)
    composite = AttentionBias.from_config(
        [{"type": "gaussian", "sigma": 1.0}, "causal"],
        num_heads=1,
    )

    expected = _bias(gaussian, coordinates, coordinates) + _bias(
        causal, coordinates, coordinates
    )
    torch.testing.assert_close(
        _bias(composite, coordinates, coordinates),
        expected,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_composite_combines_headless_gaussian_and_per_head_learned_bias(reverse):
    queries = torch.rand(2, 3, 2)
    keys = torch.rand(2, 5, 2)
    specifications = [
        {
            "type": "gaussian_distance",
            "params": {"sigma": 0.4},
        },
        {
            "type": "relative_mlp",
            "params": {"coordinate_dimensions": 2, "hidden_size": 7},
        },
    ]
    if reverse:
        specifications.reverse()
    composite = AttentionBias.from_config(specifications, num_heads=4)

    bias = _bias(composite, queries, keys)
    bias.sum().backward()

    assert bias.shape == (2, 4, 3, 5)
    learned_bias = next(
        child for child in composite.biases if tuple(child.parameters())
    )
    assert all(
        parameter.grad is not None for parameter in learned_bias.parameters()
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("learnable_sigma", "yes"),
        ("learnable_weight", 1),
        ("per_head", None),
    ],
)
def test_gaussian_learnability_options_require_booleans(field, value):
    with pytest.raises(TypeError, match=field):
        AttentionBias.from_config(
            {
                "type": "gaussian_distance",
                "params": {field: value},
            },
            num_heads=2,
        )


def test_attention_bias_mapping_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unsupported"):
        AttentionBias.from_config(
            {"type": "causal", "future_penalty_typo": -1.0},
            num_heads=1,
        )
