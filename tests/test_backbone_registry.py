from dataclasses import FrozenInstanceError

import pytest

import glinext.backbones as backbones


class DummyModel:
    pass


class DummyConfig:
    pass


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = {}
    monkeypatch.setattr(backbones, "BACKBONE_REGISTRY", registry)
    return registry


def test_normalize_backbone_type_handles_auto_case_and_hyphens():
    assert backbones.normalize_backbone_type(None) == "auto"
    assert backbones.normalize_backbone_type("") == "auto"
    assert backbones.normalize_backbone_type("QWEN3-5-TEXT") == "qwen3_5_text"


def test_register_backbone_supports_canonical_and_alias_lookup(isolated_registry):
    spec = backbones.register_backbone(
        "Dummy-Backbone",
        DummyModel,
        DummyConfig,
        aliases=(alias for alias in ("Dummy-Alias", "SECOND")),
        model_class_names=(name for name in ("DummyModel", "LegacyDummyModel")),
    )

    assert spec.name == "dummy_backbone"
    assert spec.aliases == ("dummy_alias", "second")
    assert spec.model_class_names == ("DummyModel", "LegacyDummyModel")
    assert set(isolated_registry) == {"dummy_backbone", "dummy_alias", "second"}
    assert backbones.get_backbone("DUMMY-BACKBONE") is spec
    assert backbones.get_backbone("dummy-alias") is spec

    with pytest.raises(FrozenInstanceError):
        spec.name = "changed"


def test_available_backbones_returns_sorted_canonical_names(isolated_registry):
    backbones.register_backbone("zeta", DummyModel, aliases=("first-alias",))
    backbones.register_backbone("Alpha", DummyModel, aliases=("last-alias",))

    assert backbones.available_backbones() == ("alpha", "zeta")
    assert backbones.available_backbones(include_auto=True) == (
        "auto",
        "alpha",
        "zeta",
    )


def test_get_backbone_auto_and_unknown_behavior(isolated_registry):
    backbones.register_backbone("known", DummyModel, aliases=("known-alias",))

    assert backbones.get_backbone(None) is None
    assert backbones.get_backbone("AUTO") is None

    with pytest.raises(ValueError) as exc_info:
        backbones.get_backbone("missing-backbone")

    message = str(exc_info.value)
    assert "missing-backbone" in message
    assert "Available backbones: known" in message
    assert "known-alias" not in message


@pytest.mark.parametrize(
    "name,aliases,collision",
    [
        ("new", ("taken",), "taken"),
        ("new", ("NEW",), "new"),
    ],
)
def test_failed_registration_does_not_partially_mutate_registry(
    isolated_registry,
    name,
    aliases,
    collision,
):
    backbones.register_backbone("existing", DummyModel, aliases=("taken",))
    before = dict(isolated_registry)

    with pytest.raises(ValueError, match=rf"Backbone '{collision}' is already registered"):
        backbones.register_backbone(name, DummyModel, aliases=aliases)

    assert isolated_registry == before
