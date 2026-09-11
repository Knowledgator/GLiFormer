import importlib.util
from dataclasses import fields

import gliformer
from gliformer.config import (
    JointRelexHeadConfig,
    OpenRelexHeadConfig,
    StructuringHeadConfig,
)
from gliformer.outputs import (
    GLiFormerAudioOutput,
    GLiFormerLayoutOutput,
    GLiFormerOmniOutput,
    GLiFormerOutput,
    GLiFormerTextOutput,
    GLiFormerVisionOutput,
)
from gliformer.processing.mappings import (
    OpenRelexClassMapping,
    OpenRelexItemMapping,
    StructuringClassMapping,
    StructuringItemMapping,
)
from gliformer.tasks import TASK_REGISTRY, TaskHead


def test_task_registry_lookup_and_definition_invariants():
    definitions = tuple(TASK_REGISTRY)
    names = [definition.name for definition in definitions]

    assert len(names) == len(set(names))
    for definition in definitions:
        assert definition.name in TASK_REGISTRY
        assert TASK_REGISTRY.get(definition.name) is definition
        head_class = definition.load_head_class()
        assert issubclass(head_class, TaskHead)
        assert head_class.name == definition.name

    assert "unknown" not in TASK_REGISTRY
    assert TASK_REGISTRY.get("unknown") is None
    assert TASK_REGISTRY.task_names_for_modality("unknown") == ()


def test_root_exports_canonical_task_types_and_compatibility_alias():
    expected_exports = {
        "OpenRelexHeadConfig": OpenRelexHeadConfig,
        "StructuringHeadConfig": StructuringHeadConfig,
        "OpenRelexItemMapping": OpenRelexItemMapping,
        "OpenRelexClassMapping": OpenRelexClassMapping,
        "StructuringItemMapping": StructuringItemMapping,
        "StructuringClassMapping": StructuringClassMapping,
    }

    for name, expected in expected_exports.items():
        assert getattr(gliformer, name) is expected
    assert gliformer.RelationsHeadConfig is JointRelexHeadConfig


def test_retired_set_prediction_symbols_and_modules_are_not_public():
    retired_names = {
        "SetOpenRelexHeadConfig",
        "SetStructuringHeadConfig",
        "SetOpenRelexItemMapping",
        "SetOpenRelexClassMapping",
        "SetStructuringItemMapping",
        "SetStructuringClassMapping",
    }

    assert retired_names.isdisjoint(vars(gliformer))
    assert importlib.util.find_spec("gliformer.tasks.set_open_relex") is None
    assert importlib.util.find_spec("gliformer.tasks.set_structuring") is None

    from gliformer.tasks import open_relex, structuring

    assert set(open_relex.__all__) == {
        "OpenRelexHead",
        "OpenRelexDecoder",
        "OpenRelexProcessor",
    }
    assert set(structuring.__all__) == {
        "StructuringHead",
        "StructuringDecoder",
        "StructuringProcessor",
    }


def test_public_output_exports_use_only_canonical_task_fields():
    output_classes = (
        GLiFormerTextOutput,
        GLiFormerLayoutOutput,
        GLiFormerVisionOutput,
        GLiFormerAudioOutput,
        GLiFormerOmniOutput,
        GLiFormerOutput,
    )
    for output_class in output_classes:
        assert getattr(gliformer, output_class.__name__) is output_class

    text_fields = {field.name for field in fields(GLiFormerTextOutput)}
    assert {
        "open_rel_entity_logits",
        "open_rel_logits",
        "open_rel_assignment_logits",
        "structuring_entity_logits",
        "structuring_field_logits",
        "structuring_assignment_logits",
    } <= text_fields
    retired_prefixes = (
        "set_open_rel",
        "set_open_relex",
        "set_structuring",
        "groups_",
    )
    assert not any(name.startswith(retired_prefixes) for name in text_fields)
