"""Tests for the default prompt-marker vocabulary."""

from gliformer.config import GLiFormerConfig


def test_default_prompt_marker_convention():
    config = GLiFormerConfig()

    assert config.seq_token == "[SEQ]"
    assert config.sep_token == "[SEP]"
    assert config.parent_token == "[SCHEMA]"
    assert config.ent_token == "[ENTITY]"
    assert config.cat_token == "[CLASS]"
    assert config.rel_token == "[RELATION]"
    assert config.child_token == "[FIELD]"
    assert config.structuring_child_token == "<<CHILD>>"
    assert config.structuring_end_token == "<<END>>"
    assert config.obj_token == "[OBJECT]"
    assert {
        config.ner_parent_token,
        config.cat_parent_token,
        config.open_rel_parent_token,
        config.struct_parent_token,
    } == {"[SCHEMA]"}


def test_descriptive_token_aliases_override_legacy_names():
    config = GLiFormerConfig(
        entity_token="[E]",
        class_token="[C]",
        relation_token="[R]",
        schema_token="[S]",
        field_token="[F]",
        object_token="[O]",
    )

    assert config.ent_token == "[E]"
    assert config.cat_token == "[C]"
    assert config.rel_token == "[R]"
    assert config.parent_token == "[S]"
    assert config.child_token == "[F]"
    assert config.obj_token == "[O]"
    assert config.struct_parent_token == "[S]"


def test_per_task_schema_tokens_are_descriptive():
    config = GLiFormerConfig(per_task_parents=True)

    assert config.ner_parent_token == "[ENTITY_SCHEMA]"
    assert config.cat_parent_token == "[CLASS_SCHEMA]"
    assert config.open_rel_parent_token == "[RELATION_SCHEMA]"
    assert config.struct_parent_token == "[STRUCTURE_SCHEMA]"
