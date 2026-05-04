from dataclasses import dataclass
from typing import Any, Iterable, Optional

from transformers import AutoConfig, AutoModel

from .deberta_2d import (
    Deberta2DConfig,
    Deberta2DModel,
    LayoutDebertaConfig,
    LayoutDebertaModel,
)
from .qwen3 import GLiNextQwen3Model, Qwen3BidirectionalModel
from .qwen3_5 import (
    GLiNextQwen3_5Model,
    GLiNextQwen3_5TextModel,
    QWEN3_5_AVAILABLE,
    Qwen3_5BidirectionalModel,
    Qwen3_5Config,
    Qwen3_5TextConfig,
)

try:
    AutoConfig.register(LayoutDebertaConfig.model_type, LayoutDebertaConfig)
    AutoModel.register(LayoutDebertaConfig, LayoutDebertaModel)
except ValueError:
    # Transformers raises when a notebook/process imports this module twice.
    pass

try:
    AutoModel.register(GLiNextQwen3Model.config_class, GLiNextQwen3Model, exist_ok=True)
except ValueError:
    pass

if QWEN3_5_AVAILABLE:
    try:
        AutoModel.register(Qwen3_5Config, GLiNextQwen3_5Model, exist_ok=True)
        AutoModel.register(Qwen3_5TextConfig, GLiNextQwen3_5TextModel, exist_ok=True)
    except ValueError:
        pass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_class: type
    config_class: Optional[type] = None
    aliases: tuple[str, ...] = ()
    model_class_names: tuple[str, ...] = ()


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {}


def register_backbone(
    name: str,
    model_class: type,
    config_class: Optional[type] = None,
    aliases: Iterable[str] = (),
    model_class_names: Iterable[str] = (),
) -> BackboneSpec:
    normalized = normalize_backbone_type(name)
    alias_tuple = tuple(normalize_backbone_type(alias) for alias in aliases)
    spec = BackboneSpec(
        name=normalized,
        model_class=model_class,
        config_class=config_class,
        aliases=alias_tuple,
        model_class_names=tuple(model_class_names),
    )
    for key in (normalized, *alias_tuple):
        if key in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {key!r} is already registered")
        BACKBONE_REGISTRY[key] = spec
    return spec


def normalize_backbone_type(backbone_type: Optional[str]) -> str:
    return (backbone_type or "auto").replace("-", "_").lower()


def get_backbone(backbone_type: Optional[str]) -> Optional[BackboneSpec]:
    normalized = normalize_backbone_type(backbone_type)
    if normalized == "auto":
        return None
    try:
        return BACKBONE_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GLiNExT backbone_type {backbone_type!r}. "
            f"Available backbones: {', '.join(available_backbones())}"
        ) from exc


def available_backbones(include_auto: bool = False) -> tuple[str, ...]:
    names = sorted({spec.name for spec in BACKBONE_REGISTRY.values()})
    if include_auto:
        return ("auto", *names)
    return tuple(names)


register_backbone(
    "deberta_2d",
    LayoutDebertaModel,
    LayoutDebertaConfig,
    aliases=("layout_deberta",),
    model_class_names=("LayoutDebertaModel", "Deberta2DModel"),
)
register_backbone(
    "qwen3",
    GLiNextQwen3Model,
    GLiNextQwen3Model.config_class,
    aliases=("qwen3_bidirectional",),
    model_class_names=("GLiNextQwen3Model", "Qwen3BidirectionalModel"),
)
if QWEN3_5_AVAILABLE:
    register_backbone(
        "qwen3_5",
        GLiNextQwen3_5Model,
        Qwen3_5Config,
        aliases=("qwen3_5_bidirectional",),
        model_class_names=("GLiNextQwen3_5Model", "Qwen3_5BidirectionalModel"),
    )
    register_backbone(
        "qwen3_5_text",
        GLiNextQwen3_5TextModel,
        Qwen3_5TextConfig,
        model_class_names=("GLiNextQwen3_5TextModel",),
    )

__all__ = [
    "BACKBONE_REGISTRY",
    "BackboneSpec",
    "available_backbones",
    "Deberta2DConfig",
    "Deberta2DModel",
    "get_backbone",
    "GLiNextQwen3_5Model",
    "GLiNextQwen3_5TextModel",
    "GLiNextQwen3Model",
    "LayoutDebertaConfig",
    "LayoutDebertaModel",
    "normalize_backbone_type",
    "QWEN3_5_AVAILABLE",
    "Qwen3_5BidirectionalModel",
    "Qwen3BidirectionalModel",
    "register_backbone",
]
