from collections.abc import Iterable
from dataclasses import dataclass

from transformers import AutoConfig, AutoModel

from .deberta_2d import (
    Deberta2DConfig,
    Deberta2DModel,
    LayoutDebertaConfig,
    LayoutDebertaModel,
)
from .flash_deberta import (
    ATTN_KERNELS,
    flash_kernels_available,
    is_flash_kernel,
    normalize_attn_kernel,
)
from .qwen3 import GLiFormerQwen3Model, Qwen3BidirectionalModel
from .qwen3_5 import (
    QWEN3_5_AVAILABLE,
    GLiFormerQwen3_5Model,
    GLiFormerQwen3_5TextModel,
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
    AutoModel.register(GLiFormerQwen3Model.config_class, GLiFormerQwen3Model, exist_ok=True)
except ValueError:
    pass

if QWEN3_5_AVAILABLE:
    try:
        AutoModel.register(Qwen3_5Config, GLiFormerQwen3_5Model, exist_ok=True)
        AutoModel.register(Qwen3_5TextConfig, GLiFormerQwen3_5TextModel, exist_ok=True)
    except ValueError:
        pass


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    model_class: type
    config_class: type | None = None
    aliases: tuple[str, ...] = ()
    model_class_names: tuple[str, ...] = ()


BACKBONE_REGISTRY: dict[str, BackboneSpec] = {}


def register_backbone(
    name: str,
    model_class: type,
    config_class: type | None = None,
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
    keys = (normalized, *alias_tuple)
    seen = set()
    for key in keys:
        if key in seen or key in BACKBONE_REGISTRY:
            raise ValueError(f"Backbone {key!r} is already registered")
        seen.add(key)
    for key in keys:
        BACKBONE_REGISTRY[key] = spec
    return spec


def normalize_backbone_type(backbone_type: str | None) -> str:
    return (backbone_type or "auto").replace("-", "_").lower()


def get_backbone(backbone_type: str | None) -> BackboneSpec | None:
    normalized = normalize_backbone_type(backbone_type)
    if normalized == "auto":
        return None
    try:
        return BACKBONE_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GLiFormer backbone_type {backbone_type!r}. "
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
    GLiFormerQwen3Model,
    GLiFormerQwen3Model.config_class,
    aliases=("qwen3_bidirectional",),
    model_class_names=("GLiFormerQwen3Model", "Qwen3BidirectionalModel"),
)
if QWEN3_5_AVAILABLE:
    register_backbone(
        "qwen3_5",
        GLiFormerQwen3_5Model,
        Qwen3_5Config,
        aliases=("qwen3_5_bidirectional",),
        model_class_names=("GLiFormerQwen3_5Model", "Qwen3_5BidirectionalModel"),
    )
    register_backbone(
        "qwen3_5_text",
        GLiFormerQwen3_5TextModel,
        Qwen3_5TextConfig,
        model_class_names=("GLiFormerQwen3_5TextModel",),
    )

__all__ = [
    "ATTN_KERNELS",
    "BACKBONE_REGISTRY",
    "BackboneSpec",
    "available_backbones",
    "Deberta2DConfig",
    "Deberta2DModel",
    "flash_kernels_available",
    "get_backbone",
    "GLiFormerQwen3_5Model",
    "GLiFormerQwen3_5TextModel",
    "GLiFormerQwen3Model",
    "is_flash_kernel",
    "LayoutDebertaConfig",
    "LayoutDebertaModel",
    "normalize_attn_kernel",
    "normalize_backbone_type",
    "QWEN3_5_AVAILABLE",
    "Qwen3_5BidirectionalModel",
    "Qwen3BidirectionalModel",
    "register_backbone",
]
