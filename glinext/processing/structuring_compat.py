"""Versionless compatibility adapters for historical structuring contracts.

Canonical task code should use independent set-structuring names.  The small
helpers here are the only place where legacy shared keys and mappings are
resolved, which makes their eventual deprecation measurable and removable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def legacy_task_value(
    payload: Mapping[str, Any],
    task_prefix: str,
    suffix: str,
    *,
    fallback_prefix: str | None = None,
) -> Any:
    """Read a canonical task key, optionally falling back to an old prefix."""

    value = payload.get(f"{task_prefix}_{suffix}")
    if value is None and fallback_prefix is not None:
        value = payload.get(f"{fallback_prefix}_{suffix}")
    return value


def legacy_task_mapping(
    container: object,
    mapping_attr: str,
    *,
    fallback_attr: str | None = None,
    default: Any = None,
) -> Any:
    """Resolve an independent mapping with an old shared-channel fallback."""

    if container is None:
        return default
    if hasattr(container, mapping_attr):
        mapping = getattr(container, mapping_attr)
        if mapping is not None:
            return mapping
    if fallback_attr is not None and hasattr(container, fallback_attr):
        return getattr(container, fallback_attr)
    return default


def is_legacy_set_structuring_output(
    membership_logits: object,
    field_logits: object,
) -> bool:
    """Recognize the historical combined ``(BN,A,E,C)`` decoder tensor."""

    return (
        field_logits is None
        and getattr(membership_logits, "dim", lambda: -1)() == 4
    )


__all__ = [
    "is_legacy_set_structuring_output",
    "legacy_task_mapping",
    "legacy_task_value",
]
