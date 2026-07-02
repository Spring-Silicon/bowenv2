from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gz.codec import SchemaDims
from gz.model.stub import stub

ModelFn = Callable[[Any], tuple[Any, Any]]


def _build_stub(schema_dims: SchemaDims, arch_config: dict[str, Any]) -> ModelFn:
    _ = schema_dims
    if arch_config:
        raise ValueError("stub arch_config must be empty")
    return stub


ARCHS: dict[str, Callable[[SchemaDims, dict[str, Any]], ModelFn]] = {
    "stub": _build_stub,
}


def build(name: str, schema_dims: SchemaDims, arch_config: dict[str, Any]) -> ModelFn:
    try:
        return ARCHS[name](schema_dims, arch_config)
    except KeyError as error:
        known = ", ".join(sorted(ARCHS))
        raise KeyError(f"unknown arch {name!r}; known: {known}") from error
