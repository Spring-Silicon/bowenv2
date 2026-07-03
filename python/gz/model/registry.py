from __future__ import annotations

from collections.abc import Callable
from typing import Any

from gz.codec import FeatureSchemaConfig, SchemaDims
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


def build(
    name_or_schema: str | FeatureSchemaConfig,
    schema_or_arch: SchemaDims | dict[str, Any] | object,
    arch_config: dict[str, Any] | None = None,
) -> ModelFn:
    if isinstance(name_or_schema, str):
        if arch_config is None:
            raise ValueError("missing arch_config")
        return _build_by_name(name_or_schema, schema_or_arch, arch_config)

    from gz.model.exphormer import ArchConfig, build_model

    if arch_config is not None:
        raise ValueError("new registry.build form takes only schema and arch")
    arch = schema_or_arch if isinstance(schema_or_arch, ArchConfig) else ArchConfig.from_dict(schema_or_arch)
    return build_model(name_or_schema, arch)


def _build_by_name(name: str, schema_dims: SchemaDims | object, arch_config: dict[str, Any]) -> ModelFn:
    try:
        return ARCHS[name](schema_dims, arch_config)
    except KeyError as error:
        known = ", ".join(sorted(ARCHS))
        raise KeyError(f"unknown arch {name!r}; known: {known}") from error
