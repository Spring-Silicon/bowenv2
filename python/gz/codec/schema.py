from __future__ import annotations

from dataclasses import dataclass

from gz.common.tags import FeatureSchemaHash


@dataclass(frozen=True, slots=True)
class SchemaDims:
    feature_schema_hash: FeatureSchemaHash
    batch_capacity: int
    row_count: int
    max_nodes: int
    max_edges: int
    max_actions: int
    max_subjects: int
    node_attr_dim: int
