from __future__ import annotations

from typing import NamedTuple


class GraphBatchTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    action_count: object
    action_kind: object
    action_prior: object
    subject_count: object
    action_subjects: object
    position: object
    opponent_state_present: object
    opponent_node_count: object
    opponent_node_tokens: object
    opponent_node_attrs: object
    opponent_edge_count: object
    opponent_edge_src: object
    opponent_edge_dst: object
    opponent_edge_type: object
    opponent_position: object


class GraphStateTensors(NamedTuple):
    node_count: object
    node_tokens: object
    node_attrs: object
    edge_count: object
    edge_src: object
    edge_dst: object
    edge_type: object
    position: object
