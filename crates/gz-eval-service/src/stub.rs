use gz_engine::ModelVersion;
use gz_features::{FeatureBatchView, RowOutput};

pub const STUB_MODEL_VERSION: ModelVersion = ModelVersion::from_bytes(*b"gz-stub-v1\0\0\0\0\0\0");

pub fn stub_row_outputs(view: &FeatureBatchView) -> Vec<RowOutput> {
    let row_count = view.row_count as usize;
    let max_actions = view.max_actions as usize;
    let mut rows = Vec::with_capacity(row_count);

    for row in 0..row_count {
        let node_count = u64::from(view.node_count[row]);
        let action_count = view.action_count[row] as usize;
        let value_raw = node_count
            .wrapping_mul(2_654_435_761)
            .wrapping_add((action_count as u64).wrapping_mul(40_503))
            % 4096;
        let value = ((value_raw as i64 - 2048) as f32) / 2048.0;
        let mut policy_logits = Vec::with_capacity(action_count);

        for action in 0..action_count.min(max_actions) {
            let raw = (node_count
                .wrapping_add(31u64.wrapping_mul(action as u64))
                .wrapping_add(7u64.wrapping_mul(action_count as u64)))
                % 64;
            policy_logits.push(((raw as i64 - 32) as f32) / 32.0);
        }
        rows.push(RowOutput {
            policy_logits,
            value,
        });
    }

    rows
}
