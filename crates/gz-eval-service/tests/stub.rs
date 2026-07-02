mod common;

use common::{collate, row, schema};
use gz_eval_service::{FeatureEvalBackend, STUB_MODEL_VERSION, StubBackend, stub_row_outputs};
use gz_features::FeatureBatchView;

#[test]
fn stub_outputs_match_golden_values() {
    let schema = schema("stub-golden", 4);
    let rows = [row(3, 2)];
    let (batch, _) = collate(schema, 2, &rows);
    let view = FeatureBatchView::parse(&batch).unwrap();

    let outputs = stub_row_outputs(&view);

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].value.to_bits(), 0x3e40_8000);
    assert_eq!(
        outputs[0]
            .policy_logits
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![(-0.46875f32).to_bits(), 0.5f32.to_bits()]
    );
}

#[test]
fn stub_backend_excludes_padded_rows_and_actions() {
    let schema = schema("stub-padding", 5);
    let rows = [row(4, 3), row(1, 1)];
    let (batch, action_counts) = collate(schema, 4, &rows);

    let outputs = StubBackend.eval(&batch, &action_counts).unwrap();

    assert_eq!(outputs.model_version, STUB_MODEL_VERSION);
    assert_eq!(outputs.rows.len(), 2);
    assert_eq!(outputs.rows[0].policy_logits.len(), 3);
    assert_eq!(outputs.rows[1].policy_logits.len(), 1);
}

#[test]
fn stub_formula_matches_independent_scalar_reference() {
    let schema = schema("stub-scalar", 6);
    let rows = [row(2, 1), row(5, 4), row(3, 2)];
    let (batch, _) = collate(schema, 3, &rows);
    let view = FeatureBatchView::parse(&batch).unwrap();

    let outputs = stub_row_outputs(&view);

    for (index, output) in outputs.iter().enumerate() {
        let node_count = rows[index].node_count as u64;
        let action_count = rows[index].actions.len() as u64;
        let raw_value = node_count
            .wrapping_mul(2_654_435_761)
            .wrapping_add(action_count.wrapping_mul(40_503))
            % 4096;
        assert_eq!(
            output.value.to_bits(),
            ((raw_value as i64 - 2048) as f32 / 2048.0).to_bits()
        );

        for (action, logit) in output.policy_logits.iter().enumerate() {
            let raw_logit = (node_count + 31 * action as u64 + 7 * action_count) % 64;
            assert_eq!(
                logit.to_bits(),
                ((raw_logit as i64 - 32) as f32 / 32.0).to_bits()
            );
        }
    }
}
