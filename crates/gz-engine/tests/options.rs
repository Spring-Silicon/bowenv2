use gz_engine::{MeasureConfigHash, MeasureOptions, MeasureOptionsError};

fn config_hash() -> MeasureConfigHash {
    MeasureConfigHash::from_bytes([7; 32])
}

#[test]
fn candidate_options_default_is_deterministic_and_unlimited() {
    let options = gz_engine::CandidateOptions::default();

    assert_eq!(options.max_candidates, None);
    assert!(options.deterministic_order);
}

#[test]
fn measure_options_reject_invalid_ranges() {
    assert_eq!(
        MeasureOptions::new(config_hash(), 0, None, true).unwrap_err(),
        MeasureOptionsError::ZeroSamples
    );
    assert_eq!(
        MeasureOptions::new(config_hash(), 1, Some(0), true).unwrap_err(),
        MeasureOptionsError::ZeroTimeout
    );
}

#[test]
fn measure_options_preserve_fields() {
    let options = MeasureOptions::new(config_hash(), 5, Some(100), false).unwrap();

    assert_eq!(options.config_hash, config_hash());
    assert_eq!(options.samples, 5);
    assert_eq!(options.timeout_ms, Some(100));
    assert!(!options.deterministic);
}
