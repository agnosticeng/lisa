#![cfg(test)]
use super::*;

#[test]
fn image_options_preserve_explicit_steps_and_reject_zero() {
    assert_eq!(Options::default().steps, None);
    assert!(
        Options {
            steps: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        Options {
            steps: Some(16),
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
}
