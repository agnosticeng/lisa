#![cfg(test)]
use super::*;

#[test]
fn flow_map_schedule_has_reference_endpoints_and_order() {
    for steps in [1, 4, 16, 50] {
        let t = schedule(steps);
        assert_eq!(t.len(), steps as usize + 1);
        assert_eq!(t[0], 1.0);
        assert_eq!(*t.last().unwrap(), 0.0);
        assert!(t.windows(2).all(|w| w[0] > w[1]));
    }
    assert_eq!(schedule(2)[1], 16.0 / 17.0);
}
