use super::*;

#[test]
fn lookup_follows_the_longest_most_recent_match() {
    // "1 2 3 4 5 ... 1 2 3" continues with "4 5".
    let history = [1, 2, 3, 4, 5, 9, 9, 1, 2, 3];
    assert_eq!(lookup_chain(&history, 2), (vec![4, 5], 3));
    // Never past the latest token, and none below three tokens.
    assert_eq!(lookup_chain(&history, 20).0, vec![4, 5, 9, 9, 1, 2, 3]);
    assert!(lookup_chain(&[7, 1, 2, 8, 1, 2], 4).0.is_empty());
    // The longer match wins over a more recent shorter one.
    let history = [5, 1, 2, 3, 40, 0, 1, 2, 3, 50, 5, 1, 2, 3];
    assert_eq!(lookup_chain(&history, 1), (vec![40], 4));
}

#[test]
fn lookup_rates_start_at_the_prior_and_learn() {
    let mut lookup = LookupAcceptance::default();
    assert!((lookup.rate(3, 0, 0) - 0.966).abs() < 1e-6);
    assert!((lookup.rate(70, 20, 3) - 0.999).abs() < 1e-6);
    let before = lookup.rate(3, 0, 2);
    for _ in 0..50 {
        lookup.observe(3, &[2, 2], 0);
    }
    assert!(lookup.rate(3, 0, 2) < before);
    let (rates, chances) = lookup.rates(5, &[1, 2, 3]);
    assert_eq!(rates.len(), CHAIN_TOKENS * AGREEMENTS);
    assert_eq!(chances.len(), 3);
    assert!(chances.windows(2).all(|w| w[1] <= w[0]));
}

#[test]
fn fit_recovers_an_acceptance_model() {
    // Labels drawn from slope 1.0 and intercepts 0.5 below the prior's.
    let truth = Acceptance {
        slope: 1.0,
        intercepts: [0.5, 0.2, -0.3, -0.3, -0.2, -0.4, 0.0],
    };
    let mut labels = vec![];
    let mut state = 12345u64;
    for i in 0..4000 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        let position = i % POSITIONS;
        let feature = ((i * 37 % 100) as f32 / 100.0) * 8.0 - 4.0;
        let z = truth.slope * feature + truth.intercepts[position];
        let p = 1.0 / (1.0 + (-z as f64).exp());
        labels.push(Label {
            position,
            feature,
            accepted: u < p,
            cap: 1.0,
        });
    }
    let fitted = fit(&labels, &ACCEPTANCE_PRIOR);
    assert!((fitted.slope - truth.slope).abs() < 0.12, "{fitted:?}");
    for p in 0..POSITIONS {
        assert!(
            (fitted.intercepts[p] - truth.intercepts[p]).abs() < 0.3,
            "{fitted:?}"
        );
    }
}

#[test]
fn step_choice_trades_expected_tokens_against_cost() {
    let mut costs = StepCosts::default();
    // Unmeasured: a draft's block as far as the chances fill it.
    assert_eq!(costs.choose(&[0.9; 12], 0).rows, 8);
    assert_eq!(costs.choose(&[0.9; 3], 0).rows, 4);
    // Flat costs take every row with a positive chance.
    costs.table = vec![20.0; MOST_ROWS];
    let step = costs.choose(&[0.9; 15], 0);
    assert_eq!((step.rows, step.lane_cap), (16, 16));
    // A steep row cost stops at the rows worth it.
    costs.table = (1..=MOST_ROWS).map(|r| 15.0 + 2.0 * r as f64).collect();
    let rows = costs.choose(&[0.95, 0.9, 0.8, 0.5, 0.2, 0.1, 0.05], 0).rows;
    assert!((4..=6).contains(&rows), "{rows}");
}

#[test]
fn chances_merge_draft_and_chain_best_first() {
    let last = Chances {
        draft: vec![0.9, 0.5],
        joined: vec![0.8],
    };
    assert_eq!(
        lane_chances(Some(&last), &[0.95, 0.7], 15),
        vec![0.95, 0.9, 0.7, 0.5]
    );
    assert_eq!(lane_chances(Some(&last), &[], 2), vec![0.9, 0.8]);
    assert_eq!(lane_chances(None, &[], 15), vec![1.0; POSITIONS]);
}

#[test]
fn labels_follow_the_accepted_path() {
    // Anchor 0 with children 1 (token 10) and 2 (token 11); 3 under 1.
    let tree = GrownTree {
        tokens: vec![0, 10, 11, 12],
        parents: vec![u32::MAX, 0, 0, 1],
        q: vec![1.0, 0.8, 0.1, 0.6],
        estimate: vec![1.0, 0.8, 0.1, 0.5],
        caps: vec![1.0; 4],
        candidates: vec![],
    };
    let labels = tree_labels(&tree, &[10, 13]);
    let got: Vec<(usize, bool)> = labels.iter().map(|l| (l.position, l.accepted)).collect();
    assert_eq!(got, vec![(0, true), (0, false), (1, false)]);
    let chances = Chances::of(&tree, &[11]).unwrap();
    assert_eq!(chances.joined, vec![0.1]);
    assert_eq!(chances.draft, vec![0.8, 0.5]);
}
