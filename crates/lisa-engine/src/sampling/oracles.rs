//! Frozen the reference C-reference sampling vectors, shared by host and native adapters.
use super::*;
fn tensor<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
    values: &[f32],
    shape: &[i32],
) -> D::Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
    ctx.ops.pull(&bytes, shape, ScalarType::F32).unwrap()
}
pub fn host_sparse_penalty_matches_vocab_scan<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
) {
    let source = [2.0f32, -3.0, 0.0, -0.0, 4.5, f32::NAN, f32::INFINITY];
    let logits = tensor(ctx, &source, &[source.len() as i32]);
    let history = [0, 1, 4, 1, 3, 5, 6, u32::MAX];
    for penalty in [1.0, 1.2, 0.8] {
        for window in [0, 1, 4, 64] {
            let sampler = Sampler {
                repetition_penalty: penalty,
                penalty_window: window,
                ..Default::default()
            };
            let mut expected = source;
            let tail = &history[history.len().saturating_sub(window)..];
            for (index, value) in expected.iter_mut().enumerate() {
                if penalty != 1.0 && tail.contains(&(index as u32)) {
                    *value = if *value > 0.0 {
                        *value / penalty
                    } else {
                        *value * penalty
                    };
                }
            }
            let actual = sampler.host_logits_on(ctx, &logits, &history).unwrap();
            assert_eq!(
                actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.map(f32::to_bits)
            );
        }
    }
}

pub fn exact_speculation_distribution_matches_top_k_top_p_and_min_p<
    D: HostAccessDevice + ReadbackDevice,
>(
    ctx: &SamplingContext<'_, D>,
) {
    let logits = tensor(ctx, &[3.0f32, 2.0, 0.0], &[3]);
    let mut sampler = Sampler {
        temperature: 1.0,
        top_p: 0.8,
        ..Default::default()
    };

    let weights = sampler.speculation_weights_on(ctx, &logits, &[]).unwrap();
    assert!(weights[0] > weights[1]);
    let top_p_sum = weights[0] + weights[1];
    assert!((weights[0] / top_p_sum - 1.0 / (1.0 + (-1.0f32).exp())).abs() < 1e-6);
    assert_eq!(weights[2], 0.0, "top-p must use full-vocabulary mass");

    sampler.top_k = 1;
    sampler.top_p = 1.0;
    let weights = sampler.speculation_weights_on(ctx, &logits, &[]).unwrap();
    assert_eq!(weights, vec![1.0, 0.0, 0.0]);
    assert_eq!(sampler.draw_host_on(ctx, &logits, &[]).unwrap(), 0);

    sampler.top_k = 0;
    sampler.min_p = 0.2;
    let weights = sampler.speculation_weights_on(ctx, &logits, &[]).unwrap();
    assert!(weights[0] > 0.0 && weights[1] > 0.0);
    assert_eq!(weights[2], 0.0, "min-p must cut the low-probability tail");
}

pub fn host_sampling_order_matches_c_reference<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
) {
    // Oracle tokens and post-draw RNG states from ds4.c at
    // ee7e8405232c8971000cc7ba2b945b00bf6d20d3, using
    // ds4_test_sample_logits with top_k=0, top_p=0.95, min_p=0.
    let logits = tensor(ctx, &[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0], &[6]);
    let expected = [
        (2, 285_734_347_287_933_762),
        (2, 33_554_433),
        (2, 67_108_866),
        (0, 100_663_299),
        (2, 134_217_733),
        (2, 167_772_164),
        (0, 201_326_599),
        (0, 234_881_030),
        (2, 268_435_466),
        (0, 301_989_899),
        (2, 335_544_328),
        (0, 369_098_761),
        (0, 402_653_199),
        (0, 436_207_630),
        (0, 469_762_061),
        (0, 503_316_492),
    ];
    for (seed, (expected_token, expected_state)) in expected.into_iter().enumerate() {
        let mut sampler = Sampler {
            temperature: 1.0,
            top_p: 0.95,
            seed: seed as u64,
            ..Default::default()
        };
        assert_eq!(
            sampler.draw_host_on(ctx, &logits, &[]).unwrap(),
            expected_token
        );
        assert_eq!(sampler.seed, expected_state);
    }

    // The 2048-entry fixtures exercise the reference's 512-candidate fast top-p
    // path and its full-sort fallback. Expected values come from the same
    // C hook and reference revision above.
    let mut fast_logits: Vec<f32> = (0..2048)
        .map(|i| -0.02 * i as f32 + ((i * 17) % 11) as f32 * 0.001)
        .collect();
    let mut fallback_logits: Vec<f32> =
        (0..2048).map(|i| ((i * 37) % 101) as f32 * 0.001).collect();
    let fast_tokens = [
        2, 15, 38, 80, 14, 14, 77, 75, 35, 75, 33, 71, 72, 70, 68, 66,
    ];
    let fallback_tokens = [
        1665, 1992, 1388, 107, 1760, 1831, 1764, 1835, 722, 1532, 763, 1371, 1199, 1068, 937, 1008,
    ];
    for (logits, expected_tokens) in [
        (&mut fast_logits, &fast_tokens),
        (&mut fallback_logits, &fallback_tokens),
    ] {
        let logits = tensor(ctx, logits, &[2048]);
        for (seed, expected_token) in expected_tokens.iter().copied().enumerate() {
            let mut sampler = Sampler {
                temperature: 1.0,
                top_p: 0.95,
                seed: seed as u64,
                ..Default::default()
            };
            assert_eq!(
                sampler.draw_host_on(ctx, &logits, &[]).unwrap(),
                expected_token
            );
            assert_eq!(sampler.seed, expected[seed].1);
        }
    }
}

pub fn host_exact_delta_sampling_matches_c_reference<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
) {
    // Oracle from ds4_test_speculative_delta_sample with draft token 2,
    // top_k=0, top_p=0.95, min_p=0, at the reference revision above.
    let logits = tensor(ctx, &[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0], &[6]);
    let expected = [
        (2, 285_734_347_287_933_762),
        (2, 33_554_433),
        (2, 67_108_866),
        (0, 3_378_524_379_445_251),
        (2, 134_217_733),
        (2, 167_772_164),
        (0, 6_757_048_725_336_071),
        (1, 7_883_223_518_468_102),
        (2, 268_435_466),
        (0, 10_135_573_071_226_891),
        (2, 335_544_328),
        (1, 12_387_922_657_490_953),
        (0, 13_514_097_417_117_711),
        (1, 14_640_272_210_249_742),
        (1, 15_766_447_003_381_773),
        (1, 16_892_621_796_513_804),
    ];
    for (seed, (expected_token, expected_state)) in expected.into_iter().enumerate() {
        let mut sampler = Sampler {
            temperature: 1.0,
            top_p: 0.95,
            seed: seed as u64,
            ..Default::default()
        };
        let p = sampler
            .draft_accept_prob_with_history_on(ctx, &logits, 2, &[])
            .unwrap();
        assert!((p - 0.665_240_943).abs() <= 1e-7, "p={p}");
        let token = if sampler.accept_draw_host(p) {
            2
        } else {
            sampler
                .draw_residual_with_history_on(ctx, &logits, 2, &[])
                .unwrap()
        };
        assert_eq!(token, expected_token);
        assert_eq!(sampler.seed, expected_state);
    }
}

pub fn host_filter_edges_match_c_reference<D: HostAccessDevice + ReadbackDevice>(
    ctx: &SamplingContext<'_, D>,
) {
    // Direct fixtures from ds4_test_sample_logits and
    // ds4_test_speculative_delta_sample at
    // ee7e8405232c8971000cc7ba2b945b00bf6d20d3. These cover top-k plus
    // nucleus, top-p=1 with min-p, top-k=1, the 512-candidate fast path,
    // and the full-sort fallback for both ordinary and exact draws.
    let draw_states = [
        285_734_347_287_933_762,
        33_554_433,
        67_108_866,
        100_663_299,
        134_217_733,
        167_772_164,
        201_326_599,
        234_881_030,
        268_435_466,
        301_989_899,
        335_544_328,
        369_098_761,
        402_653_199,
        436_207_630,
        469_762_061,
        503_316_492,
    ];
    let cases: [(
        &[f32],
        usize,
        f32,
        usize,
        f32,
        f32,
        u32,
        f32,
        &[u32],
        &[u32],
        &[u64],
    ); 2] = [
        (
            &[3.0, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, f32::NAN],
            8,
            0.7,
            3,
            0.8,
            0.05,
            2,
            0.806_678_653,
            &[2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2],
            &[2, 2, 2, 0, 2, 2, 0, 0, 2, 0, 2, 2, 0, 2, 2, 2],
            &[
                285_734_347_287_933_762,
                33_554_433,
                67_108_866,
                3_378_524_379_445_251,
                134_217_733,
                167_772_164,
                6_757_048_725_336_071,
                7_883_223_518_468_102,
                268_435_466,
                10_135_573_071_226_891,
                335_544_328,
                369_098_761,
                13_514_097_417_117_711,
                436_207_630,
                469_762_061,
                503_316_492,
            ],
        ),
        (
            &[3.0, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            8,
            1.0,
            0,
            1.0,
            0.01,
            2,
            0.636_408_627,
            &[0, 1, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            &[2, 2, 2, 0, 2, 2, 0, 3, 2, 0, 2, 1, 0, 4, 1, 1],
            &[
                285_734_347_287_933_762,
                33_554_433,
                67_108_866,
                3_378_524_379_445_251,
                134_217_733,
                167_772_164,
                6_757_048_725_336_071,
                7_883_223_518_468_102,
                268_435_466,
                10_135_573_071_226_891,
                335_544_328,
                12_387_922_657_490_953,
                13_514_097_417_117_711,
                14_640_272_210_249_742,
                15_766_447_003_381_773,
                16_892_621_796_513_804,
            ],
        ),
    ];

    for (
        case_index,
        (
            values,
            len,
            temperature,
            top_k,
            top_p,
            min_p,
            draft,
            expected_p,
            ordinary_tokens,
            exact_tokens,
            exact_states,
        ),
    ) in cases.iter().copied().enumerate()
    {
        let logits = tensor(ctx, values, &[len as i32]);
        for seed in 0..16 {
            let mut ordinary = Sampler {
                temperature,
                top_k,
                top_p,
                min_p,
                seed,
                ..Default::default()
            };
            assert_eq!(
                ordinary.draw_host_on(ctx, &logits, &[]).unwrap(),
                ordinary_tokens[seed as usize],
                "ordinary case={case_index} seed={seed}"
            );
            assert_eq!(ordinary.seed, draw_states[seed as usize]);
        }

        for seed in 0..16 {
            let mut exact = Sampler {
                temperature,
                top_k,
                top_p,
                min_p,
                seed,
                ..Default::default()
            };
            let p = exact
                .draft_accept_prob_with_history_on(ctx, &logits, draft, &[])
                .unwrap();
            assert!(
                (p - expected_p).abs() <= 1e-6,
                "target probability case={case_index} seed={seed}: {p}"
            );
            let token = if exact.accept_draw_host(p) {
                draft
            } else {
                exact
                    .draw_residual_with_history_on(ctx, &logits, draft, &[])
                    .unwrap()
            };
            assert_eq!(
                token, exact_tokens[seed as usize],
                "exact case={case_index} seed={seed}"
            );
            assert_eq!(
                exact.seed, exact_states[seed as usize],
                "exact RNG case={case_index} seed={seed}"
            );
        }
    }

    let logits = tensor(ctx, &[3.0f32, 2.0, 4.0, 1.0, 0.0, -1.0, -2.0, -3.0], &[8]);
    for seed in 0..16 {
        let mut sampler = Sampler {
            temperature: 0.8,
            top_k: 1,
            top_p: 0.95,
            seed,
            ..Default::default()
        };
        assert_eq!(sampler.draw_host_on(ctx, &logits, &[]).unwrap(), 2);
        assert_eq!(sampler.seed, draw_states[seed as usize]);
    }

    let fast_logits: Vec<f32> = (0..2048)
        .map(|i| -0.02 * i as f32 + ((i * 17) % 11) as f32 * 0.001)
        .collect();
    let fast_ordinary = [
        1, 11, 26, 50, 10, 10, 49, 48, 25, 48, 24, 46, 46, 45, 44, 43,
    ];
    let fast_exact = [14, 35, 15, 1, 37, 29, 2, 61, 17, 3, 7, 51, 4, 68, 53, 42];
    let fast_exact_states = [
        6_354_739_634_455_712_083,
        1_126_174_793_148_417,
        2_252_349_586_296_834,
        3_378_524_379_445_251,
        4_504_699_139_039_237,
        5_630_873_932_171_268,
        6_757_048_725_336_071,
        7_883_223_518_468_102,
        9_009_398_278_078_474,
        10_135_573_071_226_891,
        11_261_747_864_342_536,
        12_387_922_657_490_953,
        13_514_097_417_117_711,
        14_640_272_210_249_742,
        15_766_447_003_381_773,
        16_892_621_796_513_804,
    ];
    let fast = tensor(ctx, &fast_logits, &[2048]);
    for seed in 0..16 {
        let mut ordinary = Sampler {
            temperature: 0.9,
            top_p: 0.95,
            min_p: 0.2,
            seed,
            ..Default::default()
        };
        assert_eq!(
            ordinary.draw_host_on(ctx, &fast, &[]).unwrap(),
            fast_ordinary[seed as usize]
        );
        assert_eq!(ordinary.seed, draw_states[seed as usize]);

        let mut exact = Sampler {
            temperature: 0.9,
            top_p: 0.95,
            min_p: 0.2,
            seed,
            ..Default::default()
        };
        let p = exact
            .draft_accept_prob_with_history_on(ctx, &fast, 0, &[])
            .unwrap();
        assert!((p - 0.027_237_312_9).abs() <= 1e-6, "p={p}");
        let token = if exact.accept_draw_host(p) {
            0
        } else {
            exact
                .draw_residual_with_history_on(ctx, &fast, 0, &[])
                .unwrap()
        };
        assert_eq!(token, fast_exact[seed as usize], "fast exact seed={seed}");
        assert_eq!(exact.seed, fast_exact_states[seed as usize]);
    }

    let fallback_logits: Vec<f32> = (0..2048).map(|i| ((i * 37) % 101) as f32 * 0.001).collect();
    let fallback_ordinary = [
        1261, 1558, 217, 1876, 1528, 1700, 1543, 1715, 1945, 1412, 198, 1453, 1180, 1150, 1322,
        1393,
    ];
    let fallback_exact = [
        675, 1370, 698, 27, 1420, 1195, 76, 1899, 795, 125, 351, 1725, 174, 1997, 1773, 1550,
    ];
    let fallback_exact_states = [
        6_354_739_634_455_712_083,
        1_126_174_793_148_417,
        2_252_349_586_296_834,
        3_378_524_379_445_251,
        4_504_699_139_039_237,
        5_630_873_932_171_268,
        6_757_048_725_336_071,
        7_883_223_518_468_102,
        9_009_398_278_078_474,
        10_135_573_071_226_891,
        11_261_747_864_342_536,
        12_387_922_657_490_953,
        13_514_097_417_117_711,
        14_640_272_210_249_742,
        15_766_447_003_381_773,
        16_892_621_796_513_804,
    ];
    let fallback = tensor(ctx, &fallback_logits, &[2048]);
    for seed in 0..16 {
        let mut ordinary = Sampler {
            temperature: 1.3,
            top_p: 0.9,
            seed,
            ..Default::default()
        };
        assert_eq!(
            ordinary.draw_host_on(ctx, &fallback, &[]).unwrap(),
            fallback_ordinary[seed as usize]
        );
        assert_eq!(ordinary.seed, draw_states[seed as usize]);

        let mut exact = Sampler {
            temperature: 1.3,
            top_p: 0.9,
            seed,
            ..Default::default()
        };
        let p = exact
            .draft_accept_prob_with_history_on(ctx, &fallback, 100, &[])
            .unwrap();
        assert!((p - 0.000_548_244_687).abs() <= 1e-6, "p={p}");
        let token = if exact.accept_draw_host(p) {
            100
        } else {
            exact
                .draw_residual_with_history_on(ctx, &fallback, 100, &[])
                .unwrap()
        };
        assert_eq!(
            token, fallback_exact[seed as usize],
            "fallback exact seed={seed}"
        );
        assert_eq!(exact.seed, fallback_exact_states[seed as usize]);
    }
}

/// Independent host arithmetic over a vocabulary scan, exercised through the
/// actual tensor comparison/selection chain and public greedy sampler.
pub fn greedy_penalty_preserves_sign_history_window_and_rng<
    D: HostAccessDevice + ReadbackDevice,
>(
    ctx: &SamplingContext<'_, D>,
) {
    let sources = [
        vec![6.0, 5.0, -3.0, -2.0, 0.0, -0.0],
        vec![-2.0, -3.0, -8.0, -7.0],
        vec![9.0, 6.0, 5.0, -4.0],
    ];
    let histories = [vec![], vec![0], vec![0, 0, 0], vec![0, 2, 1, u32::MAX]];
    for source in sources {
        let logits = tensor(ctx, &source, &[1, 1, source.len() as i32]);
        for penalty in [0.5, 1.0, 2.0, 4.0] {
            for history in &histories {
                for window in [0, 1, 2, 64] {
                    // Both temperature-zero and top-k-one requests are greedy.
                    for (temperature, top_k) in [(0.0, 0), (0.7, 1)] {
                        let mut sampler = Sampler {
                            temperature,
                            top_k,
                            repetition_penalty: penalty,
                            penalty_window: window,
                            seed: 12345,
                            ..Default::default()
                        };
                        let mut expected = source.clone();
                        let tail = &history[history.len().saturating_sub(window)..];
                        for (id, value) in expected.iter_mut().enumerate() {
                            if penalty != 1.0 && tail.contains(&(id as u32)) {
                                *value = if *value > 0.0 {
                                    *value / penalty
                                } else {
                                    *value * penalty
                                };
                            }
                        }
                        let flat = ctx.flat_logits(&logits).unwrap();
                        let adjusted = sampler
                            .greedy_logits_on(ctx, &flat, history)
                            .unwrap()
                            .unwrap_or(flat);
                        assert_eq!(
                            ctx.ops
                                .read_f32(&adjusted)
                                .unwrap()
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .copied()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                            "penalty={penalty} history={history:?} window={window}"
                        );
                        let best = expected.iter().enumerate().fold(0, |best, (id, &value)| {
                            if value > expected[best] { id } else { best }
                        });
                        assert_eq!(sampler.draw_on(ctx, &logits, history).unwrap(), best as u32);
                        assert_eq!(sampler.seed, 12345, "greedy penalties must not consume RNG");
                        assert_eq!(
                            ctx.ops
                                .read_f32(&logits)
                                .unwrap()
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>(),
                            source.iter().copied().map(f32::to_bits).collect::<Vec<_>>(),
                            "raw logits were changed"
                        );
                    }
                }
            }
        }
    }
    for source in [vec![6.0, 5.5], vec![-2.0, -2.2]] {
        let logits = tensor(ctx, &source, &[2]);
        let mut sampler = Sampler {
            repetition_penalty: 1.2,
            ..Default::default()
        };
        assert_eq!(sampler.draw_on(ctx, &logits, &[0, 0]).unwrap(), 1);
    }
}

/// Public batched samplers must preserve each row's configured history, even
/// when every stream is greedy. The vocabulary scan is independent of the
/// tensor penalty chain; mixed the reference draws also retain their per-stream RNG law.
pub fn batched_greedy_penalty_preserves_each_history_and_sampler<
    D: HostAccessDevice + ReadbackDevice,
>(
    ctx: &SamplingContext<'_, D>,
) {
    let rows = [
        [6.0f32, 5.0, -3.0, -2.0],
        [-2.0, -3.0, -5.0, -8.0],
        [9.0, 6.0, 5.0, 4.0],
        [6.0, 5.0, 4.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [-2.0, -3.0, -4.0, -5.0],
    ];
    let values = rows.iter().flatten().copied().collect::<Vec<_>>();
    let logits = tensor(ctx, &values, &[6, 1, 4]);
    // The missing sixth entry exercises the public empty-history fallback.
    let histories = vec![
        vec![0, 0, 0],
        vec![0],
        vec![0, 2, 1],
        vec![0, u32::MAX],
        vec![],
    ];
    let expected = |row: usize, sampler: &Sampler, suppressed: Option<usize>| {
        let mut values = rows[row];
        if let Some(id) = suppressed {
            values[id] = f32::NEG_INFINITY;
        }
        let history = histories.get(row).map(Vec::as_slice).unwrap_or(&[]);
        let tail = &history[history.len().saturating_sub(sampler.penalty_window)..];
        for (id, value) in values.iter_mut().enumerate() {
            if sampler.repetition_penalty != 1.0 && tail.contains(&(id as u32)) {
                *value = if *value > 0.0 {
                    *value / sampler.repetition_penalty
                } else {
                    *value * sampler.repetition_penalty
                };
            }
        }
        values.iter().enumerate().fold(
            0,
            |best, (id, &value)| {
                if value > values[best] { id } else { best }
            },
        ) as u32
    };
    for penalty in [0.5, 1.0, 2.0] {
        for window in [0, 1, 2, 64] {
            for (temperature, top_k) in [(0.0, 0), (0.7, 1)] {
                let mut sampler = Sampler {
                    temperature,
                    top_k,
                    repetition_penalty: penalty,
                    penalty_window: window,
                    seed: 12345,
                    ..Default::default()
                };
                let want = (0..rows.len())
                    .map(|row| expected(row, &sampler, None))
                    .collect::<Vec<_>>();
                assert_eq!(
                    draw_rows(ctx, &logits, rows.len(), &mut sampler, &histories, false).unwrap(),
                    want,
                    "shared sampler penalty={penalty}, window={window}"
                );
                assert_eq!(sampler.seed, 12345);
            }
        }
    }
    let mut samplers = [1.0, 2.0, 0.5, 4.0, 1.0, 2.0]
        .into_iter()
        .enumerate()
        .map(|(row, penalty)| Sampler {
            temperature: if row % 2 == 0 { 0.7 } else { 0.0 },
            top_k: 1,
            repetition_penalty: penalty,
            penalty_window: [64, 64, 1, 0, 64, 2][row],
            seed: row as u64 + 100,
            ..Default::default()
        })
        .collect::<Vec<_>>();
    let original = samplers.clone();
    let want = samplers
        .iter()
        .enumerate()
        .map(|(row, s)| expected(row, s, None))
        .collect::<Vec<_>>();
    assert_eq!(
        draw_rows_multi(ctx, &logits, rows.len(), &mut samplers, &histories, false).unwrap(),
        want
    );
    assert_eq!(
        samplers.iter().map(|s| s.seed).collect::<Vec<_>>(),
        original.iter().map(|s| s.seed).collect::<Vec<_>>()
    );

    let mut ds4 = Sampler {
        repetition_penalty: 2.0,
        penalty_window: 2,
        seed: 444,
        ..Default::default()
    };
    let want = (0..rows.len())
        .map(|row| expected(row, &ds4, None))
        .collect::<Vec<_>>();
    assert_eq!(
        draw_rows(ctx, &logits, rows.len(), &mut ds4, &histories, true).unwrap(),
        want
    );
    assert_eq!(ds4.seed, 444);
    // Since one row requires a history adjustment, the whole mixed batch uses
    // each configured the reference draw, including its existing top-k-one RNG behavior.
    let mut ds4 = original.clone();
    let mut scalar = original;
    let mut want = Vec::new();
    for (row, sampler) in scalar.iter_mut().enumerate() {
        let one = tensor(ctx, &rows[row], &[4]);
        want.push(
            sampler
                .draw_host_on(
                    ctx,
                    &one,
                    histories.get(row).map(Vec::as_slice).unwrap_or(&[]),
                )
                .unwrap(),
        );
    }
    assert_eq!(
        draw_rows_multi(ctx, &logits, rows.len(), &mut ds4, &histories, true).unwrap(),
        want
    );
    assert_eq!(
        ds4.iter().map(|s| s.seed).collect::<Vec<_>>(),
        scalar.iter().map(|s| s.seed).collect::<Vec<_>>()
    );
}
