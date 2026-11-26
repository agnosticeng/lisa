use super::{GenDefaults, ServerConfig, can_omit_configured_sampler, sampler_from};
use serde_json::json;

fn config() -> ServerConfig {
    ServerConfig {
        addr: "127.0.0.1:0".into(),
        max_tokens: 64,
        temperature: None,
        top_p: None,
        top_k: None,
        min_p: None,
        rep_penalty: None,
        mtp_exact_sampling: false,
        gen_defaults: GenDefaults::default(),
        depth: lisa_engine::core::round_cost::AUTO_DEPTH,
        prefix_cache: 0,
        max_batch: 1,
        chat_template: None,
        model_id: "test".into(),
        supports_images: false,
    }
}

#[test]
fn http_greedy_penalties_keep_the_resolved_sampler_for_native_and_target_mtp() {
    let mut cfg = config();
    cfg.gen_defaults.rep_penalty = Some(1.2);
    // A silent request inherits the model policy even at greedy temperature.
    let inherited = sampler_from(&json!({}), &cfg);
    assert!(inherited.greedy());
    assert_eq!(inherited.repetition_penalty, 1.2);
    assert!(!can_omit_configured_sampler(&inherited, false));

    cfg.rep_penalty = Some(1.0);
    let flag = sampler_from(&json!({}), &cfg);
    assert!(can_omit_configured_sampler(&flag, false));
    assert!(!can_omit_configured_sampler(&flag, true));

    for body in [
        json!({"temperature": 0.0, "repetition_penalty": 2.0}),
        json!({"temperature": 0.7, "top_k": 1, "repetition_penalty": 0.8}),
    ] {
        let requested = sampler_from(&body, &cfg);
        assert!(requested.greedy());
        assert_eq!(
            requested.repetition_penalty,
            body["repetition_penalty"].as_f64().unwrap() as f32
        );
        for target_mtp in [false, true] {
            assert!(!can_omit_configured_sampler(&requested, target_mtp));
        }
    }
    let clean = sampler_from(&json!({"repetition_penalty": 1.0}), &cfg);
    assert!(can_omit_configured_sampler(&clean, false));
    let sampled = sampler_from(&json!({"temperature": 0.7, "top_k": 0}), &cfg);
    assert!(!can_omit_configured_sampler(&sampled, false));
    cfg.mtp_exact_sampling = true;
    let exact = sampler_from(&json!({"temperature": 0.0}), &cfg);
    assert!(!can_omit_configured_sampler(&exact, false));
}
