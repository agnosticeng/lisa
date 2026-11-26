use super::Sampler;

#[test]
fn raw_argmax_requires_a_history_independent_greedy_configuration() {
    for (temperature, top_k, greedy) in [(0.0, 0, true), (0.7, 1, true), (0.7, 0, false)] {
        for penalty in [0.5, 1.0, 1.2] {
            for window in [0, 64] {
                let sampler = Sampler {
                    temperature,
                    top_k,
                    repetition_penalty: penalty,
                    penalty_window: window,
                    ..Default::default()
                };
                assert_eq!(sampler.greedy(), greedy);
                assert_eq!(sampler.can_use_raw_argmax(), greedy && penalty == 1.0);
            }
        }
    }
}
