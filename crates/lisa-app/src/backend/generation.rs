//! Request scope for the shared generation driver. Rendering stays in backend.
use lisa_engine::conversation::GenerationPolicy;

pub(super) fn chat_policy(current_images: bool, history_images: bool) -> GenerationPolicy {
    if current_images || history_images {
        GenerationPolicy::Serial
    } else {
        GenerationPolicy::AdaptiveBlock
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_chat_allows_adaptive_generation_but_image_turns_and_followups_stay_serial() {
        assert_eq!(chat_policy(false, false), GenerationPolicy::AdaptiveBlock);
        for (current_images, history_images) in [(true, false), (false, true), (true, true)] {
            assert_eq!(
                chat_policy(current_images, history_images),
                GenerationPolicy::Serial
            );
        }
        // Clearing image history restores eligibility; the engine still checks
        // the loaded model and sampler for this new text request.
        assert_eq!(chat_policy(false, false), GenerationPolicy::AdaptiveBlock);
    }
}
