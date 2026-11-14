/// Sanitize a checkpoint tensor name:
/// drop `language_model.` / `model.language_model.`, remap `model.mtp.` ->
/// `mtp.`, drop the vision tower, and drop n-gram shards.
pub fn sanitize_name(raw: &str) -> Option<String> {
    if raw.starts_with("vision_tower.")
        || raw.starts_with("visual.")
        || raw.starts_with("model.visual.")
    {
        return None;
    }
    let mut key = raw.to_string();
    if let Some(rest) = key.strip_prefix("model.language_model.") {
        key = format!("model.{rest}");
    } else if let Some(rest) = key.strip_prefix("language_model.") {
        key = rest.to_string();
    }
    if let Some(rest) = key.strip_prefix("model.mtp.") {
        key = format!("mtp.{rest}");
    }
    if key.contains(".ngram_embedding.shard_") {
        return None;
    }
    // The checkpoint's I64 n-gram buffers are never used: the values are
    // rebuilt from configuration.
    if key.ends_with(".layer_multipliers")
        || key.ends_with(".ngram_heads_vocab_sizes")
        || key.ends_with(".ngram_heads_offsets")
    {
        return None;
    }
    // Torch stores a depthwise kernel as (C, 1, K); MLX wants (C, K, 1).
    // Handled by the caller at load time (shape inspection).
    Some(key)
}
