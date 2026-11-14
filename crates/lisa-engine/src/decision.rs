//! Backend-independent decision request interface.
use std::path::Path;

/// The interface a non-generative (encoder/decision) model exposes.
///
/// These models are not driven by the generation runtime; a caller addresses
/// them through their own API (e.g. `lisa decide`). This trait is the loading
/// seam, so a decision model can be resolved and dispatched like any other.
pub trait DecisionModel {
    /// The model family name (`"laya"`).
    fn name(&self) -> &str;
    /// The resolved model directory.
    fn dir(&self) -> &Path;
    /// A human-readable structural summary for diagnostics.
    fn summary(&self) -> String;
    /// One decision forward: `(logits per marker, action)`.
    fn decide(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)>;
    /// Run a `state` + typed `questions` request and return the answer JSON.
    fn system_one(
        &self,
        _state: &serde_json::Value,
        _questions: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        anyhow::bail!("this decision model has no typed-question interface")
    }
}
