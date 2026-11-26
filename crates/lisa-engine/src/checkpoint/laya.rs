//! File adaptation for the portable Laya configuration.
use anyhow::Context;
use lisa_models::laya::config::LayaConfig;
use std::path::Path;
pub trait LayaConfigIo: Sized {
    fn from_dir(dir: &Path) -> anyhow::Result<Self>;
}
impl LayaConfigIo for LayaConfig {
    fn from_dir(dir: &Path) -> anyhow::Result<Self> {
        let enc = read_json(&dir.join("encoder/config.json"))?;
        let agent = read_json(&dir.join("rl_agent_config.json"))?;
        let tokenizer = read_json(&dir.join("tokenizer/tokenizer_config.json")).ok();
        Self::from_values(&enc, &agent, tokenizer.as_ref())
    }
}
fn read_json(path: &Path) -> anyhow::Result<serde_json::Value> {
    let data =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&data)?)
}
