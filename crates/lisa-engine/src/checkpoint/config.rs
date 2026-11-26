//! File access for portable model configuration parsers.
use std::path::Path;

pub trait ReadConfig: Sized {
    fn parse_config(data: &str) -> anyhow::Result<Self>;
    fn from_json(path: &Path) -> anyhow::Result<Self> {
        Self::parse_config(&std::fs::read_to_string(path)?)
    }
}

impl ReadConfig for lisa_models::qwen3_5::config::Qwen35Config {
    fn parse_config(data: &str) -> anyhow::Result<Self> {
        Self::parse(data)
    }
}

impl ReadConfig for lisa_models::qwen4::config::ModelConfig {
    fn parse_config(data: &str) -> anyhow::Result<Self> {
        Self::parse(data)
    }
}
