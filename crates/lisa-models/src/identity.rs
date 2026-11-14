//! Supported checkpoint identities. These are persisted metadata contracts.
/// The implemented checkpoint layouts. Keep metadata aliases in one place so
/// the loader and desktop discovery agree about which family a checkpoint uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Qwen4,
    Qwen35,
    Qwen35Gguf,
    Glm53,
    Glm53Gguf,
    DeepSeek41,
    Flux2,
    Flux3,
    Laya,
}

impl ModelKind {
    pub fn from_model_type(name: &str) -> anyhow::Result<Self> {
        match name {
            "qwen4_exp" | "qwen4_exp_text" => Ok(Self::Qwen4),
            "qwen3_5" | "qwen3_5_text" => Ok(Self::Qwen35),
            "glm5_next" | "glm5_next_text" => Ok(Self::Glm53),
            "qwen35" => Ok(Self::Qwen35Gguf),
            "glm5-next" => Ok(Self::Glm53Gguf),
            "deepseek41" => Ok(Self::DeepSeek41),
            "flux2-klein" => Ok(Self::Flux2),
            "flux3-klein" => Ok(Self::Flux3),
            "laya" => Ok(Self::Laya),
            _ => anyhow::bail!("unsupported model type {name:?}"),
        }
    }
}
