//! Curated download entries and filesystem recognition, independent of compute.
use std::path::Path;
pub mod qwen35 {
    pub const REPO: &str = "unsloth/Qwen3.8-27B-GGUF";
    pub const ENTRY: &str = "Qwen3.8-27B-UD-Q2_K_XL.gguf";
}
pub mod glm53 {
    pub const REPO: &str = "antirez/glm-5.3-flash-gguf";
    pub const ENTRY: &str = "GLM-5.3-Flash-Q2.gguf";
}
pub mod deepseek41 {
    pub const REPO: &str = "ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF";
    pub const GGUF_ENTRY: &str =
        "IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf";
}
pub mod flux3 {
    pub const REPO: &str = "black-forest-labs/flux-3-image-klein";
}

pub mod bonsai {
    pub const REPO: &str = "prism-ml/bonsai-image-binary-4B-mlx-1bit";
    /// Conservative staged 512px working-set admission, including encoder scratch.
    pub const REQUIRED_RAM: usize = (26usize << 30) / 10;
    pub fn is_checkpoint(dir: &std::path::Path) -> bool {
        let config = dir.join("transformer-packed-mflux/quantization_config.json");
        std::fs::read(config)
            .ok()
            .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
            .is_some_and(|v| {
                v["format"] == "prism-packed-affine" && v["bits"] == 1 && v["group_size"] == 128
            })
            && dir
                .join("transformer-packed-mflux/diffusion_pytorch_model.safetensors")
                .is_file()
            && dir
                .join("text_encoder-mlx-4bit/model.safetensors")
                .is_file()
            && dir
                .join("vae/diffusion_pytorch_model.safetensors")
                .is_file()
            && dir.join("tokenizer/tokenizer.json").is_file()
    }
}

pub fn is_flux2(dir: &Path) -> bool {
    if bonsai::is_checkpoint(dir) {
        return true;
    }
    let Ok(bytes) = std::fs::read(dir.join("transformer/model.safetensors.index.json")) else {
        return false;
    };
    let Ok(index) = serde_json::from_slice::<serde_json::Value>(&bytes) else {
        return false;
    };
    index["weight_map"]
        .get("double_stream_modulation_img.linear.weight")
        .is_some()
        && dir.join("tokenizer/tokenizer.json").is_file()
        && dir.join("vae/model.safetensors.index.json").is_file()
        && dir
            .join("text_encoder/model.safetensors.index.json")
            .is_file()
}

pub fn is_flux3(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .is_some_and(|v| v["model"] == "flux-3-image-klein" && v["dit"]["flow_map"] == true)
}

pub mod olala {
    pub const REPO: &str = "OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF";
    pub const ENTRY: &str = "Olala-v1.0-mini-7A1B-q5_k_m.gguf";
}
