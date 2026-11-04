//! Explicit download selections for LISA's supported model implementations.
//! Review the checkpoint's architecture, quantization and required files before
//! adding an entry. Hub search results never extend this list automatically.

pub struct CuratedModel {
    pub repo_id: &'static str,
    pub name: &'static str,
    pub quantization: &'static str,
    pub description: &'static str,
    pub model_type: &'static str,
    /// Selected package sizes checked against Hub metadata on 2026-11-03.
    /// Used immediately at startup and when the Hub is unavailable.
    pub download_bytes: u64,
    /// Weight residency when disk-only tables make download size misleading.
    pub resident_bytes: Option<u64>,
}

pub const MODELS: &[CuratedModel] = &[
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-27B-4bit",
        name: "Qwen 3.8 27B",
        quantization: "4-bit",
        description: "Everyday chat, coding and image input. Includes an MTP head.",
        model_type: "qwen3_5",
        resident_bytes: None,
        download_bytes: 16_320_428_961,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-4B-MLX-4bit",
        name: "Qwen 3.5 4B",
        quantization: "4-bit",
        description: "Compact chat and image input for smaller Macs.",
        model_type: "qwen3_5",
        resident_bytes: None,
        download_bytes: 3_061_132_920,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        name: "Qwen 3.5 0.8B",
        quantization: "4-bit",
        description: "Smallest download for lightweight chat and image input.",
        model_type: "qwen3_5",
        resident_bytes: None,
        download_bytes: 652_031_020,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-27B-2bit",
        name: "Qwen 3.8 27B",
        quantization: "2-bit",
        description: "A smaller 27B checkpoint for memory-constrained Macs. Includes MTP.",
        model_type: "qwen3_5",
        resident_bytes: None,
        download_bytes: 8_695_542_535,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-Flash-Next-4bit",
        name: "Qwen 3.8 Flash-Next",
        quantization: "4-bit",
        description: "Large mixture-of-experts chat model with MTP. Needs a high-memory Mac.",
        model_type: "qwen4_exp",
        resident_bytes: None,
        download_bytes: 112_334_975_818,
    },
    CuratedModel {
        repo_id: lisa_engine::models::glm53::gguf::REPO,
        name: "GLM 5.3 Flash",
        quantization: "Q2 GGUF",
        description: "DS4's original GLM checkpoint with MTP. About 90 GiB of weights; requires a 128 GB Mac. Native GGUF support is experimental.",
        model_type: "glm5-next",
        resident_bytes: Some(96_505_816_384),
        download_bytes: 96_505_816_384,
    },
    CuratedModel {
        repo_id: lisa_engine::models::deepseek41::REPO,
        name: "DeepSeek V4.1 Flash Coder",
        quantization: "Mixed IQ GGUF",
        description: "Experimental text-only coding model. About 73 GB of resident weights and 111 GB of disk-only lookup data. Requires a high-memory Mac and fast SSD.",
        model_type: "deepseek41",
        download_bytes: 182_718_858_272,
        resident_bytes: Some(73_000_000_000),
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-4b-4bit",
        name: "FLUX.2 Klein 4B",
        quantization: "4-bit",
        description: "Generate images locally from a text prompt.",
        model_type: "flux2-klein",
        resident_bytes: None,
        download_bytes: 4_619_708_452,
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-9b-4bit",
        name: "FLUX.2 Klein 9B",
        quantization: "4-bit",
        description: "Larger four-step image generator. FLUX Non-Commercial License.",
        model_type: "flux2-klein",
        resident_bytes: None,
        download_bytes: 9_541_194_042,
    },
    CuratedModel {
        repo_id: "black-forest-labs/flux-3-image-klein",
        name: "FLUX 3 Image Klein",
        quantization: "BF16",
        description: "Image generation with local prompt expansion. Requires Hugging Face access and the FLUX Kommunity License.",
        model_type: "flux3-klein",
        resident_bytes: None,
        download_bytes: 20_320_953_966,
    },
];

pub fn find(repo_id: &str) -> Option<&'static CuratedModel> {
    MODELS.iter().find(|entry| entry.repo_id == repo_id)
}

pub fn rank(repo_id: &str) -> usize {
    MODELS
        .iter()
        .position(|entry| entry.repo_id == repo_id)
        .unwrap_or(usize::MAX)
}
