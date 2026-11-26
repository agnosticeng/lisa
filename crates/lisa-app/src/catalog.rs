//! Explicit download selections for LISA's supported model implementations.
//! Review the checkpoint's architecture, quantization and required files before
//! adding an entry. Hub search results never extend this list automatically.

pub struct CuratedModel {
    pub repo_id: &'static str,
    pub name: &'static str,
    /// The weights' quantization as the library badge shows it.
    pub quantization: &'static str,
    /// The family the library shows as a badge.
    pub family: &'static str,
    /// Notable capabilities the library shows as tags and filters by.
    pub features: &'static [&'static str],
    /// What the model generates: `"text"` or `"image"`.
    pub generates: &'static str,
    /// Release date (`YYYY-MM-DD`): the Hub repository's creation date.
    pub released: &'static str,
    /// Whether its kernels need Metal 4 (an Apple7 or newer GPU on macOS
    /// or iOS 26 or later).
    pub metal4: bool,
    pub description: &'static str,
    /// Selected package sizes checked against Hub metadata.
    /// Used immediately at startup and when the Hub is unavailable.
    pub download_bytes: u64,
    /// Weight residency when disk-only tables make download size misleading.
    pub resident_bytes: Option<u64>,
}

pub const MODELS: &[CuratedModel] = &[
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::inkling2::REPO,
        name: "Inkling 2 Mini",
        quantization: "Q4_K_M",
        family: "Inkling",
        features: &["Vision", "Tools"],
        generates: "text",
        released: "2026-11-20",
        metal4: true,
        description: "Fast chat, reasoning and image understanding.",
        download_bytes: 18239869695,
        resident_bytes: None,
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::olala::REPO,
        name: "Olala mini 7A1B",
        quantization: "Q5_K_M",
        family: "Olala",
        features: &["Tools"],
        generates: "text",
        released: "2026-10-30",
        metal4: false,
        description: "Lightweight multilingual chat and tools.",
        download_bytes: 5990583679,
        resident_bytes: Some(5_990_551_040),
    },
    CuratedModel {
        repo_id: "LiquidAI/LFM3-1.7B-MLX-4bit",
        name: "LFM3 1.7B",
        quantization: "MLX 4-bit",
        family: "LFM",
        features: &["Tools"],
        generates: "text",
        released: "2026-10-27",
        metal4: false,
        description: "Small, quick chat for iPhone and Mac.",
        download_bytes: 991_133_347,
        resident_bytes: None,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-27B-4bit",
        name: "Qwen 3.8 27B",
        quantization: "MLX 4-bit",
        family: "Qwen",
        features: &["Vision", "Tools"],
        generates: "text",
        released: "2026-09-05",
        metal4: false,
        description: "Everyday assistant for chat, code and images.",
        resident_bytes: None,
        download_bytes: 16_320_428_961,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-4B-MLX-4bit",
        name: "Qwen 3.5 4B",
        quantization: "MLX 4-bit",
        family: "Qwen",
        features: &["Vision", "Tools"],
        generates: "text",
        released: "2026-03-02",
        metal4: false,
        description: "Compact chat for Macs with less memory.",
        resident_bytes: None,
        download_bytes: 3_061_132_920,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        name: "Qwen 3.5 0.8B",
        quantization: "MLX 4-bit",
        family: "Qwen",
        features: &["Vision", "Tools"],
        generates: "text",
        released: "2026-03-02",
        metal4: false,
        description: "The smallest download, for quick chat.",
        resident_bytes: None,
        download_bytes: 652_031_020,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-Flash-Next-4bit",
        name: "Qwen 3.8 Flash-Next",
        quantization: "MLX 4-bit",
        family: "Qwen",
        features: &["Tools"],
        generates: "text",
        released: "2026-09-25",
        metal4: false,
        description: "Large MoE for demanding chat and code.",
        resident_bytes: None,
        download_bytes: 112_334_975_818,
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::glm53::REPO,
        name: "GLM 5.3 Flash",
        quantization: "Q2",
        family: "GLM",
        features: &["Tools"],
        generates: "text",
        released: "2026-08-27",
        metal4: false,
        description: "Large MoE for advanced reasoning and code.",
        resident_bytes: Some(96_505_816_384),
        download_bytes: 96_505_816_384,
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::deepseek41::REPO,
        name: "DeepSeek V4.1 Flash Coder",
        quantization: "IQ2_XXS",
        family: "DeepSeek",
        features: &["Tools"],
        generates: "text",
        released: "2026-11-02",
        metal4: false,
        description: "Coding specialist for high-memory Macs.",
        download_bytes: 182_718_858_272,
        resident_bytes: Some(73_000_000_000),
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::bonsai::REPO,
        name: "Bonsai Binary 4B",
        quantization: "1-bit",
        family: "Bonsai",
        features: &[],
        generates: "image",
        released: "2026-05-18",
        metal4: false,
        description: "Tiny four-step image generator.",
        download_bytes: 3_428_210_856,
        resident_bytes: Some(lisa_engine::checkpoint::catalog::bonsai::REQUIRED_RAM as u64),
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-4b-4bit",
        name: "FLUX.2 Klein 4B",
        quantization: "MLX 4-bit",
        family: "FLUX",
        features: &[],
        generates: "image",
        released: "2026-05-23",
        metal4: false,
        description: "Fast four-step image generation.",
        resident_bytes: None,
        download_bytes: 4_619_708_452,
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-9b-4bit",
        name: "FLUX.2 Klein 9B",
        quantization: "MLX 4-bit",
        family: "FLUX",
        features: &[],
        generates: "image",
        released: "2026-05-23",
        metal4: false,
        description: "Higher-quality four-step images.",
        resident_bytes: None,
        download_bytes: 9_541_194_042,
    },
    CuratedModel {
        repo_id: "black-forest-labs/flux-3-image-klein",
        name: "FLUX 3 Image Klein",
        quantization: "BF16",
        family: "FLUX",
        features: &[],
        generates: "image",
        released: "2026-10-13",
        metal4: false,
        description: "High-quality images with prompt expansion.",
        resident_bytes: None,
        download_bytes: 20_320_953_966,
    },
];

pub fn find(repo_id: &str) -> Option<&'static CuratedModel> {
    MODELS.iter().find(|entry| entry.repo_id == repo_id)
}

/// Curated repositories whose MTP head ships as an `mtp/` sidecar in the
/// same repository. Older installs may lack it and complete it on load.
pub fn ships_mtp_sidecar(repo: &str) -> bool {
    repo == "agnosticeng/Qwen3.8-27B-4bit"
}
