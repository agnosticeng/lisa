//! Explicit download selections for LISA's supported model implementations.
//! Review the checkpoint's architecture, quantization and required files before
//! adding an entry. Hub search results never extend this list automatically.

pub struct CuratedModel {
    pub repo_id: &'static str,
    pub name: &'static str,
    pub description: &'static str,
    /// Selected package sizes checked against Hub metadata.
    /// Used immediately at startup and when the Hub is unavailable.
    pub download_bytes: u64,
    /// Weight residency when disk-only tables make download size misleading.
    pub resident_bytes: Option<u64>,
}

pub const MODELS: &[CuratedModel] = &[
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::olala::REPO,
        name: "Olala mini 7A1B · 5-bit",
        description: "Multilingual text chat and tools. Original 5-bit GGUF, about 6 GB. Experimental native support.",
        download_bytes: 5990583679,
        resident_bytes: Some(5_990_551_040),
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::bonsai::REPO,
        name: "Bonsai Binary 4B",
        description: "Compact local image generation. Four steps, up to 512 × 512 pixels.",
        download_bytes: 3_428_210_856,
        resident_bytes: Some(lisa_engine::checkpoint::catalog::bonsai::REQUIRED_RAM as u64),
    },
    CuratedModel {
        repo_id: "LiquidAI/LFM3-1.7B-MLX-4bit",
        name: "LFM3 1.7B",
        description: "Compact multilingual text chat and tool use for iPhone and Mac.",
        download_bytes: 991_133_347,
        resident_bytes: None,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-27B-4bit",
        name: "Qwen 3.8 27B",
        description: "Everyday chat, coding and image input. Includes the DSpine drafter.",
        resident_bytes: None,
        download_bytes: 16_320_428_961,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-27B-2bit",
        name: "Qwen 3.8 27B",
        description: "Compact text-only Qwen 27B with the DSpine drafter.",
        resident_bytes: None,
        download_bytes: 8_695_542_535,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-4B-MLX-4bit",
        name: "Qwen 3.5 4B",
        description: "Compact chat and image input for smaller Macs.",
        resident_bytes: None,
        download_bytes: 3_061_132_920,
    },
    CuratedModel {
        repo_id: "mlx-community/Qwen3.5-0.8B-MLX-4bit",
        name: "Qwen 3.5 0.8B",
        description: "Smallest download for lightweight chat and image input.",
        resident_bytes: None,
        download_bytes: 652_031_020,
    },
    CuratedModel {
        repo_id: "agnosticeng/Qwen3.8-Flash-Next-4bit",
        name: "Qwen 3.8 Flash-Next",
        description: "Large mixture-of-experts chat model with MTP. Needs a high-memory Mac.",
        resident_bytes: None,
        download_bytes: 112_334_975_818,
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::glm53::REPO,
        name: "GLM 5.3 Flash",
        description: "DS4's original GLM checkpoint with MTP. About 90 GiB of weights; requires a 128 GB Mac. Native GGUF support is experimental.",
        resident_bytes: Some(96_505_816_384),
        download_bytes: 96_505_816_384,
    },
    CuratedModel {
        repo_id: lisa_engine::checkpoint::catalog::deepseek41::REPO,
        name: "DeepSeek V4.1 Flash Coder",
        description: "Experimental text-only coding model. About 73 GB of resident weights and 111 GB of disk-only lookup data. Requires a high-memory Mac and fast SSD.",
        download_bytes: 182_718_858_272,
        resident_bytes: Some(73_000_000_000),
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-4b-4bit",
        name: "FLUX.2 Klein 4B",
        description: "Generate images locally from a text prompt.",
        resident_bytes: None,
        download_bytes: 4_619_708_452,
    },
    CuratedModel {
        repo_id: "mlx-community/flux2-klein-9b-4bit",
        name: "FLUX.2 Klein 9B",
        description: "Larger four-step image generator. FLUX Non-Commercial License.",
        resident_bytes: None,
        download_bytes: 9_541_194_042,
    },
    CuratedModel {
        repo_id: "black-forest-labs/flux-3-image-klein",
        name: "FLUX 3 Image Klein",
        description: "Image generation with local prompt expansion. Requires Hugging Face access and the FLUX Kommunity License.",
        resident_bytes: None,
        download_bytes: 20_320_953_966,
    },
];

pub fn find(repo_id: &str) -> Option<&'static CuratedModel> {
    MODELS.iter().find(|entry| entry.repo_id == repo_id)
}

/// One shared draft download across all compatible target quantizations.
pub const DSPINE_DOWNLOAD_BYTES: u64 = 4_051_857_976;
pub const DSPINE_WEIGHT_BYTES: u64 = 4_051_831_880;

pub fn uses_dspine(repo: &str) -> bool {
    matches!(
        repo,
        "agnosticeng/Qwen3.8-27B-4bit" | "agnosticeng/Qwen3.8-27B-2bit"
    )
}

/// Fallback package size excludes MLX sidecars superseded by DSpine.
/// Sizes include each sidecar's config/tokenizer, as advertised by the Hub.
pub fn target_download_bytes(entry: &CuratedModel) -> u64 {
    let mtp = match entry.repo_id {
        "agnosticeng/Qwen3.8-27B-4bit" => 238_934_261 + 3_804,
        "agnosticeng/Qwen3.8-27B-2bit" => 238_934_237 + 3_172 + 19_989_325,
        _ => 0,
    };
    entry.download_bytes.saturating_sub(mtp)
}

pub fn bundle_download_bytes(entry: &CuratedModel) -> u64 {
    target_download_bytes(entry)
        + if uses_dspine(entry.repo_id) {
            DSPINE_DOWNLOAD_BYTES
        } else {
            0
        }
}
