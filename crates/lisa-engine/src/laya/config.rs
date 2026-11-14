//! Laya configuration I/O and request override diagnostics.
use anyhow::Context;
pub use lisa_models::laya::config::{LayaConfig, clamp_temperature};
use std::path::Path;
/// Like [`clamp_temperature`], but warn when a value is changed. Used for
/// user-supplied overrides so a clamped calibration is never silent.
pub(crate) fn clamp_temperature_warn(name: &str, orig: f32) -> f32 {
    let mut applied = orig;
    clamp_temperature(name, &mut applied);
    if applied != orig {
        if !(orig > 0.0) || !orig.is_finite() {
            eprintln!("warning: temperature {name}={orig} is not positive/finite; using {applied}");
        } else if applied > orig {
            eprintln!(
                "warning: temperature {name}={orig} clamped to {applied} (below floor 0.5; cannot sharpen further)"
            );
        } else {
            eprintln!(
                "warning: temperature {name}={orig} clamped to {applied} (above ceiling 5.0)"
            );
        }
    }
    applied
}

pub trait LayaConfigIo {
    fn from_dir(dir: &Path) -> anyhow::Result<Self>
    where
        Self: Sized;
    fn apply_overrides(
        &mut self,
        head_max_len: Option<usize>,
        max_len: Option<usize>,
        temperature: Option<Vec<f32>>,
        by_options: &[(String, f32)],
    ) -> anyhow::Result<()>;
}
impl LayaConfigIo for LayaConfig {
    fn from_dir(dir: &Path) -> anyhow::Result<Self> {
        let enc = read_json(&dir.join("encoder/config.json"))?;
        let agent = read_json(&dir.join("rl_agent_config.json"))?;
        let tokenizer = read_json(&dir.join("tokenizer/tokenizer_config.json")).ok();
        Self::from_values(&enc, &agent, tokenizer.as_ref())
    }
    /// Apply runtime overrides, re-checking the budget invariant. Precedence is
    /// `by_options` > `temperature` > shipped config: passing `temperature`
    /// clears the shipped buckets (so it can beat them), and `by_options` is
    /// inserted afterwards (so it beats `temperature`).
    fn apply_overrides(
        &mut self,
        head_max_len: Option<usize>,
        max_len: Option<usize>,
        temperature: Option<Vec<f32>>,
        by_options: &[(String, f32)],
    ) -> anyhow::Result<()> {
        if let Some(m) = max_len {
            self.max_len = m;
        }
        if let Some(h) = head_max_len {
            self.head_max_len = h;
        }
        let (h, m, mp) = (
            self.head_max_len,
            self.max_len,
            self.max_position_embeddings,
        );
        anyhow::ensure!(
            4 < h && h < m && m <= mp,
            "want 4 < head_max_len ({h}) < max_len ({m}) <= max_position_embeddings ({mp})"
        );

        if let Some(t) = temperature {
            anyhow::ensure!(
                t.len() == 3,
                "--temperature needs exactly 3 values (choice,score,noul)"
            );
            // `--temperature` wins over the shipped buckets for the three types;
            // `--temperature-by-options` is applied after and wins over this.
            self.temperature_by_options.clear();
            for (i, &v) in t.iter().enumerate() {
                self.temperature[i] = clamp_temperature_warn(["choice", "score", "noul"][i], v);
            }
        }
        for (key, v) in by_options {
            super::prompt::validate_bucket(key)?;
            self.temperature_by_options
                .insert(key.clone(), clamp_temperature_warn(key, *v));
        }
        Ok(())
    }
}

fn read_json(path: &Path) -> anyhow::Result<serde_json::Value> {
    let data =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&data)?)
}

#[cfg(test)]
mod tests {
    use super::super::prompt::{QType, effective_scale};
    use super::*;

    const ENCODER: &str = include_str!("../../tests/fixtures/laya_encoder_config.json");
    const AGENT: &str = include_str!("../../tests/fixtures/laya_rl_agent_config.json");

    /// The real English config, parsed through a temp dir (no weights needed).
    fn real_config(tag: &str) -> LayaConfig {
        let dir = std::env::temp_dir().join(format!("laya_cfg_{tag}_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("encoder")).unwrap();
        std::fs::create_dir_all(dir.join("tokenizer")).unwrap();
        std::fs::write(dir.join("encoder/config.json"), ENCODER).unwrap();
        std::fs::write(dir.join("rl_agent_config.json"), AGENT).unwrap();
        std::fs::write(
            dir.join("tokenizer/tokenizer_config.json"),
            r#"{"mask_token":"[MASK]"}"#,
        )
        .unwrap();
        let cfg = LayaConfig::from_dir(&dir).unwrap();
        let _ = std::fs::remove_dir_all(dir);
        cfg
    }

    #[test]
    fn parses_the_real_configs() {
        let cfg = real_config("parse");
        assert_eq!(cfg.num_layers, 28);
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_heads, 16);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.intermediate_size, 2624);
        assert_eq!(cfg.vocab_size, 50368);
        assert_eq!(cfg.cls_id, 50281);
        assert_eq!(cfg.layer_global.iter().filter(|g| **g).count(), 10);
        assert_eq!(cfg.n_actions, 2); // escalate + one
        assert!((cfg.rope_theta_global - 160000.0).abs() < 1.0);
    }

    #[test]
    fn temperature_override_beats_the_shipped_bucket() {
        let mut cfg = real_config("temp");
        // Shipped `choice:3-5` is ~1.76; the per-type `--temperature` must win.
        assert!((effective_scale(&cfg, QType::Choice, 5).1 - 1.76015).abs() < 1e-3);
        cfg.apply_overrides(None, None, Some(vec![0.7, 1.0, 1.0]), &[])
            .unwrap();
        assert!((effective_scale(&cfg, QType::Choice, 5).1 - 0.7).abs() < 1e-6);
        assert!(cfg.temperature_by_options.is_empty());
    }

    #[test]
    fn by_options_beats_temperature() {
        let mut cfg = real_config("byopt");
        cfg.apply_overrides(
            None,
            None,
            Some(vec![0.7, 1.0, 1.0]),
            &[("choice:3-5".to_string(), 0.9)],
        )
        .unwrap();
        let (bucket, scale) = effective_scale(&cfg, QType::Choice, 5);
        assert_eq!(bucket, "choice:3-5");
        assert!((scale - 0.9).abs() < 1e-6);
    }

    #[test]
    fn rejects_unknown_bucket_size() {
        let mut cfg = real_config("reject");
        let err = cfg
            .apply_overrides(None, None, None, &[("choice:21-40".to_string(), 0.7)])
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("choice:21-40"), "{msg}");
        assert!(msg.contains("11+"), "{msg}");
    }

    #[test]
    fn clamp_warns_and_returns_applied() {
        assert!((clamp_temperature_warn("choice:11+", 0.1) - 0.5).abs() < 1e-6);
        assert!((clamp_temperature_warn("choice", 10.0) - 5.0).abs() < 1e-6);
        assert!((clamp_temperature_warn("noul", 0.0) - 1.0).abs() < 1e-6);
        assert!((clamp_temperature_warn("choice", 1.2) - 1.2).abs() < 1e-6);
    }
}
