//! Persisted schedule identity and admission rules. No environment or file access.
use anyhow::ensure;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct Identity {
    pub schedule: String,
    pub gpu: String,
    pub architecture: String,
    pub memory_bytes: usize,
    pub target: String,
    pub draft: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum SsmMode {
    #[default]
    Capture,
    Replay,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Policy {
    pub format: String,
    pub identity: Identity,
    /// Includes the carried target token, like SGLang's BS1 uniform width.
    pub verify_width: usize,
    pub draft_width: usize,
    #[serde(default)]
    pub ssm: SsmMode,
}
impl Policy {
    pub fn validate(&self, identity: &Identity) -> anyhow::Result<()> {
        ensure!(
            self.format == "lisa-dspine-sps-metal-v1",
            "unsupported DSpine SPS profile"
        );
        ensure!(
            &self.identity == identity,
            "DSpine SPS device/checkpoint differs; retune on this model and GPU"
        );
        ensure!(
            (3..=16).contains(&self.draft_width)
                && (2..=self.draft_width).contains(&self.verify_width),
            "DSpine SPS requires draft width 3..16 and verify width 2..draft width"
        );
        Ok(())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn profile_rejects_other_devices_checkpoints_and_invalid_widths() {
        let id = Identity {
            schedule: "v1".into(),
            gpu: "GPU".into(),
            architecture: "arch".into(),
            memory_bytes: 42,
            target: "target".into(),
            draft: "draft".into(),
        };
        let mut p = Policy {
            format: "lisa-dspine-sps-metal-v1".into(),
            identity: id.clone(),
            verify_width: 16,
            draft_width: 16,
            ssm: SsmMode::Capture,
        };
        assert!(p.validate(&id).is_ok());
        for width in [0, 1, 17, usize::MAX] {
            p.verify_width = width;
            assert!(p.validate(&id).is_err());
        }
        p.verify_width = 2;
        for width in [0, 1, 2, 17] {
            p.draft_width = width;
            assert!(p.validate(&id).is_err());
        }
        p.draft_width = 16;
        let mut legacy = serde_json::to_value(&p).unwrap();
        legacy.as_object_mut().unwrap().remove("ssm");
        assert_eq!(
            serde_json::from_value::<Policy>(legacy.clone())
                .unwrap()
                .ssm,
            SsmMode::Capture
        );
        legacy["ssm"] = serde_json::json!("unknown");
        assert!(serde_json::from_value::<Policy>(legacy).is_err());
        for field in [
            "gpu",
            "architecture",
            "schedule",
            "target",
            "draft",
            "memory_bytes",
        ] {
            let mut wrong = serde_json::to_value(&id).unwrap();
            wrong[field] = if field == "memory_bytes" {
                serde_json::json!(43)
            } else {
                serde_json::json!("changed")
            };
            assert!(
                p.validate(&serde_json::from_value(wrong).unwrap()).is_err(),
                "{field}"
            );
        }
    }
}
