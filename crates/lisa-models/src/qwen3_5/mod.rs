//! Portable model definition and computation plans.
pub mod config;

use crate::decoder::{Mixer, Plan, Step};

/// Delayed residual addition is part of this model's arithmetic/dataflow law.
/// The MLP output is returned separately and added at the next input norm.
const fn decoder_steps(kind: Mixer) -> [Step; 4] {
    [
        Step::ResidualNorm {
            norm: 0,
            input: 0,
            residual: 1,
            sum: 2,
            normalized: 3,
        },
        Step::Mix {
            kind,
            input: 3,
            output: 4,
        },
        Step::ResidualNorm {
            norm: 1,
            input: 2,
            residual: 4,
            sum: 5,
            normalized: 6,
        },
        Step::FeedForward {
            input: 6,
            output: 7,
        },
    ]
}
static ATTENTION: [Step; 4] = decoder_steps(Mixer::Attention);
static GATED_DELTA: [Step; 4] = decoder_steps(Mixer::GatedDelta);

pub const fn decoder(full_attention: bool) -> Plan {
    Plan {
        steps: if full_attention {
            &ATTENTION
        } else {
            &GATED_DELTA
        },
        slots: 8,
        output: (5, 7),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hybrid_decoder_programs_have_valid_dataflow() {
        for full in [false, true] {
            super::decoder(full).validate().unwrap();
        }
    }
}

pub mod attention;

pub mod mtp;

pub mod readout;

pub mod entry;

use config::Qwen35Config;
// Bind cached state to the corrected attention launch before lazy pipelines
// compile. Legacy Qwen state could contain outputs from the invalid >1024-thread
// launch; it must not be restored merely because no new SDPA kernel ran yet.
pub fn cache_law(config: &Qwen35Config, device_arithmetic: &[u8]) -> u64 {
    let mut bytes = b"lisa.qwen35.mlx.valid-head-split.v1".to_vec();
    bytes.extend_from_slice(device_arithmetic);
    for value in [
        config.hidden_size as u64,
        config.num_hidden_layers as u64,
        config.num_attention_heads as u64,
        config.num_key_value_heads as u64,
        config.head_dim as u64,
        config.quant_group_size as u64,
        config.quant_bits as u64,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    })
}

pub mod tower;

pub mod dspine;

pub mod gguf;
