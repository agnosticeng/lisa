//! Shared CLI front-end helpers: argument types and glue used by both the
//! `lisa` (user) and `lisa-bench` (test/parity/benchmark) binaries. Pure
//! plumbing — no model code lives here.

use std::path::PathBuf;

use crate::sampler;

/// `--model`: a local model directory or a Hugging Face repo id, resolved to a
/// local directory when the arguments are parsed.
#[derive(Clone)]
pub struct ModelDir(pub PathBuf);

impl std::ops::Deref for ModelDir {
    type Target = PathBuf;
    fn deref(&self) -> &PathBuf {
        &self.0
    }
}

impl AsRef<std::path::Path> for ModelDir {
    fn as_ref(&self) -> &std::path::Path {
        &self.0
    }
}

impl std::str::FromStr for ModelDir {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        crate::models::resolve_model_dir(s)
            .map(ModelDir)
            .map_err(|e| e.to_string())
    }
}

/// MTP draft depth: 0 = serial, 2..=6 = speculative. A single draft is never
/// worth a round, so depth 1 is rejected rather than silently routed.
pub fn parse_depth(s: &str) -> Result<usize, String> {
    let v: usize = s.parse().map_err(|e| format!("{e}"))?;
    if v == 1 {
        return Err("depth 1 is not supported; use 0 (serial) or 2..=6 (MTP)".to_string());
    }
    Ok(v)
}

/// `--depth` on the generation commands: `auto` (the EV controller; the
/// default when the model ships a drafter) or a fixed `0` (serial) / `2..=6`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DepthArg {
    Auto,
    Fixed(usize),
}

impl std::str::FromStr for DepthArg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.eq_ignore_ascii_case("auto") {
            return Ok(DepthArg::Auto);
        }
        let v: usize = s.parse().map_err(|e| format!("{e}"))?;
        if v == 1 {
            return Err(
                "depth 1 is not supported; use 0 (serial), 2..=6 (MTP), or auto".to_string(),
            );
        }
        Ok(DepthArg::Fixed(v))
    }
}

impl std::fmt::Display for DepthArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DepthArg::Auto => write!(f, "auto"),
            DepthArg::Fixed(d) => write!(f, "{d}"),
        }
    }
}

/// Effective PLD depth from the `--pld` / `--pld-depth` pair (0 = off).
/// Widest draft the verify fast path actually covers.
///
/// The split-K verify lane gates `2..=7` rows — one precompiled kernel per M
/// (`warm_verify_qmm` in `lisa-mlx/src/jit/matmul.rs:605-631`); the other
/// row-fusions stop at 8 as well. A round's draft width is `S = depth + 1`, so a
/// depth above `FAST_S_MAX - 1` lands every round on the stock qmm path
/// (measured, bucket <2k: S7 = 78.0 ms → S8 = 108.0 → S9 = 136.8, i.e. 2.16× S5).
///
/// NOTE: an earlier backlog item said "align every width up to 32" — that is a
/// different geometry, not this one's. Here, rounding *up* is strictly
/// worse: it puts every draft off-lane. Clamp DOWNWARD.
pub const FAST_S_MAX: usize = 7;
/// Highest PLD depth whose round still fits the fast lane (`S = depth + 1`).
pub const PLD_DEPTH_MAX: usize = FAST_S_MAX - 1;

pub fn pld_enable(pld: bool, pld_depth: usize) -> usize {
    if !pld {
        0
    } else if pld_depth == 0 {
        4
    } else {
        if pld_depth > PLD_DEPTH_MAX {
            eprintln!(
                "[pld] width engaged at depth {PLD_DEPTH_MAX} (requested {pld_depth}; \
                 S={pld_depth} falls off the split-K verify lane, S>{FAST_S_MAX} costs ~2× )"
            );
        }
        pld_depth.min(PLD_DEPTH_MAX)
    }
}

/// Resolve a `--depth` argument against the sampler/model: `Some(policy)` runs
/// the speculative path, `None` runs serial (PLD still applies there). `auto`
/// means the EV controller, but only when the model ships a drafter and sampling
/// is greedy — otherwise it silently degrades to serial (logged by the
/// controller when it runs).
pub fn resolve_mtp_depth(
    depth: DepthArg,
    greedy: bool,
    has_drafter: bool,
) -> anyhow::Result<Option<crate::session::MtpDepth>> {
    use crate::session::MtpDepth;
    match depth {
        DepthArg::Fixed(0) => Ok(None),
        DepthArg::Fixed(d) => {
            Ok(Some(MtpDepth::Fixed(d)))
        }
        DepthArg::Auto => Ok(if greedy && has_drafter {
            Some(MtpDepth::Auto)
        } else {
            None
        }),
    }
}

pub fn build_sampler(
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rep_penalty: f32,
    seed: u64,
) -> sampler::Sampler {
    let mut s = sampler::Sampler {
        temperature,
        top_k,
        top_p,
        min_p,
        repetition_penalty: rep_penalty,
        ..Default::default()
    };
    if seed != 0 {
        s.seed = seed;
    }
    s
}
