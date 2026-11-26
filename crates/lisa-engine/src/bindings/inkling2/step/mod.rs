//! The pinned reference's Inkling step: dispatch builders of its vendored
//! kernels (`lisa-metal`'s `shaders/reference/`, provenance in its
//! `provenance.json`), its schedule, its per-device kernel calibration and
//! dense tile law, the sequence state, the target's decode/verify step and
//! prompt prefill, and the DFlash2 draft step.
pub(crate) mod calibration;
pub(crate) mod dense;
pub(crate) mod draft;
// Builders of the vendored ABI not every caller binds.
#[allow(dead_code)]
pub(crate) mod kernels;
pub(crate) mod schedule;
pub(crate) mod state;
pub(crate) mod target;
