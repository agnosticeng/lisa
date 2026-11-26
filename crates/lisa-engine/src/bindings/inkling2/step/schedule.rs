//! The pinned reference's application schedule of an Inkling step
//! (`Inkling2LoopSchedule`): 24 weight layers, layers 6-17 run three times,
//! 48 applications. Which rows each application reads and writes, which
//! block states its mixes read, and where its messages gate it. Pure.
use super::state::APPLICATIONS;

/// Hidden width, KV width (8 heads x 128) and the attention's query width.
pub(crate) const HIDDEN: u32 = 2304;
pub(crate) const KV_WIDTH: u32 = 1024;
pub(crate) const ATTENTION_WIDTH: u32 = 3072;
/// Row stride of the packed q projection: queries, gates, relative states,
/// sinks, padded to whole 64-column tiles.
pub(crate) const PACKED_WIDTH: u32 = 6592;
/// Columns of the packed projection's segments and the sinks' column.
pub(crate) const QUERY_COLUMNS: u32 = 3072;
pub(crate) const RELATIVE_COLUMNS: u32 = 384;
pub(crate) const SINK_COLUMN: u32 = 6528;
pub(crate) const SINKS: u32 = 24;
/// Rows of a full layer's key/value tap projections, and their output
/// stride (a whole tile).
pub(crate) const TAPS: u32 = 32;
pub(crate) const TAP_STRIDE: u32 = 64;
/// Experts, routes a row, expert width, the router's score stride.
pub(crate) const EXPERTS: u32 = 128;
pub(crate) const TOP_K: u32 = 8;
pub(crate) const INTERMEDIATE: u32 = 1152;
/// Vocabulary rows of the head and the tokens among them.
pub(crate) const VOCABULARY: u32 = 201_024;
pub(crate) const TOKENS: u32 = 200_058;
pub(crate) const SOFTCAP: f32 = 30.0;
/// The routing weights' scale.
pub(crate) const ROUTED_SCALE: f32 = 8.0;
/// Message gate rank and the messages a loop layer mixes (q, k, v, FFN).
pub(crate) const GATE_RANK: u32 = 64;
pub(crate) const MESSAGES: u32 = 4;
/// Draft capture slots and the slot of the final norm's rows.
pub(crate) const CAPTURES: u32 = 4;
pub(crate) const FINAL_CAPTURE: u32 = 3;

const WEIGHT_LAYERS: usize = 24;
const PERIOD: usize = 6;
const LOOP_FIRST: usize = 6;
const LOOP_END: usize = 18;
const LOOP_LAYERS: usize = LOOP_END - LOOP_FIRST;
const PASSES: usize = 3;
/// Stack planes: one block state per period.
pub(crate) const STACK_PLANES: u32 = (WEIGHT_LAYERS / PERIOD) as u32;
/// The pass before's planes a later pass's messages mix: the block state it
/// pushed after the loop's first (from stack plane [`FIRST_PASS_PLANE`]),
/// then its output.
pub(crate) const PREVIOUS_PLANES: u32 = (LOOP_LAYERS / PERIOD) as u32;
pub(crate) const FIRST_PASS_PLANE: u32 = (LOOP_FIRST / PERIOD + 1) as u32;

/// Rows a step's application reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rows {
    /// The two alternating hidden row buffers.
    Hidden(u32),
    /// A block stack plane.
    Stack(u32),
    /// A plane of the pass before's states.
    Previous(u32),
}

/// The weight layer application `a` runs.
pub(crate) fn weight_layer(a: usize) -> usize {
    if a < LOOP_FIRST {
        return a;
    }
    let looped = a - LOOP_FIRST;
    if looped < PASSES * LOOP_LAYERS {
        return LOOP_FIRST + looped % LOOP_LAYERS;
    }
    a - (PASSES - 1) * LOOP_LAYERS
}

/// Its pass of the loop from 0; 0 outside the loop.
pub(crate) fn pass(a: usize) -> usize {
    if a < LOOP_FIRST {
        return 0;
    }
    let pass = (a - LOOP_FIRST) / LOOP_LAYERS;
    if pass < PASSES { pass } else { 0 }
}

fn loop_layer(layer: usize) -> bool {
    (LOOP_FIRST..LOOP_END).contains(&layer)
}

/// Whether a weight layer attends its whole history (all but 5, 11, 17, 23
/// slide).
pub(crate) fn full_layer(layer: usize) -> bool {
    layer % PERIOD == PERIOD - 1
}

/// Whether application `a` attends its whole history.
pub(crate) fn full(a: usize) -> bool {
    full_layer(weight_layer(a))
}

/// Whether it pushes its input rows onto the block stack: its mixer's
/// output is then the residual alone.
pub(crate) fn boundary(a: usize) -> bool {
    weight_layer(a).is_multiple_of(PERIOD)
}

/// Whether messages of the pass before gate its inputs.
pub(crate) fn gated(a: usize) -> bool {
    loop_layer(weight_layer(a)) && pass(a) > 0
}

/// Whether a pass restarts at it (after copying the pass before's states).
pub(crate) fn restarts(a: usize) -> bool {
    gated(a) && weight_layer(a) == LOOP_FIRST
}

/// Block states the mix before its mixer reads beside its input.
pub(crate) fn mixer_states(a: usize) -> u32 {
    weight_layer(a).div_ceil(PERIOD) as u32
}

/// Block states the mix before its FFN reads beside its residual.
pub(crate) fn ffn_states(a: usize) -> u32 {
    (weight_layer(a) / PERIOD + 1) as u32
}

/// Queries of a weight layer's mixer mix: none in layer 0, one where it
/// slides, three (q, k, v) where it attends everything.
pub(crate) fn mixer_queries(layer: usize) -> u32 {
    if layer == 0 {
        0
    } else if full_layer(layer) {
        3
    } else {
        1
    }
}

/// The rows passed into application `a`.
pub(crate) fn input(a: usize) -> Rows {
    let layer = weight_layer(a);
    if layer.is_multiple_of(PERIOD) {
        Rows::Stack((layer / PERIOD) as u32)
    } else {
        Rows::Hidden((a & 1) as u32)
    }
}

/// The rows application `a` writes: the next one's input, a pass's output
/// where another pass follows, and the last application's.
pub(crate) fn output(a: usize) -> Rows {
    if a + 1 == APPLICATIONS {
        return Rows::Hidden(((a + 1) & 1) as u32);
    }
    let layer = weight_layer(a);
    if loop_layer(layer) && layer + 1 == LOOP_END && pass(a) + 1 < PASSES {
        return Rows::Previous(PREVIOUS_PLANES - 1);
    }
    input(a + 1)
}

/// The draft capture slot of the rows passed into application `a`.
pub(crate) fn capture_slot(a: usize) -> Option<u32> {
    match a {
        5 => Some(0),
        23 => Some(1),
        41 => Some(2),
        _ => None,
    }
}

/// The lane cap of a one-lane step of `rows` rows: the narrow cap of eight
/// while the rows fit it, else a verify block's sixteen.
pub(crate) fn lane_cap(rows: u32) -> u32 {
    if rows <= 8 { 8 } else { 16 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_matches_the_reference_checks() {
        assert_eq!(weight_layer(5), 5);
        assert_eq!(weight_layer(18), 6);
        assert_eq!(weight_layer(41), 17);
        assert_eq!(weight_layer(42), 18);
        assert_eq!(weight_layer(47), 23);
        assert_eq!(pass(42), 0);
        assert_eq!(pass(29), 1);
        assert!(restarts(18) && restarts(30) && !restarts(6));
        assert!(gated(29) && !gated(17) && !gated(42));
        assert!(full(5) && full(29) && !full(28));
        assert!(boundary(24) && !boundary(25));
        assert_eq!(input(0), Rows::Stack(0));
        assert_eq!(output(5), Rows::Stack(1));
        assert_eq!(input(18), Rows::Stack(1));
        assert_eq!(output(17), Rows::Previous(1));
        assert_eq!(output(29), Rows::Previous(1));
        assert_eq!(output(41), Rows::Stack(3));
        assert_eq!(output(23), Rows::Stack(2));
        assert_eq!(output(47), Rows::Hidden(0));
        assert_eq!(mixer_states(0), 0);
        assert_eq!(mixer_states(6), 1);
        assert_eq!(ffn_states(6), 2);
        assert_eq!(mixer_states(47), 4);
        assert_eq!((PREVIOUS_PLANES, FIRST_PASS_PLANE, STACK_PLANES), (2, 2, 4));
    }

    #[test]
    fn sliding_applications_match_the_reference_mask() {
        let sliding = (0..APPLICATIONS)
            .filter(|&a| !full(a))
            .fold(0u64, |bits, a| bits | 1 << a);
        assert_eq!(sliding, 0x7DF7_DF7D_F7DF);
        assert_eq!(
            (0..APPLICATIONS)
                .filter(|&a| super::super::state::full(a) != full(a))
                .count(),
            0
        );
    }

    #[test]
    fn gated_and_captured_applications() {
        let gated = (0..APPLICATIONS).filter(|&a| gated(a)).collect::<Vec<_>>();
        assert_eq!(gated, (18..42).collect::<Vec<_>>());
        let captured = (0..APPLICATIONS)
            .filter_map(|a| capture_slot(a).map(|s| (a, s)))
            .collect::<Vec<_>>();
        assert_eq!(captured, [(5, 0), (23, 1), (41, 2)]);
        // Captured inputs are full, non-boundary applications' hidden rows.
        for (a, _) in captured {
            assert!(full(a) && !boundary(a));
            assert_eq!(input(a), Rows::Hidden(1));
        }
        assert_eq!(
            (0..24).map(mixer_queries).collect::<Vec<_>>()[..7],
            [0, 1, 1, 1, 1, 3, 1]
        );
        assert_eq!(
            (lane_cap(1), lane_cap(8), lane_cap(9), lane_cap(16)),
            (8, 8, 16, 16)
        );
    }
}
