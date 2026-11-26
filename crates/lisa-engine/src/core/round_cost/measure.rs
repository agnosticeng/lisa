//! In-session measurement of verify-round cost at a given KV length.
use super::*;

/// Measure one model at one KV length: returns the median ms of a serial step,
/// a draft step, and a verify forward at each requested width `S`.
///
/// The KV state is built by prefilling `tokens` (the caller builds a synthetic
/// prompt of the target length), then timing over the live caches. Caches are
/// rolled back between verify passes, so the widths are comparable.
pub fn measure_at_kv<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    tokens: &[u32],
    widths: &[u32],
    steps: usize,
) -> anyhow::Result<(f64, f64, BTreeMap<u32, f64>, BTreeMap<u32, f64>)> {
    if tower.has_target_mtp() {
        return measure_target_mtp_at_kv(tower, tokens, widths, steps);
    }
    anyhow::ensure!(
        tower.has_drafter(),
        "round-cost measurement needs a model with an MTP drafter"
    );
    let warm: usize = 4;
    let mut sess = crate::conversation::Conversation::new(tower);
    let ctx_len = tower.context_window();

    // --- serial S=1 step ---
    let (logits, _multi) = sess.feed_multi(tower, tokens)?;
    let mut tok = logits.argmax_id()?;
    let mut serial: Vec<f64> = Vec::new();
    for i in 0..(steps + warm) {
        let t0 = std::time::Instant::now();
        let logits = sess.decode_token(tower, tok)?;
        let id = logits.argmax_id()?;
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= warm {
            serial.push(dt);
        }
        tok = id;
    }

    // --- one draft step ---
    // A fresh multi row from a capture forward (the [1,1] last row).
    let (_, m1) = {
        tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
        let arr = tower.runtime().tokens(&[tok as i32], &[1i32, 1])?;
        // NOTE: this forward advances the full cache by one row; the draft
        // timings below do not touch the trunk caches, so the offset drift is
        // irrelevant for the verify passes (they snapshot/restore around each
        // pass and never rely on absolute offsets).
        let (mixed, multi) = tower.forward_capture(&arr, Some(&mut sess.caches), true)?;
        let logits = tower.head(&mixed)?;
        let _ = logits.argmax_id()?;
        ((), multi)
    };
    let mut draft: Vec<f64> = Vec::new();
    {
        let mut tok_arr = tower.runtime().tokens(&[tok as i32], &[1i32, 1])?;
        let mut mul = m1.reshape(&[1, 1, m1.dim(-1)])?;
        for i in 0..(steps + warm) {
            let t0 = std::time::Instant::now();
            let (d, m) = tower.draft_step(&tok_arr, &mul)?;
            d.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if i >= warm {
                draft.push(dt);
            }
            tok_arr = d.reshape(&[1, 1])?;
            mul = m.reshape(&[1, 1, m.dim(-1)])?;
        }
    }

    // --- draft chain per depth, AS SERVED ---
    // The served round (session.rs) enqueues every chain step LAZY — the draft
    // id stays on device, the next step consumes it — and drains the whole
    // chain at the verify readback. Timing one synchronous `draft_step` (the
    // old draft cell) therefore prices enqueue-only work and under-bills the
    // chain several-fold at long KV. Here: depth steps device-fed, ONE eval of the chain tail, wall
    // over the live caches at the bucket's KV.
    let mut chain: BTreeMap<u32, f64> = BTreeMap::new();
    for &s in widths {
        let depth = (s.max(2) as usize).saturating_sub(1).min(6);
        let mut times: Vec<f64> = Vec::new();
        for i in 0..(steps + warm) {
            let mut d_arr = tower.runtime().tokens(&[tok as i32], &[1i32, 1])?;
            let mut m_arr = m1.reshape(&[1, 1, m1.dim(-1)])?;
            let t0 = std::time::Instant::now();
            for _ in 0..depth {
                let (d2, m2) = tower.draft_step(&d_arr, &m_arr)?;
                d_arr = d2.reshape(&[1, 1])?;
                m_arr = m2.reshape(&[1, 1, m2.dim(-1)])?;
            }
            // Drain like the round: one readback over the chain tail forces
            // the whole lazy graph.
            d_arr.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            m_arr.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if i >= warm {
                times.push(dt);
            }
        }
        chain.insert(depth as u32, median(&times));
    }

    // --- verify widths ---
    let mut verify: BTreeMap<u32, f64> = BTreeMap::new();
    for &s in widths {
        let s = s.max(2);
        let ids: Vec<i32> = std::iter::repeat_n(tok as i32, s as usize).collect();
        let arr = tower.runtime().tokens(&ids, &[1i32, s as i32])?;
        let mut times: Vec<f64> = Vec::new();
        for i in 0..(steps + warm) {
            // Snapshot the caches, verify wide, roll back to the snapshot.
            let snaps: Vec<_> = sess.caches.iter().map(R::snapshot_rollback).collect();
            tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
            // Measure the loaded target's production verification law; the
            // controller must not price an implicit arithmetic override.
            let t0 = std::time::Instant::now();
            let (mixed, _multi) = tower.forward_capture(&arr, Some(&mut sess.caches), true)?;
            let logits = tower.head_verified(&mixed)?;
            let top = logits.argmax_axis(-1).map_err(|e| anyhow::anyhow!("{e}"))?;
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            for (c, s) in sess.caches.iter_mut().zip(snaps.iter()) {
                R::restore_rollback(c, s)?;
            }
            if i >= warm {
                times.push(dt);
            }
        }
        verify.insert(s, median(&times));
    }
    Ok((median(&serial), median(&draft), chain, verify))
}

/// Measure GLM's target-conditioned MTP path. It has one proposal position, so
/// its only verify width is S=2; the proposal consumes a target multi-stream
/// row and a private indexer tape rather than an independent drafter cache.
pub(super) fn measure_target_mtp_at_kv<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    tokens: &[u32],
    widths: &[u32],
    steps: usize,
) -> anyhow::Result<(f64, f64, BTreeMap<u32, f64>, BTreeMap<u32, f64>)> {
    anyhow::ensure!(
        widths.iter().all(|&width| width == 2),
        "GLM target-MTP supports only verify width S=2"
    );
    anyhow::ensure!(steps > 0, "round-cost needs at least one timed step");

    const WARM: usize = 4;
    let mut sess = crate::conversation::Conversation::new(tower);
    let ctx_len = tower.context_window();
    let logits = sess.feed(tower, tokens)?;
    let mut tok = logits.argmax_id()?;

    // Price an ordinary serial decode step at this live KV length.
    let mut serial = Vec::with_capacity(steps);
    for i in 0..(steps + WARM) {
        let start = std::time::Instant::now();
        let logits = sess.decode_token(tower, tok)?;
        tok = logits.argmax_id()?;
        if i >= WARM {
            serial.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    // Seed the GLM MTP head exactly as the first target-MTP round does: feed
    // the pending target token, obtain its multi stream, and sample carry.
    let position = sess.fed.len();
    anyhow::ensure!(
        position < tower.max_position_embeddings(),
        "GLM target-MTP round-cost seed exceeds model context"
    );
    tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
    let seed_ids = tower.runtime().tokens(&[tok as i32], &[1i32, 1])?;
    let (seed_mixed, seed_multi) =
        tower.forward_capture(&seed_ids, Some(&mut sess.caches), false)?;
    let seed_logits = tower.head(&seed_mixed)?;
    let carry = seed_logits.argmax_id()?;
    sess.fed.push(tok);

    // A GLM proposal is one target-conditioned MTP step. Use a fresh private
    // tape for each sample so every timed call prices the first proposal at
    // the same absolute target position and starts from the same state.
    let mut draft = Vec::with_capacity(steps);
    for i in 0..(steps + WARM) {
        let mut cache = tower.runtime().new_draft_cache();
        let start = std::time::Instant::now();
        let proposal = tower.target_mtp_step(&mut cache, &seed_multi, carry, position, position)?;
        let _draft_id = proposal.argmax_id()?;
        if i >= WARM {
            draft.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }
    let draft_ms = median(&draft);

    // Price the exact two-row target verify path used by GLM generation. The
    // target caches are restored after each pass; the MTP tape is not read by
    // this forward and remains independent of the verify transaction.
    let mut verify = Vec::with_capacity(steps);
    let mut proposal_cache = tower.runtime().new_draft_cache();
    let proposal =
        tower.target_mtp_step(&mut proposal_cache, &seed_multi, carry, position, position)?;
    let draft_id = proposal.argmax_id()?;
    let verify_ids = tower
        .runtime()
        .tokens(&[carry as i32, draft_id as i32], &[1i32, 2])?;
    for i in 0..(steps + WARM) {
        let snapshots: Vec<_> = sess.caches.iter().map(R::snapshot_rollback).collect();
        tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
        let start = std::time::Instant::now();
        let (mixed, _) = tower.forward_capture(&verify_ids, Some(&mut sess.caches), false)?;
        let logits = tower.head(&mixed)?;
        let top = logits
            .argmax_axis(-1)
            .map_err(|error| anyhow::anyhow!("{error}"))?;
        top.eval().map_err(|error| anyhow::anyhow!("{error}"))?;
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        for (cache, snapshot) in sess.caches.iter_mut().zip(&snapshots) {
            R::restore_rollback(cache, snapshot)?;
        }
        if i >= WARM {
            verify.push(elapsed);
        }
    }

    let mut chain = BTreeMap::new();
    chain.insert(1, draft_ms);
    let mut verify_by_width = BTreeMap::new();
    verify_by_width.insert(2, median(&verify));
    Ok((median(&serial), draft_ms, chain, verify_by_width))
}

pub(super) fn sess_tail<R: SpeculativeRuntime>(
    sess: &crate::conversation::Conversation<R>,
    ctx_len: usize,
) -> Vec<i64> {
    sess.fed[sess.fed.len().saturating_sub(ctx_len)..]
        .iter()
        .map(|&t| t as i64)
        .collect()
}
