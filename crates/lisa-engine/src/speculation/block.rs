//! Block drafters: greedy tree rounds, each drafting a tree that one target step
//! verifies; sampled targets decode serially.
use super::*;

/// Block drafts use the target's ordinary Qwen sampling law, including the
/// absolute-position key and repetition history, for both serial and verify rows.
pub(super) fn draw_block_target_token<R: SpeculativeRuntime>(
    context: &R,
    sampler: Option<&mut R::Sampler>,
    logits: &R::Tensor,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return context.sampling_logits(logits)?.argmax_id();
    }
    match sampler {
        Some(sampler) => sampler.draw(context, logits, history),
        None => context.sampling_logits(logits)?.argmax_id(),
    }
}

impl<R: SpeculativeRuntime> Conversation<R> {
    /// Greedy generation in tree rounds: each round drafts a tree after the
    /// carry, verifies it in one target step and commits the path the
    /// target's picks walk. Sampled targets and models without tree
    /// verification decode serially. A fixed depth caps a round's rows at
    /// the carry and that many proposals.
    pub(super) fn generate_block_drafter(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        if !tower.supports_tree_verify()
            || stochastic
                .as_deref()
                .is_some_and(|s| !R::Sampler::can_use_raw_argmax(s))
        {
            return self.generate_block_drafter_serial(
                tower,
                tokens,
                max_tokens,
                after_prefill,
                stochastic,
                on_token,
            );
        }
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow before a tree round"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before a tree round: position {prompt_end}, limit {}",
            tower.max_position_embeddings()
        );
        let max_tokens = max_tokens.min(
            tower
                .max_position_embeddings()
                .saturating_sub(prompt_end)
                .saturating_add(1),
        );
        use super::tree;
        let most = match depth_sel {
            MtpDepth::Fixed(depth) => depth + 1,
            MtpDepth::Auto => tree::MOST_ROWS,
        }
        .min(crate::loading::tree_nodes())
        .clamp(tower.tree_min_rows(), tree::MOST_ROWS);
        let result = (|| -> anyhow::Result<Vec<u32>> {
            let logits = self.feed_for_generation(tower, tokens)?;
            if let Some(cb) = after_prefill.as_deref_mut() {
                cb(&self.caches);
            }
            let first = tower.runtime().sampling_logits(&logits)?.argmax_id()?;
            let mut generated = vec![first];
            on_token(first)?;
            let mut carry = first;
            // A request's first round takes a draft block's chances, as the
            // reference's lanes do: the last tree's belong to its request.
            if let Some(control) = tower.tree_control() {
                control.chances = None;
            }
            let ctx_len = tower.context_window();
            let mut tail: Vec<i64> = self.fed.iter().map(|&token| token as i64).collect();
            if ctx_len > 0 && tail.len() > ctx_len {
                tail.drain(0..tail.len() - ctx_len);
            }
            while generated.len() < max_tokens && !is_eos(carry) {
                if max_tokens - generated.len() < 2 {
                    // One token left: a plain step.
                    tower.set_context_tails(vec![tail.clone()]);
                    let logits = self.decode_token(tower, carry)?;
                    tail.push(i64::from(carry));
                    carry = tower.runtime().sampling_logits(&logits)?.argmax_id()?;
                    generated.push(carry);
                    on_token(carry)?;
                    continue;
                }
                let _span = tower.runtime().trace("block.tree");
                let mut control = std::mem::take(
                    tower
                        .tree_control()
                        .ok_or_else(|| anyhow::anyhow!("tree rounds need tree control"))?,
                );
                let result = (|| -> anyhow::Result<bool> {
                    let context = self.fed.len() + 1;
                    if !control.costs.measured() {
                        let _span = tower.runtime().trace("block.tree.measure");
                        control.costs.table =
                            tree_round_costs(tower, &mut self.caches, carry, most)?;
                        if tower.runtime().trace_enabled("block.round") {
                            eprintln!("[block.round] tree costs {:?}", control.costs.table);
                        }
                    }
                    let started = std::time::Instant::now();
                    // The deepest proposal a walk may retain.
                    let depth = (max_tokens - generated.len()).min(tree::MOST_ROWS) - 1;
                    let mut history = self.fed.clone();
                    history.push(carry);
                    let (chain, length) =
                        tree::lookup_chain(&history, depth.min(tree::CHAIN_TOKENS));
                    let (rates, chain_chances) = if chain.is_empty() {
                        (vec![0.0; tree::CHAIN_TOKENS * tree::AGREEMENTS], vec![])
                    } else {
                        control.lookup.rates(length, &chain)
                    };
                    let chances =
                        tree::lane_chances(control.chances.as_ref(), &chain_chances, depth);
                    let rows = control
                        .costs
                        .choose(&chances, context)
                        .rows
                        .clamp(tower.tree_min_rows(), most);
                    let plan = tree::TreePlan {
                        depth,
                        chain: chain.clone(),
                        rates,
                        children: tree::CHILDREN,
                        acceptance: control.budget.acceptance,
                        rows,
                        lane_cap: if rows > tree::NARROW_CAP {
                            tree::MOST_ROWS
                        } else {
                            tree::NARROW_CAP
                        },
                        committing: true,
                    };
                    let draft_span = tower.runtime().trace("block.tree.draft");
                    let grown = tower.draft_tree(carry, &plan, &mut self.caches)?;
                    drop(draft_span);
                    tower.set_context_tails(vec![tail.clone()]);
                    let verify_span = tower.runtime().trace("block.tree.verify");
                    let verified = tower.verify_tree(&grown.ids, &grown.tree, &mut self.caches)?;
                    let logits = tower.head_verified(&verified)?;
                    let picks = tower
                        .runtime()
                        .sampling_logits(&logits)?
                        .argmax_axis(-1)?
                        .reshape(&[-1])?
                        .read_raw_ids()?;
                    drop(verify_span);
                    // Everything the round reads, at its one wait: the
                    // commit encoded below must not wait again.
                    let learned = read_grown_tree::<R>(&grown)?;
                    let tokens = grown.ids.read_raw_ids()?;
                    let parents = grown
                        .tree
                        .reshape(&[-1])?
                        .read_raw_ids()?
                        .into_iter()
                        .take(rows)
                        .map(|p| p as i32)
                        .collect::<Vec<_>>();
                    // Walk down while the target's pick at a row is one of
                    // its children; every visited row emits its pick.
                    let mut path = vec![0usize];
                    let mut emitted = vec![];
                    let mut eos_seen = false;
                    // The round's cost sample ends at its results, as the
                    // reference's: the caller's handling of its tokens is no
                    // part of it.
                    let mut emitting = std::time::Duration::ZERO;
                    loop {
                        let node = *path.last().unwrap();
                        let token = picks[node];
                        emitted.push(token);
                        generated.push(token);
                        let handed = std::time::Instant::now();
                        on_token(token)?;
                        emitting += handed.elapsed();
                        if is_eos(token) {
                            eos_seen = true;
                            break;
                        }
                        if generated.len() >= max_tokens {
                            break;
                        }
                        match (node + 1..rows)
                            .find(|&c| parents[c] == node as i32 && tokens[c] == token)
                        {
                            Some(child) => path.push(child),
                            None => break,
                        }
                    }
                    if tower.runtime().trace_enabled("block.round") {
                        eprintln!(
                            "[block.round] tree rows={rows} chain={} emitted={} path={path:?}",
                            chain.len(),
                            emitted.len(),
                        );
                    }
                    let commit_span = tower.runtime().trace("block.tree.commit");
                    tower.commit_tree(&mut self.caches, &path)?;
                    drop(commit_span);
                    let elapsed = started.elapsed().saturating_sub(emitting);
                    // Learn from the grown tree.
                    if !eos_seen {
                        if !chain.is_empty() {
                            let (verified, accepted) =
                                tree::lookup_outcome(&chain, &learned, rows, &emitted);
                            if !verified.is_empty() {
                                control.lookup.observe(length, &verified, accepted);
                            }
                        }
                        control
                            .budget
                            .observe(&tree::tree_labels(&learned, &emitted));
                        if let Some(chances) = tree::Chances::of(&learned, &chain) {
                            control.chances = Some(chances);
                        }
                        control
                            .costs
                            .observe(rows, context, elapsed.as_secs_f64() * 1e3);
                    }
                    let committed = path.iter().map(|&n| tokens[n]).collect::<Vec<_>>();
                    self.fed.extend(&committed);
                    let last = *path.last().unwrap() as i32;
                    self.last_mixed = Some(verified.select(1, last)?.contiguous()?);
                    carry = *emitted.last().expect("a tree round emits a token");
                    tail.extend(committed.iter().map(|&t| t as i64));
                    if ctx_len > 0 && tail.len() > ctx_len {
                        tail.drain(0..tail.len() - ctx_len);
                    }
                    Ok(eos_seen)
                })();
                if let Some(slot) = tower.tree_control() {
                    *slot = control;
                }
                if result? || generated.len() >= max_tokens {
                    break;
                }
            }
            tower.set_context_tails(vec![
                self.fed[self.fed.len().saturating_sub(ctx_len)..]
                    .iter()
                    .map(|&token| token as i64)
                    .collect(),
            ]);
            Ok(generated)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    pub(super) fn generate_block_drafter_serial(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow before serial decode"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before serial decode: position {prompt_end}, limit {}",
            tower.max_position_embeddings()
        );
        let max_tokens = max_tokens.min(
            tower
                .max_position_embeddings()
                .saturating_sub(prompt_end)
                .saturating_add(1),
        );
        let result = (|| -> anyhow::Result<Vec<u32>> {
            let mut logits = self.feed_for_generation(tower, tokens)?;
            if let Some(cb) = after_prefill.as_deref_mut() {
                cb(&self.caches);
            }
            let mut mode = TargetMtpSamplingMode::Configured;
            let mut token = draw_serial_target_token(
                tower,
                stochastic.as_deref_mut(),
                &logits,
                &self.fed,
                mode,
            )?;
            let mut output = vec![token];
            mode = on_token(token)?;
            while output.len() < max_tokens && !is_eos(token) {
                logits = self.decode_token(tower, token)?;
                token = draw_serial_target_token(
                    tower,
                    stochastic.as_deref_mut(),
                    &logits,
                    &self.fed,
                    mode,
                )?;
                output.push(token);
                mode = on_token(token)?;
            }
            Ok(output)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}

/// Replays a greedy tree round takes per verified row count when measuring
/// round costs; the median counts.
const COST_REPLAYS: usize = 3;

/// Milliseconds of a tree round (draft, tree, verification, head and
/// readback) at each verified row count 1..=16, measured on the current
/// state without committing; counts past `most` and the single row take
/// their nearest measured neighbour. Tables persist per machine and model
/// next to the MTP round costs.
fn tree_round_costs<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    caches: &mut Vec<R::Cache>,
    anchor: u32,
    most: usize,
) -> anyhow::Result<Vec<f64>> {
    use super::tree;
    let key = format!(
        "{}|{}|{most}",
        crate::core::round_cost::host_chip(),
        tower.mtp_cost_key()
    );
    let path = crate::core::round_cost::default_path().with_file_name("tree_cost.json");
    let mut store: std::collections::BTreeMap<String, Vec<f64>> = std::fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_default();
    // A stored table stands while a level check at its widest row count
    // stays within a fifth of it.
    if let Some(table) = store.get(&key).filter(|t| t.len() == tree::MOST_ROWS) {
        let mut level = time_tree_rounds(tower, caches, anchor, most, COST_REPLAYS + 1)?;
        level.remove(0);
        level.sort_by(f64::total_cmp);
        let ratio = table[most - 1] / level[level.len() / 2];
        if (1.0 / (1.0 + COST_STALE)..=1.0 + COST_STALE).contains(&ratio) {
            return Ok(table.clone());
        }
    }
    let table = measure_tree_costs(tower, caches, anchor, most)?;
    store.insert(key, table.clone());
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&path, serde_json::to_string(&store)?);
    Ok(table)
}

/// A stored cost table is stale past this relative level change.
const COST_STALE: f64 = 0.2;

/// Wall milliseconds of `replays` uncommitted tree rounds of `rows` rows.
fn time_tree_rounds<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    caches: &mut Vec<R::Cache>,
    anchor: u32,
    rows: usize,
    replays: usize,
) -> anyhow::Result<Vec<f64>> {
    use super::tree;
    let plan = tree::TreePlan {
        depth: tree::MOST_ROWS - 1,
        chain: vec![],
        rates: vec![0.0; tree::CHAIN_TOKENS * tree::AGREEMENTS],
        children: tree::CHILDREN,
        acceptance: tree::ACCEPTANCE_PRIOR,
        rows,
        lane_cap: if rows > tree::NARROW_CAP {
            tree::MOST_ROWS
        } else {
            tree::NARROW_CAP
        },
        committing: false,
    };
    let mut times = vec![];
    for _ in 0..replays {
        let started = std::time::Instant::now();
        let grown = tower.draft_tree(anchor, &plan, caches)?;
        let verified = tower.verify_tree(&grown.ids, &grown.tree, caches)?;
        let logits = tower.head_verified(&verified)?;
        tower
            .runtime()
            .sampling_logits(&logits)?
            .argmax_axis(-1)?
            .reshape(&[-1])?
            .read_raw_ids()?;
        times.push(started.elapsed().as_secs_f64() * 1e3);
        tower.discard_tree(caches)?;
    }
    Ok(times)
}

/// Measure every row count's round cost: [`tree_round_costs`] without the
/// stored table.
fn measure_tree_costs<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    caches: &mut Vec<R::Cache>,
    anchor: u32,
    most: usize,
) -> anyhow::Result<Vec<f64>> {
    use super::tree;
    let mut samples = vec![vec![]; tree::MOST_ROWS + 1];
    // A first pass prepares each row count's plans; the timed passes each
    // start at another row count.
    for replay in 0..=COST_REPLAYS {
        let counts = (tower.tree_min_rows()..=most).collect::<Vec<_>>();
        let shift = replay * counts.len() / (COST_REPLAYS + 1);
        for &rows in counts[shift..].iter().chain(&counts[..shift]) {
            let elapsed = time_tree_rounds(tower, caches, anchor, rows, 1)?[0];
            if replay > 0 {
                samples[rows].push(elapsed);
            }
        }
    }
    let median = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(f64::total_cmp);
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    };
    let least = tower.tree_min_rows();
    let mut table = vec![0.0; tree::MOST_ROWS];
    for rows in least..=most {
        table[rows - 1] = median(&mut samples[rows]);
    }
    // Row counts the model does not verify cost as its fewest.
    for rows in 1..least {
        table[rows - 1] = table[least - 1];
    }
    for rows in most + 1..=tree::MOST_ROWS {
        table[rows - 1] = table[most - 1];
    }
    Ok(table)
}

/// The grown tree of a round, read back.
fn read_grown_tree<R: SpeculativeRuntime>(
    grown: &super::tree::DeviceTree<R::Tensor>,
) -> anyhow::Result<super::tree::GrownTree> {
    let size = grown.size.read_raw_ids()?[0] as usize;
    let words = |t: &R::Tensor| -> anyhow::Result<Vec<u32>> { t.reshape(&[-1])?.read_raw_ids() };
    let floats = |t: &R::Tensor| -> anyhow::Result<Vec<f32>> { t.reshape(&[-1])?.read_f32() };
    let tokens = words(&grown.tokens)?;
    let rows = words(&grown.rows)?;
    let scores = floats(&grown.scores)?;
    let caps = floats(&grown.caps)?;
    let size = size.min(tokens.len());
    Ok(super::tree::GrownTree {
        tokens: tokens[..size].to_vec(),
        parents: (0..size).map(|r| rows[2 * r + 1] & 0xffff).collect(),
        q: (0..size).map(|r| scores[2 * r]).collect(),
        estimate: (0..size).map(|r| scores[2 * r + 1]).collect(),
        caps: caps[..size].to_vec(),
        candidates: words(&grown.candidates)?,
    })
}
