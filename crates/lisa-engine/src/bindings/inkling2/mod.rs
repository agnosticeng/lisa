//! Inkling 2 Mini on the pinned reference's step (the `step` module): its
//! prefill runs prompts, its decode step every generated token and verified
//! tree row. A sequence's state (paged INT8 keys and values, convolution
//! state, the draft's context rings) lives in its cache, so prefix
//! snapshots, sessions and restores need no relayout; both paths read and
//! write that one state.
//!
//! Prompts prefill as the reference's chunks of up to 2048 rows from the
//! committed position; the draft's context takes the rows of the prompt's
//! last 2048 positions through its own prefill. The first token reads the
//! last row's head. Greedy generation runs tree rounds drafted by the
//! DFlash2 draft step; serial and sampled decoding run one-row chains of
//! the decode step. Kernel selection follows the reference's calibration of
//! the device's GPU family and cores. Image rows replace token rows after
//! the embedding norm.
pub(crate) mod step;

use super::BindingDevice;
use crate::{
    cache::{LayerCache, NativeCache, NativeState},
    language::{LanguageModel, runtime::Runtime},
    sampling::sampler::Sampler,
    tensor_ops::{TensorLaw, TensorOps},
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{
    Kernel, ManagedDevice, PackedCompute as P, ReadbackDevice, ScalarType as Dtype, TensorAccess,
    TensorOperation as T,
};
use lisa_models::inkling2::{config::Config, planes::Format};
use std::{collections::BTreeMap, sync::Arc};
use step::{
    calibration::Calibration,
    draft::DraftStep,
    kernels::{self as k, LaneRows, TreeRow, pack},
    schedule::{HIDDEN, TOKENS},
    state::{MOST_TOKENS, StepState},
    target::{Injected, PROMPT_ROWS, Step},
};

/// Rows of a prompt chunk: a verify block's.
const CHUNK: usize = k::VERIFY_ROWS as usize;
/// Tokens an extent holds beyond the prompt when first allocated.
const HEADROOM: usize = 2048;
/// Image placeholder and end-of-sequence tokens.
const IMAGE_TOKEN: u32 = 200054;
const EOS_TOKEN: i64 = 200006;

/// Loader-produced inputs; execution has no checkpoint parser or path.
pub(crate) struct Prepared<D: BindingDevice + ManagedDevice> {
    pub config: Config,
    pub weights: crate::weights::source::PreparedTensors<D::Tensor, D::Reservation>,
    pub formats: BTreeMap<String, Format>,
    pub vision: Box<dyn crate::weights::source::ImageFeatures<D::Tensor>>,
    /// The DFlash2 draft's tensors and planes, when downloaded.
    pub draft: Option<BTreeMap<String, D::Tensor>>,
}

/// A sequence's cache: the step's state once a prompt has run.
struct State<D: BindingDevice + ManagedDevice> {
    device: D,
    step: Option<StepState<D::Tensor>>,
}
impl<D: BindingDevice + ManagedDevice + 'static> NativeState for State<D> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn copy_for_write(&self) -> Result<Arc<dyn NativeState>> {
        Ok(Arc::new(Self {
            device: self.device.clone(),
            step: self
                .step
                .as_ref()
                .map(|s| s.snapshot(&self.device))
                .transpose()?,
        }))
    }
    fn position(&self) -> usize {
        self.step.as_ref().map_or(0, |s| s.committed)
    }
}

/// The image rows of the current request: bf16 `[n, hidden]` and runs of
/// (first position, first row, rows).
struct Images<T> {
    rows: T,
    runs: Vec<(usize, usize, usize)>,
}

/// The last tree verification, until its path is committed: its rows, lane
/// rows and, when the device already accepted and committed its path, the
/// path's row mask.
struct Pending<T> {
    rows: usize,
    lane: T,
    accepted: Option<T>,
}

pub struct Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    runtime: Runtime<D, Sampler<D>>,
    config: Config,
    capacity: usize,
    /// Holds the weights and the draft resident while the model is loaded;
    /// declared before them so it drops first.
    _resident: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Keeps the resident weights the steps bind alive.
    _weights: crate::weights::source::PreparedTensors<D::Tensor, D::Reservation>,
    step: Step<D::Tensor>,
    draft: Option<DraftStep<D::Tensor>>,
    vision: Box<dyn crate::weights::source::ImageFeatures<D::Tensor>>,
    images: Option<Images<D::Tensor>>,
    law: u64,
    warmed: bool,
    tree_control: crate::speculation::tree::TreeControl,
    /// The last drafted tree's verify rows as the step's tree rows and lane
    /// rows.
    tree_rows: Option<(D::Tensor, D::Tensor)>,
    /// Output a committing round has left: its verification accepts and
    /// commits on the device.
    accept: Option<usize>,
    pending: Option<Pending<D::Tensor>>,
}

/// The cache's state.
fn state<D: BindingDevice + ManagedDevice + 'static>(
    caches: &[LayerCache<D>],
) -> Result<&State<D>> {
    match caches {
        [LayerCache::Native(cache)] => cache.get::<State<D>>(),
        _ => anyhow::bail!("Inkling requires its native cache"),
    }
}

/// The cache's state, made exclusive (snapshots keep theirs).
fn state_mut<D: BindingDevice + ManagedDevice + 'static>(
    caches: &mut [LayerCache<D>],
) -> Result<&mut State<D>> {
    match caches {
        [LayerCache::Native(cache)] => cache.get_mut::<State<D>>(),
        _ => anyhow::bail!("Inkling requires its native cache"),
    }
}

fn words(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

impl<D: BindingDevice + ManagedDevice + ReadbackDevice + 'static> Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    pub(crate) fn from_prepared(
        runtime: Runtime<D, Sampler<D>>,
        prepared: Prepared<D>,
    ) -> Result<Self> {
        let Prepared {
            config,
            weights,
            formats,
            vision,
            draft,
        } = prepared;
        let device = runtime.context.device().clone();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        // The reference's row of the device: its GPU family and cores (an
        // unreported count plans for its assumed one).
        let family = device
            .apple_gpu_family()
            .context("the Inkling step needs the GPU's Apple family")?;
        let calibration = Calibration::of(family, device.gpu_cores().unwrap_or(0));
        // The weights and the draft stay wired between requests, as the
        // reference's residency heartbeat keeps them.
        let held = weights
            .tensors
            .values()
            .chain(draft.iter().flat_map(|d| d.values()))
            .collect::<Vec<_>>();
        let resident = device.hold_resident(&held);
        let step = Step::new(&ops, &weights.tensors, &formats, calibration)?;
        let draft = draft
            .map(|d| DraftStep::new(&ops, &d, &weights.tensors, calibration))
            .transpose()?;
        Ok(Self {
            runtime,
            capacity: config.context.min(MOST_TOKENS),
            config,
            _weights: weights,
            _resident: resident,
            step,
            draft,
            vision,
            images: None,
            // The step's numerical law.
            law: 0x494e_4b32_5354_0000,
            warmed: false,
            tree_control: Default::default(),
            tree_rows: None,
            accept: None,
            pending: None,
        })
    }

    fn device(&self) -> D {
        self.runtime.context.device().clone()
    }

    fn draft_layers(&self) -> usize {
        self.draft
            .as_ref()
            .map_or(0, |_| lisa_models::inkling2::dflash::LAYERS)
    }

    /// The step state of `state`, allocated or grown to hold `tokens`.
    fn reserve<'a>(
        &self,
        state: &'a mut State<D>,
        tokens: usize,
    ) -> Result<&'a mut StepState<D::Tensor>> {
        ensure!(
            tokens <= self.capacity,
            "Inkling context capacity {} exceeded",
            self.capacity
        );
        let device = self.device();
        match &mut state.step {
            Some(s) => s.reserve(&device, tokens)?,
            None => {
                let first = (tokens + HEADROOM).min(self.capacity);
                state.step = Some(StepState::new(&device, first, self.draft_layers())?);
            }
        }
        Ok(state.step.as_mut().unwrap())
    }

    /// Image runs within the positions `[first, first + rows)`, as step rows.
    fn injected_runs(&self, first: usize, rows: usize) -> Vec<(u32, u32, u32)> {
        let Some(images) = &self.images else {
            return vec![];
        };
        images
            .runs
            .iter()
            .filter_map(|&(position, source, count)| {
                let start = position.max(first);
                let end = (position + count).min(first + rows);
                (start < end).then(|| {
                    (
                        (start - first) as u32,
                        (source + start - position) as u32,
                        (end - start) as u32,
                    )
                })
            })
            .collect()
    }

    /// Run `tokens` (1 to 16) as one chain committed whole, the draft's
    /// context with them. Returns their logits F32 `[rows, 200058]`, a view
    /// of the step's scratch that the next step overwrites.
    fn chain(&self, state: &mut StepState<D::Tensor>, tokens: &[u32]) -> Result<D::Tensor> {
        let rows = tokens.len();
        ensure!((1..=CHUNK).contains(&rows), "invalid Inkling chain");
        ensure!(
            tokens.iter().all(|&t| t < TOKENS),
            "Inkling token outside vocabulary"
        );
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let ids = ops.pull(&words(tokens), &[rows as i32], Dtype::U32)?;
        let tree = ops.pull(
            &pack(&TreeRow::chain(rows as u32)?),
            &[(2 * rows) as i32],
            Dtype::U32,
        )?;
        let lane = ops.pull(
            &pack(&[LaneRows {
                first: 0,
                rows: rows as u32,
            }]),
            &[2],
            Dtype::U32,
        )?;
        let all = ops.pull(&words(&[((1u64 << rows) - 1) as u32]), &[1], Dtype::U32)?;
        let first = state.committed;
        let runs = self.injected_runs(first, rows);
        let injected = self
            .images
            .as_ref()
            .filter(|_| !runs.is_empty())
            .map(|i| Injected {
                rows: &i.rows,
                runs: &runs,
            });
        let submit = || device.submit(false);
        let outputs = self
            .step
            .forward(&ops, state, &ids, &tree, &lane, rows, injected, &submit)?;
        self.step.encode_commit(&ops, state, &all, &lane, rows)?;
        self.step.advance(state, rows);
        if let Some(draft) = &self.draft {
            draft.commit(
                &ops,
                &state.rings,
                self.step.captures(),
                rows,
                &all,
                &lane,
                first,
            )?;
        }
        outputs
            .logits
            .slice(0, 0, Some(rows as i32))?
            .slice(1, 0, Some(TOKENS as i32))
    }

    /// Prefill `tokens` from the cache's position in the reference's chunks
    /// of up to [`PROMPT_ROWS`] rows, the draft's context taking the rows of
    /// the last window's positions. Returns the logits of every row (`all`)
    /// or of the last, `[rows, 200058]`.
    fn feed(
        &mut self,
        tokens: &[u32],
        caches: &mut [LayerCache<D>],
        all: bool,
    ) -> Result<D::Tensor> {
        ensure!(!tokens.is_empty(), "empty Inkling 2 prompt");
        ensure!(
            tokens.iter().all(|&t| t < TOKENS),
            "Inkling token outside vocabulary"
        );
        self.pending = None;
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let position = state(caches)?.position();
        let state = state_mut(caches)?;
        // The rings hold the last window's positions: rows before them are
        // never read.
        let window = (position + tokens.len()).saturating_sub(k::DRAFT_WINDOW as usize);
        let mut rows = vec![];
        let mut first = 0;
        while first < tokens.len() {
            let count = (tokens.len() - first).min(PROMPT_ROWS as usize);
            let at = position + first;
            let s = self.reserve(state, at + count)?;
            let ids = ops.pull(
                &words(&tokens[first..first + count]),
                &[count as i32],
                Dtype::U32,
            )?;
            let runs = self.injected_runs(at, count);
            let injected = self
                .images
                .as_ref()
                .filter(|_| !runs.is_empty())
                .map(|i| Injected {
                    rows: &i.rows,
                    runs: &runs,
                });
            let captured = window.saturating_sub(at).min(count);
            let submit = || device.submit(false);
            self.step
                .prefill(&ops, s, &ids, count, captured, injected, &submit)?;
            if let Some(draft) = &mut self.draft
                && captured < count
            {
                draft.prefill(
                    &ops,
                    &s.rings,
                    self.step.prompt_captures()?,
                    count - captured,
                    at + captured,
                )?;
            }
            first += count;
            let heads = if all {
                (0..count).step_by(CHUNK).collect::<Vec<_>>()
            } else if first == tokens.len() {
                vec![count - 1]
            } else {
                vec![]
            };
            for row in heads {
                let n = if all { CHUNK.min(count - row) } else { 1 };
                let logits = self
                    .step
                    .prompt_logits(&ops, row as u32, n as u32)?
                    .slice(1, 0, Some(TOKENS as i32))?
                    .contiguous()?;
                device.evaluate(&logits, false)?;
                rows.push(logits);
            }
            device.submit(false)?;
        }
        D::Tensor::concatenate(&rows.iter().collect::<Vec<_>>(), 0)
    }

    /// One generated token through the decode step, committed: its logits
    /// `[1, 200058]`.
    fn decode(&mut self, token: u32, caches: &mut [LayerCache<D>]) -> Result<D::Tensor> {
        self.pending = None;
        let device = self.device();
        let position = state(caches)?.position();
        let s = self.reserve(state_mut(caches)?, position + 1)?;
        let logits = self.chain(s, &[token])?.contiguous()?;
        device.evaluate(&logits, false)?;
        device.submit(false)?;
        Ok(logits)
    }

    /// The reference cycle's tail, encoded before any readback: greedy
    /// selections, the accepted path (`decode_accept_tree`), and the
    /// target's and the draft's commits of it. The host's walk of the same
    /// selections reaches the same path; its commit then only advances
    /// counters.
    fn accept_commit(
        &mut self,
        state: &StepState<D::Tensor>,
        ids: &D::Tensor,
        tree: &D::Tensor,
        logits: &D::Tensor,
        remaining: usize,
    ) -> Result<()> {
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let pending = self.pending.as_mut().context("no Inkling verification")?;
        let rows = pending.rows;
        let flat = logits.reshape(&[rows as i32, TOKENS as i32])?;
        let picks = ops.op(
            T::Argmax {
                axis: -1,
                keep_dims: false,
            },
            &[&flat],
        )?;
        let picks = ops.op(T::Cast(Dtype::U32), &[&picks])?;
        let picks = ops.op(T::Contiguous, &[&picks])?;
        let zeros = |n| {
            ops.required(
                Kernel::Zeros {
                    shape: vec![n],
                    dtype: Dtype::U32,
                },
                &[],
            )
        };
        let (retained, accepted, retained_rows) = (zeros(1)?, zeros(1)?, zeros(1)?);
        let mut params = k::AcceptBatchParams::default();
        params.remaining[0] = remaining as u32;
        for (slot, id) in params.stop_tokens.iter_mut().zip(crate::eos::eos_ids()) {
            *slot = id;
        }
        ops.required(
            Kernel::PackedCompute(P::Vendored(Box::new(k::decode_accept_tree(&params, 1)?))),
            &[
                ids,
                tree,
                &picks,
                &retained,
                &accepted,
                &retained_rows,
                &pending.lane,
            ],
        )?;
        self.step
            .encode_commit(&ops, state, &retained_rows, &pending.lane, rows)?;
        if let Some(draft) = &self.draft {
            draft.commit(
                &ops,
                &state.rings,
                self.step.captures(),
                rows,
                &retained_rows,
                &pending.lane,
                state.committed,
            )?;
        }
        pending.accepted = Some(retained_rows);
        Ok(())
    }
}

impl<D: BindingDevice + ManagedDevice + ReadbackDevice + 'static>
    LanguageModel<Runtime<D, Sampler<D>>> for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn supports_images(&self) -> bool {
        self.vision.available()
    }
    fn clear_images(&mut self) {
        self.images = None;
    }
    fn prepare_images(&mut self, tokens: &mut Vec<u32>, images: &[image::RgbImage]) -> Result<()> {
        self.images = None;
        ensure!(
            !images.is_empty() && images.len() <= crate::vision::MAX_IMAGES,
            "invalid Inkling image count"
        );
        ensure!(
            tokens.iter().filter(|&&t| t == IMAGE_TOKEN).count() == images.len(),
            "Inkling image marker count mismatch"
        );
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let mut expanded = Vec::new();
        let mut features = vec![];
        let mut runs = vec![];
        let mut source = 0;
        for &token in tokens.iter() {
            if token != IMAGE_TOKEN {
                expanded.push(token);
                continue;
            }
            let rows = self.vision.encode(&images[features.len()])?;
            let count = rows.dim(0) as usize;
            ensure!(
                expanded.len() + count <= self.capacity,
                "image prompt exceeds Inkling context"
            );
            runs.push((expanded.len(), source, count));
            expanded.extend(std::iter::repeat_n(token, count));
            source += count;
            features.push(rows);
        }
        // The step's rows are bf16: image rows round once, as token rows do
        // after the embedding norm.
        let rows = D::Tensor::concatenate(&features.iter().collect::<Vec<_>>(), 0)?;
        ensure!(rows.dim(-1) == HIDDEN as i32, "invalid Inkling image rows");
        let rows = ops.op(T::Cast(Dtype::BF16), &[&rows])?;
        let rows = ops.op(T::Contiguous, &[&rows])?;
        self.images = Some(Images { rows, runs });
        *tokens = expanded;
        Ok(())
    }
    /// A prompt split at a chunk's end prefills its rest as the whole
    /// prompt's chunks do.
    fn prefix_cache_granularity(&self) -> Option<usize> {
        self.images.is_none().then_some(PROMPT_ROWS as usize)
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    /// The step's pages bound the capacity below the checkpoint's context.
    fn set_context_capacity(&mut self, n: usize) -> Result<()> {
        ensure!(
            n > 0 && n <= self.config.context,
            "invalid Inkling 2 context capacity"
        );
        self.capacity = n.min(MOST_TOKENS);
        Ok(())
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        self.images.is_none().then_some(self.law)
    }
    fn mtp_cost_key(&self) -> String {
        format!("inkling2-{:016x}", self.law)
    }
    fn eos_token_id(&self) -> i64 {
        EOS_TOKEN
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        vec![LayerCache::Native(NativeCache::new(State::<D> {
            device: self.device(),
            step: None,
        }))]
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn max_speculative_depth(&self) -> usize {
        if self.draft.is_some() { CHUNK - 1 } else { 0 }
    }
    fn has_drafter(&self) -> bool {
        self.draft.is_some()
    }
    fn has_block_drafter(&self) -> bool {
        self.draft.is_some()
    }
    fn supports_tree_verify(&self) -> bool {
        self.draft.is_some() && self.images.is_none()
    }
    fn tree_min_rows(&self) -> usize {
        1
    }
    fn tree_control(&mut self) -> Option<&mut crate::speculation::tree::TreeControl> {
        self.supports_tree_verify()
            .then_some(&mut self.tree_control)
    }
    fn draft_tree(
        &mut self,
        anchor: u32,
        plan: &crate::speculation::tree::TreePlan,
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<crate::speculation::tree::DeviceTree<D::Tensor>> {
        use crate::speculation::tree::{AGREEMENTS, CHAIN_TOKENS};
        ensure!(
            plan.chain.len() <= CHAIN_TOKENS && plan.rates.len() == CHAIN_TOKENS * AGREEMENTS,
            "invalid draft tree plan"
        );
        let draft = self.draft.as_ref().context("tree drafting needs DFlash2")?;
        let state = state(caches)?
            .step
            .as_ref()
            .context("tree drafting needs a prefilled cache")?;
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        // The draft selector's output is the lattice the tree reads, in its
        // own order.
        let s = draft.forward(&ops, &state.rings, anchor, state.committed)?;
        let context = &self.runtime.context;
        let mut chain = plan.chain.iter().map(|&t| t as i32).collect::<Vec<_>>();
        chain.resize(CHAIN_TOKENS, 0);
        let chain =
            crate::bindings::shared::tensor_io::pull_i32(context, &chain, &[CHAIN_TOKENS as i32])?;
        let rates = crate::bindings::shared::tensor_io::pull_f32(
            context,
            &plan.rates,
            &[CHAIN_TOKENS as i32, AGREEMENTS as i32],
        )?;
        let mut model = [0u32; 10];
        model[0] = 30f32.to_bits();
        model[1] = plan.acceptance.slope.to_bits();
        for (m, b) in model[2..9].iter_mut().zip(plan.acceptance.intercepts) {
            *m = b.to_bits();
        }
        model[9] = 0f32.to_bits();
        let kernel = Kernel::PackedCompute(P::DraftTree {
            anchor,
            depth: plan.depth as u32,
            chain: plan.chain.len().min(plan.depth) as u32,
            children: plan.children as u32,
            rows: plan.rows as u32,
            lane_cap: plan.lane_cap as u32,
            model,
            selected: true,
        });
        let handle = device
            .prepare(kernel)?
            .context("device has no draft tree selection")?;
        let outputs = device
            .dispatch(
                &handle,
                &[&s.candidates, &s.unary, &s.partial_values, &chain, &rates],
                &device.stream(),
            )?
            .context("device declined draft tree selection")?
            .output;
        let lisa_backend::Outputs::Many(outputs) = outputs else {
            anyhow::bail!("invalid draft tree outputs")
        };
        let [
            ids,
            tree,
            _sources,
            tokens,
            rows,
            scores,
            caps,
            size,
            candidates,
            verify_tree,
            lane_rows,
        ]: [D::Tensor; 11] = outputs
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid draft tree outputs"))?;
        self.tree_rows = Some((verify_tree, lane_rows));
        // A committing round accepts and commits on the device right behind
        // its verification, at most the output it has left.
        self.accept = plan.committing.then_some(plan.depth + 1);
        Ok(crate::speculation::tree::DeviceTree {
            ids,
            tree,
            tokens,
            rows,
            scores,
            caps,
            size,
            candidates,
        })
    }
    /// Softcapped logits `[1, rows, 200058]` of the drafted tree's rows.
    fn verify_tree(
        &mut self,
        ids: &D::Tensor,
        _tree: &D::Tensor,
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<D::Tensor> {
        let rows = ids.size();
        let (tree, lane) = self
            .tree_rows
            .take()
            .context("verification needs the drafted tree rows")?;
        self.pending = None;
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let committed = state(caches)?.position();
        ensure!(committed > 0, "tree verification needs a prefilled cache");
        let state = self.reserve(state_mut(caches)?, committed + rows)?;
        let ids = ids.reshape(&[rows as i32])?;
        let tree = tree.reshape(&[(2 * rows) as i32])?;
        // The draft runs while the step is encoded.
        device.submit(false)?;
        let submit = || device.submit(false);
        let outputs = self
            .step
            .forward(&ops, state, &ids, &tree, &lane, rows, None, &submit)?;
        let logits = outputs
            .logits
            .slice(0, 0, Some(rows as i32))?
            .slice(1, 0, Some(TOKENS as i32))?
            .contiguous()?
            .reshape(&[1, rows as i32, TOKENS as i32])?;
        device.evaluate(&logits, false)?;
        self.pending = Some(Pending {
            rows,
            lane,
            accepted: None,
        });
        if let Some(remaining) = self.accept.take() {
            self.accept_commit(state, &ids, &tree, &logits, remaining)?;
        }
        Ok(logits)
    }
    fn commit_tree(&mut self, caches: &mut Vec<LayerCache<D>>, path: &[usize]) -> Result<()> {
        let pending = self
            .pending
            .take()
            .context("no Inkling verification to commit")?;
        ensure!(
            !path.is_empty()
                && path[0] == 0
                && path.iter().all(|&n| n < pending.rows)
                && path.windows(2).all(|w| w[0] < w[1]),
            "invalid accepted tree path"
        );
        let mask = path.iter().fold(0u32, |m, &n| m | (1 << n));
        let state = state_mut(caches)?
            .step
            .as_mut()
            .context("missing Inkling step state")?;
        if let Some(accepted) = pending.accepted {
            debug_assert!(accepted.read_raw_ids().is_ok_and(|m| m[0] == mask));
            self.step.advance(state, path.len());
            return Ok(());
        }
        let device = self.device();
        let ops = TensorOps::new(&device, TensorLaw::Native);
        let retained = ops.pull(&words(&[mask]), &[1], Dtype::U32)?;
        let first = state.committed;
        self.step.commit(
            &ops,
            state,
            &retained,
            &pending.lane,
            path.len(),
            pending.rows,
        )?;
        if let Some(draft) = &self.draft {
            draft.commit(
                &ops,
                &state.rings,
                self.step.captures(),
                pending.rows,
                &retained,
                &pending.lane,
                first,
            )?;
        }
        Ok(())
    }
    fn discard_tree(&mut self, _caches: &mut Vec<LayerCache<D>>) -> Result<()> {
        // Rows past the committed tokens are scratch the next step
        // overwrites.
        self.pending = None;
        self.accept = None;
        Ok(())
    }
    fn kv_bytes_per_token(&self) -> usize {
        // INT8 rows of 8 heads with one F32 scale each, keys and values, in
        // every application's region.
        step::state::APPLICATIONS * 2 * 8 * (128 + 4)
    }
    fn stream_state_bytes(&self) -> usize {
        2 * 3 * step::state::STATE_COLUMNS * 2
            + self.draft_layers() * 2 * k::draft_ring_bytes() as usize
    }
    /// Logits `[1, S, 200058]` of every row; the head passes them through.
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "Inkling 2 requires one nonempty sequence"
        );
        let device = self.device();
        let ids = match device.dtype(ids) {
            Some(Dtype::I32) => ids.clone(),
            Some(Dtype::U32) => {
                TensorOps::new(&device, TensorLaw::Native).op(T::Cast(Dtype::I32), &[ids])?
            }
            _ => anyhow::bail!("Inkling 2 token IDs must be I32 or U32"),
        };
        let tokens = device
            .read_i32_through(&ids)?
            .into_iter()
            .map(|t| u32::try_from(t).context("negative Inkling 2 token"))
            .collect::<Result<Vec<_>>>()?;
        let mut fresh = self.new_caches();
        let caches = caches.unwrap_or(&mut fresh);
        // A generated token after a prompt runs the decode step, as the
        // reference's generation does; prompts run its prefill.
        let x = if tokens.len() == 1 && state(caches)?.position() > 0 {
            self.decode(tokens[0], caches)?
        } else {
            self.feed(&tokens, caches, true)?
        }
        .reshape(&[1, tokens.len() as i32, TOKENS as i32])?;
        Ok((x.clone(), x))
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        _capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        self.forward(ids, caches)
    }
    /// The last row's logits `[1, 200058]`.
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        self.feed(tokens, caches, false)
    }
    fn decode_token(&mut self, token: u32, caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        self.decode(token, caches)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let x = self.feed(tokens, caches, false)?;
        Ok((x.clone(), x))
    }
    /// Every path returns softcapped logits already.
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        if x.dim(-1) == TOKENS as i32 {
            Ok(x.clone())
        } else {
            Err(anyhow::anyhow!("Inkling rows are logits already").into())
        }
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        if self.warmed {
            return Ok(());
        }
        // The prompt shapes of a whole chunk (its scratch, held for every
        // shorter one), a long one (prefill tiles, row kernels, the direct
        // attention) and a short one, a verify block and a decode row.
        let token = seed.first().copied().unwrap_or(200000);
        let mut caches = self.new_caches();
        self.feed(&[token; PROMPT_ROWS as usize], &mut caches, false)?
            .eval()?;
        self.feed(&[token; 300], &mut caches, false)?.eval()?;
        self.feed(&[token; 40], &mut caches, false)?.eval()?;
        let position = state(&caches)?.position();
        let s = self.reserve(state_mut(&mut caches)?, position + CHUNK)?;
        self.chain(s, &[token; CHUNK])?.eval()?;
        self.decode(token, &mut caches)?.eval()?;
        self.runtime.context.device().clear_unused();
        self.warmed = true;
        Ok(())
    }
}
