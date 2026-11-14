//! Portable per-layer state, snapshots, rollback and spill encoding.
use crate::{
    attention_cache::FullAttentionCache,
    indexer_cache::{CacheDevice, TapeSnapshot},
    recurrent_cache::GdnCache,
};
use lisa_backend::{HostAccessDevice, ScalarType as Dtype, TensorAccess};
/// Model-owned session state for architectures whose cache cannot be described
/// as independent attention/GDN layers. Snapshots share immutable ownership;
/// the model obtains a private GPU copy before its next in-place write.
pub trait NativeState: std::any::Any + Send + Sync {
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn copy_for_write(&self) -> anyhow::Result<std::sync::Arc<dyn NativeState>>;
    fn position(&self) -> usize;
}

#[derive(Clone)]
pub struct NativeCache<T> {
    state: std::sync::Arc<dyn NativeState>,
    ple_conv: Option<T>,
    capture_ple_full: Option<T>,
}

impl<T: Clone> NativeCache<T> {
    pub fn new(state: impl NativeState) -> Self {
        Self {
            state: std::sync::Arc::new(state),
            ple_conv: None,
            capture_ple_full: None,
        }
    }
    pub fn position(&self) -> usize {
        self.state.position()
    }
    pub fn get<S: NativeState>(&self) -> anyhow::Result<&S> {
        self.state.as_any().downcast_ref().ok_or_else(|| {
            anyhow::anyhow!("native cache belongs to a different model architecture")
        })
    }
    pub fn get_mut<S: NativeState>(&mut self) -> anyhow::Result<&mut S> {
        anyhow::ensure!(
            self.state.as_any().is::<S>(),
            "native cache belongs to a different model architecture"
        );
        if std::sync::Arc::get_mut(&mut self.state).is_none() {
            self.state = self.state.copy_for_write()?;
        }
        std::sync::Arc::get_mut(&mut self.state)
            .ok_or_else(|| anyhow::anyhow!("native cache copy must return independent ownership"))?
            .as_any_mut()
            .downcast_mut()
            .ok_or_else(|| {
                anyhow::anyhow!("native cache belongs to a different model architecture")
            })
    }
}

/// Per-layer recurrent cache.
pub enum LayerCache<D: CacheDevice> {
    Full(FullAttentionCache<D>),
    Linear(GdnCache<D::Tensor>),
    Native(NativeCache<D::Tensor>),
}

impl<D: CacheDevice> LayerCache<D> {
    /// The PLE short-conv state and (capture) conv-input slots for this layer.
    #[allow(clippy::type_complexity)]
    pub fn ple_conv_mut(&mut self) -> (&mut Option<D::Tensor>, &mut Option<D::Tensor>) {
        match self {
            LayerCache::Native(n) => (&mut n.ple_conv, &mut n.capture_ple_full),
            LayerCache::Full(f) => (&mut f.ple_conv, &mut f.capture_ple_full),
            LayerCache::Linear(g) => (&mut g.ple_conv, &mut g.capture_ple_full),
        }
    }

    /// Clone the live state at a prefix boundary for a cross-request cache
    /// entry. Only the live KV rows are copied (the entry outlives the
    /// session's buffers); the GDN/PLE recurrent states clone the proven
    /// `snapshot_state` way.
    pub fn snapshot_prefix(&self) -> LayerState<D::Tensor> {
        match self {
            LayerCache::Native(n) => LayerState::Native(n.clone()),
            LayerCache::Full(f) => {
                let keys = f.keys.as_ref().map(|k| {
                    k.slice(2, 0, Some(f.offset as i32))
                        .and_then(|v| v.contiguous())
                        .ok()
                });
                let values = f.values.as_ref().map(|v| {
                    v.slice(2, 0, Some(f.offset as i32))
                        .and_then(|v| v.contiguous())
                        .ok()
                });
                LayerState::Full(FullPrefixState {
                    keys: keys.unwrap_or(None),
                    values: values.unwrap_or(None),
                    offset: f.offset,
                    tape: f.indexer_tape.snapshot(),
                    ple_conv: f.ple_conv.clone(),
                })
            }
            LayerCache::Linear(g) => {
                let (conv, ssm, ple_conv) = g.snapshot_state();
                LayerState::Linear(LinearPrefixState {
                    conv,
                    ssm,
                    ple_conv,
                })
            }
        }
    }

    /// Capture the small amount of state needed to undo an append-only verify
    /// transaction. Unlike `snapshot_prefix`, this does not copy live KV rows:
    /// full-attention buffers remain in place and only their logical offsets
    /// and indexer metadata are saved. Linear-attention recurrent arrays are
    /// cloned by reference, as in prefix snapshots.
    pub fn snapshot_rollback(&self) -> LayerRollbackState<D::Tensor> {
        match self {
            LayerCache::Native(n) => LayerRollbackState::Native(n.clone()),
            LayerCache::Full(f) => LayerRollbackState::Full {
                offset: f.offset,
                tape: f.indexer_tape.snapshot(),
                ple_conv: f.ple_conv.clone(),
                next_pos: f.next_pos.clone(),
            },
            LayerCache::Linear(g) => {
                let (conv, ssm, ple_conv) = g.snapshot_state();
                LayerRollbackState::Linear {
                    conv,
                    ssm,
                    ple_conv,
                }
            }
        }
    }

    /// Restore an append-only verify transaction captured by
    /// `snapshot_rollback`. Rows written past the restored boundary remain in
    /// the backing buffers and are overwritten by the next append.
    pub fn restore_rollback(
        &mut self,
        state: &LayerRollbackState<D::Tensor>,
    ) -> anyhow::Result<()> {
        match (self, state) {
            (LayerCache::Native(n), LayerRollbackState::Native(saved)) => {
                *n = saved.clone();
                Ok(())
            }
            (
                LayerCache::Full(f),
                LayerRollbackState::Full {
                    offset,
                    tape,
                    ple_conv,
                    next_pos,
                },
            ) => {
                if *offset > f.offset {
                    return Err(anyhow::Error::msg(format!(
                        "full-attention rollback boundary {offset} exceeds live offset {}",
                        f.offset
                    )));
                }
                f.offset = *offset;
                f.indexer_tape.restore(tape);
                f.ple_conv = ple_conv.clone();
                f.next_pos = next_pos.clone();
                Ok(())
            }
            (
                LayerCache::Linear(g),
                LayerRollbackState::Linear {
                    conv,
                    ssm,
                    ple_conv,
                },
            ) => {
                g.restore_state(&(conv.clone(), ssm.clone(), ple_conv.clone()));
                Ok(())
            }
            _ => Err(anyhow::Error::msg(
                "cache kind changed during verify transaction",
            )),
        }
    }

    /// Install a prefix-cache entry's state into this (fresh) cache.
    pub fn restore_prefix(&mut self, s: &LayerState<D::Tensor>) {
        match (self, s) {
            (LayerCache::Native(n), LayerState::Native(saved)) => *n = saved.clone(),
            (LayerCache::Full(f), LayerState::Full(st)) => {
                f.keys = st.keys.clone();
                f.values = st.values.clone();
                f.offset = st.offset;
                f.capacity = st.keys.as_ref().map(|k| k.dim(2) as usize).unwrap_or(0);
                f.indexer_tape.restore(&st.tape);
                f.ple_conv = st.ple_conv.clone();
                // The tape is the full history of the restored prefix, so the
                // indexer stays exact-causal-correct (contract kept uniform).
                f.indexer_ok = true;
            }
            (LayerCache::Linear(g), LayerState::Linear(st)) => {
                g.restore_state(&(st.conv.clone(), st.ssm.clone(), st.ple_conv.clone()))
            }
            _ => {}
        }
    }
}

/// Cross-request prefix state for one full-attention layer.
#[derive(Clone)]
pub struct FullPrefixState<T> {
    pub keys: Option<T>,
    pub values: Option<T>,
    pub offset: usize,
    pub tape: TapeSnapshot<T>,
    pub ple_conv: Option<T>,
}

/// Cross-request prefix state for one GDN layer.
#[derive(Clone)]
pub struct LinearPrefixState<T> {
    pub conv: Option<T>,
    pub ssm: Option<T>,
    pub ple_conv: Option<T>,
}

/// Cross-request prefix state for one layer (see [`LayerCache`]).
#[derive(Clone)]
pub enum LayerState<T> {
    Full(FullPrefixState<T>),
    Linear(LinearPrefixState<T>),
    Native(NativeCache<T>),
}

/// Lightweight append-only cache checkpoint for speculative verification.
pub enum LayerRollbackState<T> {
    Native(NativeCache<T>),
    Full {
        offset: usize,
        tape: TapeSnapshot<T>,
        ple_conv: Option<T>,
        next_pos: Option<Vec<usize>>,
    },
    Linear {
        conv: Option<T>,
        ssm: Option<T>,
        ple_conv: Option<T>,
    },
}

// --- SSD-spill codec (prefix_cache.rs) ---------------------------------
// A `LayerState` serializes to host bytes: one u8 presence tag per array,
// then dtype tag + shape + raw LE payload. `decode` rebuilds owned device
// tensors through `Device::pull`. Entries are keyed by their tokens
// and numerical law in prefix_cache.rs.

fn dtype_tag(d: Dtype) -> u8 {
    match d {
        Dtype::Bool => 1,
        Dtype::U8 => 2,
        Dtype::U16 => 3,
        Dtype::U32 => 4,
        Dtype::I8 => 5,
        Dtype::I16 => 6,
        Dtype::I32 => 7,
        Dtype::I64 => 8,
        Dtype::F16 => 9,
        Dtype::F32 => 10,
        Dtype::F64 => 11,
        Dtype::BF16 => 12,
    }
}

fn dtype_from_tag(t: u8) -> Option<Dtype> {
    Some(match t {
        1 => Dtype::Bool,
        2 => Dtype::U8,
        3 => Dtype::U16,
        4 => Dtype::U32,
        5 => Dtype::I8,
        6 => Dtype::I16,
        7 => Dtype::I32,
        8 => Dtype::I64,
        9 => Dtype::F16,
        10 => Dtype::F32,
        11 => Dtype::F64,
        12 => Dtype::BF16,
        _ => return None,
    })
}

fn enc_arr<D: HostAccessDevice>(
    device: &D,
    out: &mut Vec<u8>,
    a: &Option<D::Tensor>,
) -> anyhow::Result<()> {
    let Some(a) = a else {
        out.push(0);
        return Ok(());
    };
    out.push(1);
    let dtype = device
        .dtype(a)
        .ok_or_else(|| anyhow::anyhow!("prefix spill: unknown scalar type"))?;
    out.push(dtype_tag(dtype));
    let shape = device.shape(a);
    out.extend((shape.len() as u32).to_le_bytes());
    for &d in shape {
        out.extend(d.to_le_bytes());
    }
    let bytes = device.read_bytes(a)?;
    anyhow::ensure!(
        lisa_backend::tensor_bytes(shape, dtype) == Some(bytes.len()),
        "prefix spill: payload size mismatch"
    );
    out.extend((bytes.len() as u64).to_le_bytes());
    out.extend(bytes);
    Ok(())
}

fn take_bytes<'a>(cur: &mut &'a [u8], n: usize) -> anyhow::Result<&'a [u8]> {
    if cur.len() < n {
        anyhow::bail!("prefix spill: truncated record");
    }
    let (h, t) = cur.split_at(n);
    *cur = t;
    Ok(h)
}

fn rd_u8(cur: &mut &[u8]) -> anyhow::Result<u8> {
    Ok(take_bytes(cur, 1)?[0])
}

fn rd_u32(cur: &mut &[u8]) -> anyhow::Result<u32> {
    Ok(u32::from_le_bytes(take_bytes(cur, 4)?.try_into().unwrap()))
}

fn rd_u64(cur: &mut &[u8]) -> anyhow::Result<u64> {
    Ok(u64::from_le_bytes(take_bytes(cur, 8)?.try_into().unwrap()))
}

fn dec_arr<D: HostAccessDevice>(device: &D, cur: &mut &[u8]) -> anyhow::Result<Option<D::Tensor>> {
    if rd_u8(cur)? == 0 {
        return Ok(None);
    }
    let dtype = dtype_from_tag(rd_u8(cur)?)
        .ok_or_else(|| anyhow::anyhow!("prefix spill: bad dtype tag"))?;
    let ndim = rd_u32(cur)? as usize;
    anyhow::ensure!(ndim <= cur.len() / 4, "prefix spill: truncated shape");
    let shape = (0..ndim)
        .map(|_| Ok(i32::try_from(rd_u32(cur)?)?))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let nbytes = usize::try_from(rd_u64(cur)?)?;
    anyhow::ensure!(
        lisa_backend::tensor_bytes(&shape, dtype) == Some(nbytes),
        "prefix spill: payload size mismatch"
    );
    let bytes = take_bytes(cur, nbytes)?;
    Ok(Some(
        device.pull(
            lisa_backend::HostTensor {
                bytes,
                shape: &shape,
                dtype,
            }
            .into(),
            Default::default(),
        )?,
    ))
}

impl<T> LayerState<T> {
    /// Host-byte encoding for the SSD spill tier. Eval + readback of every
    /// array: the spill runs on eviction only, off the serve hot path.
    pub fn encode<D: HostAccessDevice<Tensor = T>>(&self, device: &D) -> anyhow::Result<Vec<u8>> {
        let mut out = Vec::new();
        match self {
            LayerState::Native(_) => {
                anyhow::bail!("this model's native cache has no SSD spill codec")
            }
            LayerState::Full(s) => {
                // Tag 3 adds the raw-tail base. Tags 0 and 2 remain readable
                // for spill records written before raw-history compaction.
                out.push(3);
                out.extend((s.offset as u64).to_le_bytes());
                enc_arr(device, &mut out, &s.keys)?;
                enc_arr(device, &mut out, &s.values)?;
                enc_arr(device, &mut out, &s.tape.raw)?;
                out.extend((s.tape.total as u64).to_le_bytes());
                out.extend((s.tape.raw_start as u64).to_le_bytes());
                enc_arr(device, &mut out, &s.tape.pooled)?;
                out.extend((s.tape.pooled_upto as u64).to_le_bytes());
                out.extend((s.tape.cr as u64).to_le_bytes());
                enc_arr(device, &mut out, &s.tape.compact)?;
                out.extend((s.tape.compact_total as u64).to_le_bytes());
                enc_arr(device, &mut out, &s.ple_conv)?;
            }
            LayerState::Linear(s) => {
                out.push(1);
                enc_arr(device, &mut out, &s.conv)?;
                enc_arr(device, &mut out, &s.ssm)?;
                enc_arr(device, &mut out, &s.ple_conv)?;
            }
        }
        Ok(out)
    }

    /// Rebuild a `LayerState` from [`LayerState::encode`] bytes, advancing
    /// `cur` past the consumed record (layers are variable-length, so the
    /// spill reader decodes in sequence off one flat buffer).
    pub fn decode<D: HostAccessDevice<Tensor = T>>(
        device: &D,
        cur: &mut &[u8],
    ) -> anyhow::Result<LayerState<T>> {
        let kind = if cur.is_empty() {
            anyhow::bail!("prefix spill: empty layer record")
        } else {
            let k = cur[0];
            *cur = &cur[1..];
            k
        };
        Ok(match kind {
            0 | 2 | 3 => {
                let has_compact = kind >= 2;
                let has_raw_start = kind >= 3;
                let offset = rd_u64(cur)? as usize;
                let keys = dec_arr(device, cur)?;
                let values = dec_arr(device, cur)?;
                let tape_raw = dec_arr(device, cur)?;
                let total = rd_u64(cur)? as usize;
                let raw_start = if has_raw_start {
                    rd_u64(cur)? as usize
                } else {
                    0
                };
                let pooled = dec_arr(device, cur)?;
                let pooled_upto = rd_u64(cur)? as usize;
                let cr = rd_u64(cur)? as usize;
                let compact = if has_compact {
                    dec_arr(device, cur)?
                } else {
                    None
                };
                let compact_total = if has_compact {
                    rd_u64(cur)? as usize
                } else {
                    0
                };
                let ple_conv = dec_arr(device, cur)?;
                LayerState::Full(FullPrefixState {
                    keys,
                    values,
                    offset,
                    tape: TapeSnapshot {
                        raw: tape_raw,
                        total,
                        raw_start,
                        pooled,
                        pooled_upto,
                        cr,
                        compact,
                        compact_total,
                    },
                    ple_conv,
                })
            }
            1 => {
                let conv = dec_arr(device, cur)?;
                let ssm = dec_arr(device, cur)?;
                let ple_conv = dec_arr(device, cur)?;
                LayerState::Linear(LinearPrefixState {
                    conv,
                    ssm,
                    ple_conv,
                })
            }
            k => anyhow::bail!("prefix spill: bad layer kind {k}"),
        })
    }
}

#[cfg(test)]
mod codec_tests {
    use super::*;
    use lisa_cpu::{CpuDevice, Tensor};

    // Frozen legacy linear-state record: conv=[1.0,-2.5] f32, no SSM/PLE.
    const LEGACY_LINEAR: &[u8] = &[
        1, 1, 10, 1, 0, 0, 0, 2, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 63, 0, 0, 32, 192, 0,
        0,
    ];

    #[test]
    fn legacy_linear_record_imports_on_cpu_and_preserves_bytes() {
        let mut input = LEGACY_LINEAR;
        let state = LayerState::decode(&CpuDevice, &mut input).unwrap();
        let LayerState::Linear(s) = &state else {
            panic!("linear record")
        };
        assert_eq!(s.conv.as_ref().unwrap().f32().unwrap(), &[1.0, -2.5]);
        assert!(s.ssm.is_none() && s.ple_conv.is_none() && input.is_empty());
        assert_eq!(state.encode(&CpuDevice).unwrap(), LEGACY_LINEAR);
        let fresh = LayerState::Linear(LinearPrefixState {
            conv: Some(Tensor::from_f32(vec![1.0, -2.5], &[2]).unwrap()),
            ssm: None,
            ple_conv: None,
        });
        assert_eq!(fresh.encode(&CpuDevice).unwrap(), LEGACY_LINEAR);
    }

    #[test]
    fn corrupt_spill_lengths_fail_before_import() {
        for len in 0..LEGACY_LINEAR.len() {
            assert!(LayerState::decode(&CpuDevice, &mut &LEGACY_LINEAR[..len]).is_err());
        }
        for (range, replacement) in [
            (3..7, u32::MAX.to_le_bytes().to_vec()),   // implausible rank
            (7..11, u32::MAX.to_le_bytes().to_vec()),  // negative dimension
            (11..19, u64::MAX.to_le_bytes().to_vec()), // oversized payload
        ] {
            let mut record = LEGACY_LINEAR.to_vec();
            record[range].copy_from_slice(&replacement);
            assert!(LayerState::decode(&CpuDevice, &mut record.as_slice()).is_err());
        }
    }
}

/// Lightweight committed session boundary; full attention retains its buffers.
#[derive(Clone)]
pub struct SessionSnapshot<T> {
    pub native: Option<NativeCache<T>>,
    pub full_offset: Option<usize>,
    pub gdn: Option<(Option<T>, Option<T>, Option<T>)>,
}
impl<D: CacheDevice> LayerCache<D> {
    pub fn checkpoint(&self) -> SessionSnapshot<D::Tensor> {
        match self {
            Self::Full(f) => SessionSnapshot {
                native: None,
                full_offset: Some(f.offset),
                gdn: None,
            },
            Self::Linear(g) => SessionSnapshot {
                native: None,
                full_offset: None,
                gdn: Some(g.snapshot_state()),
            },
            Self::Native(n) => SessionSnapshot {
                native: Some(n.clone()),
                full_offset: None,
                gdn: None,
            },
        }
    }
    pub fn restore_checkpoint(&mut self, snapshot: &SessionSnapshot<D::Tensor>) {
        match (self, snapshot) {
            (
                Self::Native(n),
                SessionSnapshot {
                    native: Some(saved),
                    ..
                },
            ) => *n = saved.clone(),
            (
                Self::Full(f),
                SessionSnapshot {
                    full_offset: Some(o),
                    ..
                },
            ) => f.restore_offset(*o),
            (Self::Linear(g), SessionSnapshot { gdn: Some(st), .. }) => g.restore_state(st),
            _ => {}
        }
    }
    pub fn serial_snapshot(
        &self,
    ) -> Option<(Option<D::Tensor>, Option<D::Tensor>, Option<D::Tensor>)> {
        match self {
            Self::Linear(g) => Some(g.snapshot_state()),
            _ => None,
        }
    }
    pub fn restore_serial_eos(
        &mut self,
        snapshot: &Option<(Option<D::Tensor>, Option<D::Tensor>, Option<D::Tensor>)>,
    ) {
        match (self, snapshot) {
            (Self::Full(f), _) => f.trim(1),
            (Self::Linear(g), Some(s)) => g.restore_state(s),
            _ => {}
        }
    }
    pub fn rollback_verified(
        &mut self,
        context: &crate::execution::DeviceContext<D>,
        advance: usize,
        keep: usize,
        captured: bool,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            keep <= advance,
            "speculative rollback exceeds verified rows"
        );
        match self {
            Self::Full(f) => f.trim(advance - keep),
            Self::Linear(g) if captured && g.capture_conv_input.is_some() => {
                let conv = g
                    .conv
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("captured convolution has no live history"))?;
                anyhow::ensure!(conv.rank() == 3, "invalid convolution history rank");
                // Use the loaded history extent, never an assumed model width.
                g.rollback_to(context, keep, conv.dim(1) as usize + 1)?;
            }
            Self::Linear(_) => {}
            Self::Native(_) => anyhow::bail!(
                "native cache speculative rollback requires a model-specific verifier"
            ),
        }
        Ok(())
    }
    pub fn full_offset(&self) -> Option<usize> {
        match self {
            Self::Full(f) => Some(f.offset),
            _ => None,
        }
    }
    pub fn has_positions(&self) -> bool {
        matches!(self, Self::Full(f) if f.next_pos.is_some())
    }
    pub fn set_mask(&mut self, mask: &D::Tensor) {
        if let Self::Full(f) = self {
            f.attn_mask = Some(mask.clone());
        }
    }
    pub fn advance_positions(&mut self) {
        if let Self::Full(f) = self {
            if let Some(v) = f.next_pos.as_mut() {
                for p in v.iter_mut() {
                    *p += 1;
                }
            }
        }
    }
    pub fn clear_positions(&mut self) {
        if let Self::Full(f) = self {
            f.next_pos = None;
            f.attn_mask = None;
        }
    }
}
