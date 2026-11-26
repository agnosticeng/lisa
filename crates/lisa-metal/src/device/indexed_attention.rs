//! Individual indexed-attention launches with explicit caller-owned storage.
use super::*;
use crate::kernels::ggml as n;
const F32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const F16: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F16],
    alignment: 2,
    ..F32
};
const KEYS: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32, ScalarType::BF16],
    alignment: 2,
    ..F32
};
const I32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32],
    ..F32
};
const U32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::U32],
    ..F32
};
// The native mask is a zero-initialized 16-bit storage array; its signed integer
// representation is opaque to arithmetic. Bind it as F16 without conversion.
pub(super) fn capabilities() -> Vec<KernelCapability> {
    use IndexedAttentionKind as K;
    let cap = |kind, inputs, mutations: &'static [usize]| KernelCapability {
        kind: KernelKind::IndexedAttention(kind),
        law: NumericalLaw::IndexedAttentionV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: if mutations.is_empty() {
            Workspace::ProviderManaged
        } else {
            Workspace::CallerOutputs { inputs: mutations }
        },
    };
    vec![
        cap(K::RotaryTable, &[], &[]),
        cap(K::Rotary, &[F32], &[0]),
        cap(K::RotaryStore, &[F32, F32], &[1]),
        cap(K::PoolPair, &[F32, F32, F32, F32], &[]),
        cap(K::IndexQuery, &[F32, F32], &[0]),
        cap(K::IndexKeys, &[F32], &[]),
        cap(K::IndexScores, &[F32, F32, KEYS], &[]),
        cap(K::TopK, &[F32], &[]),
        cap(K::Candidates, &[F32], &[]),
        cap(K::CandidateSelect, &[F32, F32, F32, U32], &[]),
        cap(
            K::Decode(false),
            &[F32, F32, F32, F32, F16, F16, F16, F32],
            &[4, 5, 6, 7],
        ),
        cap(
            K::Decode(true),
            &[F32, F32, F32, F32, F16, F16, F16, F32, F32, I32],
            &[4, 5, 6, 7],
        ),
    ]
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::IndexedAttention(op) = kernel else {
        return Ok(None);
    };
    use IndexedAttention as K;
    let out = match op {
        K::RotaryTable {
            position,
            frequencies,
        } => n::rope_coefficient_table(
            *position,
            [
                &frequencies[0].map(f32::from_bits),
                &frequencies[1].map(f32::from_bits),
            ],
            stream,
        )?,
        K::Rotary {
            position,
            stride,
            frequencies,
            inverse,
            round,
        } => {
            let mut x = a[0].clone();
            n::rope_in_place(
                &mut x,
                *position,
                *stride,
                &frequencies.map(f32::from_bits),
                *inverse,
                *round,
                stream,
            )?;
            x
        }
        K::RotaryStore {
            row,
            position,
            frequencies,
            encoding,
        } => {
            let mut x = a[1].clone();
            let format = match encoding {
                ActivationEncoding::Fp8E8M0 => n::ActivationFormat::Fp8E8M0,
                ActivationEncoding::Fp4E8M0 => n::ActivationFormat::Fp4E8M0,
                ActivationEncoding::Fp4E4M3 => n::ActivationFormat::Fp4E4M3,
            };
            n::rope_quantize_store(
                a[0],
                &mut x,
                *row,
                *position,
                &frequencies.map(f32::from_bits),
                format,
                stream,
            )?;
            x
        }
        K::PoolPair => n::pool_pair(a[0], a[1], a[2], a[3], stream)?,
        K::IndexQuery => {
            let mut x = a[0].clone();
            n::index_query_rope_quantize(&mut x, a[1], stream)?;
            x
        }
        K::IndexKeys => n::index_keys_bf16(a[0], stream)?,
        K::IndexScores => n::index_scores(a[0], a[1], a[2], stream)?,
        K::TopK { count } => n::index_topk(a[0], *count, stream)?,
        K::Candidates => n::index_candidates(a[0], stream)?.list,
        K::CandidateSelect { rows } => {
            if a[3].shape() != [2048] {
                return Ok(None);
            }
            let c = n::IndexCandidates {
                rows: *rows,
                list: a[3].clone(),
            };
            n::index_candidates_select(a[0], a[1], a[2], &c, stream)?
        }
        K::Decode {
            raw_count,
            raw_start,
            compressed,
            selected,
        } => {
            if a[4].shape() != [640, 512]
                || a[5].shape() != [640]
                || a[6].shape() != [2 * 32 * 512 + 32]
                || a[7].shape() != [64 * (512 * 32 + 64)]
            {
                return Ok(None);
            }
            let mut scratch = n::DecodeAttentionScratch {
                staged: a[4].clone(),
                mask: a[5].clone(),
                pad: a[6].clone(),
                partials: a[7].clone(),
            };
            n::attention_decode_with_rope(
                a[0],
                a[1],
                a[2],
                *raw_count,
                *raw_start,
                if *selected { Some((a[8], a[9])) } else { None },
                &mut scratch,
                Some((a[3], *compressed)),
                stream,
            )?
        }
    };
    Ok(Some(Outputs::One(out)))
}
