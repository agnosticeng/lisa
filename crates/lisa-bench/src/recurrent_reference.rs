//! Frozen pre-migration tensor-operation oracle, used only by diagnostics.
use lisa_metal::ops::indexing::IndexOp;
use lisa_metal::{Array, Dtype, ops};
/// Ops fallback matching the `gatedDeltaOps` kernel: a per-t loop.
///
/// Exact same math as the kernel modulo fp32 accumulation order.
pub fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
) -> lisa_metal::error::Result<(Array, Array)> {
    let b = q.dim(0);
    let t = q.dim(1);
    let hk = q.dim(2);
    let dk = q.dim(3);
    let hv = v.dim(2);
    let dv = v.dim(3);

    let repeat = hv / hk;
    let q = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(q.clone(), repeat, -2)?
    } else {
        q.clone()
    };
    let k = if repeat > 1 {
        ops::repeat_axis::<half::bf16>(k.clone(), repeat, -2)?
    } else {
        k.clone()
    };

    let mut state = match state {
        Some(s) if s.dtype() == Dtype::Float32 => s.clone(),
        Some(s) => s.as_dtype(Dtype::Float32)?,
        None => ops::full::<f32>(&[b, hv, dv, dk], Array::from_f32(0.0))?,
    };

    let mut ys = Vec::with_capacity(t as usize);
    for idx in 0..t {
        let q_t = q.index((.., idx));
        let k_t = k.index((.., idx));
        let v_t = v.index((.., idx));
        let g_t = g.index((.., idx));
        let beta_t = beta.index((.., idx));

        let decay = g_t.reshape(&[b, hv, 1, 1])?;
        let k_t = k_t.reshape(&[b, hv, 1, dk])?;
        let beta_t = beta_t.reshape(&[b, hv, 1])?;

        let new_state = state * decay;
        let kv_mem = (&new_state * &k_t).sum_axis(-1, None)?;
        let delta = (v_t - kv_mem) * beta_t;
        let new_state = new_state + k_t * delta.reshape(&[b, hv, dv, 1])?;
        let y = (&new_state * q_t.reshape(&[b, hv, 1, dk])?).sum_axis(-1, None)?;
        ys.push(y.as_dtype(q.dtype())?);
        state = new_state;
    }
    let y = ops::stack(&ys, 1)?;
    Ok((y, state))
}
