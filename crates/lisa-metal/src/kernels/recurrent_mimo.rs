//! Immutable F32 recurrent MIMO state and geodesic residual dispatch.
use crate::{
    Array, Dtype, Stream,
    error::{Error, Result},
    runtime::Math,
};
const SOURCE: &crate::shader_pack::Source = crate::kernel_source!("recurrent_mimo/recurrent.metal");
#[repr(C)]
struct MimoArgs {
    dqk: i32,
    rank: i32,
    dv: i32,
    heads: i32,
    tokens: i32,
    sequences: i32,
    chunks: i32,
    pad: i32,
    dynamic_stride: i64,
    static_stride: i64,
    input_stride: i64,
    output_stride: i64,
    off_k: i64,
    off_v: i64,
    off_a: i64,
    workspace_state: i64,
    workspace_decay: i64,
    epsilon: f32,
    a_floor: f32,
}
#[repr(C)]
struct GeoArgs {
    width: i32,
    rows: i32,
    dim2: i32,
    dim3: i32,
    x1: u64,
    x2: u64,
    x3: u64,
    g1: u64,
    g2: u64,
    g3: u64,
    y1: u64,
    y2: u64,
    y3: u64,
    inverse_depth: f32,
}
fn output(stream: &Stream, shape: &[usize]) -> Result<Array> {
    let bytes = shape
        .iter()
        .try_fold(4usize, |n, d| n.checked_mul(*d))
        .ok_or_else(|| Error::msg("recurrent output overflow"))?;
    let rt = stream.runtime();
    let buffer = rt.buffer_for_full_gpu_write(bytes, "recurrent_mimo_output")?;
    Ok(Array::new(crate::array::Array::from_parts(
        rt,
        buffer,
        shape,
        Dtype::Float32,
    )))
}
fn check(a: &Array, shape: &[i32]) -> Result<()> {
    if a.dtype() != Dtype::Float32
        || a.shape() != shape
        || !a.t.buffer_and_layout().1.is_contiguous()
    {
        return Err(Error::msg("invalid recurrent F32 input shape or layout"));
    }
    a.eval_enqueue()
}
/// State layout is [S(H,64,128), K(H,4,128), V(H,4,64), phase(H,32)].
pub fn step(a: &[&Array], heads: usize, epsilon: f32, stream: &Stream) -> Result<(Array, Array)> {
    if a.len() != 8
        || heads == 0
        || heads > 256
        || !epsilon.is_finite()
        || epsilon <= 0.0
        || a[0].rank() != 2
        || a[0].dim(0) < 1
    {
        return Err(Error::msg("invalid recurrent MIMO geometry"));
    }
    let h = heads as i32;
    let rows = a[0].dim(0);
    let state_len = h * (64 * 128 + 4 * 128 + 4 * 64 + 32);
    let shapes = [
        vec![rows, 1056],
        vec![rows, h * 131],
        vec![h, 8, 128],
        vec![h, 8, 64],
        vec![h, 4, 64],
        vec![2, 128],
        vec![2, h],
        vec![state_len],
    ];
    for (a, s) in a.iter().zip(shapes.iter()) {
        check(a, s)?;
    }
    let y = output(stream, &[rows as usize, heads * 64])?;
    let state = output(stream, &[state_len as usize])?;
    // Serial specialization never dereferences workspace; bind a valid storage object.
    let args = MimoArgs {
        dqk: 128,
        rank: 4,
        dv: 64,
        heads: h,
        tokens: rows,
        sequences: 1,
        chunks: 1,
        pad: 0,
        dynamic_stride: 1056,
        static_stride: (h * 131) as i64,
        input_stride: state_len as i64,
        output_stride: state_len as i64,
        off_k: (h * 64 * 128) as i64,
        off_v: (h * (64 * 128 + 4 * 128)) as i64,
        off_a: (h * (64 * 128 + 4 * 128 + 4 * 64)) as i64,
        workspace_state: 0,
        workspace_decay: 0,
        epsilon,
        a_floor: 1e-4,
    };
    let pipeline = stream.runtime().compile_with(
        SOURCE,
        "kernel_mamba3_mimo_serial_d128_v64_r4",
        Math::CompilerDefault,
    )?;
    let guard = stream.runtime().commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&pipeline);
    e.set_bytes(0, &args);
    for (i, t) in a.iter().enumerate() {
        let (b, l) = t.t.buffer_and_layout();
        e.set_input(i + 1, Some(b), l.offset * 4);
    }
    for (i, t) in [(9, &y), (10, &state)] {
        let (b, l) = t.t.buffer_and_layout();
        e.set_output(i, Some(b), l.offset * 4);
    }
    let (b, l) = state.t.buffer_and_layout();
    e.set_input(11, Some(b), l.offset * 4);
    e.set_threadgroup_memory_length(0, (3 * 4 * 128 + 3 * 4 * 64 + 3 * 32 + 2 * 4 + 3) * 4);
    e.dispatch_groups((1, heads, 1), (256, 1, 1));
    Ok((y, state))
}
pub fn geodesic(a: &[&Array], inverse_depth: f32, stream: &Stream) -> Result<Array> {
    if a.len() != 4
        || a[0].rank() != 2
        || a[0].dim(0) < 1
        || a[0].dim(1) < 1
        || !inverse_depth.is_finite()
        || inverse_depth <= 0.0
    {
        return Err(Error::msg("invalid geodesic geometry"));
    }
    check(a[0], a[0].shape())?;
    check(a[1], a[0].shape())?;
    check(a[2], &[1])?;
    check(a[3], &[1])?;
    let (rows, width) = (a[0].dim(0), a[0].dim(1));
    let stride = width as u64 * 4;
    let total = stride * rows as u64;
    let args = GeoArgs {
        width,
        rows,
        dim2: 1,
        dim3: 1,
        x1: stride,
        x2: total,
        x3: total,
        g1: stride,
        g2: total,
        g3: total,
        y1: stride,
        y2: total,
        y3: total,
        inverse_depth,
    };
    let y = output(stream, &[rows as usize, width as usize])?;
    let pipeline =
        stream
            .runtime()
            .compile_with(SOURCE, "kernel_geodesic_f32", Math::CompilerDefault)?;
    let guard = stream.runtime().commands.encoder()?;
    let e = guard.encoder();
    e.set_pipeline(&pipeline);
    e.set_bytes(0, &args);
    for (i, t) in a.iter().enumerate() {
        let (b, l) = t.t.buffer_and_layout();
        e.set_input(i + 1, Some(b), l.offset * 4);
    }
    let (b, l) = y.t.buffer_and_layout();
    e.set_output(5, Some(b), l.offset * 4);
    e.set_threadgroup_memory_length(0, 64 * 4);
    e.dispatch_groups((rows as usize, 1, 1), (256, 1, 1));
    Ok(y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lisa_backend::TensorAccess;
    fn bits(a: &Array) -> Vec<u32> {
        a.as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
    }
    #[test]
    fn mimo_serial_chunks_preserve_every_state_bit_and_input_snapshot() {
        let stream = Stream::gpu();
        let h = 2usize;
        let tokens = 3usize;
        let make = |n: usize, shape: &[i32], salt: usize| {
            Array::from_slice(
                &(0..n)
                    .map(|i| (((i * 17 + salt) % 31) as f32 - 15.) / 64.)
                    .collect::<Vec<_>>(),
                shape,
            )
        };
        let dyns = make(tokens * 1056, &[tokens as i32, 1056], 3);
        let stat = make(tokens * h * 131, &[tokens as i32, (h * 131) as i32], 5);
        let bias = make(h * 8 * 128, &[h as i32, 8, 128], 7);
        let mxz = make(h * 8 * 64, &[h as i32, 8, 64], 11);
        let mo = make(h * 4 * 64, &[h as i32, 4, 64], 13);
        let norms = Array::from_slice(&vec![1.0f32; 256], &[2, 128]);
        let misc = make(2 * h, &[2, h as i32], 17);
        let length = h * (64 * 128 + 4 * 128 + 4 * 64 + 32);
        let original = make(length, &[length as i32], 19);
        let saved = bits(&original);
        let (whole, whole_state) = step(
            &[&dyns, &stat, &bias, &mxz, &mo, &norms, &misc, &original],
            h,
            1e-6,
            &stream,
        )
        .unwrap();
        let mut state = original.clone();
        let mut rows = vec![];
        for t in 0..tokens {
            let d = dyns
                .slice(0, t as i32, Some(t as i32 + 1))
                .unwrap()
                .contiguous()
                .unwrap();
            let s = stat
                .slice(0, t as i32, Some(t as i32 + 1))
                .unwrap()
                .contiguous()
                .unwrap();
            let (y, next) = step(
                &[&d, &s, &bias, &mxz, &mo, &norms, &misc, &state],
                h,
                1e-6,
                &stream,
            )
            .unwrap();
            rows.push(y);
            state = next;
        }
        let joined = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0).unwrap();
        assert_eq!(bits(&whole), bits(&joined));
        assert_eq!(bits(&whole_state), bits(&state));
        assert_eq!(bits(&original), saved);
        assert!(whole.as_slice::<f32>().iter().all(|x| x.is_finite()));
    }
    #[test]
    fn geodesic_zero_update_retains_finite_residual() {
        let stream = Stream::gpu();
        let x = Array::from_slice(&vec![0.25f32; 1536], &[1, 1536]);
        let zero = Array::from_slice(&vec![0.0f32; 1536], &[1, 1536]);
        let scale = Array::from_slice(&[1.0f32], &[1]);
        let bias = Array::from_slice(&[0.0f32], &[1]);
        let y = geodesic(&[&x, &zero, &scale, &bias], 1., &stream).unwrap();
        assert_eq!(bits(&x), bits(&y));
    }
}
