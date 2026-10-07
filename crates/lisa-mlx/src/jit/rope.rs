use objc2_metal::MTLSize;

use crate::array::Array;

use super::compile::{
    builtin_template_def, compile_builtin_bool_consts, get_block_dims, type_string, type_to_name,
};
use super::{Device, Tensor};
use super::{MLX_ROPE_SOURCE, MLX_UTILS_PREAMBLE};
use crate::error::Result;

/// MLX `fast::rope` (`RoPE::eval_gpu`, `rope.cpp:12`). `offset` is an int array.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    device: &Device,
    x: &Tensor,
    dims: i32,
    traditional: bool,
    base: f32,
    scale: f32,
    offset: &Tensor,
    forward: bool,
) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    let rank = x.rank();
    if rank < 2 || !layout.is_contiguous() {
        crate::bail!("rope: non-contiguous or rank < 2 input");
    }
    let dims_v = x.dims().to_vec();
    let b = dims_v[0];
    let t = dims_v[rank - 2];
    let d = dims_v[rank - 1];
    let n: usize = dims_v[1..rank.saturating_sub(2)].iter().product();
    let mat_size = (t * d) as i64;
    let single = t == 1 && offset.elem_count() == 1;
    let in_dt = x.dtype();
    let in_t = type_string(in_dt)?;
    let ty = type_to_name(in_dt)?;

    let kernel_name = if single {
        format!("rope_single_{ty}")
    } else {
        format!("rope_{ty}")
    };
    let mut defs = builtin_template_def(
        &format!("rope_{ty}"),
        "rope",
        &[in_t.to_string(), "int32_t".to_string()],
    );
    defs.push_str(&builtin_template_def(
        &format!("rope_single_{ty}"),
        "rope_single",
        &[in_t.to_string()],
    ));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_ROPE_SOURCE}{defs}");
    let pipeline = compile_builtin_bool_consts(
        device,
        &source,
        &kernel_name,
        &[(1, forward), (2, traditional), (3, false)],
    )?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (in_dt).size_of(), "rope_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims_v.clone(), in_dt);

    let in_strides = layout.stride().to_vec();
    let strides: Vec<i64> = vec![
        mat_size,
        in_strides[rank - 2] as i64,
        in_strides[rank - 1] as i64,
    ];
    // out is contiguous
    let out_strides_raw = {
        let mut s = vec![1i64; rank];
        let mut acc = 1i64;
        for i in (0..rank).rev() {
            s[i] = acc;
            acc *= dims_v[i] as i64;
        }
        s
    };
    let out_strides: Vec<i64> = vec![
        mat_size,
        out_strides_raw[rank - 2],
        out_strides_raw[rank - 1],
    ];

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, l) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), l.start_offset());
    }
    enc.set_output(1, Some(&obuf), 0);
    {
        let (ms, l) = offset.buffer_and_layout();
        enc.set_input(2, Some(ms), l.start_offset());
    }
    enc.set_bytes(3, &scale);

    let (d0, d1, d2);
    if single {
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(out_strides.as_slice()),
            out_strides.as_ptr().cast(),
        );
        d0 = dims as usize / 2;
        d1 = b * n;
        d2 = 1;
    } else {
        enc.set_bytes_directly(
            4,
            std::mem::size_of_val(strides.as_slice()),
            strides.as_ptr().cast(),
        );
        enc.set_bytes_directly(
            5,
            std::mem::size_of_val(out_strides.as_slice()),
            out_strides.as_ptr().cast(),
        );
        let offset_stride: i64 = if offset.rank() > 0 {
            offset.stride()[0] as i64
        } else {
            0
        };
        enc.set_bytes(6, &offset_stride);
        let nn = n as i32;
        enc.set_bytes(7, &nn);
        d0 = dims as usize / 2;
        d1 = t;
        d2 = b * n.div_ceil(4);
    }
    let log2base = base.log2();
    enc.set_bytes(10, &log2base);

    let group = get_block_dims(d0, d1, d2, 10);
    enc.dispatch_threads_size(
        MTLSize {
            width: d0,
            height: d1,
            depth: d2,
        },
        MTLSize {
            width: group.0,
            height: group.1,
            depth: group.2,
        },
    );
    Ok(out)
}
