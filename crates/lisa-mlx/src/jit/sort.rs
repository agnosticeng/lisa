use objc2_metal::MTLSize;

use crate::array::Array;

use super::compile::{builtin_template_def, compile_builtin, type_string, type_to_name};
use super::{DType, Device, Tensor};
use super::{MLX_SORT_PREAMBLE, MLX_UTILS_PREAMBLE};
use crate::error::Result;

/// MLX `ops::argsort` along the last axis (`ArgSort::eval_gpu` →
/// `single_block_sort`, `sort.cpp:15`). Returns uint32 indices.
pub fn argsort_last(device: &Device, x: &Tensor) -> Result<Tensor> {
    argsort_axis(device, x, true)
}

/// MLX `ops::argpartition_axis` — MLX directs partition to sort
/// (`sort.cpp:342`), so this is the same kernel.
pub fn argpartition_axis(device: &Device, x: &Tensor, _kth: i32) -> Result<Tensor> {
    argsort_axis(device, x, true)
}

/// Multi-block merge sort (`gpu_merge_sort`'s `n_blocks > 1` path,
/// `mlx/backend/metal/sort.cpp`). The single-block sort handles axes up to
/// `bn*tn` (2048) elements; wider axes (a full vocabulary row, 248320) need the
/// blockwise sort followed by a sequence of partition/merge passes. The
/// intermediate buffers are ping-ponged exactly as MLX does, so the output is
/// the same stable ordering.
#[allow(clippy::too_many_arguments)]
fn argsort_multi_block(
    mdev: &Device,
    x: &Tensor,
    dims: &[usize],
    n_rows: usize,
    size_sorted_axis: usize,
    bn: usize,
    tn: usize,
    n_blocks: usize,
    argsort: bool,
) -> Result<Tensor> {
    let in_dt = x.dtype();
    let in_t = type_string(in_dt)?;
    let idx_dt = DType::U32;
    let out_t = type_string(idx_dt)?;
    let in_ty = type_to_name(in_dt)?;
    let out_ty = type_to_name(idx_dt)?;
    let arg_sort = if argsort { "true" } else { "false" };

    let sort_name = format!("sort_mbsort_{in_ty}_{out_ty}_bn{bn}_tn{tn}");
    let part_name = format!("partition_mbsort_{in_ty}_{out_ty}_bn{bn}_tn{tn}");
    let merge_name = format!("merge_mbsort_{in_ty}_{out_ty}_bn{bn}_tn{tn}");
    let targs: [String; 5] = [
        in_t.to_string(),
        out_t.to_string(),
        arg_sort.to_string(),
        bn.to_string(),
        tn.to_string(),
    ];
    let mut defs = String::new();
    defs.push_str(&builtin_template_def(&sort_name, "mb_block_sort", &targs));
    defs.push_str(&builtin_template_def(
        &part_name,
        "mb_block_partition",
        &targs,
    ));
    defs.push_str(&builtin_template_def(&merge_name, "mb_block_merge", &targs));
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SORT_PREAMBLE}{defs}");
    let sort_pl = compile_builtin(mdev, &source, &sort_name)?;
    let part_pl = compile_builtin(mdev, &source, &part_name)?;
    let merge_pl = compile_builtin(mdev, &source, &merge_name)?;

    // Intermediate buffers (shape [n_rows, size_sorted_axis], matching the
    // contiguous input) and the per-row block partitions.
    let nbytes_v = n_rows * size_sorted_axis * in_dt.size_of();
    let nbytes_i = n_rows * size_sorted_axis * idx_dt.size_of();
    let vals0 = mdev.buffer(nbytes_v, "mbsort_vals0")?;
    let vals1 = mdev.buffer(nbytes_v, "mbsort_vals1")?;
    let idxs0 = mdev.buffer(nbytes_i, "mbsort_idxs0")?;
    let idxs1 = mdev.buffer(nbytes_i, "mbsort_idxs1")?;
    let parts = mdev.buffer(n_rows * (n_blocks + 1) * idx_dt.size_of(), "mbsort_parts")?;

    // Non-sorted-axis shape/strides for the block-sort kernel. The input is
    // contiguous and the sorted axis is last, so the non-sorted axes are
    // row-major. `nc_dim == 0` (1-D input) means no row offset.
    let nc_dim = dims.len() - 1;
    let nc_shape: Vec<i32> = if nc_dim == 0 {
        vec![0]
    } else {
        dims[..nc_dim].iter().map(|&d| d as i32).collect()
    };
    let mut nc_str: Vec<i64> = vec![0; nc_dim.max(1)];
    if nc_dim > 0 {
        // Row-major strides of the non-sorted axes; the sorted axis (last) is
        // the innermost, so axis `i`'s stride is the product of `dims[i+1..]`.
        let mut st = size_sorted_axis as i64;
        for i in (0..nc_dim).rev() {
            nc_str[i] = st;
            st *= dims[i] as i64;
        }
    }
    let nc_shape_a = Array::from_slice_dt(mdev, &nc_shape, &[nc_shape.len()], DType::Int32)?;
    let nc_str_a = Array::from_slice_dt(mdev, &nc_str, &[nc_str.len()], DType::Int64)?;

    let ssa = size_sorted_axis as i32;
    let stride: i32 = 1;
    let ncd = nc_dim as i32;
    let nb = n_blocks as i32;

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();

    // Blockwise sort.
    {
        enc.set_pipeline(&sort_pl);
        let (ms, ml) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), ml.offset * in_dt.size_of());
        enc.set_output(1, Some(&vals0), 0);
        enc.set_output(2, Some(&idxs0), 0);
        enc.set_bytes(3, &ssa);
        enc.set_bytes(4, &stride);
        enc.set_bytes(5, &ncd);
        let (ncs, ncl) = nc_shape_a.buffer_and_layout();
        enc.set_input(6, Some(ncs), ncl.offset * DType::Int32.size_of());
        let (nct, ntl) = nc_str_a.buffer_and_layout();
        enc.set_input(7, Some(nct), ntl.offset * DType::Int64.size_of());
        enc.dispatch_groups_size(
            MTLSize {
                width: n_blocks,
                height: n_rows,
                depth: 1,
            },
            MTLSize {
                width: bn,
                height: 1,
                depth: 1,
            },
        );
    }

    // Partition + merge passes. `n_thr` is the partition threadgroup width;
    // for the axis sizes this path serves (a vocabulary row) `n_blocks + 1`
    // is well under the 1024-thread cap, so the partition row stride equals
    // the merge row stride (`n_blocks + 1`).
    let n_thr = (n_blocks + 1).min(1024);
    let mut ping = false;
    let mut merge_tiles = 2usize;
    let mut vals_out = vals1.clone();
    let mut idxs_out = idxs1.clone();
    while merge_tiles / 2 < n_blocks {
        let (vals_in, idxs_in) = if ping {
            (vals1.clone(), idxs1.clone())
        } else {
            (vals0.clone(), idxs0.clone())
        };
        vals_out = if ping { vals0.clone() } else { vals1.clone() };
        idxs_out = if ping { idxs0.clone() } else { idxs1.clone() };
        ping = !ping;

        let mt = merge_tiles as i32;
        enc.set_pipeline(&part_pl);
        enc.set_output(0, Some(&parts), 0);
        enc.set_input(1, Some(&vals_in), 0);
        enc.set_input(2, Some(&idxs_in), 0);
        enc.set_bytes(3, &ssa);
        enc.set_bytes(4, &mt);
        enc.set_bytes(5, &nb);
        enc.dispatch_groups_size(
            MTLSize {
                width: 1,
                height: n_rows,
                depth: 1,
            },
            MTLSize {
                width: n_thr,
                height: 1,
                depth: 1,
            },
        );

        enc.set_pipeline(&merge_pl);
        enc.set_input(0, Some(&parts), 0);
        enc.set_input(1, Some(&vals_in), 0);
        enc.set_input(2, Some(&idxs_in), 0);
        enc.set_output(3, Some(&vals_out), 0);
        enc.set_output(4, Some(&idxs_out), 0);
        enc.set_bytes(5, &ssa);
        enc.set_bytes(6, &mt);
        enc.set_bytes(7, &nb);
        enc.dispatch_groups_size(
            MTLSize {
                width: n_blocks,
                height: n_rows,
                depth: 1,
            },
            MTLSize {
                width: bn,
                height: 1,
                depth: 1,
            },
        );
        merge_tiles *= 2;
    }

    let (buf, dt) = if argsort {
        (idxs_out, idx_dt)
    } else {
        (vals_out, in_dt)
    };
    Ok(Array::from_parts(mdev, buf, dims, dt))
}

fn argsort_axis(device: &Device, x: &Tensor, argsort: bool) -> Result<Tensor> {
    let mdev = device;
    let (_s, layout) = x.buffer_and_layout();
    if x.dims().is_empty() || !layout.is_contiguous() {
        crate::bail!("argsort: non-contiguous or scalar input");
    }
    let dims = x.dims().to_vec();
    let in_dt = x.dtype();
    let size_sorted_axis = *dims.last().unwrap();
    let n_rows: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    // bn/tn selection from gpu_merge_sort.
    let tn = 4usize;
    let potential_bn = size_sorted_axis.div_ceil(tn);
    let mut bn = if potential_bn > 256 {
        512
    } else if potential_bn > 128 {
        256
    } else if potential_bn > 64 {
        128
    } else if potential_bn > 32 {
        64
    } else {
        32
    };
    if bn == 512 && in_dt.size_of() > 4 {
        bn = 256;
    }
    let n_blocks = size_sorted_axis.div_ceil(bn * tn);
    if n_blocks > 1 {
        return argsort_multi_block(
            mdev,
            x,
            &dims,
            n_rows,
            size_sorted_axis,
            bn,
            tn,
            n_blocks,
            argsort,
        );
    }
    let in_t = type_string(in_dt)?;
    let out_t = type_string(DType::U32)?;
    let in_ty = type_to_name(in_dt)?;
    let out_ty = type_to_name(DType::U32)?;
    let mut kernel_name = String::from("c");
    if argsort {
        kernel_name.push_str("arg");
    }
    kernel_name.push_str(&format!("_block_sort_{in_ty}_{out_ty}_bn{bn}_tn{tn}"));
    let lib_name = kernel_name
        .split_once('_')
        .map(|(_, r)| r)
        .unwrap_or(&kernel_name)
        .to_string();
    let a = |s: &str| s.to_string();
    let mut defs = String::new();
    for (prefix, arg_sort) in [("carg_", "true"), ("c_", "false")] {
        defs.push_str(&builtin_template_def(
            &format!("{prefix}{lib_name}"),
            "block_sort",
            &[
                a(in_t),
                a(out_t),
                a(arg_sort),
                bn.to_string(),
                tn.to_string(),
            ],
        ));
        defs.push_str(&builtin_template_def(
            &format!("n{prefix}{lib_name}"),
            "block_sort_nc",
            &[
                a(in_t),
                a(out_t),
                a(arg_sort),
                bn.to_string(),
                tn.to_string(),
            ],
        ));
    }
    let source = format!("{MLX_UTILS_PREAMBLE}{MLX_SORT_PREAMBLE}{defs}");
    let pipeline = compile_builtin(device, &source, &kernel_name)?;

    let count = x.elem_count();
    let obuf = mdev.buffer((count) as usize * (DType::U32).size_of(), "sort_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &dims.clone(), DType::U32);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    {
        let (ms, layout) = x.buffer_and_layout();
        enc.set_input(0, Some(ms), layout.offset * x.dtype().size_of());
    }
    enc.set_output(1, Some(&obuf), 0);
    let ssa = size_sorted_axis as i32;
    enc.set_bytes(2, &ssa);
    let in_stride: i32 = 1;
    enc.set_bytes(3, &in_stride);
    enc.set_bytes(4, &in_stride);
    // contiguous: min non-singleton non-sorted-axis stride (== the row stride)
    let seg: i32 = if dims.len() >= 2 {
        dims[..dims.len() - 1]
            .iter()
            .rposition(|&d| d != 1)
            .map(|i| layout.stride()[i] as i32)
            .unwrap_or(i32::MAX)
    } else {
        i32::MAX
    };
    enc.set_bytes(5, &seg);
    enc.set_bytes(6, &seg);

    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: n_rows,
            depth: 1,
        },
        MTLSize {
            width: bn,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

#[cfg(test)]
mod sort_tests {
    use super::*;
    use crate::runtime::MetalRuntime;
    use std::sync::Arc;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u32 << 24) as f32) - 0.5
    }

    /// Independent host reference: stable ascending sort by value.
    fn host_argsort(data: &[f32]) -> Vec<u32> {
        let mut idx: Vec<u32> = (0..data.len() as u32).collect();
        idx.sort_by(|&a, &b| {
            data[a as usize]
                .partial_cmp(&data[b as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        idx
    }

    #[test]
    fn multiblock_argsort_matches_host_1d() {
        let rt = Arc::new(MetalRuntime::new(4).unwrap());
        let mut seed = 0x1234_5678_9abc_def0u64;
        for n in [5000usize, 248320usize] {
            let data: Vec<f32> = (0..n).map(|_| lcg(&mut seed)).collect();
            let x = Array::from_slice(&rt, &data, &[n]).unwrap();
            let idx = argsort_axis(&rt, &x, true).unwrap();
            let got: Vec<u32> = idx.to_vec().unwrap();
            assert_eq!(got, host_argsort(&data), "argsort mismatch at n={n}");
        }
    }

    #[test]
    fn multiblock_argsort_matches_host_2d() {
        let rt = Arc::new(MetalRuntime::new(4).unwrap());
        let mut seed = 42u64;
        let (r, c) = (5usize, 6000usize);
        let data: Vec<f32> = (0..r * c).map(|_| lcg(&mut seed)).collect();
        let x = Array::from_slice(&rt, &data, &[r, c]).unwrap();
        let idx = argsort_axis(&rt, &x, true).unwrap();
        let got: Vec<u32> = idx.to_vec().unwrap();
        for row in 0..r {
            let base = row * c;
            let expect = host_argsort(&data[base..base + c]);
            assert_eq!(&got[base..base + c], expect.as_slice(), "row {row}");
        }
    }
}
