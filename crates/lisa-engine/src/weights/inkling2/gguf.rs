//! Inkling GGUF loading adapter. All checkpoint and option reads finish here;
//! the binding receives typed tensor views, the vendored planes repacked
//! beside the original payloads, and explicit controls.
use crate::{
    bindings::{
        BindingDevice,
        inkling2::{Prepared, Tower},
    },
    language::runtime::Runtime,
    sampling::sampler::Sampler,
    weights::{
        gguf::{ResidentWeights, load_resident},
        source::PreparedTensors,
    },
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ManagedDevice, ReadbackDevice, ScalarType as Dtype, TensorAccess};
use lisa_gguf::{Package, TensorType as Type};
use lisa_models::inkling2::planes::{self, Format};
use std::{collections::BTreeMap, path::Path};

/// The oldest Apple GPU family the step's kernels run on: Apple7, the M1
/// GPUs. The pinned reference supports Apple9 (M3) and newer; its kernels
/// need the Metal 4 language, which every Apple7 or newer GPU runs from OS
/// 26, and M1/M2 (Apple7/8) take the calibration's Apple9 rows.
pub const MINIMUM_GPU_FAMILY: u32 = 7;

pub(crate) fn load<D: BindingDevice + ManagedDevice + ReadbackDevice + 'static>(
    runtime: Runtime<D, Sampler<D>>,
    path: &Path,
) -> Result<Tower<D>>
where
    D::Error: From<anyhow::Error>,
{
    let device = runtime.context.device();
    let family = device.apple_gpu_family().unwrap_or(0);
    ensure!(
        family >= MINIMUM_GPU_FAMILY,
        "Inkling 2 needs an Apple M1 or newer GPU (Apple GPU family {MINIMUM_GPU_FAMILY}); \
         this device's family is {family}"
    );
    ensure!(
        device.metal4(),
        "Inkling 2's kernels need Metal 4: update to macOS 26 or iOS 26 or later"
    );
    let package = Package::open(path)?;
    let config = crate::checkpoint::inkling2::config(&package.header)?;
    let formats = formats(&package)?;
    ensure!(
        formats["token_embd.weight"].projection == lisa_models::packed::Projection::Q8,
        "the Inkling step gathers a Q8_0 token table"
    );
    let device = runtime.context.device();
    let names = formats.keys().map(String::as_str).collect::<Vec<_>>();
    let resident = load_resident(device.clone(), &package, &names)?;
    let weights = prepare_tensors(device, resident, &formats)?;
    let vision = Box::new(crate::weights::inkling2::vision::LazyEncoder::new(
        runtime.context.clone(),
        path.parent()
            .context("missing Inkling model directory")?
            .join(crate::checkpoint::catalog::inkling2::PROJECTOR),
    ));
    // The DFlash2 draft, when downloaded, drafts the tree rounds.
    let target_bytes = std::fs::metadata(path)?.len() as usize;
    let draft = crate::weights::inkling2::dflash::selection(path)?
        .map(|p| crate::weights::inkling2::dflash::load(device, &p, target_bytes))
        .transpose()?;
    Tower::from_prepared(
        runtime,
        Prepared {
            config,
            weights,
            formats,
            vision,
            draft,
        },
    )
}

fn prepare_tensors<D: BindingDevice + ManagedDevice>(
    device: &D,
    resident: ResidentWeights<D, D::Reservation>,
    formats: &BTreeMap<String, Format>,
) -> Result<PreparedTensors<D::Tensor, D::Reservation>> {
    let (reservation, original) = resident.into_parts();
    let mut tensors = BTreeMap::new();
    for (name, source) in original {
        let format = formats[&name];
        let layout = source.prepared_layout()?;
        let tensor = if format.dtype == Dtype::U8 {
            source.packed_bytes().clone()
        } else {
            source.plain_view()?
        };
        ensure!(
            tensor.shape() == layout.storage_shape && device.dtype(&tensor) == Some(layout.dtype),
            "prepared Inkling tensor layout mismatch: {name}"
        );
        tensors.insert(name, tensor);
    }
    let ops = crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native);
    prepare_expert_planes(device, &ops, formats, &mut tensors)?;
    prepare_dense_planes(&ops, formats, &mut tensors)?;
    prepare_message_queries(&ops, &mut tensors)?;
    Ok(PreparedTensors {
        _reservation: reservation,
        tensors,
    })
}

/// Each layer's routed experts repacked once into the vendored reference's
/// planes ([`planes::EXPERTS`]), beside the original payloads, with the
/// `[3, 128]` scale table the step's expert passes apply
/// ([`planes::EXPERT_SCALES`]): gate and up take the fused gate/up scale,
/// down its own.
fn prepare_expert_planes<D: BindingDevice + ManagedDevice>(
    device: &D,
    ops: &crate::tensor_ops::TensorOps<'_, D>,
    formats: &BTreeMap<String, Format>,
    tensors: &mut BTreeMap<String, D::Tensor>,
) -> Result<()> {
    use lisa_backend::{Kernel, PackedCompute as P};
    let prefixes = tensors
        .keys()
        .filter_map(|name| name.strip_suffix("ffn_gate_up_exps.weight"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let ones = device.pull(
        lisa_backend::HostTensor {
            bytes: &[1.0f32; 128]
                .iter()
                .flat_map(|x| x.to_ne_bytes())
                .collect::<Vec<_>>(),
            shape: &[128],
            dtype: Dtype::F32,
        }
        .into(),
        lisa_backend::ImportOptions::default(),
    )?;
    for prefix in prefixes {
        let gate_up_name = format!("{prefix}ffn_gate_up_exps.weight");
        let down_name = format!("{prefix}ffn_down_exps.weight");
        let gate_type = planes::ggml_type(formats[&gate_up_name].projection)
            .context("unsupported Inkling expert format")?;
        let down_type = planes::ggml_type(formats[&down_name].projection)
            .context("unsupported Inkling expert format")?;
        let gate_up = tensors[&gate_up_name].clone();
        let down = tensors[&down_name].clone();
        let planes = [
            (&gate_up, gate_type, 0, 1152, 2304),
            (&gate_up, gate_type, 1152, 1152, 2304),
            (&down, down_type, 0, 2304, 1152),
        ];
        for ((source, ggml_type, first_row, rows, columns), name) in
            planes.into_iter().zip(planes::EXPERTS)
        {
            let prepared = ops.required(
                Kernel::PackedCompute(P::RepackPlanes {
                    ggml_type,
                    first_row,
                    rows,
                    columns,
                }),
                &[source],
            )?;
            tensors.insert(format!("{prefix}{name}"), prepared);
        }
        let scale = tensors
            .get(&format!("{prefix}ffn_gate_up_exps.scale"))
            .cloned()
            .unwrap_or_else(|| ones.clone());
        // The reference's expert passes apply every scale themselves, down
        // included (its combine applies none).
        let down_scale = tensors
            .get(&format!("{prefix}ffn_down_exps.scale"))
            .cloned()
            .unwrap_or_else(|| ones.clone());
        let table =
            D::Tensor::concatenate(&[&scale, &scale, &down_scale], 0)?.reshape(&[3, 128])?;
        tensors.insert(format!("{prefix}{}", planes::EXPERT_SCALES), table);
    }
    Ok(())
}

/// Each loop layer's message queries in the order the reference's step
/// reads them (q, k, v, FFN): rows 2i and 2i + 1 of `loop_res_query` (q,
/// FFN) and of `loop_kv_res_query` (k, v), i = layer - 6, as F32
/// [`planes::MESSAGE_QUERIES`].
fn prepare_message_queries<D: BindingDevice + ManagedDevice>(
    ops: &crate::tensor_ops::TensorOps<'_, D>,
    tensors: &mut BTreeMap<String, D::Tensor>,
) -> Result<()> {
    let float = |name: &str| -> Result<D::Tensor> {
        let tensor = tensors
            .get(name)
            .with_context(|| format!("missing Inkling tensor {name}"))?;
        ensure!(
            tensor.shape() == [24, 2304],
            "Inkling message queries {name} are not [24, 2304]"
        );
        if ops.dtype(tensor) == Some(Dtype::F32) {
            return Ok(tensor.clone());
        }
        let wide = ops.op(lisa_backend::TensorOperation::Cast(Dtype::F32), &[tensor])?;
        ops.op(lisa_backend::TensorOperation::Contiguous, &[&wide])
    };
    let residual = float("loop_res_query.weight")?;
    let key_value = float("loop_kv_res_query.weight")?;
    for layer in 6..18 {
        let q = 2 * (layer - 6);
        let row = |t: &D::Tensor, r: i32| t.slice(0, r, Some(r + 1));
        let rows = [
            row(&residual, q)?,
            row(&key_value, q)?,
            row(&key_value, q + 1)?,
            row(&residual, q + 1)?,
        ];
        let queries = D::Tensor::concatenate(&rows.each_ref(), 0)?;
        let queries = ops.op(lisa_backend::TensorOperation::Contiguous, &[&queries])?;
        tensors.insert(format!("blk.{layer}.{}", planes::MESSAGE_QUERIES), queries);
    }
    Ok(())
}

/// Dense Q8_0 attention projections and the head repacked once into the
/// vendored reference's planes ([`planes::dense_name`]), beside the
/// originals.
fn prepare_dense_planes<D: BindingDevice + ManagedDevice>(
    ops: &crate::tensor_ops::TensorOps<'_, D>,
    formats: &BTreeMap<String, Format>,
    tensors: &mut BTreeMap<String, D::Tensor>,
) -> Result<()> {
    use lisa_backend::{Kernel, PackedCompute as P};
    let names = tensors
        .keys()
        .filter(|name| {
            (*name == "output.weight"
                || planes::DENSE
                    .iter()
                    .any(|s| name.ends_with(&format!(".{s}"))))
                && formats
                    .get(*name)
                    .is_some_and(|f| f.projection == lisa_models::packed::Projection::Q8)
        })
        .cloned()
        .collect::<Vec<_>>();
    for name in names {
        let source = &tensors[&name];
        let (rows, columns) = (
            source.dim(0) as usize,
            formats[&name].row_bytes as usize / 34 * 32,
        );
        let source = source.reshape(&[1, rows as i32, -1])?;
        let prepared = ops.required(
            Kernel::PackedCompute(P::RepackPlanes {
                ggml_type: 8,
                first_row: 0,
                rows: rows as u32,
                columns: columns as u32,
            }),
            &[&source],
        )?;
        tensors.insert(planes::dense_name(&name), prepared);
    }
    Ok(())
}

fn formats(package: &Package) -> Result<BTreeMap<String, Format>> {
    let mut result = BTreeMap::new();
    for (name, shape) in lisa_models::inkling2::config::tensors() {
        if name.ends_with(".scale") && package.tensor(&name).is_err() {
            let weight = name.replace(".scale", ".weight");
            ensure!(
                package.tensor(&weight)?.info.kind != Type::NVFP4,
                "missing Inkling NVFP4 scale {name}"
            );
            continue;
        }
        let t = &package.tensor(&name)?.info;
        ensure!(
            t.dimensions == shape,
            "Inkling tensor {name} shape mismatch: {:?}, expected {shape:?}",
            t.dimensions
        );
        let layout = crate::weights::gguf::prepared_layout(t)?;
        result.insert(
            name,
            Format {
                projection: layout.projection,
                dtype: layout.dtype,
                row_bytes: layout.row_bytes,
            },
        );
    }
    Ok(result)
}
