//! Model-independent host storage and individual ordered f32 operations.
//! No model names, weight names, checkpoint parsing or inference traversal.
use lisa_backend::*;
use rayon::prelude::*;
use std::sync::Arc;

#[derive(Debug)]
pub struct Error(String);
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
type Result<T> = std::result::Result<T, Error>;
fn require(ok: bool, message: &str) -> Result<()> {
    if ok {
        Ok(())
    } else {
        Err(Error(message.into()))
    }
}
#[derive(Clone)]
enum Storage {
    F32(Arc<[f32]>),
    I32(Arc<[i32]>),
}
#[derive(Clone)]
pub struct Tensor {
    shape: Vec<i32>,
    storage: Storage,
    offset: usize,
}
impl Tensor {
    pub fn from_f32(values: Vec<f32>, shape: &[i32]) -> Result<Self> {
        require(
            tensor_bytes(shape, ScalarType::F32) == values.len().checked_mul(4),
            "invalid host tensor shape",
        )?;
        Ok(Self {
            shape: shape.to_vec(),
            storage: Storage::F32(values.into()),
            offset: 0,
        })
    }
    pub fn from_i32(values: Vec<i32>, shape: &[i32]) -> Result<Self> {
        require(
            tensor_bytes(shape, ScalarType::I32) == values.len().checked_mul(4),
            "invalid host index shape",
        )?;
        Ok(Self {
            shape: shape.to_vec(),
            storage: Storage::I32(values.into()),
            offset: 0,
        })
    }
    pub fn f32(&self) -> Result<&[f32]> {
        match &self.storage {
            Storage::F32(v) => Ok(&v[self.offset..self.offset + self.len()]),
            _ => Err(Error("expected f32 tensor".into())),
        }
    }
    fn i32(&self) -> Result<&[i32]> {
        match &self.storage {
            Storage::I32(v) => Ok(&v[self.offset..self.offset + self.len()]),
            _ => Err(Error("expected i32 indices".into())),
        }
    }
    fn len(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }
    fn gather(&self, indices: impl Iterator<Item = usize>, shape: Vec<i32>) -> Result<Self> {
        match &self.storage {
            Storage::F32(v) => {
                Self::from_f32(indices.map(|i| v[self.offset + i]).collect(), &shape)
            }
            Storage::I32(v) => {
                Self::from_i32(indices.map(|i| v[self.offset + i]).collect(), &shape)
            }
        }
    }
}
#[derive(Clone, Copy, Default)]
pub struct CpuDevice;
impl ReadbackDevice for CpuDevice {
    fn read_f32(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        Ok(tensor.f32()?.to_vec())
    }
}
pub struct Handle {
    op: TensorOperation,
}
const FLOAT: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 0,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const ANY: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32, ScalarType::I32],
    ..FLOAT
};
impl Device for CpuDevice {
    type Tensor = Tensor;
    type KernelHandle = Handle;
    type Stream = ();
    type Completion = ();
    type Error = Error;
    fn capabilities(&self) -> &[KernelCapability] {
        static CAPS: std::sync::OnceLock<Vec<KernelCapability>> = std::sync::OnceLock::new();
        CAPS.get_or_init(|| {
            use TensorOperationKind as K;
            let mut caps = Vec::new();
            let mut add = |kind, inputs: &'static [TensorConstraint], variadic| {
                caps.push(KernelCapability {
                    kind: KernelKind::Tensor(kind),
                    law: NumericalLaw::HostSequentialF32,
                    inputs,
                    variadic,
                    shape: ShapeRelation::TensorOperation,
                    encoding: None,
                    workspace: Workspace::ProviderManaged,
                })
            };
            for kind in [K::Reshape, K::Transpose, K::Slice, K::Contiguous, K::Cast] {
                add(kind, &[ANY], false);
            }
            add(K::Concatenate, &[FLOAT], true);
            add(
                K::Take,
                &[
                    ANY,
                    TensorConstraint {
                        dtypes: &[ScalarType::I32],
                        ..FLOAT
                    },
                ],
                false,
            );
            for op in [
                Unary::Negate,
                Unary::Sigmoid,
                Unary::Silu,
                Unary::Exp,
                Unary::Log,
                Unary::Sqrt,
                Unary::Rsqrt,
                Unary::Erf,
                Unary::Log1p,
                Unary::Cos,
                Unary::Sin,
            ] {
                add(K::Unary(op), &[FLOAT], false);
            }
            for op in [
                Binary::Add,
                Binary::Subtract,
                Binary::Multiply,
                Binary::Divide,
                Binary::Maximum,
                Binary::Minimum,
            ] {
                add(K::Binary(op), &[FLOAT, FLOAT], false);
            }
            add(K::Softmax, &[FLOAT], false);
            for bias in [false, true] {
                let inputs = if bias {
                    &[FLOAT, FLOAT, FLOAT][..]
                } else {
                    &[FLOAT, FLOAT][..]
                };
                add(K::SequentialLinear(bias), inputs, false);
                add(K::SequentialLayerNorm(bias), inputs, false);
            }
            add(K::SequentialAttention, &[FLOAT, FLOAT, FLOAT, FLOAT], false);
            caps
        })
    }
    fn dtype(&self, tensor: &Tensor) -> Option<ScalarType> {
        Some(match tensor.storage {
            Storage::F32(_) => ScalarType::F32,
            Storage::I32(_) => ScalarType::I32,
        })
    }
    fn shape<'a>(&self, tensor: &'a Tensor) -> &'a [i32] {
        &tensor.shape
    }
    fn stream(&self) {}
    fn wait(&self, _: &()) -> Result<()> {
        Ok(())
    }
    fn pull(&self, source: TensorSource<'_>, options: ImportOptions) -> Result<Tensor> {
        require(
            options.mode == ImportMode::Copy,
            "host shared import is not implemented",
        )?;
        let h = source
            .host_tensor()
            .filter(|h| h.validate())
            .ok_or_else(|| Error("invalid tensor source".into()))?;
        match h.dtype {
            ScalarType::F32 => Tensor::from_f32(
                h.bytes
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                h.shape,
            ),
            ScalarType::I32 => Tensor::from_i32(
                h.bytes
                    .chunks_exact(4)
                    .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
                    .collect(),
                h.shape,
            ),
            // Imports preserve storage interpretation. Conversion to f32 is an
            // explicit engine adapter responsibility, not an implicit pull cast.
            _ => Err(Error("host import supports f32 and i32 storage".into())),
        }
    }
    fn prepare(&self, kernel: Kernel) -> Result<Option<Handle>> {
        if !self.capabilities().iter().any(|c| c.accepts(&kernel)) {
            return Ok(None);
        }
        let Kernel::HostTensor(op) = kernel else {
            return Ok(None);
        };
        if matches!(op,TensorOperation::Cast(dt) if !matches!(dt,ScalarType::F32|ScalarType::I32)) {
            return Ok(None);
        }
        Ok(Some(Handle { op }))
    }
    fn dispatch(
        &self,
        h: &Handle,
        inputs: &[&Tensor],
        _: &(),
    ) -> Result<Option<Dispatch<Outputs<Tensor>, ()>>> {
        require(inputs.len() == h.op.arity(), "wrong input count")?;
        Ok(Some(Dispatch {
            output: Outputs::One(execute(&h.op, inputs)?),
            completion: (),
        }))
    }
}
fn axis(axis: i32, rank: usize) -> Result<usize> {
    let axis = if axis < 0 { axis + rank as i32 } else { axis };
    require(
        axis >= 0 && (axis as usize) < rank,
        "axis exceeds tensor rank",
    )?;
    Ok(axis as usize)
}
fn count(shape: &[i32]) -> usize {
    shape.iter().map(|&d| d as usize).product()
}
fn strides(shape: &[i32]) -> Vec<usize> {
    let mut stride = 1;
    let mut out = vec![0; shape.len()];
    for i in (0..shape.len()).rev() {
        out[i] = stride;
        stride *= shape[i] as usize;
    }
    out
}
fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let sum: f32 = e.iter().sum();
    e.iter().map(|&v| v / sum.max(1e-30)).collect()
}
fn execute(op: &TensorOperation, args: &[&Tensor]) -> Result<Tensor> {
    use TensorOperation as T;
    let x = args[0];
    match op {
        T::Contiguous => Ok(x.clone()),
        T::Reshape(shape) => {
            let mut shape = shape.clone();
            if let Some(i) = shape.iter().position(|&d| d == -1) {
                let known = shape
                    .iter()
                    .filter(|&&d| d != -1)
                    .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                    .ok_or_else(|| Error("reshape overflow".into()))?;
                require(
                    known > 0 && x.len() % known == 0,
                    "invalid inferred reshape",
                )?;
                shape[i] = i32::try_from(x.len() / known)
                    .map_err(|_| Error("reshape dimension overflow".into()))?;
            }
            require(
                tensor_bytes(&shape, ScalarType::F32) == x.len().checked_mul(4),
                "reshape size differs",
            )?;
            Ok(Tensor {
                shape,
                storage: x.storage.clone(),
                offset: x.offset,
            })
        }
        T::Transpose(axes) => {
            require(axes.len() == x.shape.len(), "transpose rank differs")?;
            let axes = axes
                .iter()
                .map(|&a| axis(a, x.shape.len()))
                .collect::<Result<Vec<_>>>()?;
            let shape = axes.iter().map(|&a| x.shape[a]).collect::<Vec<_>>();
            let old = strides(&x.shape);
            let new = strides(&shape);
            x.gather(
                (0..x.len()).map(|i| {
                    axes.iter()
                        .enumerate()
                        .map(|(j, &a)| (i / new[j]) % (shape[j] as usize) * old[a])
                        .sum()
                }),
                shape.clone(),
            )
        }
        T::Slice {
            axis: a,
            start,
            end,
        } => {
            let a = axis(*a, x.shape.len())?;
            let dim = x.shape[a];
            let bound = |n: i32| {
                if n < 0 {
                    (dim + n).clamp(0, dim)
                } else {
                    n.min(dim)
                }
            };
            let start = bound(*start);
            let end = bound(*end).max(start);
            let mut shape = x.shape.clone();
            shape[a] = end - start;
            let inner = count(&shape[a + 1..]);
            let width = shape[a] as usize * inner;
            let old_width = dim as usize * inner;
            x.gather(
                (0..count(&shape))
                    .map(|i| (i / width) * old_width + start as usize * inner + i % width),
                shape.clone(),
            )
        }
        T::Cast(dtype) => match (dtype, &x.storage) {
            (ScalarType::F32, Storage::F32(_)) | (ScalarType::I32, Storage::I32(_)) => {
                Ok(x.clone())
            }
            (ScalarType::F32, Storage::I32(_)) => {
                Tensor::from_f32(x.i32()?.iter().map(|&v| v as f32).collect(), &x.shape)
            }
            (ScalarType::I32, Storage::F32(_)) => {
                Tensor::from_i32(x.f32()?.iter().map(|&v| v as i32).collect(), &x.shape)
            }
            _ => Err(Error("unsupported host cast".into())),
        },
        T::Unary(op) => Tensor::from_f32(
            x.f32()?
                .iter()
                .map(|&v| match op {
                    Unary::Negate => -v,
                    Unary::Sigmoid => 1.0 / (1.0 + (-v).exp()),
                    Unary::Silu => v / (1.0 + (-v).exp()),
                    Unary::Exp => v.exp(),
                    Unary::Log => v.ln(),
                    Unary::Sqrt => v.sqrt(),
                    Unary::Rsqrt => 1.0 / v.sqrt(),
                    Unary::Erf => libm::erff(v),
                    Unary::Log1p => v.ln_1p(),
                    Unary::Cos => v.cos(),
                    Unary::Sin => v.sin(),
                    Unary::Tanh => v.tanh(),
                })
                .collect(),
            &x.shape,
        ),
        T::Binary(op) => binary(*op, x, args[1]),
        T::Take { axis: a } => {
            let a = axis(*a, x.shape.len())?;
            let ids = args[1].i32()?;
            require(
                ids.iter().all(|&i| i >= 0 && i < x.shape[a]),
                "gather index exceeds dimension",
            )?;
            let inner = count(&x.shape[a + 1..]);
            let outer = count(&x.shape[..a]);
            let width = x.shape[a] as usize;
            let mut shape = x.shape[..a].to_vec();
            shape.extend_from_slice(&args[1].shape);
            shape.extend_from_slice(&x.shape[a + 1..]);
            x.gather(
                (0..outer).flat_map(|o| {
                    ids.iter().flat_map(move |&id| {
                        (0..inner).map(move |i| (o * width + id as usize) * inner + i)
                    })
                }),
                shape,
            )
        }
        T::Concatenate { axis: a, .. } => {
            let a = axis(*a, x.shape.len())?;
            let mut shape = x.shape.clone();
            let mut size = 0i32;
            for t in args {
                require(
                    t.shape.len() == shape.len()
                        && t.shape
                            .iter()
                            .enumerate()
                            .all(|(i, &d)| i == a || d == shape[i]),
                    "concatenate shapes differ",
                )?;
                size = size
                    .checked_add(t.shape[a])
                    .ok_or_else(|| Error("concatenate overflow".into()))?;
            }
            shape[a] = size;
            let outer = count(&shape[..a]);
            let inner = count(&shape[a + 1..]);
            // All current arithmetic paths concatenate floating point values.
            let data = args.iter().map(|a| a.f32()).collect::<Result<Vec<_>>>()?;
            let mut values = Vec::with_capacity(count(&shape));
            for row in 0..outer {
                for (t, v) in args.iter().zip(&data) {
                    let width = t.shape[a] as usize * inner;
                    values.extend_from_slice(&v[row * width..(row + 1) * width]);
                }
            }
            Tensor::from_f32(values, &shape)
        }
        T::Softmax { axis: a } => {
            let a = axis(*a, x.shape.len())?;
            require(
                a + 1 == x.shape.len() && x.shape[a] > 0,
                "host softmax requires a nonempty final axis",
            )?;
            Tensor::from_f32(
                x.f32()?
                    .chunks(x.shape[a] as usize)
                    .flat_map(softmax)
                    .collect(),
                &x.shape,
            )
        }
        T::SequentialLinear { bias } => linear(x, args[1], bias.then(|| args[2])),
        T::SequentialLayerNorm { epsilon, bias } => {
            layer_norm(x, args[1], bias.then(|| args[2]), f32::from_bits(*epsilon))
        }
        T::SequentialAttention => attention(x, args[1], args[2], args[3]),
        _ => Err(Error("unsupported host operation".into())),
    }
}
fn binary(op: Binary, x: &Tensor, y: &Tensor) -> Result<Tensor> {
    let rank = x.shape.len().max(y.shape.len());
    let mut shape = vec![1; rank];
    let mut xs = vec![0; rank];
    let mut ys = vec![0; rank];
    for (t, stride) in [(x, &mut xs), (y, &mut ys)] {
        let raw = strides(&t.shape);
        let offset = rank - t.shape.len();
        for (i, &d) in t.shape.iter().enumerate() {
            let i = i + offset;
            require(
                shape[i] == 1 || d == 1 || shape[i] == d,
                "binary broadcast dimensions differ",
            )?;
            if shape[i] == 1 {
                shape[i] = d;
            }
            if d != 1 {
                stride[i] = raw[i - offset];
            }
        }
    }
    let output_stride = strides(&shape);
    let a = x.f32()?;
    let b = y.f32()?;
    let values = (0..count(&shape))
        .map(|i| {
            let offset = |s: &[usize]| {
                (0..rank)
                    .map(|j| (i / output_stride[j]) % (shape[j] as usize) * s[j])
                    .sum::<usize>()
            };
            let a = a[offset(&xs)];
            let b = b[offset(&ys)];
            match op {
                Binary::Add => a + b,
                Binary::Subtract => a - b,
                Binary::Multiply => a * b,
                Binary::Divide => a / b,
                Binary::Maximum => a.max(b),
                Binary::Minimum => a.min(b),
                Binary::Power => a.powf(b),
            }
        })
        .collect();
    Tensor::from_f32(values, &shape)
}
fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> Result<Tensor> {
    require(
        w.shape.len() == 2
            && !x.shape.is_empty()
            && x.shape.last() == w.shape.last()
            && w.shape[0] > 0
            && w.shape[1] > 0,
        "linear geometry differs",
    )?;
    let input = w.shape[1] as usize;
    let output = w.shape[0] as usize;
    let xdata = x.f32()?;
    let weight = w.f32()?;
    let bias = b.map(Tensor::f32).transpose()?;
    require(
        bias.is_none_or(|b| b.len() == output),
        "linear bias differs",
    )?;
    let mut y = vec![0.0; xdata.len() / input * output];
    y.par_chunks_mut(output).enumerate().for_each(|(r, o)| {
        let xr = &xdata[r * input..(r + 1) * input];
        for (oc, ov) in o.iter_mut().enumerate() {
            let wr = &weight[oc * input..(oc + 1) * input];
            let mut acc = 0.0;
            for k in 0..input {
                acc += xr[k] * wr[k];
            }
            *ov = acc + bias.map_or(0.0, |b| b[oc]);
        }
    });
    let mut shape = x.shape.clone();
    *shape.last_mut().unwrap() = output as i32;
    Tensor::from_f32(y, &shape)
}
fn layer_norm(x: &Tensor, w: &Tensor, b: Option<&Tensor>, eps: f32) -> Result<Tensor> {
    let cols =
        *x.shape
            .last()
            .ok_or_else(|| Error("normalization requires a final axis".into()))? as usize;
    require(
        cols > 0 && w.shape == [cols as i32] && b.is_none_or(|b| b.shape == [cols as i32]),
        "normalization geometry differs",
    )?;
    let input = x.f32()?;
    let weight = w.f32()?;
    let bias = b.map(Tensor::f32).transpose()?;
    let mut y = vec![0.0; input.len()];
    y.par_chunks_mut(cols).enumerate().for_each(|(r, o)| {
        let xr = &input[r * cols..(r + 1) * cols];
        let mean = xr.iter().sum::<f32>() / cols as f32;
        let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for i in 0..cols {
            let n = (xr[i] - mean) * inv;
            o[i] = n * weight[i] + bias.map_or(0.0, |b| b[i]);
        }
    });
    Tensor::from_f32(y, &x.shape)
}
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    require(
        q.shape.len() == 3 && k.shape == q.shape && v.shape == q.shape,
        "ordered attention requires equal [heads,S,D] shapes",
    )?;
    let (heads, s, d) = (
        q.shape[0] as usize,
        q.shape[1] as usize,
        q.shape[2] as usize,
    );
    require(
        s > 0 && d > 0 && mask.shape == [s as i32, s as i32],
        "invalid attention mask or dimensions",
    )?;
    let (q, k, v, mask) = (q.f32()?, k.f32()?, v.f32()?, mask.f32()?);
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0; heads * s * d];
    out.par_chunks_mut(s * d).enumerate().for_each(|(h, o)| {
        let qh = &q[h * s * d..(h + 1) * s * d];
        let kh = &k[h * s * d..(h + 1) * s * d];
        let vh = &v[h * s * d..(h + 1) * s * d];
        let mut scores = vec![0.0; s];
        for qi in 0..s {
            let qr = &qh[qi * d..(qi + 1) * d];
            for ki in 0..s {
                let kr = &kh[ki * d..(ki + 1) * d];
                let mut acc = 0.0;
                for di in 0..d {
                    acc += qr[di] * kr[di];
                }
                scores[ki] = acc * scale + mask[qi * s + ki];
            }
            let p = softmax(&scores);
            let row = &mut o[qi * d..(qi + 1) * d];
            for ki in 0..s {
                let pv = p[ki];
                let vr = &vh[ki * d..(ki + 1) * d];
                for di in 0..d {
                    row[di] += pv * vr[di];
                }
            }
        }
    });
    Tensor::from_f32(out, &[heads as i32, s as i32, d as i32])
}

#[cfg(test)]
mod tests {
    use super::*;
    fn run(op: TensorOperation, args: &[&Tensor]) -> Result<Tensor> {
        let handle = CpuDevice.prepare(Kernel::HostTensor(op))?.unwrap();
        CpuDevice
            .dispatch(&handle, args, &())?
            .unwrap()
            .output
            .into_one()
            .map_err(|e| Error(e.into()))
    }
    #[test]
    fn import_is_exact_and_capabilities_do_not_claim_gpu_arithmetic() {
        let values = [0x80000000u32, 0x3f800001, 0x7fc01234];
        let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let tensor = CpuDevice
            .pull(
                HostTensor {
                    bytes: &bytes,
                    shape: &[3],
                    dtype: ScalarType::F32,
                }
                .into(),
                ImportOptions::default(),
            )
            .unwrap();
        assert_eq!(
            tensor
                .f32()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values
        );
        assert!(
            CpuDevice
                .prepare(Kernel::Tensor(TensorOperation::Unary(Unary::Negate)))
                .unwrap()
                .is_none()
        );
        assert!(
            CpuDevice
                .prepare(Kernel::HostTensor(TensorOperation::Transpose(vec![0, 0])))
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn tensor_layout_and_broadcast_follow_logical_axes() {
        let x = Tensor::from_f32(vec![0., 1., 2., 3., 4., 5.], &[2, 3]).unwrap();
        let t = run(TensorOperation::Transpose(vec![1, 0]), &[&x]).unwrap();
        assert_eq!(t.f32().unwrap(), &[0., 3., 1., 4., 2., 5.]);
        let t = run(
            TensorOperation::Slice {
                axis: -1,
                start: 1,
                end: 3,
            },
            &[&x],
        )
        .unwrap();
        assert_eq!(t.f32().unwrap(), &[1., 2., 4., 5.]);
        let indices = Tensor::from_i32(vec![1, 0], &[2]).unwrap();
        let t = run(TensorOperation::Take { axis: 0 }, &[&x, &indices]).unwrap();
        assert_eq!(t.f32().unwrap(), &[3., 4., 5., 0., 1., 2.]);
        let y = Tensor::from_f32(vec![10., 20., 30.], &[3]).unwrap();
        let t = run(TensorOperation::Binary(Binary::Add), &[&x, &y]).unwrap();
        assert_eq!(t.f32().unwrap(), &[10., 21., 32., 13., 24., 35.]);
        let invalid = Tensor::from_i32(vec![2], &[1]).unwrap();
        assert!(run(TensorOperation::Take { axis: 0 }, &[&x, &invalid]).is_err());
    }
}

impl lisa_backend::HostAccessDevice for CpuDevice {
    fn read_bytes(&self, tensor: &Tensor) -> anyhow::Result<Vec<u8>> {
        Ok(match &tensor.storage {
            Storage::F32(_) => tensor.f32()?.iter().flat_map(|v| v.to_ne_bytes()).collect(),
            Storage::I32(_) => tensor.i32()?.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        })
    }
    fn layout(&self, tensor: &Tensor) -> lisa_backend::TensorLayout {
        let mut stride = 1;
        let mut strides = vec![0; tensor.shape.len()];
        for (i, dim) in tensor.shape.iter().enumerate().rev() {
            strides[i] = stride;
            stride *= *dim as usize;
        }
        lisa_backend::TensorLayout {
            strides,
            offset: tensor.offset,
        }
    }
    fn evaluate(&self, _: &Tensor, _: bool) -> anyhow::Result<()> {
        Ok(())
    }
}

impl StorageDevice for CpuDevice {
    fn max_allocation_bytes(&self) -> usize {
        isize::MAX as usize
    }
    fn byte_view(
        &self,
        source: &Tensor,
        offset: usize,
        shape: &[i32],
        dtype: ScalarType,
    ) -> Result<Tensor> {
        require(
            self.dtype(source) == Some(dtype),
            "CPU byte views require the existing scalar interpretation",
        )?;
        let bytes =
            tensor_bytes(shape, dtype).ok_or_else(|| Error("invalid byte view shape".into()))?;
        require(
            offset % dtype.size_of() == 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|n| n <= source.len() * dtype.size_of()),
            "byte view exceeds source",
        )?;
        Ok(Tensor {
            shape: shape.to_vec(),
            storage: source.storage.clone(),
            offset: source.offset + offset / dtype.size_of(),
        })
    }
}

#[cfg(test)]
mod erased_device_tests {
    use super::*;
    use lisa_backend::erased::DeviceHandle;
    #[test]
    fn selected_device_preserves_views_and_rejects_foreign_handles() -> anyhow::Result<()> {
        let device = DeviceHandle::new(CpuDevice);
        let other = DeviceHandle::new(CpuDevice);
        assert!(
            device
                .prepare(Kernel::Tensor(TensorOperation::Cast(ScalarType::I32)))?
                .is_none()
        );
        let bytes: Vec<u8> = [99.0f32, 1.5, 2.5, 88.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let tensor = device.pull(
            HostTensor {
                bytes: &bytes,
                shape: &[4],
                dtype: ScalarType::F32,
            }
            .into(),
            Default::default(),
        )?;
        let view = device.byte_view(&tensor, 4, &[2], ScalarType::F32)?;
        drop(tensor);
        assert_eq!(device.read_f32(&view)?, [1.5, 2.5]);
        assert_eq!(device.read_f32_through(&view)?, [1.5, 2.5]);
        assert!(other.read_f32_through(&view).is_err());
        assert!(device.byte_view(&view, 4, &[2], ScalarType::F32).is_err());
        let kernel = device
            .prepare(Kernel::HostTensor(TensorOperation::Cast(ScalarType::I32)))?
            .unwrap();
        let result = device
            .dispatch(&kernel, &[&view], &device.stream())?
            .unwrap();
        device.wait(&result.completion)?;
        assert!(other.wait(&result.completion).is_err());
        let output = result.output.into_one().unwrap();
        assert_eq!(
            device.read_bytes(&output)?,
            [1i32, 2]
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>()
        );
        assert!(other.read_bytes(&output).is_err());
        assert!(device.dispatch(&kernel, &[&view], &other.stream()).is_err());
        assert!(other.dispatch(&kernel, &[&view], &other.stream()).is_err());
        let other_kernel = other
            .prepare(Kernel::HostTensor(TensorOperation::Cast(ScalarType::I32)))?
            .unwrap();
        assert!(
            other
                .dispatch(&other_kernel, &[&view], &other.stream())
                .is_err()
        );
        assert!(
            device
                .clone()
                .dispatch(&kernel, &[&view], &device.stream())?
                .is_some()
        );
        Ok(())
    }
}
