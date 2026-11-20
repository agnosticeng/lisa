//! Shape-carrying expressions that construct dataflow, never device tensors.
//! The engine resolves weight requests and executes the resulting graph.
use crate::graph::{Builder, Graph, Value};
use anyhow::{Context, ensure};
use lisa_backend::{Binary, Kernel, ScalarType as Dtype, TensorOperation as T, Unary};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub shape: Vec<i32>,
    pub dtype: Dtype,
}
pub type Schema = BTreeMap<String, TensorInfo>;
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeightLayout {
    Original,
    TransposedDense,
}
impl WeightLayout {
    pub fn preparation(self) -> Option<Graph> {
        match self {
            Self::Original => None,
            Self::TransposedDense => {
                let mut b = Builder::new(1, 0, 0);
                let v = b.one(
                    Kernel::Tensor(T::Transpose(vec![1, 0])),
                    vec![Value::Input(0)],
                );
                let v = b.one(Kernel::Tensor(T::Contiguous), vec![v]);
                Some(b.finish(vec![v]))
            }
        }
    }
}
pub struct Weight {
    pub name: String,
    pub layout: WeightLayout,
}
pub struct Program {
    pub graph: Graph,
    pub weights: Vec<Weight>,
}
struct State {
    graph: Option<Builder>,
    inputs: usize,
    weights: Vec<Weight>,
}
#[derive(Clone)]
pub struct Trace(Rc<RefCell<State>>);
#[derive(Clone)]
pub struct Tensor {
    trace: Trace,
    value: Value,
    shape: Vec<i32>,
    dtype: Dtype,
}
impl Trace {
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(State {
            graph: Some(Builder::new(0, 0, 0)),
            inputs: 0,
            weights: vec![],
        })))
    }
    fn tensor(&self, value: Value, shape: Vec<i32>, dtype: Dtype) -> Tensor {
        Tensor {
            trace: self.clone(),
            value,
            shape,
            dtype,
        }
    }
    pub fn input(&self, shape: &[i32], dtype: Dtype) -> Tensor {
        let mut s = self.0.borrow_mut();
        let i = s.inputs;
        s.inputs += 1;
        self.tensor(Value::Input(i), shape.to_vec(), dtype)
    }
    pub fn constant(&self, values: &[f32], shape: &[i32]) -> Tensor {
        let v = self
            .0
            .borrow_mut()
            .graph
            .as_mut()
            .unwrap()
            .constant_f32(values, shape);
        self.tensor(v, shape.to_vec(), Dtype::F32)
    }
    pub fn constant_bf16(&self, values: &[f32], shape: &[i32]) -> Tensor {
        let v = self
            .0
            .borrow_mut()
            .graph
            .as_mut()
            .unwrap()
            .constant_bf16(values, shape);
        self.tensor(v, shape.to_vec(), Dtype::BF16)
    }
    pub fn weight(&self, name: &str, info: &TensorInfo, transposed: bool) -> Tensor {
        let layout = if transposed {
            WeightLayout::TransposedDense
        } else {
            WeightLayout::Original
        };
        let mut s = self.0.borrow_mut();
        let i = if let Some(i) = s
            .weights
            .iter()
            .position(|w| w.name == name && w.layout == layout)
        {
            i
        } else {
            s.weights.push(Weight {
                name: name.into(),
                layout,
            });
            s.weights.len() - 1
        };
        let mut shape = info.shape.clone();
        if transposed {
            shape.swap(0, 1);
        }
        self.tensor(Value::Weight(i), shape, info.dtype)
    }
    pub fn finish(self, outputs: &[Tensor]) -> anyhow::Result<Program> {
        ensure!(
            outputs.iter().all(|v| Rc::ptr_eq(&self.0, &v.trace.0)),
            "mixed model graph contexts"
        );
        let mut s = self.0.borrow_mut();
        let mut graph = s
            .graph
            .take()
            .context("graph already finished")?
            .finish(outputs.iter().map(|v| v.value).collect::<Vec<_>>());
        graph.inputs = s.inputs;
        graph.weights = s.weights.len();
        graph.validate().map_err(anyhow::Error::msg)?;
        Ok(Program {
            graph,
            weights: std::mem::take(&mut s.weights),
        })
    }
}
fn promote(a: Dtype, b: Dtype) -> Dtype {
    let rank = |d| match d {
        Dtype::F32 => 5,
        Dtype::BF16 => 4,
        Dtype::F16 => 3,
        Dtype::I64 => 2,
        Dtype::U32 | Dtype::I32 => 1,
        _ => 0,
    };
    if rank(a) >= rank(b) { a } else { b }
}
fn broadcast(a: &[i32], b: &[i32]) -> anyhow::Result<Vec<i32>> {
    let rank = a.len().max(b.len());
    let mut out = vec![1; rank];
    for i in 0..rank {
        let x = a.iter().rev().nth(i).copied().unwrap_or(1);
        let y = b.iter().rev().nth(i).copied().unwrap_or(1);
        ensure!(x == y || x == 1 || y == 1, "invalid model broadcast");
        out[rank - 1 - i] = if x == 1 { y } else { x };
    }
    Ok(out)
}
impl Tensor {
    pub fn shape(&self) -> &[i32] {
        &self.shape
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub fn dim(&self, axis: i32) -> i32 {
        self.shape[self.axis(axis).expect("model axis")]
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn axis(&self, axis: i32) -> anyhow::Result<usize> {
        let a = if axis < 0 {
            self.rank() as i32 + axis
        } else {
            axis
        };
        ensure!(a >= 0 && a < self.rank() as i32, "model axis outside rank");
        Ok(a as usize)
    }
    fn record(
        &self,
        kernel: Kernel,
        inputs: &[&Tensor],
        shape: Vec<i32>,
        dtype: Dtype,
    ) -> anyhow::Result<Self> {
        ensure!(
            inputs.iter().all(|x| Rc::ptr_eq(&self.trace.0, &x.trace.0)),
            "mixed model graph contexts"
        );
        let v = self
            .trace
            .0
            .borrow_mut()
            .graph
            .as_mut()
            .unwrap()
            .one(kernel, inputs.iter().map(|x| x.value).collect::<Vec<_>>());
        Ok(self.trace.tensor(v, shape, dtype))
    }
    fn op(&self, op: T, shape: Vec<i32>, dtype: Dtype) -> anyhow::Result<Self> {
        self.record(Kernel::Tensor(op), &[self], shape, dtype)
    }
    pub fn scalar_bf16(&self, value: f32) -> Self {
        self.trace.constant_bf16(&[value], &[])
    }
    pub fn scalar(&self, value: f32) -> Self {
        self.trace.constant(&[value], &[])
    }
    pub fn reshape(&self, shape: &[i32]) -> anyhow::Result<Self> {
        ensure!(
            shape.iter().all(|&n| n >= -1) && shape.iter().filter(|&&n| n == -1).count() <= 1,
            "invalid symbolic reshape"
        );
        let total = self
            .shape
            .iter()
            .try_fold(1i64, |n, &d| n.checked_mul(d as i64))
            .context("tensor size overflow")?;
        let known = shape
            .iter()
            .filter(|&&n| n != -1)
            .try_fold(1i64, |n, &d| n.checked_mul(d as i64))
            .context("reshape size overflow")?;
        let mut resolved = shape.to_vec();
        if let Some(i) = resolved.iter().position(|&n| n == -1) {
            ensure!(
                known > 0 && total % known == 0,
                "invalid inferred dimension"
            );
            resolved[i] = i32::try_from(total / known)?;
        } else {
            ensure!(total == known, "reshape size differs");
        }
        self.op(T::Reshape(shape.to_vec()), resolved, self.dtype)
    }
    pub fn transpose_axes(&self, axes: &[i32]) -> anyhow::Result<Self> {
        ensure!(
            axes.len() == self.rank() && T::Transpose(axes.to_vec()).validate(),
            "invalid model transpose"
        );
        let shape = axes
            .iter()
            .map(|&a| self.axis(a).map(|i| self.shape[i]))
            .collect::<anyhow::Result<Vec<_>>>()?;
        self.op(T::Transpose(axes.to_vec()), shape, self.dtype)
    }
    pub fn slice(&self, axis: i32, start: i32, end: i32) -> anyhow::Result<Self> {
        let a = self.axis(axis)?;
        ensure!(
            start >= 0 && end >= start && end <= self.shape[a],
            "invalid model slice"
        );
        let mut shape = self.shape.clone();
        shape[a] = end - start;
        self.op(T::Slice { axis, start, end }, shape, self.dtype)
    }
    pub fn contiguous(&self) -> anyhow::Result<Self> {
        self.op(T::Contiguous, self.shape.clone(), self.dtype)
    }
    pub fn as_dtype(&self, dtype: Dtype) -> anyhow::Result<Self> {
        self.op(T::Cast(dtype), self.shape.clone(), dtype)
    }
    pub fn expand_dims(&self, axis: i32) -> anyhow::Result<Self> {
        let a = if axis < 0 {
            self.rank() as i32 + 1 + axis
        } else {
            axis
        };
        ensure!(a >= 0 && a <= self.rank() as i32, "invalid expand axis");
        let mut shape = self.shape.clone();
        shape.insert(a as usize, 1);
        self.reshape(&shape)
    }
    pub fn broadcast_to(&self, shape: &[i32]) -> anyhow::Result<Self> {
        ensure!(
            broadcast(&self.shape, shape)? == shape,
            "invalid model expansion"
        );
        self.op(T::Broadcast(shape.to_vec()), shape.to_vec(), self.dtype)
    }
    fn binary(&self, rhs: &Self, op: Binary) -> anyhow::Result<Self> {
        self.record(
            Kernel::Tensor(T::Binary(op)),
            &[self, rhs],
            broadcast(&self.shape, &rhs.shape)?,
            promote(self.dtype, rhs.dtype),
        )
    }
    pub fn add(&self, rhs: &Self) -> anyhow::Result<Self> {
        self.binary(rhs, Binary::Add)
    }
    pub fn subtract(&self, rhs: &Self) -> anyhow::Result<Self> {
        self.binary(rhs, Binary::Subtract)
    }
    pub fn multiply(&self, rhs: &Self) -> anyhow::Result<Self> {
        self.binary(rhs, Binary::Multiply)
    }
    pub fn sigmoid(&self) -> anyhow::Result<Self> {
        self.op(T::Unary(Unary::Sigmoid), self.shape.clone(), self.dtype)
    }
    pub fn rsqrt(&self) -> anyhow::Result<Self> {
        self.op(T::Unary(Unary::Rsqrt), self.shape.clone(), self.dtype)
    }
    pub fn mean_axis(&self, axis: i32, keep_dims: bool) -> anyhow::Result<Self> {
        let a = self.axis(axis)?;
        let mut shape = self.shape.clone();
        if keep_dims {
            shape[a] = 1;
        } else {
            shape.remove(a);
        }
        self.op(T::Mean { axis, keep_dims }, shape, self.dtype)
    }
    pub fn take_axis(&self, indices: &Self, axis: i32) -> anyhow::Result<Self> {
        let a = self.axis(axis)?;
        let mut shape = self.shape[..a].to_vec();
        shape.extend_from_slice(indices.shape());
        shape.extend_from_slice(&self.shape[a + 1..]);
        self.record(
            Kernel::Tensor(T::Take { axis }),
            &[self, indices],
            shape,
            self.dtype,
        )
    }
    pub fn matmul(&self, rhs: &Self) -> anyhow::Result<Self> {
        ensure!(
            self.rank() >= 2 && rhs.rank() >= 2 && self.dim(-1) == rhs.dim(-2),
            "invalid symbolic matrix product"
        );
        let mut shape = broadcast(&self.shape[..self.rank() - 2], &rhs.shape[..rhs.rank() - 2])?;
        shape.extend([self.dim(-2), rhs.dim(-1)]);
        self.record(
            Kernel::Tensor(T::Matmul),
            &[self, rhs],
            shape,
            promote(self.dtype, rhs.dtype),
        )
    }
    pub fn rms(&self, weight: &Self, epsilon: f32) -> anyhow::Result<Self> {
        self.record(
            Kernel::Tensor(T::RmsNorm {
                epsilon: epsilon.to_bits(),
            }),
            &[self, weight],
            self.shape.clone(),
            self.dtype,
        )
    }
    pub fn rotary(&self, dimensions: i32, base: f32, offset: i32) -> anyhow::Result<Self> {
        self.op(
            T::Rotary {
                dimensions,
                base: base.to_bits(),
                offset,
            },
            self.shape.clone(),
            self.dtype,
        )
    }
    pub fn affine(&self, weight: &Self, scales: &Self, biases: &Self) -> anyhow::Result<Self> {
        ensure!(
            weight.rank() == 2
                && scales.rank() == 2
                && biases.shape == scales.shape
                && scales.dim(0) == weight.dim(0)
                && weight.dim(-1) > 0
                && scales.dim(-1) > 0,
            "invalid FLUX affine metadata"
        );
        let k = self.dim(-1);
        ensure!(k > 0, "empty FLUX projection input");
        let bits = weight
            .dim(-1)
            .checked_mul(32)
            .context("packed width overflow")?
            / k;
        let group_size = k / scales.dim(-1);
        ensure!(
            matches!((bits, group_size), (4, 64) | (1, 128))
                && weight.dim(-1) * 32 == k * bits
                && scales.dim(-1) * group_size == k,
            "invalid FLUX affine geometry"
        );
        let mut shape = self.shape.clone();
        *shape.last_mut().unwrap() = weight.dim(0);
        {
            let dtype = promote(self.dtype, scales.dtype);
            self.record(
                Kernel::AffineLinear { group_size, bits },
                &[
                    &self.as_dtype(dtype)?,
                    weight,
                    &scales.as_dtype(dtype)?,
                    &biases.as_dtype(dtype)?,
                ],
                shape,
                dtype,
            )
        }
    }
    pub fn dequantize(&self, scales: &Self, biases: &Self) -> anyhow::Result<Self> {
        let mut shape = self.shape.clone();
        *shape.last_mut().context("packed tensor rank")? *= 8;
        self.record(
            Kernel::AffineDequantize {
                group_size: 64,
                bits: 4,
            },
            &[self, scales, biases],
            shape,
            scales.dtype,
        )
    }
}
pub fn concatenate(inputs: &[&Tensor], axis: i32) -> anyhow::Result<Tensor> {
    let first = inputs.first().context("empty symbolic concatenation")?;
    let a = first.axis(axis)?;
    let mut shape = first.shape.clone();
    shape[a] = 0;
    let dtype = first.dtype;
    for input in inputs {
        ensure!(
            input.rank() == first.rank()
                && input
                    .shape
                    .iter()
                    .enumerate()
                    .all(|(i, &d)| i == a || d == first.shape[i]),
            "invalid symbolic concatenation"
        );
        shape[a] = shape[a]
            .checked_add(input.shape[a])
            .context("concatenation overflow")?;
        ensure!(dtype == input.dtype, "concatenation requires equal dtypes");
    }
    first.record(
        Kernel::Tensor(T::Concatenate {
            axis,
            inputs: inputs.len(),
        }),
        inputs,
        shape,
        dtype,
    )
}
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> anyhow::Result<Tensor> {
    ensure!(
        [k, v]
            .into_iter()
            .chain(mask)
            .all(|x| Rc::ptr_eq(&q.trace.0, &x.trace.0)),
        "mixed attention graph contexts"
    );
    if let Some(mask) = mask {
        ensure!(
            mask.shape() == [1, 1, q.dim(2), k.dim(2)],
            "invalid attention mask shape"
        );
    }
    let shape = |x: &Tensor| {
        x.shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("attention rank must be four"))
    };
    let dtype = if [q.dtype, k.dtype, v.dtype].contains(&Dtype::F32) {
        Dtype::F32
    } else {
        q.dtype
    };
    let geometry = crate::media_attention::Geometry {
        q: shape(q)?,
        k: shape(k)?,
        v: shape(v)?,
        scale: scale.to_bits(),
        output_dtype: dtype,
        mask: mask.is_some(),
    };
    let mut inputs = vec![q.value, k.value, v.value];
    if let Some(m) = mask {
        inputs.push(m.value);
    }
    let out = q
        .trace
        .0
        .borrow_mut()
        .graph
        .as_mut()
        .unwrap()
        .append(crate::media_attention::graph(&geometry)?, &inputs, &[], &[])
        .map_err(anyhow::Error::msg)?[0];
    Ok(q.trace
        .tensor(out, vec![q.dim(0), q.dim(1), q.dim(2), v.dim(-1)], dtype))
}
pub struct Weights<'a> {
    pub trace: Trace,
    pub schema: &'a Schema,
}
impl Weights<'_> {
    pub fn get(&self, name: &str) -> anyhow::Result<Tensor> {
        let info = self
            .schema
            .get(name)
            .with_context(|| format!("missing FLUX model weight {name}"))?;
        Ok(self.trace.weight(name, info, false))
    }
    pub fn lin(&self, key: &str, x: &Tensor) -> anyhow::Result<Tensor> {
        let name = format!("{key}.weight");
        let info = self
            .schema
            .get(&name)
            .with_context(|| format!("missing FLUX projection {name}"))?;
        let mut y = if info.dtype == Dtype::U32 {
            x.affine(
                &self.get(&name)?,
                &self.get(&format!("{key}.scales"))?,
                &self.get(&format!("{key}.biases"))?,
            )?
        } else {
            ensure!(
                info.shape.len() == 2 && x.rank() > 0 && x.dim(-1) == info.shape[1],
                "invalid FLUX dense projection"
            );
            let mut shape = x.shape.clone();
            *shape.last_mut().unwrap() = info.shape[0];
            x.reshape(&[-1, info.shape[1]])?
                .matmul(&self.trace.weight(&name, info, true))?
                .reshape(&shape)?
        };
        let bias = format!("{key}.bias");
        if self.schema.contains_key(&bias) {
            y = y.add(&self.get(&bias)?)?;
        }
        Ok(y)
    }
}
pub fn silu(x: &Tensor) -> anyhow::Result<Tensor> {
    x.multiply(&x.sigmoid()?)
}
pub fn layer_norm(x: &Tensor) -> anyhow::Result<Tensor> {
    let x = x.as_dtype(Dtype::F32)?;
    let xc = x.subtract(&x.mean_axis(-1, true)?)?;
    let inv = xc
        .multiply(&xc)?
        .mean_axis(-1, true)?
        .add(&x.scalar(1e-6))?
        .rsqrt()?;
    xc.multiply(&inv)?.as_dtype(Dtype::BF16)
}

pub fn count(schema: &Schema, prefix: &str, suffix: &str) -> usize {
    (0..128)
        .take_while(|i| schema.contains_key(&format!("{prefix}.{i}.{suffix}")))
        .count()
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}
