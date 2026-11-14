//! Bounded bidirectional attention. Scaling precedes the f32 dot product;
//! this numerical law is distinct from language-tower attention.
use crate::{graph::Value, program::Build};
use lisa_backend::{Binary, ScalarType, TensorOperation as T};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub q: [i32; 4],
    pub k: [i32; 4],
    pub v: [i32; 4],
    pub output_dtype: ScalarType,
    pub scale: u32,
    pub mask: bool,
}
impl Geometry {
    pub fn validate(&self) -> anyhow::Result<()> {
        let (q, k, v) = (&self.q, &self.k, &self.v);
        anyhow::ensure!(
            q.iter().chain(k).chain(v).all(|&d| d > 0)
                && q[0] == 1
                && k[0] == 1
                && v[0] == 1
                && q[3] == k[3]
                && q[1] % k[1] == 0
                && k[1] == v[1]
                && k[2] == v[2],
            "invalid media attention geometry"
        );
        anyhow::ensure!(
            f32::from_bits(self.scale).is_finite(),
            "invalid media attention scale"
        );
        anyhow::ensure!(
            matches!(
                self.output_dtype,
                ScalarType::F32 | ScalarType::F16 | ScalarType::BF16
            ),
            "invalid media attention dtype"
        );
        Ok(())
    }
}
pub fn append(
    p: &mut Build<'_>,
    g: &Geometry,
    q: Value,
    k: Value,
    v: Value,
    mask: Option<Value>,
) -> anyhow::Result<Value> {
    g.validate()?;
    anyhow::ensure!(
        g.mask == mask.is_some(),
        "media attention mask binding mismatch"
    );
    let (qh, nq, dk, nk, dv) = (g.q[1], g.q[2], g.q[3], g.k[2], g.v[3]);
    let repeat = (qh / g.k[1]) as usize;
    let k = p.cast(k, ScalarType::F32);
    let k = p.op(
        T::Repeat {
            count: repeat,
            axis: 1,
        },
        vec![k],
    );
    let k = p.transpose(k, &[0, 1, 3, 2]);
    let k = p.contiguous(k);
    let k = p.reshape(k, &[qh, dk, nk]);
    let v = p.cast(v, ScalarType::F32);
    let v = p.op(
        T::Repeat {
            count: repeat,
            axis: 1,
        },
        vec![v],
    );
    let v = p.reshape(v, &[qh, nk, dv]);
    let mut chunks = Vec::new();
    for start in (0..nq).step_by(256) {
        let end = (start + 256).min(nq);
        let rows = end - start;
        let part = p.slice(q, 2, start, end);
        let part = p.cast(part, ScalarType::F32);
        let scale = p.scalar(f32::from_bits(g.scale), ScalarType::F32);
        let part = p.binary(Binary::Multiply, part, scale);
        let part = p.reshape(part, &[qh, rows, dk]);
        let mut scores = p.op(T::Matmul, vec![part, k]);
        if let Some(mask) = mask {
            let mask = p.slice(mask, 2, start, end);
            let mask = p.reshape(mask, &[1, rows, nk]);
            scores = p.binary(Binary::Add, scores, mask);
        }
        let probs = p.op(T::Softmax { axis: -1 }, vec![scores]);
        let out = p.op(T::Matmul, vec![probs, v]);
        let out = p.cast(out, g.output_dtype);
        chunks.push(p.reshape(out, &[1, qh, rows, dv]));
    }
    Ok(p.concat(chunks, 2))
}
pub fn graph(g: &Geometry) -> anyhow::Result<crate::graph::Graph> {
    let schema = Default::default();
    let mut p = Build::new(3 + usize::from(g.mask), &schema);
    let out = append(
        &mut p,
        g,
        Value::Input(0),
        Value::Input(1),
        Value::Input(2),
        g.mask.then_some(Value::Input(3)),
    )?;
    Ok(p.finish(vec![out])?.graph)
}
