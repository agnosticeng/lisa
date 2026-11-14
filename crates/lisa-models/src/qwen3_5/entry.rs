//! Tower input law: rotary context, original affine embeddings, optional image splice.
use crate::{
    graph::{Builder, Graph, Value},
    vision::prompt::Splice,
};
use lisa_backend::{Binary, Kernel, ScalarType, TensorOperation as T};
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub positions_shape: Vec<i32>,
    pub multimodal: bool,
    pub rotary: usize,
    pub base: u32,
    pub embedding: (i32, i32),
    pub splice: Option<Splice>,
}
impl Plan {
    /// Inputs: IDs, embedding triple, positions, optional image feature arrays.
    /// Outputs: embeddings, BF16 cosine, BF16 sine.
    pub fn graph(&self) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            self.positions_shape.len() == if self.multimodal { 3 } else { 2 },
            "invalid position geometry"
        );
        let mut g = Builder::new(5 + self.splice.as_ref().map_or(0, |s| s.spans.len()), 0, 0);
        let frequencies = crate::position::frequencies(
            self.rotary,
            f32::from_bits(self.base),
            &self.positions_shape,
            self.multimodal,
        )?;
        let freq = g
            .append(frequencies, &[Value::Input(4)], &[], &[])
            .map_err(anyhow::Error::msg)?;
        let shape = vec![
            self.positions_shape[0],
            1,
            self.positions_shape[1],
            self.rotary as i32,
        ];
        let cos = g.one(Kernel::Tensor(T::Reshape(shape.clone())), vec![freq[0]]);
        let sin = g.one(Kernel::Tensor(T::Reshape(shape)), vec![freq[1]]);
        let cos = g.one(Kernel::Tensor(T::Cast(ScalarType::BF16)), vec![cos]);
        let sin = g.one(Kernel::Tensor(T::Cast(ScalarType::BF16)), vec![sin]);
        let embed = crate::embedding::affine(self.embedding.0, self.embedding.1);
        let mut x = g
            .append(
                embed,
                &(0..4).map(Value::Input).collect::<Vec<_>>(),
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        if let Some(splice) = &self.splice {
            let mut inputs = vec![x];
            inputs.extend((5..5 + splice.spans.len()).map(Value::Input));
            x = g
                .append(splice.graph()?, &inputs, &[], &[])
                .map_err(anyhow::Error::msg)?[0];
        }
        Ok(g.finish(vec![x, cos, sin]))
    }
}
pub fn capture_hidden() -> Graph {
    let mut g = Builder::new(2, 0, 0);
    let out = g.one(
        Kernel::Tensor(T::Binary(Binary::Add)),
        vec![Value::Input(0), Value::Input(1)],
    );
    g.finish(vec![out])
}
