//! Embedded Qwen4 draft input computation and block ordering.
use crate::{
    graph::{Graph, Value as V},
    program::Build,
    sequence::{Call, Plan},
};
use lisa_backend::{Binary, Kernel};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Input {
    pub hidden: i32,
    pub streams: i32,
    pub embedding: (i32, i32),
    pub projection: [(i32, i32); 2],
    pub epsilon: [u32; 2],
}
impl Input {
    pub fn graph(self, b: i32, s: i32) -> anyhow::Result<Graph> {
        anyhow::ensure!(
            b > 0 && s > 0 && self.hidden > 0 && self.streams > 0,
            "invalid draft input geometry"
        );
        let schema = Default::default();
        let mut p = Build::new(13, &schema);
        let mut e = p
            .graph
            .append(
                crate::embedding::affine(self.embedding.0, self.embedding.1),
                &[V::Input(0), V::Input(2), V::Input(3), V::Input(4)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        e = p
            .graph
            .append(
                crate::normalization::rms(f32::from_bits(self.epsilon[0]), false),
                &[e, V::Input(5)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let linear = |p: &mut Build<'_>, x, slot, quant: (i32, i32)| {
            p.graph.one(
                Kernel::AffineLinear {
                    group_size: quant.0,
                    bits: quant.1,
                },
                vec![x, V::Input(slot), V::Input(slot + 1), V::Input(slot + 2)],
            )
        };
        e = linear(&mut p, e, 7, self.projection[0]);
        let h = p
            .graph
            .append(
                crate::normalization::rms(f32::from_bits(self.epsilon[1]), false),
                &[V::Input(1), V::Input(6)],
                &[],
                &[],
            )
            .map_err(anyhow::Error::msg)?[0];
        let h = p.reshape(h, &[b, s, self.streams, self.hidden]);
        let h = linear(&mut p, h, 10, self.projection[1]);
        let e = p.reshape(e, &[b, s, 1, self.hidden]);
        let x = p.binary(Binary::Add, e, h);
        let x = p.reshape(x, &[b, s, self.streams * self.hidden]);
        Ok(p.finish(vec![x])?.graph)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Input,
    Decoder,
    Finish,
}
pub fn schedule() -> Plan<Stage> {
    Plan {
        inputs: 2,
        slots: 8,
        calls: vec![
            Call {
                operation: Stage::Input,
                inputs: vec![0, 1],
                outputs: vec![2],
            },
            Call {
                operation: Stage::Decoder,
                inputs: vec![2],
                outputs: vec![3, 4, 5],
            },
            Call {
                operation: Stage::Finish,
                inputs: vec![3, 4, 5],
                outputs: vec![6, 7],
            },
        ],
        outputs: vec![6, 7],
    }
}
pub const FINAL_EPSILON: f32 = 1e-6;
pub struct Names {
    pub norm_embedding: &'static str,
    pub norm_hidden: &'static str,
    pub fc_embedding: &'static str,
    pub fc_hidden: &'static str,
    pub mixer: &'static str,
    pub layer: &'static str,
}
pub fn names() -> Names {
    Names {
        norm_embedding: "mtp.pre_fc_norm_embedding",
        norm_hidden: "mtp.pre_fc_norm_hidden",
        fc_embedding: "mtp.fc_embedding",
        fc_hidden: "mtp.fc_hidden",
        mixer: "mtp.hyper_connection_mixer",
        layer: "mtp.layers.0",
    }
}
