//! Qwen35 computation over original packed blocks, with a distinct F32 law.
//! Tensor names and geometry are model data; no format parser or device is used.
use crate::{
    attention_cache::Append,
    graph::{Graph, Value},
    program::{Build, Program, Schema},
};
use anyhow::{Context, ensure};
use lisa_backend::{Binary, Kernel, OrderedF32 as F, ScalarType as D, TensorOperation as T, Unary};
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Projection {
    pub encoding: u32,
    pub input: i32,
    pub output: i32,
}
pub type Projections = BTreeMap<String, Projection>;
fn project(
    b: &mut Build<'_>,
    specs: &Projections,
    x: Value,
    name: &str,
    rows: i32,
) -> anyhow::Result<Value> {
    let p = specs
        .get(name)
        .with_context(|| format!("missing packed projection {name}"))?;
    ensure!(
        p.input > 0 && p.output > 0 && rows > 0,
        "invalid packed projection geometry"
    );
    let x = b.reshape(x, &[rows, p.input]);
    let x = b.contiguous(x);
    let w = b.weight(name);
    let y = b.graph.one(
        Kernel::OrderedF32(F::BlockLinear {
            encoding: p.encoding,
        }),
        vec![w, x],
    );
    Ok(b.reshape(y, &[1, rows, p.output]))
}
fn rotary(b: &mut Build<'_>, x: Value, theta: u32, offset: usize) -> anyhow::Result<Value> {
    Ok(b.op(
        T::Rotary {
            dimensions: 64,
            base: theta,
            offset: i32::try_from(offset)?,
        },
        vec![x],
    ))
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Cache {
    Attention(Append),
    Delta {
        conv: bool,
        state: bool,
        capture: bool,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layer {
    pub index: usize,
    pub rows: i32,
    pub epsilon: u32,
    pub theta: u32,
    pub cache: Cache,
}
impl Layer {
    /// Input: activation, existing K/V or convolution/state when present.
    /// Output: activation, K/V storage or convolution/state, then optional
    /// captured recurrent states and full convolution input.
    pub fn program(&self, schema: &Schema, specs: &Projections) -> anyhow::Result<Program> {
        ensure!(
            self.rows > 0 && self.rows <= 32767 && self.index <= 64,
            "invalid Qwen block geometry"
        );
        ensure!(
            full_attention(self.index) == matches!(self.cache, Cache::Attention(_)),
            "Qwen block/cache kind differs"
        );
        let eps = f32::from_bits(self.epsilon);
        ensure!(eps.is_finite() && eps > 0., "invalid Qwen RMS epsilon");
        let inputs = match &self.cache {
            Cache::Attention(c) => 1 + 2 * usize::from(c.previous_batch.is_some()),
            Cache::Delta { conv, state, .. } => 1 + usize::from(*conv) + usize::from(*state),
        };
        let mut b = Build::new(inputs, schema);
        let prefix = format!("blk.{}", self.index);
        let x = Value::Input(0);
        let n = b.rms(x, &format!("{prefix}.attn_norm.weight"), eps);
        let (y, state) = match &self.cache {
            Cache::Attention(c) => self.attention(&mut b, specs, n, &prefix, c)?,
            Cache::Delta {
                conv,
                state,
                capture,
            } => self.delta(&mut b, specs, n, &prefix, *conv, *state, *capture)?,
        };
        let residual = b.binary(Binary::Add, x, y);
        let n = b.rms(
            residual,
            &format!("{prefix}.post_attention_norm.weight"),
            eps,
        );
        let gate = project(
            &mut b,
            specs,
            n,
            &format!("{prefix}.ffn_gate.weight"),
            self.rows,
        )?;
        let up = project(
            &mut b,
            specs,
            n,
            &format!("{prefix}.ffn_up.weight"),
            self.rows,
        )?;
        let ff = b.graph.one(
            Kernel::OrderedF32(F::SiluProduct { reverse: false }),
            vec![gate, up],
        );
        let ff = project(
            &mut b,
            specs,
            ff,
            &format!("{prefix}.ffn_down.weight"),
            self.rows,
        )?;
        let y = b.binary(Binary::Add, residual, ff);
        let mut outputs = vec![y];
        outputs.extend(state);
        b.finish(outputs)
    }
    fn attention(
        &self,
        b: &mut Build<'_>,
        p: &Projections,
        x: Value,
        prefix: &str,
        c: &Append,
    ) -> anyhow::Result<(Value, Vec<Value>)> {
        ensure!(
            c.batch == 1
                && c.heads == 4
                && c.width == 256
                && c.rows == self.rows
                && c.dtype == D::F32,
            "invalid Qwen F32 cache"
        );
        let t = self.rows;
        let eps = f32::from_bits(self.epsilon);
        let qg = project(b, p, x, &format!("{prefix}.attn_q.weight"), t)?;
        let qg = b.reshape(qg, &[1, t, 24, 512]);
        let q = b.slice(qg, -1, 0, 256);
        let q = b.contiguous(q);
        let q = b.rms(q, &format!("{prefix}.attn_q_norm.weight"), eps);
        let q = b.transpose(q, &[0, 2, 1, 3]);
        let q = b.contiguous(q);
        let gate = b.slice(qg, -1, 256, 512);
        let gate = b.contiguous(gate);
        let gate = b.reshape(gate, &[1, t, 6144]);
        let k = project(b, p, x, &format!("{prefix}.attn_k.weight"), t)?;
        let k = b.reshape(k, &[1, t, 4, 256]);
        let k = b.rms(k, &format!("{prefix}.attn_k_norm.weight"), eps);
        let k = b.transpose(k, &[0, 2, 1, 3]);
        let k = b.contiguous(k);
        let v = project(b, p, x, &format!("{prefix}.attn_v.weight"), t)?;
        let v = b.reshape(v, &[1, t, 4, 256]);
        let v = b.transpose(v, &[0, 2, 1, 3]);
        let v = b.contiguous(v);
        let q = rotary(b, q, self.theta, c.offset)?;
        let k = rotary(b, k, self.theta, c.offset)?;
        let mut input = vec![k, v];
        if c.previous_batch.is_some() {
            input.extend([Value::Input(1), Value::Input(2)]);
        }
        let cache = b
            .graph
            .append(c.graph().map_err(anyhow::Error::msg)?, &input, &[], &[])
            .map_err(anyhow::Error::msg)?;
        let y = b.graph.one(
            Kernel::OrderedF32(F::VectorAttention {
                scale: (1.0f32 / 16.).to_bits(),
                causal: t > 1,
            }),
            vec![q, cache[2], cache[3]],
        );
        let y = b.transpose(y, &[0, 2, 1, 3]);
        let y = b.contiguous(y);
        let y = b.reshape(y, &[1, t, 6144]);
        let gate = b.unary(Unary::Sigmoid, gate);
        let y = b.binary(Binary::Multiply, y, gate);
        let y = project(b, p, y, &format!("{prefix}.attn_output.weight"), t)?;
        Ok((y, cache[..2].to_vec()))
    }
    fn delta(
        &self,
        b: &mut Build<'_>,
        p: &Projections,
        x: Value,
        prefix: &str,
        has_conv: bool,
        has_state: bool,
        capture: bool,
    ) -> anyhow::Result<(Value, Vec<Value>)> {
        let t = self.rows;
        let eps = f32::from_bits(self.epsilon);
        let qkv = project(b, p, x, &format!("{prefix}.attn_qkv.weight"), t)?;
        let previous = if has_conv {
            Value::Input(1)
        } else {
            b.graph.one(
                Kernel::Zeros {
                    shape: vec![1, 3, 10240],
                    dtype: D::F32,
                },
                vec![],
            )
        };
        let full = b.concat(vec![previous, qkv], 1);
        let conv_state = b.slice(full, 1, t, t + 3);
        let conv_state = b.contiguous(conv_state);
        let w = b.weight(&format!("{prefix}.ssm_conv1d.weight"));
        let w = b.reshape(w, &[10240, 4, 1]);
        let conv = b
            .graph
            .one(Kernel::OrderedF32(F::DepthwiseConv4Silu), vec![full, w]);
        let mut qk = vec![];
        for start in [0, 2048] {
            let a = b.slice(conv, -1, start, start + 2048);
            let a = b.contiguous(a);
            let a = b.reshape(a, &[t, 16, 128]);
            let a = b.contiguous(a);
            let a = b.graph.one(
                Kernel::OrderedF32(F::UnitRms {
                    epsilon: (eps / 128.).to_bits(),
                }),
                vec![a],
            );
            let scale = b.scalar(1. / 128f32.sqrt(), D::F32);
            qk.push(b.binary(Binary::Multiply, a, scale));
        }
        let v = b.slice(conv, -1, 4096, 10240);
        let v = b.contiguous(v);
        let v = b.reshape(v, &[t, 48, 128]);
        let alpha = project(b, p, x, &format!("{prefix}.ssm_alpha.weight"), t)?;
        let beta = project(b, p, x, &format!("{prefix}.ssm_beta.weight"), t)?;
        let dt = b.weight(&format!("{prefix}.ssm_dt.bias"));
        let a = b.weight(&format!("{prefix}.ssm_a"));
        let gates = b
            .graph
            .call(Kernel::OrderedF32(F::DecayGates), vec![alpha, beta, dt, a]);
        let mut state = if has_state {
            Value::Input(1 + usize::from(has_conv))
        } else {
            b.graph.one(
                Kernel::Zeros {
                    shape: vec![48, 128, 128],
                    dtype: D::F32,
                },
                vec![],
            )
        };
        let mut history = vec![];
        let y = if capture {
            let mut ys = vec![];
            for i in 0..t {
                let mut args = vec![];
                for v in [qk[0], qk[1], v, gates[0], gates[1]] {
                    let v = b.slice(v, 0, i, i + 1);
                    args.push(b.contiguous(v));
                }
                args.push(state);
                let out = b.graph.call(Kernel::OrderedF32(F::Delta128), args);
                ys.push(out[0]);
                state = out[1];
                history.push(b.reshape(state, &[1, 48, 128, 128]));
            }
            b.concat(ys, 0)
        } else {
            let out = b.graph.call(
                Kernel::OrderedF32(F::Delta128),
                vec![qk[0], qk[1], v, gates[0], gates[1], state],
            );
            state = out[1];
            out[0]
        };
        let mut states = vec![conv_state, state];
        if capture {
            states.extend([b.concat(history, 0), full]);
        }
        let y = b.rms(y, &format!("{prefix}.ssm_norm.weight"), eps);
        let z = project(b, p, x, &format!("{prefix}.attn_gate.weight"), t)?;
        let z = b.reshape(z, &[t, 48, 128]);
        let y = b.graph.one(
            Kernel::OrderedF32(F::SiluProduct { reverse: true }),
            vec![y, z],
        );
        let y = b.reshape(y, &[1, t, 6144]);
        let y = project(b, p, y, &format!("{prefix}.ssm_out.weight"), t)?;
        Ok((y, states))
    }
}
pub fn projection(
    schema: &Schema,
    specs: &Projections,
    name: &str,
    shape: &[i32],
) -> anyhow::Result<Program> {
    ensure!(
        !shape.is_empty() && shape.iter().all(|&n| n > 0),
        "invalid projection input"
    );
    let p = specs.get(name).context("missing packed projection")?;
    ensure!(
        shape.last() == Some(&p.input),
        "packed projection input width differs"
    );
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1i32, |a, &b| a.checked_mul(b))
        .context("projection rows overflow")?;
    let mut b = Build::new(1, schema);
    let y = project(&mut b, specs, Value::Input(0), name, rows)?;
    let mut out = shape.to_vec();
    *out.last_mut().unwrap() = p.output;
    let y = b.reshape(y, &out);
    b.finish(vec![y])
}
pub fn norm(schema: &Schema, name: &str, epsilon: f32) -> anyhow::Result<Program> {
    let mut b = Build::new(1, schema);
    let y = b.rms(Value::Input(0), name, epsilon);
    b.finish(vec![y])
}
pub fn select_token() -> Graph {
    let mut b = crate::graph::Builder::new(1, 0, 0);
    let y = b.one(
        Kernel::Tensor(T::Argmax {
            axis: -1,
            keep_dims: false,
        }),
        vec![Value::Input(0)],
    );
    let y = b.one(Kernel::Tensor(T::Reshape(vec![1, 1])), vec![y]);
    b.finish(vec![y])
}

pub const TARGET_LAYERS: usize = 64;
pub const DRAFT_LAYER: usize = 64;
pub const CACHE_HEADS: usize = 4;
pub const CACHE_WIDTH: usize = 256;
pub const CACHE_LAW: &str = "qwen35-gguf-f32-v1";
pub fn initial_capacity(context: usize) -> usize {
    context.min(4096)
}
pub fn cache_law(identity: &[u8]) -> u64 {
    identity.iter().fold(0xcbf29ce484222325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    })
}
pub const CHUNK: usize = 16;
pub const MAX_DRAFT_DEPTH: usize = 2;
pub const COST_KEY: &str = "qwen35-gguf-q2-xl-f32-v2";
pub fn full_attention(index: usize) -> bool {
    index == 64 || (index + 1) % 4 == 0
}
pub fn embedding_shape(rows: usize) -> anyhow::Result<Vec<i32>> {
    Ok(vec![1, i32::try_from(rows)?, 5120])
}
pub fn token_readback() -> Graph {
    let mut b = crate::graph::Builder::new(1, 0, 0);
    let ids = b.one(Kernel::Tensor(T::Cast(D::U32)), vec![Value::Input(0)]);
    let ids = b.one(Kernel::Tensor(T::Reshape(vec![-1])), vec![ids]);
    b.finish(vec![ids])
}
pub fn draft_chunks(rows: i32) -> anyhow::Result<Vec<(i32, i32)>> {
    ensure!(rows > 0, "empty draft input");
    Ok((0..rows)
        .step_by(CHUNK)
        .map(|start| (start, (start + CHUNK as i32).min(rows)))
        .collect())
}
pub fn draft_slice(rows: i32, start: i32, end: i32) -> anyhow::Result<Graph> {
    ensure!(
        start >= 0 && start < end && end <= rows,
        "invalid draft chunk"
    );
    let mut b = crate::graph::Builder::new(2, 0, 0);
    let ids = b.one(
        Kernel::Tensor(T::Slice {
            axis: 1,
            start,
            end,
        }),
        vec![Value::Input(0)],
    );
    let hidden = b.one(
        Kernel::Tensor(T::Reshape(vec![1, rows, 5120])),
        vec![Value::Input(1)],
    );
    let hidden = b.one(
        Kernel::Tensor(T::Slice {
            axis: 1,
            start,
            end,
        }),
        vec![hidden],
    );
    let hidden = b.one(Kernel::Tensor(T::Contiguous), vec![hidden]);
    Ok(b.finish(vec![ids, hidden]))
}
pub fn draft_input(
    schema: &Schema,
    specs: &Projections,
    rows: i32,
    epsilon: f32,
) -> anyhow::Result<Program> {
    let mut b = Build::new(2, schema);
    let e = b.rms(Value::Input(0), "blk.64.nextn.enorm.weight", epsilon);
    let h = b.reshape(Value::Input(1), &[1, rows, 5120]);
    let h = b.rms(h, "blk.64.nextn.hnorm.weight", epsilon);
    let x = b.concat(vec![e, h], -1);
    let x = project(&mut b, specs, x, "blk.64.nextn.eh_proj.weight", rows)?;
    b.finish(vec![x])
}
pub fn draft_finish(schema: &Schema, rows: i32, epsilon: f32) -> anyhow::Result<Program> {
    ensure!(rows > 0, "empty draft output");
    let mut b = Build::new(1, schema);
    let x = b.op(
        T::Select {
            axis: 1,
            index: rows - 1,
        },
        vec![Value::Input(0)],
    );
    let x = b.contiguous(x);
    let x = b.rms(x, "blk.64.nextn.shared_head_norm.weight", epsilon);
    b.finish(vec![x])
}

/// Parser-neutral checkpoint metadata used by the model's schema contract.
pub struct TensorInfo {
    pub dimensions: Vec<u64>,
    pub encoding: u32,
}
pub struct Checkpoint {
    pub architecture: String,
    pub integers: BTreeMap<String, u64>,
    pub reals: BTreeMap<String, f64>,
    pub tensors: BTreeMap<String, TensorInfo>,
}
pub struct Config {
    pub eps: f32,
    pub theta: f32,
    pub context: usize,
    pub eos: i64,
}
pub fn validate(p: &Checkpoint) -> anyhow::Result<Config> {
    ensure!(
        p.architecture == "qwen35",
        "expected qwen35 GGUF architecture"
    );
    for (key, want) in [
        ("block_count", 65),
        ("embedding_length", 5120),
        ("feed_forward_length", 17408),
        ("attention.head_count", 24),
        ("attention.head_count_kv", 4),
        ("attention.key_length", 256),
        ("attention.value_length", 256),
        ("nextn_predict_layers", 1),
        ("ssm.conv_kernel", 4),
        ("ssm.state_size", 128),
        ("ssm.group_count", 16),
        ("ssm.time_step_rank", 48),
        ("ssm.inner_size", 6144),
        ("full_attention_interval", 4),
        ("rope.dimension_count", 64),
    ] {
        ensure!(
            p.integers.get(&format!("qwen35.{key}")).copied() == Some(want),
            "unsupported Qwen GGUF {key}; expected {want}"
        );
    }
    let tensor = |name: &str, dims: &[u64], plain: bool| -> anyhow::Result<()> {
        let t = p
            .tensors
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        ensure!(
            t.dimensions == dims,
            "{name}: expected {dims:?}, got {:?}",
            t.dimensions
        );
        ensure!(
            if plain {
                t.encoding == 0
            } else {
                matches!(
                    t.encoding,
                    8 | 10 | 11 | 12 | 13 | 14 | 16 | 17 | 18 | 19 | 21 | 22 | 23 | 29
                )
            },
            "unsupported Qwen GGUF {name} type {:?}",
            t.encoding
        );
        Ok(())
    };
    tensor("token_embd.weight", &[5120, 248320], false)?;
    ensure!(
        p.tensors["token_embd.weight"].encoding == 10,
        "Qwen GGUF embedding currently requires original Q2_K blocks"
    );
    tensor("output.weight", &[5120, 248320], false)?;
    tensor("output_norm.weight", &[5120], true)?;
    for i in 0..65 {
        let t = |n: &str, d: &[u64], plain| tensor(&format!("blk.{i}.{n}"), d, plain);
        for n in ["attn_norm.weight", "post_attention_norm.weight"] {
            t(n, &[5120], true)?;
        }
        for n in ["ffn_gate.weight", "ffn_up.weight"] {
            t(n, &[5120, 17408], false)?;
        }
        t("ffn_down.weight", &[17408, 5120], false)?;
        if i == 64 || (i + 1) % 4 == 0 {
            for (n, d) in [
                ("attn_q.weight", vec![5120, 12288]),
                ("attn_k.weight", vec![5120, 1024]),
                ("attn_v.weight", vec![5120, 1024]),
                ("attn_output.weight", vec![6144, 5120]),
            ] {
                t(n, &d, false)?;
            }
            for n in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                t(n, &[256], true)?;
            }
        } else {
            t("attn_qkv.weight", &[5120, 10240], false)?;
            t("attn_gate.weight", &[5120, 6144], false)?;
            t("ssm_out.weight", &[6144, 5120], false)?;
            for n in ["ssm_alpha.weight", "ssm_beta.weight"] {
                t(n, &[5120, 48], false)?;
            }
            for n in ["ssm_a", "ssm_dt.bias"] {
                t(n, &[48], true)?;
            }
            t("ssm_conv1d.weight", &[4, 10240], true)?;
            t("ssm_norm.weight", &[128], true)?;
        }
    }
    for n in ["enorm", "hnorm", "shared_head_norm"] {
        tensor(&format!("blk.64.nextn.{n}.weight"), &[5120], true)?;
    }
    tensor("blk.64.nextn.eh_proj.weight", &[10240, 5120], false)?;
    let float = |key: &str| -> anyhow::Result<f32> {
        let n = p
            .reals
            .get(key)
            .copied()
            .context("missing Qwen float metadata")? as f32;
        ensure!(n.is_finite() && n > 0.0, "invalid {key}");
        Ok(n)
    };
    let context = p
        .integers
        .get("qwen35.context_length")
        .copied()
        .context("missing Qwen context length")?;
    ensure!(
        context > 0 && context <= i32::MAX as u64,
        "invalid Qwen context length"
    );
    Ok(Config {
        eps: float("qwen35.attention.layer_norm_rms_epsilon")?,
        theta: float("qwen35.rope.freq_base")?,
        context: context as usize,
        eos: p
            .integers
            .get("tokenizer.ggml.eos_token_id")
            .copied()
            .context("missing EOS")? as i64,
    })
}
