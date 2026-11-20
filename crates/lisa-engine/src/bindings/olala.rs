//! Original packed GGUF binding; no model payload is requantized or expanded.
use super::BindingDevice;
use crate::{
    cache::{LayerCache, NativeCache, NativeState},
    core::{capture::Capture, graph::PreparedGraph},
    language::LanguageModel,
    language_runtime::Runtime,
    sampler::Sampler,
    weights::gguf::{ResidentWeights, load_resident},
};
use anyhow::{Context, Result, ensure};
use lisa_backend::{ManagedDevice, ReadbackDevice, ScalarType as Dtype, TensorAccess};
use lisa_gguf::{Package, TensorType as Type};
use lisa_models::olala::{
    config::{Config, Role},
    graph::{self, Definition, Format},
};
use std::{collections::BTreeMap, path::Path, sync::Arc};
#[derive(Clone)]
struct State<T> {
    position: usize,
    layers: Vec<Vec<T>>,
}
impl<T: Clone + Send + Sync + 'static> NativeState for State<T> {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn copy_for_write(&self) -> Result<Arc<dyn NativeState>> {
        Ok(Arc::new(self.clone()))
    }
    fn position(&self) -> usize {
        self.position
    }
}
struct Layer<D: BindingDevice> {
    weights: Vec<D::Tensor>,
    plan: PreparedGraph<D>,
    has_history: bool,
}
pub struct Tower<D: BindingDevice + ManagedDevice + ReadbackDevice> {
    runtime: Runtime<D, Sampler<D>>,
    config: Config,
    capacity: usize,
    _resident: ResidentWeights<D, D::Reservation>,
    embedding: lisa_gguf::Tensor,
    formats: BTreeMap<String, Format>,
    layers: Vec<Layer<D>>,
    head: PreparedGraph<D>,
    capture: Option<Capture<D::Tensor>>,
    warmed: bool,
}
fn formats(package: &Package, c: &Config) -> Result<BTreeMap<String, Format>> {
    let mut result = BTreeMap::new();
    for (name, shape, role) in lisa_models::olala::config::tensors(c) {
        let t = &package.tensor(&name)?.info;
        ensure!(
            t.dimensions == shape,
            "Olala tensor {name} shape mismatch: {:?}, expected {shape:?}",
            t.dimensions
        );
        let valid = match role {
            Role::Parameter => t.kind == Type::F32,
            Role::ExpertUp => t.kind == Type::Q8_0,
            Role::ExpertDown => matches!(t.kind, Type::Q5_K | Type::Q6_K),
            Role::Projection => matches!(
                t.kind,
                Type::F32 | Type::BF16 | Type::Q5_K | Type::Q6_K | Type::Q8_0
            ),
        };
        ensure!(
            valid,
            "unsupported Olala tensor encoding for {name}: {:?}",
            t.kind
        );
        let dtype = match t.kind {
            Type::F32 => Dtype::F32,
            Type::BF16 => Dtype::BF16,
            _ => Dtype::U8,
        };
        result.insert(
            name,
            Format {
                projection: super::packed::projection(t.kind)?,
                dtype,
                row_bytes: i32::try_from(t.row_bytes()?)?,
            },
        );
    }
    Ok(result)
}
fn bind<D: BindingDevice + ManagedDevice>(
    device: &D,
    resident: &ResidentWeights<D, D::Reservation>,
    def: &Definition,
) -> Result<Vec<D::Tensor>> {
    def.weights
        .iter()
        .map(|spec| {
            let w = resident
                .tensors
                .get(&spec.name)
                .with_context(|| format!("missing Olala tensor {}", spec.name))?;
            let mut t = if spec.dtype == Dtype::U8 {
                w.packed_bytes().clone()
            } else {
                w.plain_view()?
            };
            if spec.dtype == Dtype::U8
                && t.rank() == 2
                && spec.shape.len() == 2
                && t.dim(1) == spec.shape[1]
                && t.dim(0) < spec.shape[0]
                && (t.dim(0) + 7) / 8 * 8 == spec.shape[0]
            {
                let shape = [spec.shape[0] - t.dim(0), t.dim(1)];
                let zero_bytes = vec![0u8; shape[0] as usize * shape[1] as usize];
                let zero = device.pull(
                    lisa_backend::HostTensor {
                        bytes: &zero_bytes,
                        shape: &shape,
                        dtype: Dtype::U8,
                    }
                    .into(),
                    lisa_backend::ImportOptions::default(),
                )?;
                t = D::Tensor::concatenate(&[&t, &zero], 0)?;
            }
            ensure!(
                t.shape() == spec.shape,
                "Olala tensor view shape mismatch: {}",
                spec.name
            );
            Ok(t)
        })
        .collect()
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    pub fn load(runtime: Runtime<D, Sampler<D>>, path: &Path) -> Result<Self> {
        ensure!(
            path.is_file(),
            "Olala currently requires an original GGUF file"
        );
        let package = Package::open(path)?;
        let config = crate::checkpoint::olala::config(&package.header)?;
        let formats = formats(&package, &config)?;
        let device = runtime.context.device();
        let embedding = package.tensor("token_embd.weight")?.disk_reader()?;
        let names = formats
            .keys()
            .filter(|k| k.as_str() != "token_embd.weight")
            .map(String::as_str)
            .collect::<Vec<_>>();
        let resident = load_resident(device.clone(), &package, &names)?;
        let mut layers = vec![];
        for i in 0..config.layers.len() {
            let def = graph::layer(&config, i, false, &formats);
            let weights = bind(device, &resident, &def)?;
            let plan = PreparedGraph::new(device.clone(), def.graph, weights.clone())?;
            layers.push(Layer {
                weights,
                plan,
                has_history: false,
            });
        }
        let def = graph::head(&formats);
        let weights = bind(device, &resident, &def)?;
        let head = PreparedGraph::new(device.clone(), def.graph, weights)?;
        Ok(Self {
            capacity: config.context,
            config,
            runtime,
            _resident: resident,
            embedding,
            formats,
            layers,
            head,
            capture: None,
            warmed: false,
        })
    }
    fn step(&mut self, token: u32, caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        ensure!(token < 151936, "Olala token outside vocabulary");
        ensure!(caches.len() == 1, "invalid Olala cache count");
        let LayerCache::Native(cache) = &mut caches[0] else {
            anyhow::bail!("Olala requires its native cache")
        };
        let old = cache.get::<State<D::Tensor>>()?;
        let position = old.position;
        ensure!(position < self.capacity, "Olala context capacity exceeded");
        let mut next = old.clone();
        let device = self.runtime.context.device();
        let mut bytes = vec![0; self.embedding.info.row_bytes()? as usize];
        self.embedding.read_rows(&[token as u64], &mut bytes)?;
        let mut emb = vec![0.; 1536];
        lisa_gguf::quant::decode(self.embedding.info.kind, &bytes, &mut emb)?;
        let input = super::tensor_io::pull_f32(&self.runtime.context, &emb, &[1, 1536])?;
        let ops = crate::tensor_ops::TensorOps::new(device, crate::tensor_ops::TensorLaw::Native);
        use lisa_backend::{Binary as B, TensorOperation as T, Unary as U};
        let sq = ops.op(T::Binary(B::Multiply), &[&input, &input])?;
        let sum = ops.op(
            T::Sum {
                axis: -1,
                keep_dims: true,
            },
            &[&sq],
        )?;
        let norm = ops.op(T::Unary(U::Sqrt), &[&sum])?;
        let norm = ops.op(T::Binary(B::Maximum), &[&norm, &ops.scalar(1e-12)?])?;
        let mut x = ops.op(T::Binary(B::Divide), &[&input, &norm])?;
        let unused = ops.scalar(0.)?;
        // Only the empty-history branch changes the graph. Keep position as
        // runtime data so attention plans and their constants survive decode.
        let has_history = position > 0;
        let log_position = ops.op(
            T::Unary(U::Log),
            &[&ops.scalar((position + 1).min(self.config.window) as f32)?],
        )?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            if !self.config.layers[i] && layer.has_history != has_history {
                let def = graph::layer(&self.config, i, has_history, &self.formats);
                layer.plan = PreparedGraph::new(device.clone(), def.graph, layer.weights.clone())?;
                layer.has_history = has_history;
            }
            let mut state = next.layers[i].clone();
            if state.is_empty() {
                let shape = if self.config.layers[i] {
                    vec![self.config.recurrent_floats() as i32]
                } else {
                    vec![12, 128]
                };
                let zero = ops.required(
                    lisa_backend::Kernel::Zeros {
                        shape,
                        dtype: Dtype::F32,
                    },
                    &[],
                )?;
                state = if self.config.layers[i] {
                    vec![zero]
                } else {
                    vec![unused.clone(), unused.clone(), zero.clone(), zero]
                };
            }
            let args = [
                &x,
                state.first().unwrap_or(&unused),
                state.get(1).unwrap_or(&unused),
                state.get(2).unwrap_or(&unused),
                state.get(3).unwrap_or(&unused),
                &log_position,
            ];
            let mut out = layer.plan.run(&args, &mut [])?.into_iter();
            x = out.next().context("missing Olala output")?;
            next.layers[i] = out.collect();
            if let Some(c) = &self.capture {
                if c.wants(i) {
                    c.store(&format!("p{position}.layer{i}"), &x);
                    for (j, s) in next.layers[i].iter().enumerate() {
                        c.store(&format!("p{position}.layer{i}.state{j}"), s);
                    }
                }
            }
            if i % 4 == 3 {
                // Keep the serial device queue fed without a host round trip
                // every four layers. Completion-aware tensor leases retain all
                // inputs; prefill/readback supplies the token-boundary wait.
                device.submit(false)?;
            }
        }
        next.position += 1;
        *cache.get_mut::<State<D::Tensor>>()? = next;
        Ok(x)
    }
}
impl<D: BindingDevice + ManagedDevice + ReadbackDevice> LanguageModel<Runtime<D, Sampler<D>>>
    for Tower<D>
where
    D::Error: From<anyhow::Error>,
{
    fn runtime(&self) -> &Runtime<D, Sampler<D>> {
        &self.runtime
    }
    fn max_position_embeddings(&self) -> usize {
        self.capacity
    }
    fn set_context_capacity(&mut self, n: usize) -> Result<()> {
        ensure!(
            n > 0 && n <= self.config.context,
            "invalid Olala context capacity"
        );
        self.capacity = n;
        Ok(())
    }
    fn cache_numerical_law(&self) -> Option<u64> {
        Some(0x4f4c_414c_4100_0001)
    }
    fn eos_token_id(&self) -> i64 {
        151645
    }
    fn new_caches(&self) -> Vec<LayerCache<D>> {
        vec![LayerCache::Native(NativeCache::new(State::<D::Tensor> {
            position: 0,
            layers: vec![vec![]; 36],
        }))]
    }
    fn batch_decode_ok(&self) -> bool {
        false
    }
    fn max_speculative_depth(&self) -> usize {
        0
    }
    fn kv_bytes_per_token(&self) -> usize {
        7 * 12 * 128 * 2 * 4
    }
    fn stream_state_bytes(&self) -> usize {
        29 * self.config.recurrent_floats() * 4 + 7 * 2 * 12 * 128 * 4
    }
    fn forward(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(
            ids.rank() == 2 && ids.dim(0) == 1 && ids.dim(1) > 0,
            "Olala requires one nonempty sequence"
        );
        ensure!(
            matches!(
                self.runtime.context.device().dtype(ids),
                Some(Dtype::I32 | Dtype::U32)
            ),
            "Olala token IDs must be I32 or U32"
        );
        let ids_i32 = if self.runtime.context.device().dtype(ids) == Some(Dtype::U32) {
            Some(
                crate::tensor_ops::TensorOps::new(
                    self.runtime.context.device(),
                    crate::tensor_ops::TensorLaw::Native,
                )
                .op(lisa_backend::TensorOperation::Cast(Dtype::I32), &[ids])?,
            )
        } else {
            None
        };
        let tokens = self
            .runtime
            .context
            .device()
            .read_i32_through(ids_i32.as_ref().unwrap_or(ids))?;
        let mut fresh = self.new_caches();
        let cache = caches.unwrap_or(&mut fresh);
        let mut rows = vec![];
        for token in tokens {
            ensure!(token >= 0, "negative Olala token");
            rows.push(self.step(token as u32, cache)?);
            if ids.dim(1) > 1 {
                // Bound in-flight recurrent temporaries for callers that request
                // every hidden row instead of the last-row prefill interface.
                self.runtime.context.device().submit(true)?;
                self.runtime.context.device().trim_unused();
            }
        }
        let x = D::Tensor::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?;
        let x = x.reshape(&[1, ids.dim(1), 1536])?;
        Ok((x.clone(), x))
    }
    fn forward_capture(
        &mut self,
        ids: &D::Tensor,
        caches: Option<&mut Vec<LayerCache<D>>>,
        capture: bool,
    ) -> Result<(D::Tensor, D::Tensor)> {
        ensure!(!capture, "Olala speculative verification is unsupported");
        self.forward(ids, caches)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache<D>>) -> Result<D::Tensor> {
        ensure!(!tokens.is_empty(), "empty Olala prompt");
        let mut last = None;
        for &token in tokens {
            last = Some(self.step(token, caches)?);
            self.runtime.context.device().submit(true)?;
            self.runtime.context.device().trim_unused();
        }
        Ok(last.unwrap())
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache<D>>,
    ) -> Result<(D::Tensor, D::Tensor)> {
        let x = self.prefill(tokens, caches)?;
        Ok((x.clone(), x))
    }
    fn head(&self, x: &D::Tensor) -> std::result::Result<D::Tensor, D::Error> {
        (|| -> Result<D::Tensor> {
            ensure!(x.size() == 1536, "Olala head expects one row");
            let x = x.reshape(&[1, 1536])?;
            let out = self
                .head
                .run(&[&x], &mut [])?
                .into_one()
                .map_err(anyhow::Error::msg)?;
            if let Some(c) = &self.capture {
                c.store("logits", &out);
            }
            Ok(out)
        })()
        .map_err(Into::into)
    }
    fn warmup(&mut self, seed: &[u32]) -> Result<()> {
        if self.warmed {
            return Ok(());
        }
        // Compile both empty-history and continuation paths once. Replaying
        // the entire request here doubled CLI prompt work before first token.
        let tokens = [seed.first().copied().unwrap_or(151643); 2];
        let mut cache = self.new_caches();
        let x = self.prefill(&tokens, &mut cache)?;
        self.head(&x)?.eval()?;
        self.runtime.context.device().clear_unused();
        self.warmed = true;
        Ok(())
    }
    fn enable_logit_capture(&mut self) -> Result<()> {
        self.enable_stage_capture(vec![], vec![])
    }
    fn enable_stage_capture(&mut self, layers: Vec<usize>, keys: Vec<String>) -> Result<()> {
        ensure!(
            layers.iter().all(|&i| i < 36),
            "invalid Olala capture layer"
        );
        self.capture = Some(Capture {
            layers,
            key_filters: keys,
            preserve_dispatch: true,
            ..Default::default()
        });
        Ok(())
    }
    fn write_logit_capture(&self, path: &str) -> Result<()> {
        self.capture
            .as_ref()
            .context("Olala capture not enabled")?
            .flush(path)
    }
}
