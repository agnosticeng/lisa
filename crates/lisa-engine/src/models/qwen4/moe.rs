//! Mixture-of-experts block: fp32 router, top-10 of 512, quantized
//! gather-GEMM experts, shared expert with sigmoid gate.

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype};

use crate::core::loader::TensorSource;
use crate::core::quant::QuantizedLinear;

/// A quantized SwiGLU expert stack: gate_proj/up_proj [E, H, K/8],
/// down_proj [E, K, H/8].
pub struct SwitchGlu {
    pub gate_proj: Array,
    pub gate_scales: Array,
    pub gate_biases: Array,
    pub up_proj: Array,
    pub up_scales: Array,
    pub up_biases: Array,
    pub down_proj: Array,
    pub down_scales: Array,
    pub down_biases: Array,
}

impl SwitchGlu {
    pub fn load<S: TensorSource>(src: &mut S, prefix: &str) -> anyhow::Result<Self> {
        Ok(Self {
            gate_proj: src.get(&format!("{prefix}.gate_proj.weight"))?,
            gate_scales: src.get(&format!("{prefix}.gate_proj.scales"))?,
            gate_biases: src.get(&format!("{prefix}.gate_proj.biases"))?,
            up_proj: src.get(&format!("{prefix}.up_proj.weight"))?,
            up_scales: src.get(&format!("{prefix}.up_proj.scales"))?,
            up_biases: src.get(&format!("{prefix}.up_proj.biases"))?,
            down_proj: src.get(&format!("{prefix}.down_proj.weight"))?,
            down_scales: src.get(&format!("{prefix}.down_proj.scales"))?,
            down_biases: src.get(&format!("{prefix}.down_proj.biases"))?,
        })
    }
}

/// Sparse MoE block.
pub struct SparseMoeBlock {
    pub gate: Array, // bf16 [512, 2560] router weight, NOT quantized
    pub switch_mlp: SwitchGlu,
    pub shared_gate_proj: QuantizedLinear,
    pub shared_up_proj: QuantizedLinear,
    pub shared_down_proj: QuantizedLinear,
    pub shared_expert_gate: QuantizedLinear,
    pub top_k: usize,
    /// Fused shared `gate|up` quantized weight `[2N, K]` for the decode kernel
    /// (concatenated once; the per-row GEMV is unaffected by the concat).
    pub fused_shared_gate_up: std::sync::OnceLock<(Array, Array, Array)>,
}

impl SparseMoeBlock {
    pub fn load<S: TensorSource>(src: &mut S, prefix: &str, top_k: usize) -> anyhow::Result<Self> {
        Ok(Self {
            gate: src.get_bf16(&format!("{prefix}.gate.weight"))?,
            switch_mlp: SwitchGlu::load(src, &format!("{prefix}.switch_mlp"))?,
            shared_gate_proj: QuantizedLinear::load(src, &format!("{prefix}.shared_expert"), "gate_proj")?,
            shared_up_proj: QuantizedLinear::load(src, &format!("{prefix}.shared_expert"), "up_proj")?,
            shared_down_proj: QuantizedLinear::load(src, &format!("{prefix}.shared_expert"), "down_proj")?,
            shared_expert_gate: QuantizedLinear::load(src, prefix, "shared_expert_gate")?,
            top_k,
            fused_shared_gate_up: std::sync::OnceLock::new(),
        })
    }

    /// Router logits `[B, S, E]` f32. Decode uses `track_router_gemv`, wide
    /// windows (32..=1024 rows) the NAX `track_router_bf16_storage`, else the
    /// reference float32 matmul.
    fn router_logits(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hidden = x.dim(-1);
        let rows = b * s;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        // `router_gemv` is one vector; batched windows take the matmul path.
        if b == 1 && s == 1 {
            let xf = x.as_dtype(Dtype::Float32)?.reshape(&[hidden])?;
            if let Some(o) = lisa_mlx::kernels::router_gemv(&xf, &self.gate, &stream) {
                return o.reshape(&[b, s, -1]);
            }
        }
        if rows >= 32
            && rows <= 1024
            && hidden == 2560
            && self.gate.shape() == [512, 2560]
        {
            if let Some(o) = lisa_mlx::prefill_indirect::router(x, &self.gate, rows, &stream) {
                let o = o.reshape(&[b, s, 512])?;
                return Ok(o);
            }
        }
        ops::matmul(&x.as_dtype(Dtype::Float32)?, self.gate.transpose_axes(&[1, 0])?)
    }

    /// The engine's wide-window (S > 8) MoE: the NAX indirect GEMM over a tile
    /// table (`track_prefill_tile_table` -> `track_prefill_indirect_gate_up` ->
    /// `track_prefill_indirect_down`) then the sorted combine.
    fn forward_prefill(&self, x: &Array) -> Option<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hidden = x.dim(-1);
        let rows = b * s;
        let top_k = self.top_k as i32;
        let e = self.gate.dim(0);
        if e != 512 || hidden != 2560 {
            return None;
        }
        let sm = &self.switch_mlp;
        if sm.gate_proj.shape() != [512, 640, 320] {
            return None;
        }
        let br = rows * top_k;
        if br < 2048 || br >= 512 * 64 {
            return None;
        }
        let stream = lisa_mlx::Stream::thread_local_or_default();

        let logits = self.router_logits(x).ok()?;
        let neg = -&logits;
        let indices = ops::argpartition_axis(&neg, (top_k - 1) as i32, -1).ok()?;
        let indices = indices.index((.., .., 0..top_k)).contiguous().ok()?;
        let selected = logits.take_along_axis(&indices, -1).ok()?;
        let weights = ops::softmax_axis(&selected, -1, true).ok()?;

        let idx_flat = indices.reshape(&[br]).ok()?;
        // Stable counting sort over the 512 expert buckets (two launches); the
        // MLX argsort chain (14 launches/layer) is the shape fallback.
        let counting =
            lisa_mlx::kernels::route_counting_sort(&idx_flat, e as usize, self.top_k, &stream);
        let (sorted_idx, token_idx, inv_order) = match counting {
            Some(t) => t,
            None => {
                let order = ops::argsort(&idx_flat).ok()?;
                let inv = ops::argsort(&order).ok()?;
                let ti = ops::floor_divide(&order, &Array::from_int(top_k))
                    .ok()?
                    .as_dtype(Dtype::Uint32)
                    .ok()?;
                let si = idx_flat
                    .take_axis(&order, 0)
                    .ok()?
                    .as_dtype(Dtype::Uint32)
                    .ok()?;
                (si, ti, inv)
            }
        };

        let max_t = lisa_mlx::prefill_indirect::max_tiles(br, e);
        // NOTE: a plain `.eval()` here is a per-layer CPU/GPU barrier and costs
        // ~25% of prefill throughput. Keep it behind the timing flags only.
        if lisa_mlx::env_flag("LISA_PROFILE") || lisa_mlx::env_flag("LISA_DEBUG_INDIRECT") {
            let _ = sorted_idx.eval();
        }
        let t_ind = std::time::Instant::now();
        let tiles = lisa_mlx::prefill_indirect::tile_table(&sorted_idx, br, e, &stream)?;
        if lisa_mlx::env_flag("LISA_DEBUG_INDIRECT") {
            let _ = tiles.eval();
            eprintln!("[indirect] rows={br} maxT={max_t} tiles0..8={:?}", &tiles.as_slice::<u32>()[..8].to_vec());
        }
        let activated = lisa_mlx::prefill_indirect::gate_up(
            x, &sm.gate_proj, &sm.gate_scales, &sm.gate_biases,
            &sm.up_proj, &sm.up_scales, &sm.up_biases,
            &sorted_idx, &token_idx, &tiles, max_t, 640, hidden, br, &stream,
        )?;
        let down = lisa_mlx::prefill_indirect::down(
            &activated, &sm.down_proj, &sm.down_scales, &sm.down_biases,
            &sorted_idx, &tiles, max_t, hidden, 640, br, &stream,
        )?;
        let down = down.reshape(&[br, hidden]).ok()?;
        if lisa_mlx::env_flag("LISA_PROFILE") {
            let _ = down.eval();
            eprintln!("      indirect experts {:>7.2} ms", t_ind.elapsed().as_secs_f64() * 1e3);
        }
        return self.prefill_combine(x, &down, &weights, &inv_order, &sorted_idx, rows, top_k, hidden, &stream);
    }

    /// Shared expert + the sorted combine (same kernels as the fused path).
    #[allow(clippy::too_many_arguments)]
    fn prefill_combine(
        &self,
        x: &Array,
        down: &Array,
        weights: &Array,
        inv_order: &Array,
        _sorted_idx: &Array,
        rows: i32,
        top_k: i32,
        hidden: i32,
        stream: &lisa_mlx::Stream,
    ) -> Option<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let br = rows * top_k;
        let sg = self.shared_expert_gate.forward(x).ok()?;
        let shared_gate = self.shared_gate_proj.forward(x).ok()?;
        let shared_up = self.shared_up_proj.forward(x).ok()?;
        let shared = {
            let g2 = shared_gate.reshape(&[rows, shared_gate.dim(-1)]).ok()?;
            let u2 = shared_up.reshape(&[rows, shared_up.dim(-1)]).ok()?;
            match lisa_mlx::kernels::swiglu2(&g2, &u2, &stream) {
                Some(r) => r.reshape(shared_gate.shape()).ok()?,
                None => crate::core::norm::bf16_silu(&shared_gate).ok()?.multiply(&shared_up).ok()?,
            }
        };
        let shared = self.shared_down_proj.forward(&shared).ok()?;
        let out = lisa_mlx::kernels::moe_combine(
            down,
            &weights.reshape(&[br]).ok()?,
            &inv_order.reshape(&[br]).ok()?,
            &shared.reshape(&[rows, hidden]).ok()?,
            &sg.reshape(&[rows]).ok()?,
            top_k, hidden, rows, &stream,
        )?;
        out.reshape(&[b, s, hidden]).ok()
    }

    /// The engine's one-token decode MoE: `track_moe_route_1` ->
    /// `track_moe_gate_up_reuse_2row` -> `track_moe_down_combine_1`.
    fn forward_decode(&self, x: &Array) -> Option<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let kd = x.dim(-1);
        let rows = b * s;
        let e = self.gate.dim(0);
        let top_k = self.top_k as i32;
        let hidden = kd;
        let stream = lisa_mlx::Stream::thread_local_or_default();

        let x_flat = x.reshape(&[rows, kd]).ok()?;
        // The router: the S=1 GEMV, or the matmul/NAX router for the 2..8
        // window. (The GEMV only accepts one row, so inlining it here used to
        // bail the whole decode path for the MTP verify window.)
        let logits = self.router_logits(x).ok()?.reshape(&[rows, e]).ok()?;
        let sg = &self.shared_expert_gate;
        let (idx, w, gate) = lisa_mlx::moe_decode::route(
            &logits, &x_flat, &sg.weight, &sg.scales, &sg.biases, top_k, e, kd, rows, &stream,
        )?;
        let br = rows * top_k;
        let idx_flat = idx.reshape(&[br]).ok()?;
        let xrow_v: Vec<u32> = (0..br).map(|i| (i / top_k) as u32).collect();
        let xrow = Array::from_slice(&xrow_v, &[br]);
        let (fsw, fss, fsb) = self.fused_shared_gate_up.get_or_init(|| {
            let cat = |a: &Array, b: &Array| lisa_mlx::ops::concatenate(&[a, b], 0).unwrap();
            (
                cat(&self.shared_gate_proj.weight, &self.shared_up_proj.weight),
                cat(&self.shared_gate_proj.scales, &self.shared_up_proj.scales),
                cat(&self.shared_gate_proj.biases, &self.shared_up_proj.biases),
            )
        });
        let sm = &self.switch_mlp;
        let n = fsw.dim(0) / 2;
        let (wg, wu, wd) = (
            (&sm.gate_proj, &sm.gate_scales, &sm.gate_biases),
            (&sm.up_proj, &sm.up_scales, &sm.up_biases),
            (&sm.down_proj, &sm.down_scales, &sm.down_biases),
        );
        let sdd = (&self.shared_down_proj.weight, &self.shared_down_proj.scales, &self.shared_down_proj.biases);
        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static MOED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = MOED.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r >= 4000 {
                let d = |name: &str, a: &lisa_mlx::Array| {
                    if let Ok(f) = a.as_dtype(lisa_mlx::Dtype::Float32) {
                        {
                                let v = f.as_slice::<f32>();
                            let v: &[f32] = v;
                            let sum: f32 = v.iter().sum();
                            eprintln!("[moe] #{r} {name} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..6.min(v.len())]);
                        }
                    }
                };
                d("router", &logits);
                d("gate", &gate);
            }
        }
        let w_flat = w.reshape(&[br]).ok()?;
        let gate_flat = gate.reshape(&[rows]).ok()?;
        let prof = lisa_mlx::env_flag("LISA_PROFILE_MOE");
        let mut tt = std::time::Instant::now();
        let out = if s == 1 {
            let act = lisa_mlx::moe_decode::gate_up_act(
                wg.0, wg.1, wg.2, wu.0, wu.1, wu.2, fsw, fss, fsb,
                &x_flat, &idx_flat, &xrow, br, n, kd, rows, &stream,
            )?;
            if prof { let _ = act.eval(); eprintln!("      moe.gate_up  {:>6.2} ms", tt.elapsed().as_secs_f64()*1e3); tt = std::time::Instant::now(); }
            let dc = lisa_mlx::moe_decode::down_combine(
                wd.0, wd.1, wd.2, sdd.0, sdd.1, sdd.2,
                &act, &idx_flat, &w_flat, &gate_flat, top_k, n, hidden, br, rows, &stream,
            )?;
            if prof { let _ = dc.eval(); eprintln!("      moe.down     {:>6.2} ms", tt.elapsed().as_secs_f64()*1e3); }
            dc
        } else {
            let act = lisa_mlx::moe_decode::gate_up_act_wide(
                wg.0, wg.1, wg.2, wu.0, wu.1, wu.2, fsw, fss, fsb,
                &x_flat, &idx_flat, &xrow, br, n, kd, rows, &stream,
            )?;
            if prof { let _ = act.eval(); eprintln!("      moe.gate_up_w {:>6.2} ms", tt.elapsed().as_secs_f64()*1e3); tt = std::time::Instant::now(); }
            let r = lisa_mlx::moe_decode::down_combine_wide(
                wd.0, wd.1, wd.2, sdd.0, sdd.1, sdd.2,
                &act, &idx_flat, &w_flat, &gate_flat, top_k, n, hidden, br, rows, &stream,
            )?;
            if prof { let _ = r.eval(); eprintln!("      moe.down_w   {:>6.2} ms", tt.elapsed().as_secs_f64()*1e3); }
            r
        };
        out.reshape(&[b, s, hidden]).ok()
    }

    /// x: [B, S, hidden] -> [B, S, hidden]
    pub fn forward(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        // Narrow windows (S <= 8) use the engine's fused decode path.
        if x.dim(1) <= 8 {
            if let Some(out) = self.forward_decode(x) {
                return Ok(out);
            }
        }
        // Wide windows use the engine's NAX indirect GEMM (tile table).
        if x.dim(1) > 8 {
            if let Some(out) = self.forward_prefill(x) {
                return Ok(out);
            }
        }
        static FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);
        let dump_dir = std::env::var("LISA_DUMP_DIR").ok();
        let first = FIRST.swap(false, std::sync::atomic::Ordering::SeqCst) && dump_dir.is_some();
        let dump = |name: &str, arr: &Array| {
            if first {
                let dir = dump_dir.as_ref().unwrap();
                let f = arr.as_dtype(Dtype::Float32).unwrap();
                let a = f.as_slice::<f32>();
                let _ = std::fs::write(
                    format!("{dir}/rs_{name}.bin"),
                    unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u8, a.len() * 4) },
                );
            }
        };
        let prof = lisa_mlx::env_flag("LISA_PROFILE");
        let tstart = std::time::Instant::now();
        let b = x.dim(0);
        let s = x.dim(1);
        let hidden = x.dim(2);
        dump("moe_in", x);

        // The router runs in float32; the softmax is taken AFTER selection,
        // over the selected logits only (the reference order).
        let logits = self.router_logits(x)?;
        dump("moe_routerlogits", &logits);
        let neg = -&logits;
        let indices = ops::argpartition_axis(&neg, (self.top_k - 1) as i32, -1)?;
        let indices = indices.index((.., .., 0..self.top_k as i32));
        let selected = logits.take_along_axis(&indices, -1)?;
        let weights = ops::softmax_axis(&selected, -1, true)?;
        if prof { let _ = weights.eval(); eprintln!("      moe.router {:>6.2} ms", tstart.elapsed().as_secs_f64()*1e3); }
        dump("moe_indices", &indices);
        dump("moe_weights", &weights);

        // The reference SwitchGLU path: gatherSort -> gather-GEMM over the
        // sorted slots -> scatterUnsort -> the weighted sum in the original
        // slot order with the K = top_k col_reduce_small association.
        let rows = b * s;
        let x_rows = x.reshape(&[rows, hidden])?;
        let idx_flat = indices.reshape(&[rows * self.top_k as i32])?;

        let order = ops::argsort(&idx_flat)?;
        let inv_order = ops::argsort(&order)?;
        let token_idx = ops::floor_divide(&order, &Array::from_int(self.top_k as i32))?;
        // gatherSort indexes the ORIGINAL rows (one row per token), not the
        // repeated array: x_sorted[i] = x_rows[order[i] / top_k].
        let x_sorted = x_rows.take_axis(&token_idx, 0)?;
        let sorted_idx = idx_flat.take_axis(&order, 0)?;
        if prof { let _ = order.eval(); let _ = x_sorted.eval(); let _ = sorted_idx.eval(); eprintln!("      moe.sort   {:>6.2} ms", tstart.elapsed().as_secs_f64()*1e3); }
        let t_gather = std::time::Instant::now();

        // The fork's exact production call: the 3-D [rows, 1, K] lhs with the
        // SORTED rhs indices (one row per index) and the sorted_indices hint
        // ON — the aligned branch, which avoids the M-count defect.
        let gather = |inp: &Array, key: &str| -> lisa_mlx::error::Result<Array> {
            let (sw, ss, sb) = match key {
                "gate_proj" => (&self.switch_mlp.gate_proj, &self.switch_mlp.gate_scales, &self.switch_mlp.gate_biases),
                "up_proj" => (&self.switch_mlp.up_proj, &self.switch_mlp.up_scales, &self.switch_mlp.up_biases),
                _ => (&self.switch_mlp.down_proj, &self.switch_mlp.down_scales, &self.switch_mlp.down_biases),
            };
            ops::gather_qmm(
                &inp.reshape(&[inp.dim(0), 1, inp.dim(-1)])?,
                sw, ss, Some(sb), None, Some(&sorted_idx), true,
                crate::core::quant::GROUP_SIZE, crate::core::quant::BITS, true,
            )
                .map(|a| a.reshape(&[a.dim(0), -1]).unwrap())
        };
        let gate_out = gather(&x_sorted, "gate_proj")?;
        let up_out = gather(&x_sorted, "up_proj")?;
        dump("moe_gate_out", &gate_out);
        dump("moe_up_out", &up_out);
        let act = crate::core::norm::bf16_silu(&gate_out)?.multiply(&up_out)?;
        dump("moe_act_sorted", &act);
        let down_sorted = gather(&act, "down_proj")?;
        if prof { let _ = down_sorted.eval(); eprintln!("      moe.experts {:>6.2} ms", t_gather.elapsed().as_secs_f64()*1e3); }
        dump("moe_down_sorted", &down_sorted);

        // Shared expert (SwiGLU + sigmoid gate) -- computed here because the
        // fused combine kernel folds it in.
        let sg = self.shared_expert_gate.forward(x)?;
        dump("moe_sharedlogit", &sg);
        let shared_gate = self.shared_gate_proj.forward(x)?;
        let shared_up = self.shared_up_proj.forward(x)?;
        let shared = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            let g2 = shared_gate.reshape(&[rows, shared_gate.dim(-1)])?;
            let u2 = shared_up.reshape(&[rows, shared_up.dim(-1)])?;
            match lisa_mlx::kernels::swiglu2(&g2, &u2, &stream) {
                Some(r) => r.reshape(shared_gate.shape())?,
                None => crate::core::norm::bf16_silu(&shared_gate)?.multiply(&shared_up)?,
            }
        };
        let shared = self.shared_down_proj.forward(&shared)?;
        dump("moe_shared", &shared);
        dump("moe_sharedgate", &sg);

        let k = self.top_k as i32;
        let rows = b * s;
        // Fused combine (`track_p12_moe_sorted_combine`): one launch over the
        // sorted rows -- no scatter copy, no f32 [B,S,K,H] product.
        let fused: Option<Array> = {
            let stream = lisa_mlx::Stream::thread_local_or_default();
            lisa_mlx::kernels::moe_combine(
                &down_sorted,
                &weights.reshape(&[rows * k])?,
                &inv_order.reshape(&[rows * k])?,
                &shared.reshape(&[rows, hidden])?,
                &sg.reshape(&[rows])?,
                k,
                hidden,
                rows,
                &stream,
            )
        };
        let out = match fused {
            Some(r) => r.reshape(&[b, s, hidden])?,
            None => {
                // Fallback: scatter + the K = top_k col_reduce_small association.
                let down = down_sorted.take_axis(&inv_order, 0)?;
                let down = down.reshape(&[b, s, k, hidden])?;
                let prod = down.multiply(&weights.expand_dims(-1)?)?;
                let prod_f = prod.as_dtype(Dtype::Float32)?;
                let ty = k.min(8);
                let mut partials: Vec<Array> = Vec::with_capacity(ty as usize);
                for y in 0..ty {
                    let mut acc: Option<Array> = None;
                    let mut rr = y;
                    while rr < k {
                        let row = prod_f.index((.., .., rr, ..)).contiguous().unwrap();
                        acc = Some(match acc {
                            None => row,
                            Some(a) => row.add(&a)?,
                        });
                        rr += ty;
                    }
                    partials.push(acc.unwrap());
                }
                let mut total = partials[0].clone();
                for p in partials.iter().skip(1) {
                    total = p.add(&total)?;
                }
                let routed = total.as_dtype(x.dtype())?;
                let shared = shared.multiply(&ops::sigmoid(&sg)?)?;
                routed.add(&shared)?
            }
        };
        dump("moe_out", &out);
        if prof { let _ = out.eval(); eprintln!("      moe.combine {:>6.2} ms", t_gather.elapsed().as_secs_f64()*1e3); }
        Ok(out)
    }
}

