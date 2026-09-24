//! Generation loop: prefill, decode, greedy and temperature sampling.

use std::sync::OnceLock;
use std::time::Instant;

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Array;

use crate::core::cache::LayerCache;
use crate::models::LanguageModel;
use crate::core::sampler::Sampler;

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl GenerationStats {
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_seconds.max(1e-9)
    }
    pub fn decode_tps(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_seconds.max(1e-9)
    }
}

static EOS_IDS: OnceLock<Vec<u32>> = OnceLock::new();

pub fn set_eos_ids(ids: Vec<u32>) {
    let _ = EOS_IDS.set(ids);
}

pub fn is_eos(id: u32) -> bool {
    EOS_IDS.get().map_or(false, |ids| ids.contains(&id))
}

/// Greedy (temperature 0) or temperature-scaled argmax generation.
///
/// Returns the generated ids (excluding the prompt).
pub fn generate(
    tower: &mut dyn LanguageModel,
    prompt: &[u32],
    max_tokens: usize,
    sampler: &mut Sampler,
    on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, GenerationStats)> {
    // Compile/warm every kernel before the timed section (the engine warms at
    // init too; without this the first prefill pays ~0.45s of Metal JIT).
    anyhow::ensure!(
        prompt.len() + max_tokens <= tower.max_position_embeddings(),
        "context overflow: prompt {} + max_tokens {} > max_position_embeddings {}",
        prompt.len(),
        max_tokens,
        tower.max_position_embeddings()
    );
    if std::env::var("LISA_NO_WARMUP").is_err() {
        tower.warmup(prompt)?;
    }
    let mut caches: Vec<LayerCache> = tower.new_caches();

    // Prefill in windows (bounded activations for long prompts).
    let t0 = Instant::now();
    let last = tower.prefill(prompt, &mut caches)?;
    let logits = tower.head(&last)?;
    let token = sampler.draw(&logits, prompt)?;
    if lisa_mlx::env_flag("LISA_DUMP_LOGITS") {
        let l = logits.as_dtype(lisa_mlx::Dtype::Float32)?;
        let v: Vec<f32> = l.as_slice::<f32>().to_vec();
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
        eprintln!(
            "[logits] prefill top8: {:?}",
            idx[..8].iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
        );
    }

    // Force eval of the whole prefill graph before timing decode.
    lisa_mlx::transforms::eval([&logits]).map_err(|e| anyhow::anyhow!("{e}"))?;
    let prefill_seconds = t0.elapsed().as_secs_f64();

    let mut generated = Vec::new();
    generated.push(token);
    on_token(token)?;

    let mut decode_seconds = 0f64;
    let profile = lisa_mlx::env_flag("LISA_PROFILE_DECODE");
    let mut build_t = 0f64;
    let mut eval_t = 0f64;
    while generated.len() < max_tokens && !is_eos(*generated.last().unwrap()) {
        let token = *generated.last().unwrap();
        let start = Instant::now();
        let step = Array::from_slice(&[token as i32], &[1i32, 1]);
        let (mixed, _) = tower.forward(&step, Some(&mut caches))?;
        let last = mixed.index((.., mixed.dim(1) - 1, ..));
        let logits = tower.head(&last)?;
        let mut history: Vec<u32> = prompt.to_vec();
        history.extend_from_slice(&generated);
        let token = sampler.draw(&logits, &history)?;
        if lisa_mlx::env_flag("LISA_DUMP_LOGITS") {
            let l = logits.as_dtype(lisa_mlx::Dtype::Float32)?;
            let v: Vec<f32> = l.as_slice::<f32>().to_vec();
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            eprintln!(
                "[logits] gen#{} top5: {:?}",
                generated.len(),
                idx[..5].iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
            );
        }
        let mid = Instant::now();
        build_t += (mid - start).as_secs_f64();
        eval_t += mid.elapsed().as_secs_f64();
        decode_seconds += start.elapsed().as_secs_f64();
        generated.push(token);
        on_token(token)?;
        // Release pooled temporaries periodically: the pool only sweeps
        // buffer pool from `wait_until_completed`, so a long decode would pin
        // every intermediate it ever allocated.
        if generated.len() % 32 == 0 && std::env::var("LISA_NO_TRIM").is_err() {
            lisa_mlx::memory::trim_cache();
        }
    }

    if profile {
        let waited_ms = lisa_mlx::runtime_wait_ns() as f64 / 1e6;
        if std::env::var("LISA_LABEL_TRACE").is_ok() {
            let mut lc = lisa_mlx::runtime_label_counts();
            lc.sort_by(|a, b| b.1.cmp(&a.1));
            eprintln!("[labels] {:?}", &lc[..12.min(lc.len())]);
        }
        eprintln!("[decode] dispatches={} encoder_opens={} gpu_wait_ms={:.0}",
            lisa_mlx::runtime_dispatch_count(),
            lisa_mlx::runtime_encoder_count(),
            waited_ms / 1e6 * 1e3);
        eprintln!(
            "[decode] build {:.2} ms/tok, eval {:.2} ms/tok, total {:.2} ms/tok",
            build_t * 1e3 / generated.len().max(1) as f64,
            eval_t * 1e3 / generated.len().max(1) as f64,
            decode_seconds * 1e3 / generated.len().max(1) as f64
        );
    }
    let stats = GenerationStats {
        prompt_tokens: prompt.len(),
        generated_tokens: generated.len(),
        prefill_seconds,
        decode_seconds,
    };
    Ok((generated, stats))
}

pub fn sample(logits: &Array, temperature: f32) -> anyhow::Result<Array> {
    if temperature <= 1e-5 {
        return lisa_mlx::ops::indexing::argmax(logits, None).map_err(|e| anyhow::anyhow!("{e}"));
    }
    let scaled = logits / temperature;
    lisa_mlx::ops::indexing::argmax(&scaled, None).map_err(|e| anyhow::anyhow!("{e}"))
}
