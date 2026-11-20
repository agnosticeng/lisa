//! Diagnostic capture for the pinned MLX-VLM LFM3 comparison (not a benchmark).
use anyhow::{Context, Result};
use lisa_backend::TensorAccess;
use lisa_engine::language::LanguageModel;
fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    anyhow::ensure!(
        args.len() == 4,
        "usage: lfm3-capture MODEL TOKENS.json CHUNK OUTPUT.npz"
    );
    let tokens: Vec<u32> = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let schedule = args[2]
        .split(':')
        .map(str::parse::<usize>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let chunk = schedule[0];
    let tail = schedule.get(1).copied().unwrap_or(0);
    anyhow::ensure!(
        tail <= tokens.len() && schedule.len() <= 2,
        "invalid decode tail"
    );
    anyhow::ensure!(chunk > 0, "chunk must be positive");
    let context = lisa_serve::application::context()?;
    let runtime = lisa_serve::application::runtime(context.clone());
    let mut model =
        lisa_engine::bindings::lfm3::Tower::load(runtime, std::path::Path::new(&args[0]))?;
    model.enable_stage_capture((0..24).collect(), vec![])?;
    let mut caches = model.new_caches();
    let mut last = None;
    let mut replay = None;
    for batch in tokens[..tokens.len() - tail]
        .chunks(chunk)
        .chain(tokens[tokens.len() - tail..].chunks(1))
    {
        let ids = lisa_engine::bindings::tensor_io::pull_i32(
            &context,
            &batch.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            &[1, batch.len() as i32],
        )?;
        let snapshots = caches.iter().map(|c| c.checkpoint()).collect::<Vec<_>>();
        let (out, _) = model.forward(&ids, Some(&mut caches))?;
        replay = Some((snapshots, ids));
        last = Some(out.select(1, batch.len() as i32 - 1)?);
    }
    let logits = model.head(&last.context("empty token list")?)?;
    println!("argmax={}", logits.argmax_id()?);
    model.write_logit_capture(&args[3])?;
    let expected = logits.read_f32()?;
    let (snapshots, ids) = replay.context("no replay snapshot")?;
    for (cache, snapshot) in caches.iter_mut().zip(&snapshots) {
        cache.restore_checkpoint(snapshot);
    }
    let (out, _) = model.forward(&ids, Some(&mut caches))?;
    let replayed = model.head(&out.select(1, ids.dim(1) - 1)?)?.read_f32()?;
    anyhow::ensure!(expected == replayed, "snapshot replay changed LFM3 logits");
    println!("snapshot replay: exact");
    lisa_serve::application::memory_line("LFM3 capture");
    lisa_metal::trace::finish();
    Ok(())
}
