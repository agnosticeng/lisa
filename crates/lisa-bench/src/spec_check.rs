//! `spec-check`: greedy speculative decoding must reproduce serial greedy
//! decoding token for token. Reports both rates; pair with
//! `--trace block.round` for per-round acceptance.
use lisa_bench::{generate, sampler, session};

#[derive(clap::Args)]
pub struct Args {
    /// A Hugging Face repo id, read from (or downloaded to) the Hub cache.
    #[arg(long)]
    pub model: String,
    /// User prompts, rendered with the checkpoint's chat template.
    #[arg(long, required = true)]
    pub prompt: Vec<String>,
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    /// Speculative depth (`auto` or a fixed depth the model accepts).
    #[arg(long, default_value = "auto")]
    pub depth: lisa_bench::cli::DepthArg,
}

fn render(model: &std::path::Path, prompt: &str) -> anyhow::Result<(String, bool)> {
    if let Some(source) = lisa_engine::checkpoint::tokenizer::load_chat_template(model)? {
        let template = lisa_serve::ChatTemplate::new(&source)?;
        let messages = [lisa_serve::ChatMessage {
            role: "user".into(),
            content: prompt.into(),
            reasoning_content: None,
            tool_calls: vec![],
            tool_call_id: None,
            parts: None,
        }];
        return Ok((template.render(&messages, true, &[], false)?, false));
    }
    Ok((
        lisa_bench::tokenizer::generation_prompt(&[("user".to_string(), prompt.to_string())]),
        false,
    ))
}

pub fn run(args: Args, model: std::path::PathBuf) -> anyhow::Result<()> {
    let mut tower = match lisa_bench::loading::load_dir(&model)? {
        lisa_bench::loading::Loaded::Language(m) => m,
        _ => anyhow::bail!("spec-check expects a language model"),
    };
    anyhow::ensure!(tower.has_drafter(), "the model loaded no drafter");
    let tok = lisa_engine::checkpoint::tokenizer::load(&model)?;
    generate::set_eos_ids_for_model(tok.im_end_ids.clone(), tower.as_ref());
    let depth = lisa_engine::cli::resolve_mtp_depth(args.depth, true, true)?
        .ok_or_else(|| anyhow::anyhow!("spec-check needs a speculative depth"))?;
    let mut failures = 0;
    for (index, prompt) in args.prompt.iter().enumerate() {
        let (text, special) = render(&model, prompt)?;
        let ids = if special {
            tok.encode_special(&text)?
        } else {
            tok.encode(&text, false)?
        };
        if index == 0 {
            // Compile kernels outside the measured runs.
            let mut greedy = sampler::Sampler::default();
            generate::generate(&mut *tower, &ids, 8, &mut greedy, &mut |_| Ok(()))?;
            let mut s = session::Session::new(&mut *tower);
            s.generate_mtp_depth(&mut *tower, &ids, 16, depth, None, None, &mut |_| Ok(()))?;
        }
        let mut greedy = sampler::Sampler::default();
        let (serial, stats) =
            generate::generate(&mut *tower, &ids, args.max_tokens, &mut greedy, &mut |_| {
                Ok(())
            })?;
        let mut s = session::Session::new(&mut *tower);
        let started = std::time::Instant::now();
        let mut first = None;
        let drafted = s.generate_mtp_depth(
            &mut *tower,
            &ids,
            args.max_tokens,
            depth,
            None,
            None,
            &mut |_| {
                first.get_or_insert_with(std::time::Instant::now);
                Ok(())
            },
        )?;
        let decode_secs = first.map_or(0., |f| f.elapsed().as_secs_f64());
        let total_secs = started.elapsed().as_secs_f64();
        let mismatch = serial
            .iter()
            .zip(&drafted)
            .position(|(a, b)| a != b)
            .or((serial.len() != drafted.len()).then(|| serial.len().min(drafted.len())));
        if let Some(at) = mismatch {
            failures += 1;
            if let (Some(&a), Some(&b)) = (serial.get(at), drafted.get(at)) {
                // The target's logits at the divergence: prefill the prompt
                // and the shared tokens, then one decode step, as serial does.
                let mut context = ids.clone();
                context.extend_from_slice(&serial[..at]);
                let mut s = session::Session::new(&mut *tower);
                let (prefix, last) = context.split_at(context.len() - 1);
                s.feed_multi(&mut *tower, prefix)?;
                let logits = s.feed(&mut *tower, last)?.reshape(&[-1])?;
                let values = logits
                    .as_dtype(lisa_metal::Dtype::Float32)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
                    .as_slice::<f32>()
                    .to_vec();
                println!(
                    "  divergence at {at}: serial token {a} logit {:.4}, drafted token {b} logit {:.4}, gap {:.4}",
                    values[a as usize],
                    values[b as usize],
                    values[a as usize] - values[b as usize],
                );
            }
        }
        println!(
            "spec-check prompt {index}: {} prompt tokens | serial {} tok, decode {:.1} tok/s | drafted {} tok, decode {:.1} tok/s, total {:.2}s | equal={} first_mismatch={mismatch:?}",
            ids.len(),
            serial.len(),
            stats.decode_tps(),
            drafted.len(),
            (drafted.len().saturating_sub(1)) as f64 / decode_secs.max(1e-9),
            total_secs,
            mismatch.is_none(),
        );
    }
    anyhow::ensure!(
        failures == 0,
        "{failures} prompt(s) diverged from serial greedy"
    );
    Ok(())
}
