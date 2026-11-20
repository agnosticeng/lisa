//! Opt-in same-checkpoint checks. Missing data is an error when explicitly run.
use lisa_backend::TensorAccess;

#[test]
#[ignore = "requires LISA_OLALA_MODEL pointing to the pinned Q5_K_M GGUF"]
fn olala_prefix_forks_and_chunked_prefill_are_exact() -> anyhow::Result<()> {
    let path = std::env::var_os("LISA_OLALA_MODEL")
        .ok_or_else(|| anyhow::anyhow!("set LISA_OLALA_MODEL to the pinned Q5_K_M GGUF"))?;
    let path = std::path::Path::new(&path);
    anyhow::ensure!(
        lisa_engine::checkpoint::model_type_of(path)? == "olala",
        "expected Olala"
    );
    let crate::Loaded::Language(mut model) = crate::loading::load_dir(path)? else {
        anyhow::bail!("expected a language model");
    };
    let tokenizer = lisa_engine::tokenizer::Tokenizer::load(path)?;
    let tokens = tokenizer.encode_plain(
        "The capital of France is Paris. Explain why it is known for art and architecture.",
    )?;
    let (equal, logits, hidden) =
        lisa_engine::session_check::verify_incremental(model.as_mut(), &tokens, 4)?;
    assert!(equal);
    assert_eq!(logits, 0.);
    assert_eq!(hidden, 0.);

    let mut caches = model.new_caches();
    model.prefill(&tokens[..5], &mut caches)?;
    let saved: Vec<_> = caches.iter().map(|c| c.snapshot_prefix()).collect();
    let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
    let first = model.prefill(&tokens[5..], &mut caches)?;
    let first = bits(model.head(&first)?.read_f32()?);
    for (c, s) in caches.iter_mut().zip(&saved) {
        c.restore_prefix(s);
    }
    model.prefill(&[40, 1091, 13], &mut caches)?;
    for (c, s) in caches.iter_mut().zip(&saved) {
        c.restore_prefix(s);
    }
    let repeated = model.prefill(&tokens[5..], &mut caches)?;
    assert_eq!(first, bits(model.head(&repeated)?.read_f32()?));
    Ok(())
}
