use super::*;
use crate::models::flux3::tests::compare;

#[test]
#[ignore = "requires LISA_FLUX2_MODEL and LISA_FLUX2_FIXTURES; one GPU job at a time"]
fn klein9b_reference_stages() -> anyhow::Result<()> {
    let dir = PathBuf::from(std::env::var("LISA_FLUX2_MODEL")?);
    let fixtures = PathBuf::from(std::env::var("LISA_FLUX2_FIXTURES")?);
    let expected = weights::Weights::load_file(&fixtures.join("stages.safetensors"), |_| true)?;
    let prompt = std::fs::read_to_string(fixtures.join("prompt.txt"))?;
    compare(
        "klein9b.encoder",
        &text::encode(&dir, &prompt)?,
        &expected.get("text")?,
        0.03,
    )?;
    lisa_mlx::memory::clear_cache();
    let model = dit::Dit::load(&dir)?;
    let velocity = model.forward(
        &expected.get("latent")?.as_dtype(Dtype::Bfloat16)?,
        &expected.get("text")?.as_dtype(Dtype::Bfloat16)?,
        500.0,
        4,
        6,
    )?;
    compare(
        "klein9b.denoiser",
        &velocity,
        &expected.get("velocity")?,
        0.03,
    )?;
    Ok(())
}

#[test]
fn image_options_preserve_explicit_steps_and_reject_zero() {
    assert_eq!(Options::default().steps, None);
    assert!(
        Options {
            steps: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        Options {
            steps: Some(16),
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
}
