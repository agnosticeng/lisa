//! Frozen token IDs from DS4 b3f50180 --raw --dump-tokens with the pinned Q2 file.
use super::*;
#[test]
#[ignore = "requires LISA_GLM53_GGUF; missing checkpoint is an error"]
fn glm53_gguf_checkpoint_and_ds4_tokenizer() -> Result<()> {
    let path = std::env::var("LISA_GLM53_GGUF").context("set LISA_GLM53_GGUF")?;
    let package = lisa_gguf::Package::open(std::path::Path::new(&path))?;
    crate::models::glm53::gguf::weights::validate(&package)?;
    let tok = Tokenizer::from_header(&package.header)?;
    let cases: &[(&str, &[u32])] = &[
        ("Hi", &[13041]),
        (
            "What's Lisa's plan? I'm ready—we're going!",
            &[
                3838, 594, 28456, 594, 3119, 30, 358, 2776, 5527, 84519, 2299, 2087, 0,
            ],
        ),
        (
            "    int x = 1234567;\n\n",
            &[262, 526, 856, 284, 220, 108714, 100461, 21, 22, 401],
        ),
        (
            "émoji🙂中abc 中文abc .foo!\n",
            &[
                963, 6355, 7754, 152402, 98322, 13674, 117604, 13674, 659, 7974, 4894,
            ],
        ),
        ("a\r\n   \tb\u{b}  ", &[64, 319, 262, 2233, 199, 256]),
        (
            "हिन्दी العربية русский Ελληνικά 日本語 한국어",
            &[
                93158, 42069, 101, 30366, 99, 43397, 137930, 145232, 128228, 137790, 130624,
                108478, 131513, 147180, 30953,
            ],
        ),
        (" abc foo end", &[4102, 13674, 134137, 7974, 378, 101, 408]),
        (
            "[gMASK]<|user|>literal</think>",
            &[154822, 154827, 35606, 154842],
        ),
        (
            "١٢٣٤٥６７８９",
            &[
                152464, 152465, 152466, 152467, 152468, 59084, 64648, 64540, 66505,
            ],
        ),
        ("x   123", &[87, 256, 220, 108714]),
    ];
    for &(input, expected) in cases {
        assert_eq!(tok.encode_rendered(input)?, expected, "{input:?}");
    }
    assert_eq!(
        tok.encode_rendered("[gMASK]<sop><|user|>Hi<|assistant|><think></think>")?,
        [154822, 154824, 154827, 13041, 154828, 154841, 154842]
    );
    Ok(())
}
