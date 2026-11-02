# GLM tokenizer regression fixtures

These portable inputs belong to the tokenizer tests in `../../tokenizer.rs`.
The real-tokenizer test remains enabled by `LISA_GLM53_MODEL=<snapshot>`; it
resolves these files from `CARGO_MANIFEST_DIR`, independently of the current
working directory. Missing files fail the enabled test.

- `prompt-row-1.u32` contains 8,193 little-endian u32 IDs encoded by the MLX
  snapshot tokenizer with its original `ignore_merges=true` setting.
- `prompt-row-1.ds4.u32` contains the 8,195 IDs produced by DS4's `glm4` BPE
  tokenizer for the decoded source row. The test decodes the source IDs with
  the model tokenizer, then checks the complete DS4 token sequence.
- `tokenizer-ds4-merge-cases-20261101.json` contains twelve focused merge-rank
  examples and the source revision and tokenizer identity.
- `manifest.json` records the byte length and SHA-256 of every input.

The prose row comes from Alessandro Manzoni's public-domain *I promessi sposi*
text in DS4's `speed-bench/promessi_sposi.txt`, source token range
`[15984, 24177)`, beginning at a paragraph boundary. Source SHA-256:
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
The original source IDs were generated with `tokenizers` 0.22.2. DS4 captures
use revision `c97964b8aa77034a8e17afb10ca7a686c40ab511`; the MLX tokenizer JSON
SHA-256 is `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`.

The files were preserved byte-for-byte from the 2026-11-01 tokenizer campaign.
No model weights are included.
