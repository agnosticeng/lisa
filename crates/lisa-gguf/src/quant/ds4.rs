//! Original DS4 MXFP8 row decoder (ds4_engram.c, MIT).
//! Revision b3f50180ec2c7324892be1f50bc83555ccb8d5e7; see LICENSE-DS4.
//! This custom 264-byte row is explicitly requested by the adapter. It does
//! not change the standard GGUF meaning of tensor kind 24 (I8).
use anyhow::{Result, ensure};
pub fn decode_ds4_mxfp8(packed: &[u8], output: &mut [f32; 256]) -> Result<()> {
    ensure!(packed.len() == 264, "invalid MXFP8 Engram row");
    for i in 0..256 {
        let byte = packed[i];
        let exponent = (byte >> 3) & 15;
        let mantissa = byte & 7;
        let scale = packed[256 + i / 32];
        ensure!(
            byte & 127 != 127 && scale != 255,
            "non-finite MXFP8 Engram value"
        );
        let base = if exponent == 0 {
            mantissa as f32 * 2f32.powi(-9)
        } else {
            (8 + mantissa) as f32 * 2f32.powi(exponent as i32 - 10)
        };
        // Scale in f64 then round once to f32, matching ldexpf even when the
        // scale itself would overflow or underflow as a standalone f32.
        let value = ((if byte & 128 != 0 {
            -(base as f64)
        } else {
            base as f64
        }) * 2f64.powi(scale as i32 - 127)) as f32;
        let rounded = half::bf16::from_f32(value).to_f32();
        ensure!(rounded.is_finite(), "overflow in MXFP8 Engram row");
        output[i] = rounded;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engram_mxfp8_special_values() {
        let mut row = [0u8; 264];
        row[256..].fill(127);
        row[0] = 0x38; // 1.0
        row[1] = 0xb8; // -1.0
        row[2] = 0x01; // minimum E4M3 subnormal
        let mut out = [0f32; 256];
        decode_ds4_mxfp8(&row, &mut out).unwrap();
        assert_eq!(&out[..3], &[1.0, -1.0, 1.0 / 512.0]);
        row[0] = 0x7f;
        assert!(decode_ds4_mxfp8(&row, &mut out).is_err());
    }
}
