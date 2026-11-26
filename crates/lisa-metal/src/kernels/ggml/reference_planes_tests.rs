use super::*;
use half::f16;

/// Deterministic xorshift64* stream.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 56) as u8
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// E4M3 magnitude of a byte's low seven bits (bias 7, subnormals m 2^-9,
/// 0x7F read as the finite 480, as both decoders read it).
fn e4m3(byte: u8) -> f64 {
    let (e, m) = ((byte >> 3) & 15, byte & 7);
    if e == 0 {
        m as f64 * 2f64.powi(-9)
    } else {
        (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
    }
}
/// A GGUF NVFP4 scale byte: 0x7F is zero, bit 7 is ignored.
fn ue4m3(byte: u8) -> f64 {
    if byte == 0x7F { 0.0 } else { e4m3(byte & 0x7F) }
}
fn e2m1(code: u8) -> f64 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 != 0 { -magnitude } else { magnitude }
}
fn half_at(bytes: &[u8], at: usize) -> f64 {
    f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f64()
}
fn scale_min_k4(j: usize, q: &[u8]) -> (f64, f64) {
    if j < 4 {
        ((q[j] & 63) as f64, (q[j + 4] & 63) as f64)
    } else {
        (
            ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)) as f64,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as f64,
        )
    }
}

/// Row `row` of expert `expert` of a stored GGUF tensor, decoded exactly.
fn gguf_row(t: &GgufExpertTensor, expert: usize, row: usize, out: &mut [f64]) {
    let rb = t.row_bytes().unwrap();
    let bytes = &t.bytes[(expert * t.rows + row) * rb..][..rb];
    for (c, v) in out.iter_mut().enumerate().take(t.columns) {
        *v = match t.ggml_type {
            GGML_TYPE_NVFP4 => {
                let block = &bytes[c / 64 * 36..][..36];
                let sub = c % 64 / 16;
                let code = block[4 + sub * 8 + c % 8] >> if c % 16 < 8 { 0 } else { 4 };
                ue4m3(block[sub]) * e2m1(code & 15)
            }
            GGML_TYPE_Q8_0 => {
                let block = &bytes[c / 32 * 34..][..34];
                half_at(block, 0) * block[2 + c % 32] as i8 as f64
            }
            GGML_TYPE_Q5_K => {
                let block = &bytes[c / 256 * 176..][..176];
                let (j, l) = (c % 256 / 32, c % 32);
                let lo = (block[48 + 32 * (j / 2) + l] >> (4 * (j % 2))) & 15;
                let hi = (block[16 + l] >> j) & 1;
                let (sc, m) = scale_min_k4(j, &block[4..16]);
                half_at(block, 0) * sc * (lo + 16 * hi) as f64 - half_at(block, 2) * m
            }
            _ => unreachable!(),
        };
    }
}

fn quant_slot(e: usize) -> usize {
    8 * ((e >> 2) & 3) + 4 * (e >> 4) + (e & 3)
}
fn tile_index(row: usize, block: usize, blocks: usize) -> usize {
    (row / TILE_ROWS * blocks + block) * TILE_ROWS + row % TILE_ROWS
}

/// Element (expert, row, column) of repacked planes, decoded from the plane
/// layout of `QuantFormat.h` alone.
fn plane_value(planes: &RepackedPlanes, bytes: &[u8], expert: usize, row: usize, c: usize) -> f64 {
    let f = planes.format();
    let n = expert * planes.rows() + row;
    let groups = planes.columns() / 32;
    let (g, slot) = (c / 32, quant_slot(c % 32));
    let p0 = &bytes[tile_index(n, g, groups) * f.plane0_bytes..][..f.plane0_bytes];
    let meta = &bytes[planes.meta_offset()
        + tile_index(n, g / f.meta_groups, groups / f.meta_groups) * f.meta_bytes..]
        [..f.meta_bytes];
    match f.id {
        37 | 44 => {
            let code = (p0[slot / 2] >> (4 * (slot % 2))) & 15;
            let j = g % f.meta_groups;
            let scale = meta[2 * j + c % 32 / 16];
            let gs = f32::from_le_bytes(meta[f.meta_bytes - 4..].try_into().unwrap()) as f64;
            gs * e4m3(scale) * e2m1(code)
        }
        6 => half_at(meta, 0) * p0[slot] as i8 as f64,
        3 => {
            let word = u32::from_le_bytes(p0[slot / 8 * 4..][..4].try_into().unwrap());
            let i = slot % 8;
            let lo = (word >> ((i & 1) * 16 + 4 * (i >> 1))) & 15;
            let p1 = &bytes[planes.plane1_offset() + tile_index(n, g, groups) * 4..][..4];
            let hi = (u32::from_le_bytes(p1.try_into().unwrap()) >> slot) & 1;
            let (sc, m) = scale_min_k4(g % 8, &meta[4..16]);
            half_at(meta, 0) * sc * (lo + 16 * hi) as f64 - half_at(meta, 2) * m
        }
        _ => unreachable!(),
    }
}

/// Random valid stored bytes: NVFP4 scales mostly in `[0x18, 0x30)`, some
/// with bit 7 set, and the special 0x7F (zero) and 0xFF (480); Q8_0 and Q5_K with small positive
/// half scales.
fn tensor_bytes(
    rng: &mut Rng,
    ggml_type: u32,
    experts: usize,
    rows: usize,
    columns: usize,
) -> Vec<u8> {
    let probe = GgufExpertTensor {
        bytes: &[],
        ggml_type,
        experts,
        rows,
        columns,
    };
    let rb = probe.row_bytes().unwrap();
    let mut bytes = vec![0u8; experts * rows * rb];
    let small_half = |rng: &mut Rng, scale: f32| {
        f16::from_f32(scale * (0.5 + 0.5 * (rng.unit() + 1.0) / 2.0)).to_le_bytes()
    };
    match ggml_type {
        GGML_TYPE_NVFP4 => {
            for block in bytes.as_chunks_mut::<36>().0 {
                for s in 0..4 {
                    block[s] = match rng.below(1024) {
                        0..=15 => 0x7F,
                        16 => 0xFF,
                        17..=31 => 0x98 + rng.below(0x18) as u8,
                        _ => 0x18 + rng.below(0x18) as u8,
                    };
                }
                for b in &mut block[4..] {
                    *b = rng.byte();
                }
            }
        }
        GGML_TYPE_Q8_0 => {
            for block in bytes.as_chunks_mut::<34>().0 {
                block[..2].copy_from_slice(&small_half(rng, 1.0 / 64.0));
                for b in &mut block[2..] {
                    *b = rng.byte();
                }
            }
        }
        GGML_TYPE_Q5_K => {
            for block in bytes.as_chunks_mut::<176>().0 {
                block[..2].copy_from_slice(&small_half(rng, 1.0 / 256.0));
                block[2..4].copy_from_slice(&small_half(rng, 1.0 / 256.0));
                for b in &mut block[4..] {
                    *b = rng.byte();
                }
            }
        }
        _ => unreachable!(),
    }
    bytes
}

#[test]
fn repacked_planes_decode_to_the_stored_tensor() {
    let stream = Stream::gpu();
    let mut rng = Rng(0x5eed_0001);
    // (type, experts, rows, columns, first row, rows taken): NVFP4 of whole
    // 256 blocks and of 64-element blocks (Inkling's down width, 1152),
    // Q5_K and Q8_0, with row ranges as the gate/up split takes them.
    for (ggml_type, experts, rows, columns, first, taken, format) in [
        (GGML_TYPE_NVFP4, 3, 256, 2304, 128, 128, PLANE_NVFP4),
        (GGML_TYPE_NVFP4, 3, 128, 1152, 0, 128, PLANE_NVFP464),
        (GGML_TYPE_NVFP4, 2, 64, 320, 0, 64, PLANE_NVFP464),
        (GGML_TYPE_Q5_K, 3, 128, 512, 64, 64, PLANE_Q5K),
        (GGML_TYPE_Q8_0, 3, 128, 1152, 0, 128, PLANE_Q80),
    ] {
        let bytes = tensor_bytes(&mut rng, ggml_type, experts, rows, columns);
        let tensor = GgufExpertTensor {
            bytes: &bytes,
            ggml_type,
            experts,
            rows,
            columns,
        };
        let planes = repack_planes(&tensor, first, taken, &stream).unwrap();
        assert_eq!(planes.format(), format);
        let groups = columns / 32;
        assert_eq!(
            planes.bytes(),
            experts
                * taken
                * (groups * (format.plane0_bytes + format.plane1_bytes)
                    + groups / format.meta_groups * format.meta_bytes)
        );
        let plane_bytes = planes.storage().as_slice::<u8>().to_vec();
        let mut stored = vec![0f64; columns];
        for expert in 0..experts {
            for row in 0..taken {
                gguf_row(&tensor, expert, first + row, &mut stored);
                for (c, &want) in stored.iter().enumerate() {
                    let got = plane_value(&planes, &plane_bytes, expert, row, c);
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "type {ggml_type} expert {expert} row {row} column {c}: {got} vs {want}"
                    );
                }
            }
        }
    }
}
