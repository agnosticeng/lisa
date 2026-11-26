//! Checkpoint-defined n-gram hash geometry and EOS-delimited row selection.
//! No storage access or device allocation.
pub struct NgramConstants {
    pub sizes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub multipliers: Vec<i64>,
    pub rows_per_shard: usize,
    pub shard_count: usize,
}

fn splitmix64(mut v: u64) -> u64 {
    v = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    v = (v ^ (v >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    v = (v ^ (v >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    v ^ (v >> 31)
}

fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut d: usize = 3;
    while d.checked_mul(d).is_some_and(|dd| dd <= value) {
        if value.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

fn nth_prime_after(start: usize, count: usize) -> usize {
    let mut p = start;
    for _ in 0..count {
        p += 1;
        while !is_prime(p) {
            p += 1;
        }
    }
    p
}

impl NgramConstants {
    pub fn new(
        ngram_size: usize,
        heads_per_ngram: usize,
        vocab_size: usize,
        ngram_vocab_size_base: usize,
        divisible_by: usize,
        split_parts: usize,
        seed: i64,
        ple_layer_index: usize,
    ) -> Self {
        let ngram_heads = (ngram_size - 1) * heads_per_ngram;
        let mut sizes = Vec::with_capacity(ngram_heads);
        let mut offsets = Vec::with_capacity(ngram_heads);
        let mut total: usize = 0;
        for head in 0..ngram_heads {
            let global = ple_layer_index * ngram_heads + head;
            let size = nth_prime_after(ngram_vocab_size_base - 1, global + 1);
            sizes.push(size as i64);
            offsets.push(total as i64);
            total += size;
        }
        let padded = total.div_ceil(divisible_by) * divisible_by;
        let shard_count = split_parts;
        let rows_per_shard = padded.div_ceil(shard_count);

        let gamma: u64 = 0x9E37_79B9_7F4A_7C15;
        let max_long = i64::MAX as u64;
        let half = 1u64.max((max_long / vocab_size.max(1) as u64) / 2);
        let base_seed = (seed as u64).wrapping_add(10007u64.wrapping_mul(ple_layer_index as u64));
        let mut multipliers = Vec::with_capacity(ngram_size);
        for i in 0..ngram_size {
            let mixed = splitmix64(base_seed.wrapping_add(gamma.wrapping_mul((i + 1) as u64)));
            multipliers.push((2u64.wrapping_mul(mixed % half).wrapping_add(1)) as i64);
        }
        Self {
            sizes,
            offsets,
            multipliers,
            rows_per_shard,
            shard_count,
        }
    }
}

impl NgramConstants {
    pub fn row_ids(
        &self,
        history: &[i64],
        new_count: usize,
        eos_token_id: i64,
        ngram_size: usize,
        heads_per_ngram: usize,
    ) -> Vec<i64> {
        let eos = eos_token_id;
        let t = history.len();
        // previous[t] = position of the last EOS strictly before t, or -1.
        let mut previous = vec![-1i64; t];
        let mut last = -1i64;
        for (i, &tok) in history.iter().enumerate() {
            previous[i] = last;
            if tok == eos {
                last = i as i64;
            }
        }
        let shifted = |s: usize, ti: usize| -> i64 {
            if s == 0 {
                return history[ti];
            }
            let in_segment = ti as i64 - (previous[ti] + 1);
            let source = ti as i64 - s as i64;
            if in_segment >= s as i64 && source >= 0 {
                history[source as usize]
            } else {
                eos
            }
        };

        let ngram_heads = (ngram_size - 1) * heads_per_ngram;
        let mut out = Vec::with_capacity(new_count * ngram_heads);
        for ti in t.saturating_sub(new_count)..t {
            for ngram in 2..=ngram_size {
                let mut mixed = shifted(0, ti).wrapping_mul(self.multipliers[0]);
                for p in 1..ngram {
                    mixed ^= shifted(p, ti).wrapping_mul(self.multipliers[p]);
                }
                let low = (ngram - 2) * heads_per_ngram;
                for head in low..low + heads_per_ngram {
                    let size = self.sizes[head];
                    let mut r = mixed % size;
                    if r != 0 && (r < 0) != (size < 0) {
                        r += size;
                    }
                    out.push(r + self.offsets[head]);
                }
            }
        }
        out
    }
}

impl NgramConstants {
    /// Request rows share a rectangular token window; previous histories may
    /// have different lengths. The hash law is applied independently per row.
    pub fn batch_ids(
        &self,
        ids: &[Vec<i64>],
        history: &[Vec<i64>],
        eos: i64,
        size: usize,
        heads: usize,
    ) -> anyhow::Result<Vec<i64>> {
        anyhow::ensure!(
            !ids.is_empty() && ids.len() == history.len(),
            "invalid n-gram batch"
        );
        let count = ids[0].len();
        anyhow::ensure!(
            count > 0 && ids.iter().all(|row| row.len() == count),
            "invalid n-gram token window"
        );
        let mut out = Vec::new();
        for (row, previous) in ids.iter().zip(history) {
            let mut tokens = previous.clone();
            tokens.extend_from_slice(row);
            out.extend(self.row_ids(&tokens, count, eos, size, heads));
        }
        Ok(out)
    }
}
/// Physical layout of one original affine n-gram shard, never requantized.
pub fn table_rows(weight: &[i32], scales: &[i32], biases: &[i32]) -> anyhow::Result<usize> {
    anyhow::ensure!(
        weight.len() == 2
            && weight[0] > 0
            && weight[1] == 20
            && scales == [weight[0], 5]
            && biases == scales,
        "invalid n-gram affine storage geometry"
    );
    Ok(weight[0] as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn checkpoint_hash_geometry_and_eos_boundaries() {
        // Frozen independent Python integer oracle: explicit 64-bit wrapping,
        // signed XOR and Python modulo. Covers consecutive EOS and chunk tails.
        let c = NgramConstants::new(3, 8, 248320, 20_000_000, 128, 128, 0, 0);
        assert_eq!(
            c.sizes,
            [
                20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171
            ]
        );
        assert_eq!(
            c.offsets,
            [
                0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                300001275
            ]
        );
        assert_eq!(
            c.multipliers,
            [4788054244585, 5075510189727, 24189832309785]
        );
        assert_eq!(c.rows_per_shard, 2500012);
        assert_eq!(c.shard_count, 128);
        let history = [42, 248044, 248044, 1, 99, 248044, 17];
        let expected = [
            7784742, 25825759, 45790359, 74798177, 80643814, 109460509, 122874277, 141111757,
            178419460, 194363695, 207308574, 238638683, 245631466, 273483518, 289625171, 311764813,
            3433720, 27997241, 51169813, 72609090, 86197738, 104250696, 131508297, 148184264,
            172729053, 187400719, 202260503, 236597571, 253091678, 270457999, 289774038, 307807279,
            18066362, 32697499, 50067959, 74448077, 95402405, 102398889, 132904658, 146932754,
            174590358, 196286984, 213661443, 226745839, 248799404, 271920135, 286530913, 301362920,
            1487290, 28647717, 53172172, 68563908, 99881718, 110522424, 126672327, 148558043,
            170538844, 198251385, 211592489, 235482715, 248359507, 262095397, 286865098, 312144638,
            1291400, 31248675, 56227694, 67198687, 85174216, 111166198, 130154186, 142138260,
            170986621, 196702474, 202164401, 235885436, 250083796, 265362198, 284028448, 319159295,
            10792645, 24504341, 52250792, 76093255, 83170100, 119052472, 133054026, 145388555,
            164953376, 188966359, 214728705, 237604205, 241854830, 266319340, 287855114, 315889570,
            6865253, 37320032, 43491690, 65189619, 90484199, 105783860, 138922245, 150125846,
            163661467, 182925333, 216409719, 232137102, 250789459, 270300902, 283662344, 311901042,
        ];
        assert_eq!(c.row_ids(&history, history.len(), 248044, 3, 8), expected);
        for count in 0..=history.len() {
            assert_eq!(
                c.row_ids(&history, count, 248044, 3, 8),
                expected[(history.len() - count) * 16..]
            );
        }
        assert!(c.row_ids(&[], 0, 248044, 3, 8).is_empty());
    }
    #[test]
    fn signed_wrapping_hash_uses_positive_modulo() {
        let c = NgramConstants {
            sizes: vec![7, 11],
            offsets: vec![0, 7],
            multipliers: vec![i64::MAX, 3, -7],
            rows_per_shard: 18,
            shard_count: 1,
        };
        assert_eq!(c.row_ids(&[2, 3], 2, 0, 3, 1), [5, 16, 3, 10]);
    }
}
