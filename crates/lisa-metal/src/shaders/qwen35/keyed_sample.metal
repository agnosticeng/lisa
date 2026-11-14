// Keyed Gumbel-max sampling (specs/08 §5): the whole temp>0 draw in
// ONE kernel, no vocab-wide sort or partition. The token at absolute
// position p is argmax_i(v_i/T + g(seed, p, i)) where g is Gumbel noise
// keyed by a splitmix64 hash of (seed, position, token id). The noise is a
// pure function of (seed, position), so a verify row draws exactly what the
// serial step draws. Over the full vocabulary this is exact multinomial
// sampling from softmax(v/T) (no top-k/top-p window), with ties going to
// the earlier token (same convention as argmax).
//
// Grid: ONE threadgroup of 1024 threads per row; each thread strides the
// vocabulary, then (score, index) reduces through a simd butterfly and the
// simdgroup partials. The host reads back the single i32.

inline uint tf_mix(ulong x) {
  x ^= x >> 30; x *= 0xBF58476D1CE4E5B9UL; x ^= x >> 27; x *= 0x94D049BB133111EBUL;
  return uint(x ^ (x >> 31)) + uint((x ^ (x >> 31)) >> 32);
}
inline float tf_uniform(ulong seed, uint pos, uint id) {
  ulong x = tf_mix2(seed + 0x9E3779B97F4A7C15UL);
  x = tf_mix2(x ^ (ulong(pos) * 0xD1B54A32D192ED03UL));
  x = tf_mix2(x ^ ulong(id));
  return (float(uint(x >> 40)) + 0.5f) * (1.0f / 16777216.0f);
}

  constexpr uint TG = 1024;
  constexpr uint NSG = TG / 32;
  const uint t = thread_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const ulong seed = (ulong(seed_lo) & 0xFFFFFFFFUL) | (ulong(seed_hi) << 32);
  const uint position = uint(pos_i);
  const float inv_t = inv_temperature;

  threadgroup float fsh[NSG];
  threadgroup uint ish[NSG];

  float bs = -INFINITY;
  uint bi = 0;
  for (uint i = t; i < V; i += TG) {
    const float u = tf_uniform(seed, position, i);
    const float score = static_cast<float>(L[i]) * inv_t - metal::log(-metal::log(u));
    if (score > bs || (score == bs && i < bi)) {
      bs = score;
      bi = i;
    }
  }
  for (uint off = 16u; off >= 1u; off >>= 1u) {
    const float os = simd_shuffle(bs, lane ^ off);
    const uint oi = simd_shuffle(bi, lane ^ off);
    if (os > bs || (os == bs && oi < bi)) {
      bs = os;
      bi = oi;
    }
  }
  if (sg == 0) {
    fsh[lane] = -INFINITY;
    ish[lane] = 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    fsh[sg] = bs;
    ish[sg] = bi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float m = -INFINITY;
    uint mi = 0u;
    for (uint s = 0; s < NSG; ++s) {
      if (fsh[s] > m || (fsh[s] == m && ish[s] < mi)) {
        m = fsh[s];
        mi = ish[s];
      }
    }
    out[0] = static_cast<int>(mi);
  }
