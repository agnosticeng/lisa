// GLM-5.3 KDA (Kimi Delta Attention) vectorized-gate recurrence.
//
// Port of the gated-delta recurrence with PER-Dk gating (g: [B, T, Hv, Dk],
// unlike the GDN scalar gate [B, T, Hv]). This mirrors ds4's
// glm53_kda_prefill_recurrence operation order: one float4 per lane, dot,
// SIMD reduction, decay, FMA update, then query dot and SIMD reduction.
// Inputs q/k/v arrive post conv+silu+l2norm; beta contains raw logits and is
// sigmoid-transformed in this kernel to match DS4's prefill recurrence; state
// layout is [B, Hv, Dv, Dk].
//
// Template: InT/StT dtypes, Dk/Dv/Hk/Hv dims, CAPTURE (speculative
// state-per-position output).

auto n = thread_position_in_grid.z;
auto b_idx = n / Hv;
auto hv_idx = n % Hv;
auto hk_idx = hv_idx / (Hv / Hk);

// q, k: [B, T, Hk, Dk]
auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

// v, y: [B, T, Hv, Dv]
auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
y += b_idx * T * Hv * Dv + hv_idx * Dv;

auto lane = thread_index_in_simdgroup;
auto dv_idx = thread_position_in_grid.y;

// g (vectorized): [B, T, Hv, Dk]
auto g_ = g + (b_idx * T * Hv + hv_idx) * Dk;
auto beta_ = beta + b_idx * T * Hv;

// state_in/out: [B, Hv, Dv, Dk]
auto i_state = state_in + (n * Dv + dv_idx) * Dk;
auto o_state = state_out + (n * Dv + dv_idx) * Dk;

auto k0 = lane * 4;
float4 h = *((device const float4 *)(i_state + k0));

for (int t = 0; t < T; ++t) {
  if (true) {
    auto q4 = *((device const float4 *)(q_ + k0));
    auto k4 = *((device const float4 *)(k_ + k0));
    auto decay4 = *((device const float4 *)(g_ + k0));
    h *= decay4;
    auto kv_mem = simd_sum(dot(h, k4));
    const float beta_value = 1.0f / (1.0f + exp(-beta_[hv_idx]));
    auto delta = (v_[dv_idx] - kv_mem) * beta_value;
    h = fma(k4, float4(delta), h);
    auto out = simd_sum(dot(h, q4));
    if (thread_index_in_simdgroup == 0) {
      y[dv_idx] = static_cast<InT>(out);
    }
  } else {
    y[dv_idx] = static_cast<InT>(0);
  }
  // Speculative capture: write the state after EVERY position (the
  // output is then [B*T, Hv, Dv, Dk], slot b*T + t). Without capture
  // this writes only the final state.
  if (CAPTURE || t == T - 1) {
    const uint slot = CAPTURE ? (b_idx * T + t) : b_idx;
    device StT* o_seq = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
    o_seq[k0 + 0] = static_cast<StT>(h[0]);
    o_seq[k0 + 1] = static_cast<StT>(h[1]);
    o_seq[k0 + 2] = static_cast<StT>(h[2]);
    o_seq[k0 + 3] = static_cast<StT>(h[3]);
  }
  // Increment data pointers to next time step
  q_ += Hk * Dk;
  k_ += Hk * Dk;
  v_ += Hv * Dv;
  y += Hv * Dv;
  g_ += Hv * Dk;
  beta_ += Hv;
}
