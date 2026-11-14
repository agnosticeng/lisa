// GDN recurrence, four value rows per simdgroup (`track_gdn_rows`, the prefill
// form). Same per-row operation order as the two-row kernel, so the outputs are
// identical regardless of how many rows share a simdgroup.

const uint dv_idx = 4 * thread_position_in_grid.y;
if (dv_idx >= Dv) { return; }
const uint n = thread_position_in_grid.z;
const uint b_idx = n / Hv;
const uint hv_idx = n % Hv;
const uint hk_idx = hv_idx / (Hv / Hk);
constexpr int n_per_t = Dk / 32;
static_assert(n_per_t == 4 && metal::is_same<StT, float>::value, "vector load shape");
const int T_ = T;
const uint dk_idx = thread_position_in_threadgroup.x;
const device InT* q_ = q + b_idx * T_ * Hk * Dk + hk_idx * Dk + 4 * dk_idx;
const device InT* k_ = k + b_idx * T_ * Hk * Dk + hk_idx * Dk + 4 * dk_idx;
const device InT* v_ = v + b_idx * T_ * Hv * Dv + hv_idx * Dv + dv_idx;
device InT* y_ = y + b_idx * T_ * Hv * Dv + hv_idx * Dv + dv_idx;
const device float* g_ = g + b_idx * T_ * Hv;
const device float* beta_ = beta + b_idx * T_ * Hv;
const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk + 4 * dk_idx;
uint kq_off = 0, vy_off = 0, gb_off = hv_idx;

float state0[n_per_t], state1[n_per_t], state2[n_per_t], state3[n_per_t];
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 0 * Dk); state0[0] = s4.x; state0[1] = s4.y; state0[2] = s4.z; state0[3] = s4.w; }
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 1 * Dk); state1[0] = s4.x; state1[1] = s4.y; state1[2] = s4.z; state1[3] = s4.w; }
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 2 * Dk); state2[0] = s4.x; state2[1] = s4.y; state2[2] = s4.z; state2[3] = s4.w; }
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 3 * Dk); state3[0] = s4.x; state3[1] = s4.y; state3[2] = s4.z; state3[3] = s4.w; }

for (int t = 0; t < T_; ++t) {
    float keys[n_per_t], queries[n_per_t];
    {
        const metal::vec<InT, 4> kk = *reinterpret_cast<const device metal::vec<InT, 4>*>(k_ + kq_off);
        const metal::vec<InT, 4> qq = *reinterpret_cast<const device metal::vec<InT, 4>*>(q_ + kq_off);
        for (int i = 0; i < n_per_t; ++i) {
            keys[i] = static_cast<float>(kk[i]);
            queries[i] = static_cast<float>(qq[i]);
        }
    }
    const float decay = g_[gb_off];
    const float gate_beta = beta_[gb_off];
    const metal::vec<InT, 4> vv = *reinterpret_cast<const device metal::vec<InT, 4>*>(v_ + vy_off);

    float kv_mem0, kv_mem1, kv_mem2, kv_mem3;
    {
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        float kv_compensation0, kv_compensation1, kv_compensation2, kv_compensation3;
        state0[0] = state0[0] * decay; kv_mem0 = 0.0f + state0[0] * keys[0]; kv_compensation0 = 0.0f;
        state1[0] = state1[0] * decay; kv_mem1 = 0.0f + state1[0] * keys[0]; kv_compensation1 = 0.0f;
        state2[0] = state2[0] * decay; kv_mem2 = 0.0f + state2[0] * keys[0]; kv_compensation2 = 0.0f;
        state3[0] = state3[0] * decay; kv_mem3 = 0.0f + state3[0] * keys[0]; kv_compensation3 = 0.0f;
        for (int i = 1; i < n_per_t; ++i) {
            const float key = keys[i];
            { state0[i] = state0[i] * decay; auto product = state0[i] * key; auto corrected = product - kv_compensation0; auto next_sum = kv_mem0 + corrected; if (i + 1 < n_per_t) { kv_compensation0 = (next_sum - kv_mem0) - corrected; } kv_mem0 = next_sum; }
            { state1[i] = state1[i] * decay; auto product = state1[i] * key; auto corrected = product - kv_compensation1; auto next_sum = kv_mem1 + corrected; if (i + 1 < n_per_t) { kv_compensation1 = (next_sum - kv_mem1) - corrected; } kv_mem1 = next_sum; }
            { state2[i] = state2[i] * decay; auto product = state2[i] * key; auto corrected = product - kv_compensation2; auto next_sum = kv_mem2 + corrected; if (i + 1 < n_per_t) { kv_compensation2 = (next_sum - kv_mem2) - corrected; } kv_mem2 = next_sum; }
            { state3[i] = state3[i] * decay; auto product = state3[i] * key; auto corrected = product - kv_compensation3; auto next_sum = kv_mem3 + corrected; if (i + 1 < n_per_t) { kv_compensation3 = (next_sum - kv_mem3) - corrected; } kv_mem3 = next_sum; }
        }
    }
    kv_mem0 = simd_sum(kv_mem0);
    kv_mem1 = simd_sum(kv_mem1);
    kv_mem2 = simd_sum(kv_mem2);
    kv_mem3 = simd_sum(kv_mem3);
    const float delta0 = (static_cast<float>(vv[0]) - kv_mem0) * gate_beta;
    const float delta1 = (static_cast<float>(vv[1]) - kv_mem1) * gate_beta;
    const float delta2 = (static_cast<float>(vv[2]) - kv_mem2) * gate_beta;
    const float delta3 = (static_cast<float>(vv[3]) - kv_mem3) * gate_beta;

    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        const float key = keys[i];
        const float query = queries[i];
        state0[i] = state0[i] + key * delta0; out0 += state0[i] * query;
        state1[i] = state1[i] + key * delta1; out1 += state1[i] * query;
        state2[i] = state2[i] + key * delta2; out2 += state2[i] * query;
        state3[i] = state3[i] + key * delta3; out3 += state3[i] * query;
    }
    out0 = simd_sum(out0);
    out1 = simd_sum(out1);
    out2 = simd_sum(out2);
    out3 = simd_sum(out3);

    if (dk_idx == 0) {
        y_[vy_off + 0] = static_cast<InT>(out0);
        y_[vy_off + 1] = static_cast<InT>(out1);
        y_[vy_off + 2] = static_cast<InT>(out2);
        y_[vy_off + 3] = static_cast<InT>(out3);
    }
    if (CAPTURE || t == T_ - 1) {
        const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
        device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk + 4 * dk_idx;
        *reinterpret_cast<device float4*>((device float*)o_state + 0 * Dk) = float4(state0[0], state0[1], state0[2], state0[3]);
        *reinterpret_cast<device float4*>((device float*)o_state + 1 * Dk) = float4(state1[0], state1[1], state1[2], state1[3]);
        *reinterpret_cast<device float4*>((device float*)o_state + 2 * Dk) = float4(state2[0], state2[1], state2[2], state2[3]);
        *reinterpret_cast<device float4*>((device float*)o_state + 3 * Dk) = float4(state3[0], state3[1], state3[2], state3[3]);
    }
    kq_off += Hk * Dk; vy_off += Hv * Dv; gb_off += Hv;
}