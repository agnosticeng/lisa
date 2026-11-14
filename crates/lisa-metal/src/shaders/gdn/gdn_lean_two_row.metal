// GDN recurrence, two value rows per simdgroup (`track_gdn_lean_two_row`).
//
// The recurrence is the sequential GDN step: state in registers, Kahan-style
// compensated f32 accumulation over the key axis, then the delta-rule update.
// Each value row's operations are emitted in the same order as the one-row
// form, so the output is independent of how many rows share a simdgroup.

const uint dv_idx = 2 * thread_position_in_grid.y;
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

float state0[n_per_t], state1[n_per_t];
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 0 * Dk); state0[0] = s4.x; state0[1] = s4.y; state0[2] = s4.z; state0[3] = s4.w; }
{ const float4 s4 = *reinterpret_cast<const device float4*>((const device float*)i_state + 1 * Dk); state1[0] = s4.x; state1[1] = s4.y; state1[2] = s4.z; state1[3] = s4.w; }

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
    const metal::vec<InT, 2> vv = *reinterpret_cast<const device metal::vec<InT, 2>*>(v_ + vy_off);

    float kv_mem0, kv_mem1;
    {
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        float kv_compensation0, kv_compensation1;
        state0[0] = state0[0] * decay; kv_mem0 = 0.0f + state0[0] * keys[0]; kv_compensation0 = 0.0f;
        state1[0] = state1[0] * decay; kv_mem1 = 0.0f + state1[0] * keys[0]; kv_compensation1 = 0.0f;
        for (int i = 1; i < n_per_t; ++i) {
            const float key = keys[i];
            { state0[i] = state0[i] * decay; auto product = state0[i] * key; auto corrected = product - kv_compensation0; auto next_sum = kv_mem0 + corrected; if (i + 1 < n_per_t) { kv_compensation0 = (next_sum - kv_mem0) - corrected; } kv_mem0 = next_sum; }
            { state1[i] = state1[i] * decay; auto product = state1[i] * key; auto corrected = product - kv_compensation1; auto next_sum = kv_mem1 + corrected; if (i + 1 < n_per_t) { kv_compensation1 = (next_sum - kv_mem1) - corrected; } kv_mem1 = next_sum; }
        }
    }
    kv_mem0 = simd_sum(kv_mem0);
    kv_mem1 = simd_sum(kv_mem1);
    const float delta0 = (static_cast<float>(vv[0]) - kv_mem0) * gate_beta;
    const float delta1 = (static_cast<float>(vv[1]) - kv_mem1) * gate_beta;

    float out0 = 0.0f, out1 = 0.0f;
    for (int i = 0; i < n_per_t; ++i) {
        const float key = keys[i];
        const float query = queries[i];
        state0[i] = state0[i] + key * delta0; out0 += state0[i] * query;
        state1[i] = state1[i] + key * delta1; out1 += state1[i] * query;
    }
    out0 = simd_sum(out0);
    out1 = simd_sum(out1);

    if (dk_idx == 0) {
        y_[vy_off + 0] = static_cast<InT>(out0);
        y_[vy_off + 1] = static_cast<InT>(out1);
    }
    if (CAPTURE || t == T_ - 1) {
        const uint slot = CAPTURE ? (b_idx * T_ + t) : b_idx;
        device StT* o_state = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk + 4 * dk_idx;
        *reinterpret_cast<device float4*>((device float*)o_state + 0 * Dk) = float4(state0[0], state0[1], state0[2], state0[3]);
        *reinterpret_cast<device float4*>((device float*)o_state + 1 * Dk) = float4(state1[0], state1[1], state1[2], state1[3]);
    }
    kq_off += Hk * Dk; vy_off += Hv * Dv; gb_off += Hv;
}