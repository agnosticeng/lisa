// Commit an accepted prefix from compact ReplaySSM delta/key/decay history.
// Same multiplication boundary and delta update as gdn_lean_two_row.metal.
const uint dv_idx = 2 * thread_position_in_grid.y;
if (dv_idx >= Dv) { return; }
const uint n = thread_position_in_grid.z;
const uint b = n / Hv, hv = n % Hv, hk = hv / (Hv / Hk);
const uint lane = thread_position_in_threadgroup.x;
const uint offset = (n * Dv + dv_idx) * Dk + 4 * lane;
float4 s0 = *reinterpret_cast<const device float4*>(state_in + offset);
float4 s1 = *reinterpret_cast<const device float4*>(state_in + offset + Dk);
for (int t = 0; t < N; ++t) {
    const float decay = g[(b * T + t) * Hv + hv];
    const float4 key = float4(*reinterpret_cast<const device metal::vec<InT, 4>*>(
        k + ((b * T + t) * Hk + hk) * Dk + 4 * lane));
    const float d0 = delta[((b * T + t) * Hv + hv) * Dv + dv_idx];
    const float d1 = delta[((b * T + t) * Hv + hv) * Dv + dv_idx + 1];
    {
        #pragma clang fp reassociate(off)
        #pragma clang fp contract(off)
        s0 = s0 * decay;
        s1 = s1 * decay;
    }
    s0 = s0 + key * d0;
    s1 = s1 + key * d1;
}
*reinterpret_cast<device float4*>(state_out + offset) = s0;
*reinterpret_cast<device float4*>(state_out + offset + Dk) = s1;
