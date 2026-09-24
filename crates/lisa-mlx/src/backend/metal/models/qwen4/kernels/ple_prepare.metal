// PLE prepare step for the S=1 fusion.
        constexpr uint H = 2560;
        constexpr uint W = 4 * H;
        const uint hc = threadgroup_position_in_grid.y;
        const uint lid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint d = lid * 4;
        const uint base = hc * H + d;
        threadgroup float partials[32];
        const float eps = as_type<float>((uint)EPS_BITS);
        float acc = 0.0f;
        for (uint i = 0; i < 4; ++i) { float k = float(key[base + i]); acc += k * k; }
        const float ik = metal::precise::rsqrt(ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) { float q = float(query[base + i]); acc += q * q; }
        const float iq = metal::precise::rsqrt(ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        InT dot = InT(0);
        for (uint i = 0; i < 4; ++i) {
            InT k = InT(float(key[base + i]) * ik);
            k = k * keyScale[base + i];
            InT q = InT(float(query[base + i]) * iq);
            q = q * queryScale[base + i];
            InT product = k * q;
            dot = product + dot;
        }
        dot = InT(0) + dot;
        dot = simd_sum(dot);
        if (sg == 0 && lane >= 20) { partials[lane] = 0.0f; }
        if (lane == 0) { partials[sg] = float(dot); }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        dot = simd_sum(InT(partials[lane]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        InT gate = dot / InT(as_type<float>((uint)DIVISOR_BITS));
        InT magnitude = metal::abs(gate);
        magnitude = metal::max(magnitude, InT(1e-6f));
        magnitude = metal::sqrt(magnitude);
        InT direction = InT((gate > InT(0)) - (gate < InT(0)));
        gate = magnitude * direction;
        const InT activation = mlx_sigmoid(gate);
        InT g[4];
        acc = 0.0f;
        for (uint i = 0; i < 4; ++i) {
            g[i] = activation * value[d + i];
            gated[base + i] = g[i];
            float v = float(g[i]);
            acc += v * v;
        }
        const float iv = metal::precise::rsqrt(ple_row_sum(acc, partials, lane, sg) / float(H) + eps);
        for (uint i = 0; i < 4; ++i) {
            InT n = InT(float(g[i]) * iv);
            full[9 * W + base + i] = n * convScale[base + i];
        }
        for (uint t = 0; t < 9; ++t) {
            for (uint i = 0; i < 4; ++i) {
                full[t * W + base + i] = convState[t * W + base + i];
            }
        }
    