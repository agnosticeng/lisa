// One-token GDN: conv prep, state transition and gated normalization in one launch.
        static_assert(Dk == 128 && Dv == 128, "GDN head geometry");
        const uint n = threadgroup_position_in_grid.z;
        const uint b_idx = n / Hv;
        const uint hv_idx = n % Hv;
        const uint hk_idx = hv_idx / (Hv / Hk);
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lane = thread_index_in_simdgroup;
        constexpr int KM1 = KC - 1;
        threadgroup InT q_shared[Dk];
        threadgroup InT k_shared[Dk];
        threadgroup InT v_shared[Dv];
        threadgroup float gb_shared[2];
        float4 next_state;
        if constexpr (metal::is_same<StT, float>::value) {
            const device StT* first_state = state_in + (n * Dv + sg * RPS) * Dk;
            next_state = *reinterpret_cast<const device float4*>(first_state + 4 * lane);
        }
        if (sg < 3) {
            const uint vec = sg == 0 ? hk_idx : (sg == 1 ? Hk + hk_idx : 2 * Hk + hv_idx);
            const device InT* proj_b = proj + b_idx * PW;
            const device InT* cst_b = conv_state + b_idx * KM1 * CONV_DIM;
            auto win = [&](int r, uint ch) -> float {
                if (r < KM1) { return static_cast<float>(cst_b[(uint)(r * CONV_DIM) + ch]); }
                return static_cast<float>(proj_b[(uint)((r - KM1) * PW) + ch]);
            };
            float thread_x[4];
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                const uint ch = vec * 128 + lane * 4 + i;
                float cacc = 0.0f;
                for (int j = 0; j < KC; ++j) {
                    cacc += win(j, ch) * conv_w[ch * KC + j];
                }
                const InT c0 = static_cast<InT>(cacc);
                const InT c1 = mlx_silu(c0);
                thread_x[i] = static_cast<float>(c1);
                acc += thread_x[i] * thread_x[i];
            }
            if (sg < 2) {
                acc = simd_sum(acc);
                const float inv_mean = metal::precise::rsqrt(acc / 128.0f + 1e-6f);
                const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
                const InT q_mul = static_cast<InT>(inv_scale * inv_scale);
                const InT k_mul = static_cast<InT>(inv_scale);
                for (int i = 0; i < 4; ++i) {
                    const InT normalized = static_cast<InT>(thread_x[i] * inv_mean);
                    const uint d = lane * 4 + i;
                    if (sg == 0) { q_shared[d] = q_mul * normalized; }
                    else { k_shared[d] = k_mul * normalized; }
                }
            } else {
                for (int i = 0; i < 4; ++i) { v_shared[lane * 4 + i] = static_cast<InT>(thread_x[i]); }
            }
            if (sg == 2 || hv_idx % (Hv / Hk) == 0) {
                device InT* o_conv = conv_out + b_idx * KM1 * CONV_DIM;
                for (int i = 0; i < 4; ++i) {
                    const uint ch = vec * 128 + lane * 4 + i;
                    for (int j = 0; j < KM1; ++j) {
                        o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win(1 + j, ch));
                    }
                }
            }
        }
        if (sg == 0 && lane == 0) {
            const device InT* row = proj + b_idx * PW;
            const InT b_raw = row[B_OFF + hv_idx];
            gb_shared[1] = static_cast<float>(mlx_sigmoid(b_raw));
            const InT ax = row[A_OFF + hv_idx] + dt_bias[hv_idx];
            const InT sp = mlx_logaddexp0(ax);
            gb_shared[0] = metal::precise::exp(neg_exp_alog[hv_idx] * sp);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const threadgroup InT* q_ = q_shared;
        const threadgroup InT* k_ = k_shared;
        const threadgroup InT* v_ = v_shared;
        const float gate_decay = gb_shared[0];
        const float gate_beta = gb_shared[1];
        const float4 local_q = float4(
            static_cast<float>(q_[4 * lane]), static_cast<float>(q_[4 * lane + 1]),
            static_cast<float>(q_[4 * lane + 2]), static_cast<float>(q_[4 * lane + 3]));
        const float4 local_k = float4(
            static_cast<float>(k_[4 * lane]), static_cast<float>(k_[4 * lane + 1]),
            static_cast<float>(k_[4 * lane + 2]), static_cast<float>(k_[4 * lane + 3]));
        threadgroup InT y_shared[Dv];
        for (int r = 0; r < RPS; ++r) {
            const uint dv_idx = sg * RPS + r;
            const device StT* i_state = state_in + (n * Dv + dv_idx) * Dk;
            float state[4];
            constexpr bool vec4 = metal::is_same<StT, float>::value;
            if constexpr (vec4) {
                const float4 s4 = next_state;
                if (r + 1 < RPS) {
                    next_state = *reinterpret_cast<const device float4*>(i_state + Dk + 4 * lane);
                }
                state[0] = s4.x; state[1] = s4.y; state[2] = s4.z; state[3] = s4.w;
            } else {
                for (int i = 0; i < 4; ++i) { state[i] = static_cast<float>(i_state[4 * lane + i]); }
            }
            float kv_mem;
            {
                #pragma clang fp reassociate(off)
                #pragma clang fp contract(off)
                state[0] = state[0] * gate_decay;
                kv_mem = 0.0f + state[0] * local_k[0];
                float kv_compensation = 0.0f;
                for (int i = 1; i < 4; ++i) {
                    state[i] = state[i] * gate_decay;
                    auto product = state[i] * local_k[i];
                    auto corrected = product - kv_compensation;
                    auto next_sum = kv_mem + corrected;
                    if (i + 1 < 4) { kv_compensation = (next_sum - kv_mem) - corrected; }
                    kv_mem = next_sum;
                }
            }
            kv_mem = simd_sum(kv_mem);
            const float delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * gate_beta;
            float out = 0.0f;
            for (int i = 0; i < 4; ++i) {
                state[i] = state[i] + local_k[i] * delta;
                out += state[i] * local_q[i];
            }
            out = simd_sum(out);
            if (lane == 0) {
                const InT value = static_cast<InT>(out);
                y_shared[dv_idx] = value;
            }
            device StT* o_state = state_out + (n * Dv + dv_idx) * Dk;
            if constexpr (vec4) {
                *reinterpret_cast<device float4*>(o_state + 4 * lane) = float4(state[0], state[1], state[2], state[3]);
            } else {
                for (int i = 0; i < 4; ++i) { o_state[4 * lane + i] = static_cast<StT>(state[i]); }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float thread_x[4];
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
                thread_x[i] = static_cast<float>(y_shared[lane * 4 + i]);
                acc += thread_x[i] * thread_x[i];
            }
            acc = simd_sum(acc);
            const float inv_mean = metal::precise::rsqrt(acc / (float)Dv + as_type<float>((uint)EPS_BITS));
            const auto w4 = *reinterpret_cast<const device vec<InT, 4>*>(w + lane * 4);
            const auto z4 = *reinterpret_cast<const device vec<InT, 4>*>(proj + (b_idx * PW + Z_OFF + hv_idx * Dv + lane * 4));
            vec<InT, 4> out4;
            for (int i = 0; i < 4; ++i) {
                InT normalized = w4[i] * static_cast<InT>(thread_x[i] * inv_mean);
                const float zg = mlx_sigmoid(static_cast<float>(z4[i]));
                out4[i] = static_cast<InT>(zg * static_cast<float>(normalized));
            }
            *reinterpret_cast<device vec<InT, 4>*>(gated + (n * Dv + lane * 4)) = out4;
        }
