// GDN input prep, fused-projection form (`track_gdn_prep`): the causal
// depthwise conv + Q/K L2-norm + gates, with the two 48-wide gate rows read
// from offsets in the concatenated `proj` instead of their own buffers.
        constexpr int KM1 = KC - 1;
        constexpr int N_READS = 4;
        constexpr int VEC_Q = Hk;
        constexpr int VEC_K = 2 * Hk;
        const uint lane = thread_position_in_threadgroup.x;
        const uint vec = thread_position_in_grid.y;
        const uint bt = thread_position_in_grid.z;
        const uint b = bt / T;
        const uint t = bt % T;
        const device InT* proj_b = proj + (uint)(b * T * PROJ_W);
        const device InT* cst_b = conv_state + (uint)(b * KM1 * CONV_DIM);
        auto win = [&](int r, uint ch) -> float {
            if (r < KM1) { return static_cast<float>(cst_b[(uint)(r * CONV_DIM) + ch]); }
            return static_cast<float>(proj_b[(uint)((r - KM1) * PROJ_W) + ch]);
        };
        float thread_x[N_READS];
        float acc = 0.0f;
        for (int i = 0; i < N_READS; ++i) {
            const uint ch = vec * 128 + lane * N_READS + i;
            float cacc = 0.0f;
            for (int j = 0; j < KC; ++j) {
                cacc += win((int)t + j, ch) * conv_w[ch * KC + j];
            }
            const InT c0 = static_cast<InT>(cacc);
            const InT c1 = mlx_silu(c0);
            thread_x[i] = static_cast<float>(c1);
            acc += thread_x[i] * thread_x[i];
        }
        if (vec < VEC_K) {
            acc = simd_sum(acc);
            const float inv_mean = metal::precise::rsqrt(acc / 128.0f + 1e-6f);
            const float inv_scale = metal::rsqrt(static_cast<float>(Dk));
            const InT q_mul = static_cast<InT>(inv_scale * inv_scale);
            const InT k_mul = static_cast<InT>(inv_scale);
            for (int i = 0; i < N_READS; ++i) {
                const InT n = static_cast<InT>(thread_x[i] * inv_mean);
                const uint d = lane * N_READS + i;
                if (vec < VEC_Q) {
                    qn[(uint)((bt * Hk + vec) * Dk) + d] = q_mul * n;
                } else {
                    kn[(uint)((bt * Hk + (vec - VEC_Q)) * Dk) + d] = k_mul * n;
                }
            }
        } else {
            for (int i = 0; i < N_READS; ++i) {
                const uint d = lane * N_READS + i;
                vv[(uint)((bt * Hv + (vec - VEC_K)) * Dv) + d] = static_cast<InT>(thread_x[i]);
            }
        }
        if (vec == 0) {
            const device InT* row = proj_b + (uint)(t * PROJ_W);
            for (int hh = lane; hh < Hv; hh += 32) {
                const InT b_raw = row[B_OFF + hh];
                beta[bt * Hv + hh] = static_cast<float>(mlx_sigmoid(b_raw));
                const InT ax = row[A_OFF + hh] + dt_bias[hh];
                const InT sp = mlx_logaddexp0(ax);
                g[bt * Hv + hh] = metal::precise::exp(neg_exp_alog[hh] * sp);
            }
        }
        if (t == (uint)(T - 1)) {
            const uint slot = b;
            device InT* o_conv = conv_out + (uint)(slot * KM1 * CONV_DIM);
            for (int i = 0; i < N_READS; ++i) {
                const uint ch = vec * 128 + lane * N_READS + i;
                for (int j = 0; j < KM1; ++j) {
                    o_conv[(uint)(j * CONV_DIM) + ch] = static_cast<InT>(win((int)t + 1 + j, ch));
                }
            }
        }
