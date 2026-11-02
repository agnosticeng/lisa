// Fused one-token GLM-5.3 KDA decode, ported from ds4's
// kernel_glm53_kda_decode. It consumes F32 projections and planar conv/state
// buffers, updates both recurrent caches, and emits the gated RMS-normalized
// attention output in one dispatch.
        constexpr uint D = 128u;
        constexpr uint HISTORY = 3u;
        const uint head = threadgroup_position_in_grid.y;
        const uint tid = thread_position_in_threadgroup.x;
        const uint lane = thread_index_in_simdgroup;
        const uint sg = simdgroup_index_in_threadgroup;
        if (head >= (uint)n_heads || tid >= D) return;

        threadgroup float scratch[5u * D + 16u];
        threadgroup float *sq = scratch;
        threadgroup float *sk = sq + D;
        threadgroup float *sd = sk + D;
        threadgroup float *sv = sd + D;
        threadgroup float *so = sv + D;
        threadgroup float *reduce_q = so + D;
        threadgroup float *reduce_k = reduce_q + 4u;
        threadgroup float *reduce_o = reduce_k + 4u;
        threadgroup float *beta_shared = reduce_o + 4u;

        const uint projection = (uint)n_heads * D;
        const uint channel = head * D + tid;
        const ulong input_base = channel;
        const ulong history = (ulong)HISTORY * projection;
        device const float *q_state = conv_state_in;
        device const float *k_state = conv_state_in + history;
        device const float *v_state = conv_state_in + 2u * history;

        float q_acc = 0.0f, k_acc = 0.0f, v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel],
                        q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel],
                        k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel],
                        v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[input_base];
        const float k_new = k[input_base];
        const float v_new = v[input_base];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        // Shift each projection's three-entry convolution history.
        conv_state_out[(ulong)head * D + tid] = q_state[projection + channel];
        conv_state_out[(ulong)head * D + projection + tid] = q_state[2u * projection + channel];
        conv_state_out[(ulong)head * D + 2u * projection + tid] = q_new;
        conv_state_out[history + (ulong)head * D + tid] = k_state[projection + channel];
        conv_state_out[history + (ulong)head * D + projection + tid] = k_state[2u * projection + channel];
        conv_state_out[history + (ulong)head * D + 2u * projection + tid] = k_new;
        conv_state_out[2u * history + (ulong)head * D + tid] = v_state[projection + channel];
        conv_state_out[2u * history + (ulong)head * D + projection + tid] = v_state[2u * projection + channel];
        conv_state_out[2u * history + (ulong)head * D + 2u * projection + tid] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base] + dt_bias[channel];
        sd[tid] = exp(lower_bound *
            (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
        if (tid == 0u) {
            beta_shared[0] = 1.0f /
                (1.0f + exp(-raw_beta[head]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup |
                            mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
        const float k_scale = rsqrt(k_total + 1.0e-6f);
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint k0 = lane * 4u;
        const float4 q4 = *((threadgroup float4 *)(sq + k0));
        const float4 k4 = *((threadgroup float4 *)(sk + k0));
        const float4 decay4 = *((threadgroup float4 *)(sd + k0));
        const ulong state_head = ((ulong)head * D * D);
        for (uint value = sg; value < D; value += 4u) {
            device const float *state_row = state_in + state_head + (ulong)value * D + k0;
            device float *out_state_row = state_out + state_head + (ulong)value * D + k0;
            float4 h = *((device const float4 *)state_row) * decay4;
            const float hk = simd_sum(dot(h, k4));
            const float delta_v = (sv[value] - hk) * beta_shared[0];
            h = fma(k4, float4(delta_v), h);
            *((device float4 *)out_state_row) = h;
            const float hq = simd_sum(dot(h, q4));
            if (lane == 0u) so[value] = hq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup |
                            mem_flags::mem_device);

        float o_sumsq = simd_sum(so[tid] * so[tid]);
        if (lane == 0u) reduce_o[sg] = o_sumsq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
        o_total = simd_sum(o_total);
        const float o_scale = rsqrt(o_total / (float)D + norm_eps);
        const float output_gate_value =
            1.0f / (1.0f + exp(-output_gate[input_base]));
        out[input_base] = so[tid] * o_scale * output_norm[tid] * output_gate_value;
