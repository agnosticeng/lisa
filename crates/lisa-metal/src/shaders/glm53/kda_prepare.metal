// GLM-5.3 KDA prefill prepare, following DS4's one-thread-per-channel
// schedule and per-head serial token loop. Inputs and outputs are F32.

        const uint head = thread_position_in_grid.x / 128u;
        if (head >= (uint)n_heads) return;
        constexpr uint D = 128u;
        constexpr uint HISTORY = 3u;
        const uint projection = (uint)n_heads * D;
        const uint history = HISTORY * projection;
        const uint tid = thread_index_in_threadgroup;
        const ushort lane = thread_index_in_simdgroup;
        const ushort sg = simdgroup_index_in_threadgroup;
        const uint channel = head * D + tid;
        threadgroup float scratch[2u * D + 8u];
        threadgroup float *sq = scratch;
        threadgroup float *sk = sq + D;
        threadgroup float *reduce_q = sk + D;
        threadgroup float *reduce_k = reduce_q + 4u;

        device float *q_state = conv_state_out + channel;
        device float *k_state = conv_state_out + history + channel;
        device float *v_state = conv_state_out + 2u * history + channel;
        for (uint w = 0; w < HISTORY; w++) {
            q_state[(ulong)w * projection] =
                conv_state_in[(ulong)w * projection + channel];
            k_state[(ulong)w * projection] =
                conv_state_in[history + (ulong)w * projection + channel];
            v_state[(ulong)w * projection] =
                conv_state_in[2u * history + (ulong)w * projection + channel];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        for (uint token = 0; token < (uint)n_rows; token++) {
            const ulong index = (ulong)token * projection + channel;
            float q_acc = 0.0f;
            float k_acc = 0.0f;
            float v_acc = 0.0f;
            for (uint w = 0; w < HISTORY; w++) {
                q_acc = fma(q_state[(ulong)w * projection],
                            q_conv[(ulong)channel * 4u + w], q_acc);
                k_acc = fma(k_state[(ulong)w * projection],
                            k_conv[(ulong)channel * 4u + w], k_acc);
                v_acc = fma(v_state[(ulong)w * projection],
                            v_conv[(ulong)channel * 4u + w], v_acc);
            }
            const float q_new = q_in[index];
            const float k_new = k_in[index];
            const float v_new = v_in[index];
            q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
            k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
            v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

            q_state[0] = q_state[projection];
            q_state[projection] = q_state[2u * projection];
            q_state[2u * projection] = q_new;
            k_state[0] = k_state[projection];
            k_state[projection] = k_state[2u * projection];
            k_state[2u * projection] = k_new;
            v_state[0] = v_state[projection];
            v_state[projection] = v_state[2u * projection];
            v_state[2u * projection] = v_new;

            sq[tid] = q_acc / (1.0f + exp(-q_acc));
            sk[tid] = k_acc / (1.0f + exp(-k_acc));
            v_out[index] = v_acc / (1.0f + exp(-v_acc));
            const float gate = gate_in[index] + dt_bias[channel];
            g_out[index] = exp(lower_bound *
                (1.0f / (1.0f + exp(-exp(a_log[head]) * gate))));
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

            const float q_sumsq = simd_sum(sq[tid] * sq[tid]);
            const float k_sumsq = simd_sum(sk[tid] * sk[tid]);
            if (lane == 0u) {
                reduce_q[sg] = q_sumsq;
                reduce_k[sg] = k_sumsq;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
            float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
            q_total = simd_sum(q_total);
            k_total = simd_sum(k_total);
            q_out[index] = sq[tid] * rsqrt(q_total + 1.0e-6f) *
                0x1.6a09e6p-4f;
            k_out[index] = sk[tid] * rsqrt(k_total + 1.0e-6f);
            threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        }
