// GDN gated-delta recurrence (custom kernel and ops fallback).
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;

        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;

        // g: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        // state_in: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          if (true) {
            float kv_mem = 0.0f;
            {
              // Preserve Kahan summation under Metal's default fast math.
              #pragma clang fp reassociate(off)
              #pragma clang fp contract(off)
              float kv_compensation = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                auto product = state[i] * k_[s_idx];
                auto corrected = product - kv_compensation;
                auto next_sum = kv_mem + corrected;
                kv_compensation = (next_sum - kv_mem) - corrected;
                kv_mem = next_sum;
              }
            }
            kv_mem = simd_sum(kv_mem);

            auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

            float out = 0.0f;
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = state[i] + k_[s_idx] * delta;
              out += state[i] * q_[s_idx];
            }
            out = simd_sum(out);
            if (thread_index_in_simdgroup == 0) {
              y[dv_idx] = static_cast<InT>(out);
            }
          } else {
            y[dv_idx] = static_cast<InT>(0);
          }
          // Speculative capture: write the state after EVERY position (the
          // output is then [B*T, Hv, Dv, Dk], slot b*T + t). Without capture
          // this writes only the final state, exactly as before.
          if (CAPTURE || t == T - 1) {
            const uint slot = CAPTURE ? (b_idx * T + t) : b_idx;
            device StT* o_seq = state_out + ((slot * Hv + hv_idx) * Dv + dv_idx) * Dk;
            for (int i = 0; i < n_per_t; ++i) {
              o_seq[n_per_t * dk_idx + i] = static_cast<StT>(state[i]);
            }
          }
          // Increment data pointers to next time step
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
    