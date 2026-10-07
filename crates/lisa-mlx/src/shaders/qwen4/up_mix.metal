// Hyper-connection mixer up projection + sigmoid-gated stream fold, 2..8-token
// window.
        auto sigmoid = [&](T value) -> T {
            if constexpr (metal::is_same<T, bfloat16_t>::value) {
                return sigmoid_lut[as_type<ushort>(static_cast<bfloat16_t>(value))];
            } else {
                return mlx_sigmoid(value);
            }
        };
        const int tile = (int)threadgroup_position_in_grid.y;
        const int d0 = 2 * tile;
        const uint sg = simdgroup_index_in_threadgroup;
        const uint lid = thread_index_in_simdgroup;
        threadgroup T products[8][VPT];
        const int s = (int)sg * 4 + (int)(lid / 8);
        const int row = d0 + (s & 1) + H * (s >> 1);
        float r[VPT];
        qmv_wide_reg_full<T, GS, BITS, VPT, 8, false>(wu, su, bu, act, LW, VPT, row, lid, r);
        if ((lid % 8) == 0 && (s >> 1) < HC) {
            for (int v = 0; v < VPT; ++v) {
                const T weight = static_cast<T>(r[v]);
                products[s][v] = sigmoid(weight) * normed[(size_t)v * (size_t)(HC * H) + (size_t)row];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            for (int v = 0; v < VPT; ++v) {
                T acc = T(0);
                for (int s = 0; s < HC; ++s) {
                    const T p = products[s * 2 + (int)t][v];
                    acc = acc + p;
                }
                input[(size_t)v * (size_t)H + (size_t)d] = acc;
                if (EMIT_F32) { inputF[(size_t)v * (size_t)H + (size_t)d] = static_cast<float>(acc); }
            }
        }
        if (HAS_INJECT && tile == 0 && t < (uint)(HC * VPT)) {
            const T x = inj[t];
            inject[t] = T(2) * sigmoid(x);
        }
