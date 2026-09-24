// Hyper-connection mixer up projection + sigmoid-gated stream fold, one-token
// window (PACKED_ROWS selects the contiguous-row GEMV vs the strided one).
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
        threadgroup T products[8][1];
        float r[4];
        if constexpr (PACKED_ROWS) {
            qmv_reg<T, GS, BITS, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, LW, tile * 8 + (int)sg * 4, lid, r);
        } else {
            int rows[4];
            for (int i = 0; i < 4; ++i) { const int s = (int)sg * 4 + i; rows[i] = d0 + (s & 1) + H * (s >> 1); }
            qmv_reg_rows<T, GS, BITS, false, (LW % get_pack_factor<BITS, 32>()) == 0>(wu, su, bu, act, LW, rows, lid, r);
        }
        if (lid < 4 && (int)(sg * 2 + lid / 2) < HC) {
            const int slot = (int)sg * 4 + (int)lid;
            const float low = metal::select(r[0], r[1], (lid & 1u) != 0);
            const float high = metal::select(r[2], r[3], (lid & 1u) != 0);
            const T weight = static_cast<T>(metal::select(low, high, (lid & 2u) != 0));
            const int row = d0 + (slot & 1) + H * (slot >> 1);
            products[slot][0] = sigmoid(weight) * normed[row];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint t = sg * 32 + lid;
        if (t < 2) {
            const int d = d0 + (int)t;
            T acc = T(0);
            for (int s = 0; s < HC; ++s) {
                const T p = products[s * 2 + (int)t][0];
                acc = acc + p;
            }
            input[(size_t)d] = acc;
            if (EMIT_F32) { inputF[(size_t)d] = static_cast<float>(acc); }
        }
        if (HAS_INJECT && tile == 0 && t < (uint)HC) {
            const T x = inj[t];
            inject[t] = T(2) * sigmoid(x);
        }
