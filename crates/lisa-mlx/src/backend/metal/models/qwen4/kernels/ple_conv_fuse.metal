// PLE fused conv variant (S=1 prepare + convolution).
        constexpr uint W = 10240;
        const uint c = threadgroup_position_in_grid.z;
        const uint lane = thread_index_in_simdgroup;
        if (simdgroup_index_in_threadgroup != 0) { return; }
        float product = 0.0f;
        if (lane < 4) {
            product = float(full[(lane * 3) * W + c]) * float(weight[c * 4 + lane]);
        }
        float acc = simd_broadcast(product, 0);
        acc += simd_broadcast(product, 1);
        acc += simd_broadcast(product, 2);
        acc += simd_broadcast(product, 3);
        if (lane == 0) {
            InT convolved = InT(acc);
            InT activated = mlx_silu(convolved);
            out[c] = gated[c] + activated;
        }
    