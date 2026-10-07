// MoE prefill sorted combine (expert fold) kernel.
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (d >= H) return;
        float prod[K];
        for (int k = 0; k < K; ++k) {
            prod[k] = static_cast<float>(
                routed[static_cast<uint>(inverse_order[row * K + k]) * H + d]) * w[row * K + k];
        }
        const InT r = static_cast<InT>(mlx_colsum_small_f32<K>(prod));
        const InT sg = mlx_sigmoid(gate[row]);
        const InT sh = sg * shared[row * H + d];
        out[row * H + d] = r + sh;
    