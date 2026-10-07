// PLE causal depthwise convolution (dilation = ngram_size).
        const uint c = thread_position_in_grid.x;
        const uint t = thread_position_in_grid.y;
        const uint b = thread_position_in_grid.z;
        if (c >= (uint)W) return;
        const device InT* fb = full + (size_t)b * (size_t)(NIN) * (size_t)W;
        float acc = 0.0f;
        for (int j = 0; j < KC; ++j) {
            acc += static_cast<float>(fb[(size_t)(t + (uint)(j * DIL)) * (size_t)W + c])
                 * static_cast<float>(convw[c * KC + j]);
        }
        const size_t o = ((size_t)b * (size_t)S + (size_t)t) * (size_t)W + c;
        out[o] = gated[o] + mlx_silu(static_cast<InT>(acc));
    