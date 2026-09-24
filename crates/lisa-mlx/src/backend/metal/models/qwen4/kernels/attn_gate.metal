// Attention output gate: `sigmoid(gate) * attn` fused with the head reshape.
        const uint j = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (j >= HQ * D) return;
        const InT a = att[row * HQ * D + j];
        const InT g = gateb[row * HQ * D + j];
        out[row * HQ * D + j] = a * mlx_sigmoid(g);
    