// Two-input SwiGLU (`silu(gate) * up`).
        const uint i = thread_position_in_grid.x;
        if (i >= (uint)(B * F)) return;
        out[i] = mlx_silu(gate[i]) * up[i];
    