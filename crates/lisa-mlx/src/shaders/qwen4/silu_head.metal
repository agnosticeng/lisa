// Fused bf16 silu over the mixer low-rank head.
        const uint j = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (j >= LMIX) return;
        out[row * LMIX + j] = mlx_silu(lo[row * LW + j]);
    