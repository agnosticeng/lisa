// Builds the 65536-entry bf16 sigmoid lookup table (mixer fast sigmoid).
        const uint i = thread_position_in_grid.x;
        table[i] = mlx_sigmoid(as_type<bfloat16_t>(ushort(i)));
