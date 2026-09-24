// Wide-window router partial-sum reduction.
        const uint i = thread_position_in_grid.y * 512 + thread_position_in_grid.x;
        float total = 0.0f;
        total += partials[i];
        total += partials[M * 512 + i];
        y[i] = total;
        