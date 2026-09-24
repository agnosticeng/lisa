// Hyper-connection mixer: mean-fold the HC streams and scale by the inject weights.
        const uint d = thread_position_in_grid.x;
        const uint row = thread_position_in_grid.y;
        if (d >= H) return;
        InT acc = InT(0);
        for (int s = 0; s < HC; ++s) {
            const uint i = row * W + s * H + d;
            InT sg = mlx_sigmoid(w[i]);
            InT p = sg * normed[i];
            acc = acc + p;
        }
        input[row * H + d] = acc;
        if (HAS_INJECT && d < HC) {
            InT x = inj[row * LW + (LW - HC) + d];
            InT sg = mlx_sigmoid(x);
            inject[row * HC + d] = InT(2) * sg;
        }
    