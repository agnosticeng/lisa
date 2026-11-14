// Fused sigmoid output gate: `sigmoid(gate) * up` (bf16), the attention-tail
// chain `ops::sigmoid(g)` → `multiply(a)` in one launch.
//
// Parity note: the composed `Sigmoid` unary kernel runs its arithmetic per-op
// in bf16 (each of exp, 1+, 1/ and 1-y rounds to bf16), so the fused kernel
// reproduces those rounding points explicitly — a plain float-chain sigmoid
// differs on ~0.03% of bf16 inputs (measured: 21/65546 random + specials).
        const uint i = thread_position_in_grid.x;
        if (i >= (uint)(B * F)) return;
        InT x = gate[i];
        InT ax = static_cast<InT>(metal::abs(static_cast<float>(x)));
        InT e = static_cast<InT>(metal::precise::exp(static_cast<float>(ax)));
        InT t = static_cast<InT>(1.0f + static_cast<float>(e));
        InT y = static_cast<InT>(1.0f / static_cast<float>(t));
        InT s = (x < InT(0)) ? y : static_cast<InT>(1.0f - static_cast<float>(y));
        out[i] = s * up[i];
