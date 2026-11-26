// Adapted from the native DS4 clamped SwiGLU port in ../glm53/swiglu.metal.
// Copyright (c) 2026 The ds4.c authors. MIT; see LICENSE-DS4 and provenance.
// Compile with the same CompilerDefault math as that standalone activation.
{
    if ((int)thread_position_in_grid.x >= n) return;
    const uint i = thread_position_in_grid.x;
    float x0 = gate[i];
    float x1 = up[i];
    if (limit > 1.0e-6f) {
        x0 = min(x0, limit);
        x1 = clamp(x1, -limit, limit);
    }
    const float silu = x0 / (1.0f + exp(-x0));
    // The standalone dispatch writes an F32 activation before the separate
    // Multiply. A volatile thread value prevents reassociation across it.
    volatile float activation = silu * x1 * alpha;
    out[i] = activation * weights[i / width];
}
