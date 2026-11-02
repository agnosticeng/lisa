// GLM-5.3 clamped SwiGLU, fused (ds4 glu.metal `kernel_swiglu_flat_f32` port).
// Body only — the framework generates the kernel signature from the wrapper's
// param list (n, limit, alpha, gate, up, out) and injects thread_position_in_grid.
//
// gate [N] f32, up [N] f32 -> out [N] f32, one thread per element. The
// activation stays F32 like ds4's GLM execution graph.
// alpha is 1.0 for glm53 but kept as a constant for parity with ds4's arg.
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
    out[i] = silu * x1 * alpha;
}
