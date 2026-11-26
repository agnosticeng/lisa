// GLM MTP residual + routed-expert + shared-expert sum, following DS4's
// kernel_add3_f32. Preserve the source order: (residual + routed) + shared.
{
    const uint i = thread_position_in_grid.x;
    if (i < (uint)n) output[i] = a[i] + b[i] + c[i];
}
