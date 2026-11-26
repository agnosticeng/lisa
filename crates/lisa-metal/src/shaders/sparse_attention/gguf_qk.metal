// DS4 GLM QK-lowrank serial dots, original split Q8_0 K matrix.
{
    uint head = threadgroup_position_in_grid.x;
    uint tid = thread_index_in_threadgroup;
    device const float *qh = q + head * 256u;
    for (uint j = tid; j < 512u; j += threads_per_threadgroup.x) {
        device const char *row = (device const char *)weight + ((ulong)head * 512u + j) * 272u;
        output[head * 512u + j] = glm_q8_0_dot_row_dev_f32(row, qh, 256u);
    }
}
