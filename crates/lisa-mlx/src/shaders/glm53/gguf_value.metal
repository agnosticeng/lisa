// DS4 Q8_0 value projection for the split wide-window attention path.
{
    const uint head = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup;
    const uint nth = threads_per_threadgroup.x;
    for (uint j = tid; j < 512u; j += nth) shared[j] = lora[head * 512u + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint d = tid; d < 256u; d += nth) {
        device const char *row = (device const char *)weight + ((ulong)head * 256u + d) * 544u;
        output[head * 256u + d] = glm_q8_0_dot_row_tg_f32_fast(row,shared,512u);
    }
}
