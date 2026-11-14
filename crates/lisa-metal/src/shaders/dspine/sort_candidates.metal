// DSpine candidate ordering. Tencent DSpine design; see NOTICE.
// Selection has already happened in the ordinary top-32 kernel. Only integer
// ordering is fused here, so score arithmetic and cutoff ties are unchanged.
uint row = thread_position_in_grid.x;
if (row >= N) return;
uint ids[16];
for (uint i = 0; i < 16; ++i) {
    uint id = uint(selected[row * 32 + i]);
    uint j = i;
    while (j > 0 && ids[j - 1] > id) {
        ids[j] = ids[j - 1];
        --j;
    }
    ids[j] = id;
}
for (uint i = 0; i < 16; ++i) candidates[row * 16 + i] = ids[i];
