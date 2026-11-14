// Native counterpart of Tencent DSpine's serial_chain, revision
// 321b0fc1388c4cdf03483ab73856975f7c0c0be7. See NOTICE.
// The parallel refinement table already contains each conditional argmax.
// Resolve its small causal dependency without per-row gathers or host reads.
if (thread_position_in_grid.x != 0) return;
uint sender = 0;
for (uint row = 0; row < N; ++row) {
    sender = uint(choices[row * K + sender]);
    tokens[row] = uint(candidates[row * K + sender]);
}
