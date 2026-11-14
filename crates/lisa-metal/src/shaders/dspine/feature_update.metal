// Tencent DSpine feature_finish write stage. Normalization keeps the original
// MLX reduction kernels; this kernel fuses only elementwise operations.
#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= B * T * H) return;
uint row = i / H;
float update = (float(rms[row]) * float(delta[i])) * float(row % T != 0);
updated[i] = bfloat16_t(float(raw[i]) + update);
