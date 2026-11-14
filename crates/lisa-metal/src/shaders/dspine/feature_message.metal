// Tencent DSpine feature_message design, with Lisa/MLX's exact sigmoid law.
// Keep separate add and multiply boundaries from the composed reference.
#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
if (i >= B * T * R) return;
uint row = i / R;
uint c = i % R;
if (row % T == 0) { message[i] = bfloat16_t(0.0f); return; }
float q = float(packed[row * (3 * R) + c]);
float s = float(packed[(row - 1) * (3 * R) + R + c]);
float v = float(packed[(row - 1) * (3 * R) + 2 * R + c]);
float gate = (q + s) + float(bias[c]);
float y = 1.0f / (1.0f + metal::exp(metal::abs(gate)));
float sig = gate < 0.0f ? y : 1.0f - y;
message[i] = bfloat16_t(sig * v);
