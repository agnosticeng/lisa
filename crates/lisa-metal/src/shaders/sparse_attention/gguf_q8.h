// DS4 b3f50180, original Q8_0 dot helpers. See gguf_provenance.json.
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
static inline float glm_q8_0_weight_at(
        device const char *row,
        uint col) {
    const uint block = col >> 5;
    const uint qi = col & 31u;
    device const char *block_base = row + (uint64_t)block * 34u;
    const float d = (float)(*((device const half *)block_base));
    device const int8_t *qs = (device const int8_t *)(block_base + 2u);
    return d * (float)qs[qi];
}

static inline float glm_q8_0_dot_row_tg_f32(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_512(
        device const char *row,
        threadgroup const float *x) {
    float acc = 0.0f;
    for (uint block = 0; block < 16u; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        FOR_UNROLL (uint qi = 0; qi < 32u; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

static inline float glm_q8_0_dot_row_tg_f32_fast(
        device const char *row,
        threadgroup const float *x,
        uint n_cols) {
    if (n_cols == 512u) {
        return glm_q8_0_dot_row_tg_f32_512(row, x);
    }
    return glm_q8_0_dot_row_tg_f32(row, x, n_cols);
}

static inline float glm_q8_0_dot_row_dev_f32(
        device const char *row,
        device const float *x,
        uint n_cols) {
    float acc = 0.0f;
    const uint n_blocks = (n_cols + 31u) >> 5;
    for (uint block = 0; block < n_blocks; block++) {
        device const char *block_base = row + (uint64_t)block * 34u;
        const float d = (float)(*((device const half *)block_base));
        device const int8_t *qs = (device const int8_t *)(block_base + 2u);
        const uint base = block << 5;
        const uint count = min(32u, n_cols - base);
        for (uint qi = 0; qi < count; qi++) {
            acc += d * (float)qs[qi] * x[base + qi];
        }
    }
    return acc;
}

