// Exact-arithmetic helpers shared by the engine kernels: bf16 `mlx_sigmoid`/`mlx_silu`, `mlx_colsum_small_f32`, `mlx_logaddexp0`.
template <typename T>
METAL_FUNC T mlx_sigmoid(T x) {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
}
template <typename T>
METAL_FUNC T mlx_silu(T x) {
    T s = mlx_sigmoid(x);
    return x * s;
}
template <int K>
METAL_FUNC float mlx_colsum_small_f32(thread const float* r) {
    constexpr int TY = K < 8 ? K : 8;
    float t[TY];
    for (int y = 0; y < TY; ++y) {
        float acc = 0.0f;
        for (int rr = y; rr < K; rr += TY) { acc = r[rr] + acc; }
        t[y] = acc;
    }
    float total = t[0];
    for (int j = 1; j < TY; ++j) { total = t[j] + total; }
    return total;
}
template <typename T>
METAL_FUNC T mlx_logaddexp0(T x) {
    T y = 0;
    if (metal::isnan(x)) { return metal::numeric_limits<T>::quiet_NaN(); }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
}
