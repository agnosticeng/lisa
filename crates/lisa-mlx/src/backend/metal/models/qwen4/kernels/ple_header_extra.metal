// Extra PLE kernel helpers.
template <typename T>
METAL_FUNC T mlx_maximum(T x, T y) {
    if (metal::isnan(x)) { return x; }
    return x > y ? x : y;
}
template <typename T>
METAL_FUNC T mlx_sign(T x) {
    return static_cast<T>((x > T(0)) - (x < T(0)));
}
template <typename T> METAL_FUNC T mlx_abs_t(T x) { return metal::abs(x); }
template <typename T> METAL_FUNC T mlx_sqrt_t(T x) { return metal::precise::sqrt(x); }
