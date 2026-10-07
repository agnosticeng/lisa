// Flattened MLX NAX quantized header closure (AOT for `qmm_nax`/`gather_qmm_rhs_nax`).
// Auto-generated flattened MLX Metal header closure. Do not hand-edit;
// regenerate from the pinned MLX headers.
#line 1 "mlx/backend/metal/kernels/bf16.h"


#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}

#line 1 "mlx/backend/metal/kernels/bf16_math.h"


///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal

#line 1 "mlx/backend/metal/kernels/complex.h"


#include <metal_stdlib>


using namespace metal;

template <typename T>
struct complex_t;

template <typename T>
static constexpr constant bool is_complex_v = false;

template <typename T>
static constexpr constant bool is_complex_v<complex_t<T>> = true;

// Metal accepts explicit bfloat casts that is_convertible_v reports as false.
template <typename From, typename To>
static constexpr constant bool is_lane_convertible_v =
    is_convertible_v<From, To> ||
    (is_same_v<To, bfloat16_t> && is_convertible_v<From, float>) ||
    (is_same_v<From, bfloat16_t> && is_convertible_v<float, To>);

template <typename T>
struct complex_t {
  using value_type = T;

  T real;
  T imag;

  // Constructors
  constexpr complex_t(T real, T imag) thread : real(real), imag(imag) {};
  constexpr complex_t() thread : real(0), imag(0) {};
  constexpr complex_t() threadgroup : real(0), imag(0) {};

  // Conversions from scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) thread : real(static_cast<T>(x)),
                                    imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) threadgroup : real(static_cast<T>(x)),
                                         imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) device : real(static_cast<T>(x)),
                                    imag(static_cast<T>(0)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(U x) constant : real(static_cast<T>(x)),
                                      imag(static_cast<T>(0)) {}

  // Conversions between complex types
  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) thread : real(static_cast<T>(x.real)),
                                               imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) threadgroup
      : real(static_cast<T>(x.real)),
        imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) device : real(static_cast<T>(x.real)),
                                               imag(static_cast<T>(x.imag)) {}

  template <
      typename U,
      typename = typename enable_if<
          !is_same_v<U, T> && is_lane_convertible_v<U, T>>::type>
  constexpr complex_t(complex_t<U> x) constant : real(static_cast<T>(x.real)),
                                                 imag(static_cast<T>(x.imag)) {}

  // Conversions to and from two-lane vectors (the FFT lane representation)
  constexpr complex_t(vec<T, 2> v) thread : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) threadgroup : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) device : real(v.x), imag(v.y) {};
  constexpr complex_t(vec<T, 2> v) constant : real(v.x), imag(v.y) {};

  constexpr operator vec<T, 2>() const thread {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const threadgroup {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const device {
    return vec<T, 2>(real, imag);
  }

  constexpr operator vec<T, 2>() const constant {
    return vec<T, 2>(real, imag);
  }

  // Conversions to scalar types
  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const thread {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const threadgroup {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const device {
    return static_cast<U>(real);
  }

  template <
      typename U,
      typename = typename enable_if<
          !is_complex_v<U> && is_lane_convertible_v<T, U>>::type>
  constexpr operator U() const constant {
    return static_cast<U>(real);
  }
};

using complex32_t = complex_t<half>;
using complex64_t = complex_t<float>;

static_assert(sizeof(complex32_t) == 2 * sizeof(half));
static_assert(sizeof(complex64_t) == 2 * sizeof(float));
static_assert(sizeof(complex_t<bfloat16_t>) == 2 * sizeof(bfloat16_t));

template <typename T>
constexpr complex_t<T> operator-(complex_t<T> x) {
  return {-x.real, -x.imag};
}

template <typename T>
constexpr bool operator>=(complex_t<T> a, complex_t<T> b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

template <typename T>
constexpr bool operator>(complex_t<T> a, complex_t<T> b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

template <typename T>
constexpr bool operator<=(complex_t<T> a, complex_t<T> b) {
  return operator>=(b, a);
}

template <typename T>
constexpr bool operator<(complex_t<T> a, complex_t<T> b) {
  return operator>(b, a);
}

template <typename T>
constexpr bool operator==(complex_t<T> a, complex_t<T> b) {
  return a.real == b.real && a.imag == b.imag;
}

template <typename T>
constexpr complex_t<T> operator+(complex_t<T> a, complex_t<T> b) {
  return {a.real + b.real, a.imag + b.imag};
}

template <typename T>
constexpr thread complex_t<T>& operator+=(
    thread complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <typename T>
constexpr threadgroup complex_t<T>& operator+=(
    threadgroup complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <typename T>
constexpr device complex_t<T>& operator+=(
    device complex_t<T>& a,
    complex_t<T> b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator+(U a, complex_t<T> b) {
  return {static_cast<T>(a) + b.real, b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator+(complex_t<T> a, U b) {
  return {a.real + static_cast<T>(b), a.imag};
}

template <typename T>
constexpr complex_t<T> operator-(complex_t<T> a, complex_t<T> b) {
  return {a.real - b.real, a.imag - b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator-(U a, complex_t<T> b) {
  return {static_cast<T>(a) - b.real, -b.imag};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator-(complex_t<T> a, U b) {
  return {a.real - static_cast<T>(b), a.imag};
}

template <typename T>
constexpr complex_t<T> operator*(complex_t<T> a, complex_t<T> b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

template <typename T>
constexpr complex_t<T> operator/(complex_t<T> a, complex_t<T> b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

template <
    typename T,
    typename U,
    enable_if_t<!is_complex_v<U> && is_lane_convertible_v<U, T>, bool> = true>
constexpr complex_t<T> operator/(U a, complex_t<T> b) {
  auto scalar = static_cast<T>(a);
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = scalar * b.real;
  auto y = -scalar * b.imag;
  return {x / denom, y / denom};
}

template <typename T>
constexpr complex_t<T> operator%(complex_t<T> a, complex_t<T> b) {
  auto real = a.real - (b.real * static_cast<int64_t>(a.real / b.real));
  auto imag = a.imag - (b.imag * static_cast<int64_t>(a.imag / b.imag));
  if (real != 0 && (real < 0 != b.real < 0)) {
    real += b.real;
  }
  if (imag != 0 && (imag < 0 != b.imag < 0)) {
    imag += b.imag;
  }
  return {real, imag};
}

static_assert(
    (complex_t<half>{1.0h, 2.0h} * complex_t<half>{3.0h, 4.0h}).real == -5.0h);
static_assert(
    (complex_t<bfloat16_t>{bfloat16_t(1.0f), bfloat16_t(2.0f)} *
     complex_t<bfloat16_t>{bfloat16_t(3.0f), bfloat16_t(4.0f)})
        .real == bfloat16_t(-5.0f));

#line 1 "mlx/backend/metal/kernels/defines.h"


#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

#line 1 "mlx/backend/metal/kernels/logging.h"


#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const thread {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif

#line 1 "mlx/backend/metal/kernels/utils.h"


#include <metal_math>


typedef half float16_t;

// Work per thread values for different types. The values here are expected to
// match get_work_per_thread in mlx/backend/metal/utils.h
template <typename U>
struct WorkPerThread {
  static_assert(sizeof(U) <= 8, "Type too large");
  static constexpr int constant n = 8 / sizeof(U);
};

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

template <typename T>
struct Limits<complex_t<T>> {
  inline static constexpr constant complex_t<T> max = complex_t<T>(
      metal::numeric_limits<T>::infinity(),
      metal::numeric_limits<T>::infinity());
  inline static constexpr constant complex_t<T> min = complex_t<T>(
      -metal::numeric_limits<T>::infinity(),
      -metal::numeric_limits<T>::infinity());
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    IdxT elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_1(uint elem, constant const int64_t& stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_2(uint2 elem, constant const int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_3(uint3 elem, constant const int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 2> elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  vec<IdxT, 2> loc = {
      IdxT(
          elem.x * IdxT(a_strides[ndim - 1]) +
          IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
      IdxT(
          elem.x * IdxT(b_strides[ndim - 1]) +
          elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 3> elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  vec<IdxT, 3> loc = {
      IdxT(elem.x * IdxT(a_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(a_strides[ndim - 2])),
      IdxT(elem.x * IdxT(b_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(b_strides[ndim - 2])),
      IdxT(elem.x * IdxT(c_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(c_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    loc.z += l * IdxT(c_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) thread : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int* shape, const constant int64_t* strides) thread {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides)
      thread {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() thread {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) thread : dim(dim) {}

  void next(const constant int* shape, const constant int64_t* strides) thread {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides)
      thread {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() thread {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) thread {}

  void next(const constant int*, const constant int64_t* strides) thread {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int*, const constant int64_t* strides)
      thread {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() thread {
    return offset;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}

inline complex64_t log1p(complex64_t in) {
  float x = in.real;
  float y = in.imag;
  float zabs = metal::precise::sqrt(x * x + y * y);
  float theta = metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = metal::sqrt((x + 1) * (x + 1) + y * y);
    return {metal::log(z0), theta};
  }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_down(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_down(data.real, delta), simd_shuffle_down(data.imag, delta));
}

inline uint64_t simd_shuffle_up(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_up(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_up(bool data, uint16_t delta) {
  return simd_shuffle_up(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_up(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_up(data.real, delta), simd_shuffle_up(data.imag, delta));
}

inline uint64_t
simd_shuffle_and_fill_up(uint64_t data, uint64_t filling, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline int64_t
simd_shuffle_and_fill_up(int64_t data, int64_t filling, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}

inline complex64_t simd_shuffle_and_fill_up(
    complex64_t data,
    complex64_t filling,
    uint16_t delta) {
  return complex64_t(
      simd_shuffle_and_fill_up(data.real, filling.real, delta),
      simd_shuffle_and_fill_up(data.imag, filling.imag, delta));
}

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

inline complex64_t simd_shuffle(complex64_t data, uint16_t lane) {
  return complex64_t(
      simd_shuffle(data.real, lane), simd_shuffle(data.imag, lane));
}

// std::conditional is not included with Metal
template <bool condition, typename T, typename U>
struct ConditionalType {
  using type = U;
};

template <typename T, typename U>
struct ConditionalType<true, T, U> {
  using type = T;
};

///////////////////////////////////////////////////////////////////////////////
// Type casting utils
///////////////////////////////////////////////////////////////////////////////

template <typename U, typename T>
inline U cast_to(T val) {
  return static_cast<U>(val);
}

template <>
inline bool cast_to<bool, float>(float val) {
  return (as_type<uint32_t>(val) & 0x7FFFFFFF) != 0;
}

template <>
inline bool cast_to<bool, bfloat16_t>(bfloat16_t val) {
  return (as_type<uint16_t>(val) & 0x7FFF) != 0;
}

template <>
inline bool cast_to<bool, complex64_t>(complex64_t val) {
  return cast_to<bool, float>(val.real) || cast_to<bool, float>(val.imag);
}

#line 1 "mlx/backend/metal/kernels/steel/defines.h"


#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")

#line 1 "mlx/backend/metal/kernels/steel/utils/type_traits.h"


#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable

#line 1 "mlx/backend/metal/kernels/steel/utils/integral_constant.h"


#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable

#line 1 "mlx/backend/metal/kernels/steel/gemm/nax.h"


#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
      thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) const thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() thread {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const thread {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(
      thread metal::vec<T, kRowsPerThread>& vals) thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

#line 1 "mlx/backend/metal/kernels/steel/gemm/params.h"


///////////////////////////////////////////////////////////////////////////////
// GEMM param classes
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

struct GEMMParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldd;

  const int tiles_n;
  const int tiles_m;

  const int64_t batch_stride_a;
  const int64_t batch_stride_b;
  const int64_t batch_stride_d;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;

  const int batch_ndim;
};

struct GEMMSpiltKParams {
  const int M;
  const int N;
  const int K;

  const int lda;
  const int ldb;
  const int ldc;

  const int tiles_n;
  const int tiles_m;

  const int split_k_partitions;
  const int split_k_partition_stride;
  const int split_k_partition_size;

  const int swizzle_log;
  const int gemm_k_iterations_aligned;
};

struct GEMMAddMMParams {
  const int ldc;
  const int fdc;

  const int64_t batch_stride_c;

  const float alpha;
  const float beta;
};

} // namespace steel
} // namespace mlx

#line 1 "mlx/backend/metal/kernels/steel/utils.h"


#include <metal_stdlib>

METAL_FUNC ulong2 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
  }
  return ulong2(loc_a, loc_b);
}

METAL_FUNC ulong3 elem_to_loc_broadcast(
    uint elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  ulong loc_a{0};
  ulong loc_b{0};
  ulong loc_c{0};
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    int pos_in_dim = (elem % shape[i]);
    elem /= shape[i];
    loc_a += pos_in_dim * a_strides[i];
    loc_b += pos_in_dim * b_strides[i];
    loc_c += pos_in_dim * c_strides[i];
  }
  return ulong3(loc_a, loc_b, loc_c);
}

#line 1 "mlx/backend/metal/kernels/steel/gemm/transforms.h"



///////////////////////////////////////////////////////////////////////////////
// Transforms and Epilogues
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

template <typename OutT, typename InT>
struct TransformNone {
  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT) {
    return static_cast<OutT>(x);
  }
};

template <typename OutT, typename InT>
struct TransformAdd {
  TransformAdd(const float, const float) thread {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  static METAL_FUNC OutT apply(InT x, OutT c) {
    return static_cast<OutT>(x) + c;
  }
};

template <typename OutT, typename InT>
struct TransformAxpby {
  const float alpha;
  const float beta;

  TransformAxpby(const float alpha_, const float beta_) thread : alpha(alpha_),
                                                                 beta(beta_) {}

  static METAL_FUNC OutT apply(InT x) {
    return static_cast<OutT>(x);
  }

  METAL_FUNC OutT apply(InT x, OutT c) const thread {
    return static_cast<OutT>(
        x * static_cast<InT>(alpha) + (static_cast<OutT>(beta) * c));
  }
};

template <typename T>
struct AccumHelper {
  typedef float accum_type;
};

struct BlockSwizzle {
  static METAL_FUNC int2
  swizzle(uint3 tid [[threadgroup_position_in_grid]], const int swizzle_log) {
    const int tid_x = (tid.x) >> swizzle_log;
    const int tid_y =
        ((tid.y) << swizzle_log) + ((tid.x) & ((1 << swizzle_log) - 1));
    return int2(tid_x, tid_y);
  }
};

} // namespace steel
} // namespace mlx

#line 1 "mlx/backend/metal/kernels/steel/gemm/gemm_nax.h"



using namespace metal;

namespace mlx::steel {

template <
    typename T,
    short SM,
    short SN,
    short SK,
    short BK,
    bool transpose_a,
    bool transpose_b,
    bool kAlignedM,
    bool kAlignedN,
    bool kAlignedK,
    typename AccumType = float>
auto gemm_loop(
    const device T* A,
    const device T* B,
    int lda,
    int ldb,
    int K,
    int gemm_k_iterations_aligned,
    const short sgp_sm,
    const short sgp_sn) {
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  constexpr int RA = transpose_a ? TK : TM;
  constexpr int CA = transpose_a ? TM : TK;

  constexpr int RB = transpose_b ? TN : TK;
  constexpr int CB = transpose_b ? TK : TN;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  const bool has_output = sgp_sm > 0 && sgp_sn > 0;

  int gemm_k_iterations_ = gemm_k_iterations_aligned;

  STEEL_PRAGMA_NO_UNROLL
  for (int kk0 = 0; kk0 < gemm_k_iterations_; kk0++) {
    threadgroup_barrier(mem_flags::mem_none);
    if constexpr (!kAlignedM || !kAlignedN) {
      if (!has_output)
        continue;
    }

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;
      const int k = kk1;

      volatile int compiler_barrier;

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      if constexpr (kAlignedM) {
        Atile.load(A + A_offset, lda);
      } else {
        const short rmax = transpose_a ? SK : sgp_sm;
        const short cmax = transpose_a ? sgp_sm : SK;
        Atile.load_safe(A + A_offset, lda, short2(cmax, rmax));
      }

      if constexpr (kAlignedN) {
        Btile.load(B + B_offset, ldb);
      } else {
        const short rmax = transpose_b ? sgp_sn : SK;
        const short cmax = transpose_b ? SK : sgp_sn;
        Btile.load_safe(B + B_offset, ldb, short2(cmax, rmax));
      }

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    A += transpose_a ? (BK * lda) : BK;
    B += transpose_b ? BK : (BK * ldb);
  }

  if constexpr (!kAlignedK) {
    simdgroup_barrier(mem_flags::mem_none);
    if constexpr (!kAlignedM || !kAlignedN) {
      if (!has_output)
        return Dtile;
    }

    const short rem_bk = K - gemm_k_iterations_ * BK;

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < rem_bk; kk1 += SK) {
      NAXTile<T, RA, CA> Atile;
      NAXTile<T, RB, CB> Btile;

      const int k = kk1;
      const short psk = max(0, rem_bk - k);

      const short2 Aklims =
          transpose_a ? short2(sgp_sm, psk) : short2(psk, sgp_sm);
      const short2 Bklims =
          transpose_b ? short2(psk, sgp_sn) : short2(sgp_sn, psk);

      const int A_offset = transpose_a ? k * lda : k;
      const int B_offset = transpose_b ? k : k * ldb;

      Atile.load_safe(A + A_offset, lda, Aklims);
      Btile.load_safe(B + B_offset, ldb, Bklims);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});
    }
  }

  return Dtile;
}

} // namespace mlx::steel

#line 1 "mlx/backend/metal/kernels/quantized_utils.h"

#include <metal_simdgroup>
#include <metal_stdlib>

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_aligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    loader_a.load_unsafe();
    loader_b.load_unsafe();

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <
    bool rows_aligned,
    bool cols_aligned,
    bool transpose,
    typename T,
    typename mma_t,
    typename loader_a_t,
    typename loader_b_t>
METAL_FUNC void gemm_loop_unaligned(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const int k_iterations,
    const short tgp_bm,
    const short tgp_bn,
    const short tgp_bk) {
  for (int k = 0; k < k_iterations; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Load elements into threadgroup memory
    if (rows_aligned) {
      loader_a.load_unsafe();
    } else {
      loader_a.load_safe(short2(tgp_bk, tgp_bm));
    }
    if (cols_aligned) {
      loader_b.load_unsafe();
    } else {
      loader_b.load_safe(
          transpose ? short2(tgp_bk, tgp_bn) : short2(tgp_bn, tgp_bk));
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Multiply and accumulate threadgroup elements
    mma_op.mma(As, Bs);

    // Prepare for next iteration
    loader_a.next();
    loader_b.next();
  }
}

template <typename T, typename mma_t, typename loader_a_t, typename loader_b_t>
METAL_FUNC void gemm_loop_finalize(
    threadgroup T* As,
    threadgroup T* Bs,
    thread mma_t& mma_op,
    thread loader_a_t& loader_a,
    thread loader_b_t& loader_b,
    const short2 tile_a,
    const short2 tile_b) {
  loader_a.load_safe(tile_a);
  loader_b.load_safe(tile_b);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mma_op.mma(As, Bs);
}

#line 1 "mlx/backend/metal/kernels/quantized_nax.h"

#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;
using namespace mlx::steel;

constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

template <typename U, int N, int bits>
inline void
dequantize(const device uint8_t* w, U scale, U bias, threadgroup U* w_local) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  const float s = float(scale);
  const float b = float(bias);

  if (bits == 2) {
    float sc[4] = {s, s / 4.0f, s / 16.0f, s / 64.0f};
    for (int i = 0; i < (N / 4); i++) {
      w_local[4 * i] = static_cast<U>(sc[0] * (w[i] & 0x03) + b);
      w_local[4 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0x0c) + b);
      w_local[4 * i + 2] = static_cast<U>(sc[2] * (w[i] & 0x30) + b);
      w_local[4 * i + 3] = static_cast<U>(sc[3] * (w[i] & 0xc0) + b);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 3 * i;

      w_local[0] = static_cast<U>((w[0] & 0x7) * s + b);
      w_local[1] = static_cast<U>(((w[0] & 0x38) >> 3) * s + b);
      w_local[2] =
          static_cast<U>((((w[0] & 0xc0) >> 6) + ((w[1] & 0x1) << 2)) * s + b);
      w_local[3] = static_cast<U>(((w[1] & 0xe) >> 1) * s + b);
      w_local[4] = static_cast<U>(((w[1] & 0x70) >> 4) * s + b);
      w_local[5] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0x3) << 1)) * s + b);
      w_local[6] = static_cast<U>(((w[2] & 0x1c) >> 2) * s + b);
      w_local[7] = static_cast<U>(((w[2] & 0xe0) >> 5) * s + b);
    }
  }

  else if (bits == 4) {
    float sc[2] = {s, s / 16.0f};
    for (int i = 0; i < (N / 2); i++) {
      w_local[2 * i] = static_cast<U>(sc[0] * (w[i] & 0x0f) + b);
      w_local[2 * i + 1] = static_cast<U>(sc[1] * (w[i] & 0xf0) + b);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      w_local += 8 * i;
      w += 5 * i;

      w_local[0] = static_cast<U>((w[0] & 0x1f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] & 0xe0) >> 5) + ((w[1] & 0x3) << 3)) * s + b);
      w_local[2] = static_cast<U>(((w[1] & 0x7c) >> 2) * s + b);
      w_local[3] =
          static_cast<U>((((w[1] & 0x80) >> 7) + ((w[2] & 0xf) << 1)) * s + b);
      w_local[4] =
          static_cast<U>((((w[2] & 0xf0) >> 4) + ((w[3] & 0x1) << 4)) * s + b);
      w_local[5] = static_cast<U>(((w[3] & 0x3e) >> 1) * s + b);
      w_local[6] =
          static_cast<U>((((w[3] & 0xc0) >> 6) + ((w[4] & 0x7) << 2)) * s + b);
      w_local[7] = static_cast<U>(((w[4] & 0xf8) >> 3) * s + b);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      w_local += 4 * i;
      w += 3 * i;
      w_local[0] = static_cast<U>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<U>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<U>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<U>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      w_local[i] = static_cast<U>(s * w[i] + b);
    }
  }
}

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader {
  static_assert(
      BCOLS <= group_size,
      "The group size should be larger than the columns");
  static_assert(
      group_size % BCOLS == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short group_steps = group_size / BCOLS;

  const int src_ld;
  const int tile_stride;
  short group_step_cnt;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_step_cnt(0),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size),
        biases(biases_ + bi * src_ld / group_size) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      if (group_steps > 1) {
        group_step_cnt++;
        if (group_step_cnt == group_steps) {
          group_step_cnt = 0;
          scales++;
          biases++;
        }
      } else {
        scales++;
        biases++;
      }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct QuantizedBlockLoader<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    32,
    bits> {
  MLX_MTL_CONST short group_size = 32;

  static_assert(
      BCOLS % group_size == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        biases(biases_ + bi * src_ld / group_size + group_id) {}

  void load_unsafe() const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          src + i * bytes_per_pack, scale, bias, dst + i * pack_factor);
    }
  }

  void load_safe(short2 src_tile_dim) const thread {
    if (BCOLS_PACKED * BROWS < tgp_size && bi >= BROWS) {
      return;
    }

    if (reduction_dim == 1 && bi >= src_tile_dim.x) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    if (reduction_dim == 0 && bi >= src_tile_dim.y) {
      for (int i = 0; i < n_reads * pack_factor; i++) {
        dst[i] = T(0);
      }
      return;
    }

    T scale = *scales;
    T bias = *biases;
    for (int i = 0; i < n_reads; i++) {
      dequantize<T, pack_factor, bits>(
          (device uint8_t*)(src + i * bytes_per_pack),
          scale,
          bias,
          dst + i * pack_factor);
    }
  }

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      // if (group_steps > 1) {
      //   group_step_cnt++;
      //   if (group_step_cnt == group_steps) {
      //     group_step_cnt = 0;
      //     scales++;
      //     biases++;
      //   }
      // } else {
      scales += n_groups;
      biases += n_groups;
      // }
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    device T*& y,
    int output_stride,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx = tid.z;
  uint32_t w_idx = tid.z;
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <typename T>
METAL_FUNC void adjust_matrix_offsets(
    const device T*& x,
    const device uint32_t*& w,
    const device T*& scales,
    const device T*& biases,
    const device uint32_t* lhs_indices,
    const device uint32_t* rhs_indices,
    device T*& y,
    int output_stride,
    const constant int& batch_ndims,
    const constant int* batch_shape,
    const constant int64_t* lhs_strides,
    const constant int64_t* rhs_strides,
    const constant int& x_batch_ndims,
    const constant int* x_shape,
    const constant int64_t* x_strides,
    const constant int& w_batch_ndims,
    const constant int* w_shape,
    const constant int64_t* w_strides,
    const constant int64_t* s_strides,
    const constant int64_t* b_strides,
    uint3 tid [[threadgroup_position_in_grid]]) {
  // Set the input/output matrices
  uint32_t x_idx;
  uint32_t w_idx;
  if (batch_ndims == 1) {
    x_idx = lhs_indices[tid.z * lhs_strides[0]];
    w_idx = rhs_indices[tid.z * rhs_strides[0]];
  } else {
    ulong2 idx = elem_to_loc_broadcast(
        tid.z, batch_shape, lhs_strides, rhs_strides, batch_ndims);
    x_idx = lhs_indices[idx.x];
    w_idx = rhs_indices[idx.y];
  }
  if (x_batch_ndims == 1) {
    x += x_idx * x_strides[0];
  } else {
    x += elem_to_loc(x_idx, x_shape, x_strides, x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += w_idx * w_strides[0];
    scales += w_idx * s_strides[0];
    biases += w_idx * b_strides[0];
  } else {
    ulong3 idx = elem_to_loc_broadcast(
        w_idx, w_shape, w_strides, s_strides, b_strides, w_batch_ndims);
    w += idx.x;
    scales += idx.y;
    biases += idx.z;
  }
  y += tid.z * output_stride;
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_t_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  (void)lid;

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BN,
      BK,
      BK_padded,
      1,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  x += y_row * static_cast<int64_t>(K);
  wl += y_col * K_w;
  scales += y_col * K_g;
  biases += y_col * K_g;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = true;

  const short sgp_sm = min(int(SM), M - (y_row + tm));
  const bool is_unaligned_sm = (sgp_sm != SM);

  const short sgp_sn = aligned_N ? SN : min(int(SN), N - (y_col + tn));

  const short tgp_bn = aligned_N ? BN : min(BN, int(N - (y_col)));
  const bool is_unaligned_bn = aligned_N ? false : (tgp_bn != BN);

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
    dispatch_bool(aligned_N || !is_unaligned_bn, [&](auto kAlignedN) {
      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if constexpr (kAlignedN.value) {
          loader_w.load_unsafe();
        } else {
          loader_w.load_safe(short2(BK, tgp_bn));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          if constexpr (kAlignedM.value) {
            Atile.load(x + kk1, K);
          } else {
            Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
          }

          Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

          tile_matmad_nax(
              Dtile,
              Atile,
              metal::bool_constant<transpose_a>{},
              Btile,
              metal::bool_constant<transpose_b>{});

          (void)compiler_barrier;
        }

        x += BK;
        loader_w.next();
      }

      // Store results to device memory
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if constexpr (kAlignedM.value && kAlignedN.value) {
        Dtile.store(y + tm * N + tn, N);
      } else if (kAlignedM.value && sgp_sn == SN) {
        Dtile.store(y + tm * N + tn, N);
      } else {
        Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
      }
    });
  });
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
METAL_FUNC void qmm_n_nax_tgp_impl(
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device T* x,
    device T* y,
    threadgroup T* Ws,
    const constant int& K,
    const constant int& N,
    const constant int& M,
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  static_assert(BK >= SIMD_SIZE, "BK should be larger than SIMD_SIZE");
  static_assert(BK % SIMD_SIZE == 0, "BK should be divisible by SIMD_SIZE");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      BK,
      BN,
      BN_padded,
      0,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  // Set the block
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;

  // Here w is [K, N]: packed and group-quantized along N, with row stride N.
  x += y_row * static_cast<int64_t>(K);
  wl += y_col * bytes_per_pack / pack_factor;
  scales += y_col / group_size;
  biases += y_col / group_size;
  y += y_row * static_cast<int64_t>(N) + y_col;

  // Make the weight loader
  loader_w_t loader_w(wl, scales, biases, N, Ws, simd_gid, simd_lid);

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_gid / WN);
  const short tn = SN * (simd_gid % WN);

  const short sgp_sm = min(int(SM), M - (y_row + tm));

  const short ldb_tgp = BN_padded;

  constexpr bool transpose_a = false;
  constexpr bool transpose_b = false;

  using AccumType = float;

  NAXTile<AccumType, TM, TN> Dtile;
  Dtile.clear();

  x += tm * K;

  for (int k = 0; k < K; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader_w.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);

    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      NAXTile<T, TM, TK> Atile;
      NAXTile<T, TK, TN> Btile;

      volatile int compiler_barrier;

      if (sgp_sm == SM) {
        Atile.load(x + kk1, K);
      } else {
        Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
      }

      Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * ldb_tgp);

      tile_matmad_nax(
          Dtile,
          Atile,
          metal::bool_constant<transpose_a>{},
          Btile,
          metal::bool_constant<transpose_b>{});

      (void)compiler_barrier;
    }

    x += BK;
    loader_w.next();
  }

  // Store results to device memory
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgp_sm == SM) {
    Dtile.store(y + tm * N + tn, N);
  } else {
    Dtile.store_safe(y + tm * N + tn, N, short2(SN, sgp_sm));
  }
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const bool batched,
    const int BM = 64,
    const int BK = 32,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Ws[BN * BK_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool batched,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    device T* y [[buffer(4)]],
    const constant int& K [[buffer(5)]],
    const constant int& N [[buffer(6)]],
    const constant int& M [[buffer(7)]],
    const constant int& x_batch_ndims [[buffer(8)]],
    const constant int* x_shape [[buffer(9)]],
    const constant int64_t* x_strides [[buffer(10)]],
    const constant int& w_batch_ndims [[buffer(11)]],
    const constant int* w_shape [[buffer(12)]],
    const constant int64_t* w_strides [[buffer(13)]],
    const constant int64_t* s_strides [[buffer(14)]],
    const constant int64_t* b_strides [[buffer(15)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  if (batched) {
    adjust_matrix_offsets<T>(
        x,
        w,
        scales,
        biases,
        y,
        M * N,
        x_batch_ndims,
        x_shape,
        x_strides,
        w_batch_ndims,
        w_shape,
        w_strides,
        s_strides,
        b_strides,
        tid);
  }

  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const bool aligned_N,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_t_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BK_padded = (BK + 16 / sizeof(T));

  threadgroup T Ws[BN * BK_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_t_nax_tgp_impl<T, group_size, bits, aligned_N, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    const int group_size,
    const int bits,
    const int BM = 64,
    const int BK = 64,
    const int BN = 64,
    const int WM = 2,
    const int WN = 2>
[[kernel]] void affine_gather_qmm_n_nax(
    const device uint32_t* w [[buffer(0)]],
    const device T* scales [[buffer(1)]],
    const device T* biases [[buffer(2)]],
    const device T* x [[buffer(3)]],
    const device uint32_t* lhs_indices [[buffer(4)]],
    const device uint32_t* rhs_indices [[buffer(5)]],
    device T* y [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    const constant int& x_batch_ndims [[buffer(10)]],
    const constant int* x_shape [[buffer(11)]],
    const constant int64_t* x_strides [[buffer(12)]],
    const constant int& w_batch_ndims [[buffer(13)]],
    const constant int* w_shape [[buffer(14)]],
    const constant int64_t* w_strides [[buffer(15)]],
    const constant int64_t* s_strides [[buffer(16)]],
    const constant int64_t* b_strides [[buffer(17)]],
    const constant int& batch_ndims [[buffer(18)]],
    const constant int* batch_shape [[buffer(19)]],
    const constant int64_t* lhs_strides [[buffer(20)]],
    const constant int64_t* rhs_strides [[buffer(21)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  (void)lid;

  constexpr int BN_padded = (BN + 16 / sizeof(T));

  threadgroup T Ws[BK * BN_padded];

  adjust_matrix_offsets<T>(
      x,
      w,
      scales,
      biases,
      lhs_indices,
      rhs_indices,
      y,
      M * N,
      batch_ndims,
      batch_shape,
      lhs_strides,
      rhs_strides,
      x_batch_ndims,
      x_shape,
      x_strides,
      w_batch_ndims,
      w_shape,
      w_strides,
      s_strides,
      b_strides,
      tid);
  qmm_n_nax_tgp_impl<T, group_size, bits, BM, BK, BN, WM, WN>(
      w, scales, biases, x, y, Ws, K, N, M, tid, lid, simd_gid, simd_lid);
}

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose>
[[kernel]] void affine_gather_qmm_rhs_nax(
    const device T* x [[buffer(0)]],
    const device uint32_t* w [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device uint32_t* indices [[buffer(4)]],
    device T* y [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BN_padded = (BN + 16 / sizeof(T));

  using loader_w_t = QuantizedBlockLoader<
      T,
      transpose ? BN : BK,
      transpose ? BK : BN,
      transpose ? BK_padded : BN_padded,
      transpose,
      WM * WN * SIMD_SIZE,
      group_size,
      bits>;

  threadgroup T Ws[transpose ? BN * BK_padded : BK * BN_padded];

  // Compute the block
  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int N_w = N * bytes_per_pack / pack_factor;
  const int N_g = N / group_size;
  const int K_it = K / BK;
  const size_t stride_w = transpose ? N * K_w : K * N_w;
  const size_t stride_s = transpose ? N * K_g : K * N_g;
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const size_t y_row_long = size_t(y_row);
  const size_t y_col_long = size_t(y_col);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));

  // Calculate the final tiles in the case that K is not aligned
  const int k_remain = K - K_it * BK;
  const short2 tile_w =
      transpose ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

  // Move x and output to the correct block
  auto wl = (const device uint8_t*)w;
  x += y_row_long * K;
  y += y_row_long * N + y_col_long;
  wl += transpose ? y_col_long * K_w : y_col * bytes_per_pack / pack_factor;
  scales += transpose ? y_col_long * K_g : y_col / group_size;
  biases += transpose ? y_col_long * K_g : y_col / group_size;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;

  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;

  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);

  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, (M - (y_row + tm)))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, (N - (y_col + tn)))));

  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);

  constexpr short BR = transpose ? TN : TK;
  constexpr short BC = transpose ? TK : TN;

  using AccumType = float;

  // Do as many matmuls as necessary
  uint32_t index;
  short offset;
  uint32_t index_next = indices[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (indices[y_row + n] != index) {
        offset_next = n;
        index_next = indices[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);

    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    const device T* xn = x + tm * K;

    // Prepare threadgroup loading operations
    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        transpose ? K : N,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            loader_w.load_unsafe();
          } else {
            loader_w.load_safe(
                transpose ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
          }

          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              if constexpr (kAlignedM.value) {
                Atile.load(xn + kk1, K);
              } else {
                Atile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }

          xn += BK;
          loader_w.next();
        }

        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> Atile;
              NAXTile<T, BR, BC> Btile;

              volatile int compiler_barrier;

              const short psk = min(int(SK), max(0, (BK - kk1)));
              Atile.load_safe(xn + kk1, K, short2(psk, sgp_sm));

              if constexpr (transpose) {
                Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);
              } else {
                Btile.template load<T, BN_padded, 1>(Ws + tn + kk1 * BN_padded);
              }

              tile_matmad_nax(
                  Dtile,
                  Atile,
                  metal::bool_constant<false>{},
                  Btile,
                  metal::bool_constant<transpose>{});

              (void)compiler_barrier;
            }
          }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Store results to device memory
        if constexpr (kAlignedN.value) {
          if (m_lo_lim == 0 && m_hi_lim == SM) {
            Dtile.store(y + tm * N + tn, N);
          } else {
            Dtile.store_slice(
                y + tm * N + tn, N, short2(0, m_lo_lim), short2(SN, m_hi_lim));
          }
        } else {
          Dtile.store_slice(
              y + tm * N + tn,
              N,
              short2(0, m_lo_lim),
              short2(sgp_sn, m_hi_lim));
        }
      });
    });
  }
}

