// Contiguous materialisation of a strided view: `src` is indexed through the
// element strides in `CopyParams`; `dst` is row-major (last dim varies fastest),
// one thread per element. Rank is fixed at 8.

#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

struct CopyParams { uint ndim; uint shape[8]; uint strides[8]; };

template <typename T>
[[kernel]] void copy_strided(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant CopyParams& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rem = gid;
    ulong off = 0;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        const uint idx = rem % p.shape[d];
        rem /= p.shape[d];
        off += (ulong)idx * (ulong)p.strides[d];
    }
    dst[gid] = src[off];
}

template [[host_name("copy_strided_uint8_t")]] [[kernel]] decltype(copy_strided<uchar>) copy_strided<uchar>;
template [[host_name("copy_strided_uint16_t")]] [[kernel]] decltype(copy_strided<uint16_t>) copy_strided<uint16_t>;
template [[host_name("copy_strided_uint32_t")]] [[kernel]] decltype(copy_strided<uint>) copy_strided<uint>;
template [[host_name("copy_strided_int8_t")]] [[kernel]] decltype(copy_strided<int8_t>) copy_strided<int8_t>;
template [[host_name("copy_strided_int16_t")]] [[kernel]] decltype(copy_strided<int16_t>) copy_strided<int16_t>;
template [[host_name("copy_strided_int32_t")]] [[kernel]] decltype(copy_strided<int>) copy_strided<int>;
template [[host_name("copy_strided_int64_t")]] [[kernel]] decltype(copy_strided<int64_t>) copy_strided<int64_t>;
template [[host_name("copy_strided_float16_t")]] [[kernel]] decltype(copy_strided<half>) copy_strided<half>;
template [[host_name("copy_strided_float")]] [[kernel]] decltype(copy_strided<float>) copy_strided<float>;
template [[host_name("copy_strided_bfloat16_t")]] [[kernel]] decltype(copy_strided<bfloat16_t>) copy_strided<bfloat16_t>;