// Strided -> strided copy (`slice_assign` into a view): `src` and `dst` are both
// described by the iteration shape with their own element strides. One thread
// per element; the rank is fixed at 8 (the maximum an Array view carries).

#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;

struct Copy2Params { uint ndim; uint shape[8]; uint src_strides[8]; uint dst_strides[8]; };

template <typename T>
[[kernel]] void copy2_strided(
    const device T* src [[buffer(0)]],
    device T* dst [[buffer(1)]],
    constant Copy2Params& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint rem = gid;
    ulong so = 0;
    ulong doff = 0;
    for (int d = int(p.ndim) - 1; d >= 0; --d) {
        const uint idx = rem % p.shape[d];
        rem /= p.shape[d];
        so += (ulong)idx * (ulong)p.src_strides[d];
        doff += (ulong)idx * (ulong)p.dst_strides[d];
    }
    dst[doff] = src[so];
}

template [[host_name("copy2_strided_uint8_t")]] [[kernel]] decltype(copy2_strided<uchar>) copy2_strided<uchar>;
template [[host_name("copy2_strided_uint16_t")]] [[kernel]] decltype(copy2_strided<uint16_t>) copy2_strided<uint16_t>;
template [[host_name("copy2_strided_uint32_t")]] [[kernel]] decltype(copy2_strided<uint>) copy2_strided<uint>;
template [[host_name("copy2_strided_int8_t")]] [[kernel]] decltype(copy2_strided<int8_t>) copy2_strided<int8_t>;
template [[host_name("copy2_strided_int16_t")]] [[kernel]] decltype(copy2_strided<int16_t>) copy2_strided<int16_t>;
template [[host_name("copy2_strided_int32_t")]] [[kernel]] decltype(copy2_strided<int>) copy2_strided<int>;
template [[host_name("copy2_strided_int64_t")]] [[kernel]] decltype(copy2_strided<int64_t>) copy2_strided<int64_t>;
template [[host_name("copy2_strided_float16_t")]] [[kernel]] decltype(copy2_strided<half>) copy2_strided<half>;
template [[host_name("copy2_strided_float")]] [[kernel]] decltype(copy2_strided<float>) copy2_strided<float>;
template [[host_name("copy2_strided_bfloat16_t")]] [[kernel]] decltype(copy2_strided<bfloat16_t>) copy2_strided<bfloat16_t>;