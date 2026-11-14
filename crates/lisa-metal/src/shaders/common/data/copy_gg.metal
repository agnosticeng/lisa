// General-general strided copy (used by `concatenate` to write each input into
// a strided slice of the output). `elem_to_loc_*` and `cast_to` come from the
// MLX utils preamble, which is prepended to this source at compile time.
//
// One kernel per rank (1/2/3) and one explicit instantiation per element type;
// the host selects `copy_gg_nd{rank}_{type}`.

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd1(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t& src_stride [[buffer(3)]],
    constant const int64_t& dst_stride [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_1<IdxT>(index, src_stride);
  auto dst_idx = elem_to_loc_1<IdxT>(index, dst_stride);
  dst[dst_idx] = cast_to<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd2(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint2 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_2<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_2<IdxT>(index, dst_strides);
  dst[dst_idx] = cast_to<U>(src[src_idx]);
}

template <typename T, typename U, typename IdxT = int64_t>
[[kernel]] void copy_gg_nd3(
    device const T* src [[buffer(0)]],
    device U* dst [[buffer(1)]],
    constant const int64_t* src_strides [[buffer(3)]],
    constant const int64_t* dst_strides [[buffer(4)]],
    uint3 index [[thread_position_in_grid]]) {
  auto src_idx = elem_to_loc_3<IdxT>(index, src_strides);
  auto dst_idx = elem_to_loc_3<IdxT>(index, dst_strides);
  dst[dst_idx] = cast_to<U>(src[src_idx]);
}

template [[host_name("copy_gg_nd1_uint8_t")]] [[kernel]] decltype(copy_gg_nd1<uchar, uchar>) copy_gg_nd1<uchar, uchar>;
template [[host_name("copy_gg_nd2_uint8_t")]] [[kernel]] decltype(copy_gg_nd2<uchar, uchar>) copy_gg_nd2<uchar, uchar>;
template [[host_name("copy_gg_nd3_uint8_t")]] [[kernel]] decltype(copy_gg_nd3<uchar, uchar>) copy_gg_nd3<uchar, uchar>;

template [[host_name("copy_gg_nd1_uint16_t")]] [[kernel]] decltype(copy_gg_nd1<uint16_t, uint16_t>) copy_gg_nd1<uint16_t, uint16_t>;
template [[host_name("copy_gg_nd2_uint16_t")]] [[kernel]] decltype(copy_gg_nd2<uint16_t, uint16_t>) copy_gg_nd2<uint16_t, uint16_t>;
template [[host_name("copy_gg_nd3_uint16_t")]] [[kernel]] decltype(copy_gg_nd3<uint16_t, uint16_t>) copy_gg_nd3<uint16_t, uint16_t>;

template [[host_name("copy_gg_nd1_uint32_t")]] [[kernel]] decltype(copy_gg_nd1<uint, uint>) copy_gg_nd1<uint, uint>;
template [[host_name("copy_gg_nd2_uint32_t")]] [[kernel]] decltype(copy_gg_nd2<uint, uint>) copy_gg_nd2<uint, uint>;
template [[host_name("copy_gg_nd3_uint32_t")]] [[kernel]] decltype(copy_gg_nd3<uint, uint>) copy_gg_nd3<uint, uint>;

template [[host_name("copy_gg_nd1_int8_t")]] [[kernel]] decltype(copy_gg_nd1<int8_t, int8_t>) copy_gg_nd1<int8_t, int8_t>;
template [[host_name("copy_gg_nd2_int8_t")]] [[kernel]] decltype(copy_gg_nd2<int8_t, int8_t>) copy_gg_nd2<int8_t, int8_t>;
template [[host_name("copy_gg_nd3_int8_t")]] [[kernel]] decltype(copy_gg_nd3<int8_t, int8_t>) copy_gg_nd3<int8_t, int8_t>;

template [[host_name("copy_gg_nd1_int16_t")]] [[kernel]] decltype(copy_gg_nd1<int16_t, int16_t>) copy_gg_nd1<int16_t, int16_t>;
template [[host_name("copy_gg_nd2_int16_t")]] [[kernel]] decltype(copy_gg_nd2<int16_t, int16_t>) copy_gg_nd2<int16_t, int16_t>;
template [[host_name("copy_gg_nd3_int16_t")]] [[kernel]] decltype(copy_gg_nd3<int16_t, int16_t>) copy_gg_nd3<int16_t, int16_t>;

template [[host_name("copy_gg_nd1_int32_t")]] [[kernel]] decltype(copy_gg_nd1<int, int>) copy_gg_nd1<int, int>;
template [[host_name("copy_gg_nd2_int32_t")]] [[kernel]] decltype(copy_gg_nd2<int, int>) copy_gg_nd2<int, int>;
template [[host_name("copy_gg_nd3_int32_t")]] [[kernel]] decltype(copy_gg_nd3<int, int>) copy_gg_nd3<int, int>;

template [[host_name("copy_gg_nd1_int64_t")]] [[kernel]] decltype(copy_gg_nd1<int64_t, int64_t>) copy_gg_nd1<int64_t, int64_t>;
template [[host_name("copy_gg_nd2_int64_t")]] [[kernel]] decltype(copy_gg_nd2<int64_t, int64_t>) copy_gg_nd2<int64_t, int64_t>;
template [[host_name("copy_gg_nd3_int64_t")]] [[kernel]] decltype(copy_gg_nd3<int64_t, int64_t>) copy_gg_nd3<int64_t, int64_t>;

template [[host_name("copy_gg_nd1_float16_t")]] [[kernel]] decltype(copy_gg_nd1<half, half>) copy_gg_nd1<half, half>;
template [[host_name("copy_gg_nd2_float16_t")]] [[kernel]] decltype(copy_gg_nd2<half, half>) copy_gg_nd2<half, half>;
template [[host_name("copy_gg_nd3_float16_t")]] [[kernel]] decltype(copy_gg_nd3<half, half>) copy_gg_nd3<half, half>;

template [[host_name("copy_gg_nd1_float")]] [[kernel]] decltype(copy_gg_nd1<float, float>) copy_gg_nd1<float, float>;
template [[host_name("copy_gg_nd2_float")]] [[kernel]] decltype(copy_gg_nd2<float, float>) copy_gg_nd2<float, float>;
template [[host_name("copy_gg_nd3_float")]] [[kernel]] decltype(copy_gg_nd3<float, float>) copy_gg_nd3<float, float>;


template [[host_name("copy_gg_nd1_bfloat16_t")]] [[kernel]] decltype(copy_gg_nd1<bfloat16_t, bfloat16_t>) copy_gg_nd1<bfloat16_t, bfloat16_t>;
template [[host_name("copy_gg_nd2_bfloat16_t")]] [[kernel]] decltype(copy_gg_nd2<bfloat16_t, bfloat16_t>) copy_gg_nd2<bfloat16_t, bfloat16_t>;
template [[host_name("copy_gg_nd3_bfloat16_t")]] [[kernel]] decltype(copy_gg_nd3<bfloat16_t, bfloat16_t>) copy_gg_nd3<bfloat16_t, bfloat16_t>;