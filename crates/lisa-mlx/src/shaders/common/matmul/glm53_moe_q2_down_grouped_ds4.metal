using namespace metal;

// DS4's production Q2 grouped down path is the standard 64-output by
// 32-route by 32-K simdgroup matmul with an F16 RHS. Decode only the MLX
// affine-Q2 weight representation differently, then preserve DS4's staging,
// simdgroup accumulation, and route-major output layout.
template <typename M, bool VECTOR_LOAD, int OUTPUT_TILE = 64>
[[kernel]] void glm53_moe_q2_down_grouped_ds4(
    const device ulong* address_table [[buffer(0)]],
    const device half* x [[buffer(1)]],
    const device int* route_ids [[buffer(2)]],
    const device int* counts [[buffer(3)]],
    const device int* work_count [[buffer(4)]],
    const device int* work_items [[buffer(5)]],
    device float* y [[buffer(6)]],
    const constant int& n_tokens [[buffer(7)]],
    const constant int& top_k [[buffer(8)]],
    const constant int& n_experts [[buffer(9)]],
    const constant int& out_dim [[buffer(10)]],
    const constant int& in_dim [[buffer(11)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = OUTPUT_TILE;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    constexpr int GROUP_SIZE = 128;

    const uint work_index = tgpig.x;
    if (work_index >= (uint)work_count[0]) return;
    const int expert = work_items[work_index * 2u];
    const int route_start = work_items[work_index * 2u + 1u];
    if (expert < 0 || expert >= n_experts || route_start < 0) return;

    const int route_count = counts[expert];
    const int output_row = int(tgpig.y) * NR0;
    if (route_start >= route_count || output_row >= out_dim) return;
    const int valid_routes = min(NR1, route_count - route_start);
    const int valid_outputs = min(NR0, out_dim - output_row);
    // The second pair of simdgroups owns routes 16..31. Keep its lanes at
    // threadgroup barriers, but skip their MMA when this expert tile has no
    // rows in that half.
    const bool mma_active = 16 * int(sgitg / 2) < valid_routes;

    const size_t expert_base = (size_t)expert * 9u;
    const device uint8_t* weight = reinterpret_cast<const device uint8_t*>(
        address_table[expert_base + 6u]);
    const device M* scales = reinterpret_cast<const device M*>(
        address_table[expert_base + 7u]);
    const device M* biases = reinterpret_cast<const device M*>(
        address_table[expert_base + 8u]);
    const int weight_row_bytes = in_dim / 4;
    const int scale_groups = in_dim / GROUP_SIZE;

    // The 128-output experiment computes two independent DS4 64-row tiles
    // in one threadgroup. Each output keeps the same K accumulation sequence.
    threadgroup half storage[OUTPUT_TILE == 128 ? 9216 : 4096];
    threadgroup half* sa = storage;
    threadgroup half* sb = storage + (OUTPUT_TILE / 64) * 64 * NK;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 ma1[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    simdgroup_float8x8 mc1[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        if constexpr (OUTPUT_TILE == 128) {
            mc1[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (int loop_k = 0; loop_k < in_dim; loop_k += NK) {
        // Match DS4's A-fragment ownership and shared-memory swizzle.
        const int output = int(tiitg) / NL0;
        const int il0 = int(tiitg) % NL0;
        const int sx_base = 2 * il0;
        const int sy = output / 8;
        const int lx = output % 8;
        // Mapped safetensors weight origins may be byte-aligned but not
        // four-byte aligned. Only issue uchar4 loads when the tensor origin
        // itself satisfies the vector type's alignment; otherwise preserve
        // the scalar byte path.
        const bool vector_load_aligned =
            VECTOR_LOAD && (address_table[expert_base + 6u] & 3ul) == 0ul;
        for (short output_block = 0; output_block < OUTPUT_TILE / 64; ++output_block) {
            const int weight_row = output_row + output + output_block * 64;
            const int scale_index = weight_row * scale_groups + loop_k / GROUP_SIZE;
            const float scale = float(scales[scale_index]);
            const float bias = float(biases[scale_index]);
            threadgroup half* sa_block = sa + (size_t)output_block * 64u * NK;
            if (vector_load_aligned) {
                const device uchar4* weight_bytes = reinterpret_cast<const device uchar4*>(
                    weight + (size_t)weight_row * (size_t)weight_row_bytes);
                const int global_k_base = loop_k + 16 * il0;
                const uchar4 packed = weight_bytes[global_k_base / 16];
                for (short byte_lane = 0; byte_lane < 4; ++byte_lane) {
                    const uint8_t packed_byte = packed[byte_lane];
                    for (short code_lane = 0; code_lane < 4; ++code_lane) {
                        const int i = 4 * byte_lane + code_lane;
                        const int code = int((packed_byte >> (2 * code_lane)) & 0x03u);
                        const half value = half(float(code) * scale + bias);
                        const int sx = sx_base + i / 8;
                        const int ly = i % 8;
                        const int block = 8 * sx + sy;
                        sa_block[64 * block + 8 * ly + lx] = value;
                    }
                }
            } else {
                for (short i = 0; i < 16; ++i) {
                    const int local_k = 16 * il0 + i;
                    const int global_k = loop_k + local_k;
                    const uint8_t packed = weight[
                        (size_t)weight_row * (size_t)weight_row_bytes + (size_t)(global_k / 4)];
                    const int code = int((packed >> ((global_k & 3) * 2)) & 0x03u);
                    const half value = half(float(code) * scale + bias);
                    const int sx = sx_base + i / 8;
                    const int ly = i % 8;
                    const int block = 8 * sx + sy;
                    sa_block[64 * block + 8 * ly + lx] = value;
                }
            }
        }

        // Match DS4's B-fragment ownership, with already-F16 routed inputs.
        const int route = int(tiitg) / NL1;
        const int k_block = int(tiitg) % NL1;
        const int route_block = route / 8;
        const int route_lane = route % 8;
        const int b_block = 4 * k_block + route_block;
        const size_t route_index = (size_t)expert * (size_t)n_tokens +
            (size_t)(route_start + route);
        const int slot = route < valid_routes ? route_ids[route_index] : -1;
        for (short i = 0; i < 8; ++i) {
            const int global_k = loop_k + 8 * k_block + i;
            const half value = route < valid_routes
                ? x[(size_t)slot * (size_t)in_dim + (size_t)global_k]
                : half(0.0f);
            sb[64 * b_block + 8 * route_lane + i] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsma1 = sa + 64 * NK + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        if (mma_active) {
            for (short ik = 0; ik < NK / 8; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                    if constexpr (OUTPUT_TILE == 128) {
                        simdgroup_load(ma1[i], lsma1 + 64 * i, 8, 0, false);
                    }
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 2; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 8; ++i) {
                    simdgroup_multiply_accumulate(
                        mc[i], mb[i / 4], ma[i % 4], mc[i]);
                    if constexpr (OUTPUT_TILE == 128) {
                        simdgroup_multiply_accumulate(
                            mc1[i], mb[i / 4], ma1[i % 4], mc1[i]);
                    }
                }
                lsma += 8 * 64;
                if constexpr (OUTPUT_TILE == 128) {
                    lsma1 += 8 * 64;
                }
                lsmb += 4 * 64;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* result = reinterpret_cast<threadgroup float*>(storage);
    threadgroup float* temp = result + 32 * (sgitg & 1) +
        (16 * (sgitg >> 1)) * NR0;
    threadgroup float* temp1 = temp + 64;
    if (mma_active) {
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(
                mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
            if constexpr (OUTPUT_TILE == 128) {
                simdgroup_store(
                    mc1[i], temp1 + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short route = sgitg; route < valid_routes; route += 4) {
        const int slot = route_ids[(size_t)expert * (size_t)n_tokens +
            (size_t)(route_start + route)];
        threadgroup const float* row = result + (size_t)route * NR0;
        for (int col = tiisg; col < valid_outputs; col += 32) {
            y[(size_t)slot * (size_t)out_dim + (size_t)(output_row + col)] = row[col];
        }
    }

    (void)top_k;
}

template [[host_name("glm53_moe_q2_down_grouped_ds4_f32_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<float, true>)
        glm53_moe_q2_down_grouped_ds4<float, true>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_bf16_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<bfloat16_t, true>)
        glm53_moe_q2_down_grouped_ds4<bfloat16_t, true>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_f32_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<float, false>)
        glm53_moe_q2_down_grouped_ds4<float, false>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_bf16_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<bfloat16_t, false>)
        glm53_moe_q2_down_grouped_ds4<bfloat16_t, false>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_f32_nr1_32_nr0_128_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<float, true, 128>)
        glm53_moe_q2_down_grouped_ds4<float, true, 128>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_bf16_nr1_32_nr0_128_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<bfloat16_t, true, 128>)
        glm53_moe_q2_down_grouped_ds4<bfloat16_t, true, 128>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_f32_scalar_nr1_32_nr0_128_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<float, false, 128>)
        glm53_moe_q2_down_grouped_ds4<float, false, 128>;
template [[host_name("glm53_moe_q2_down_grouped_ds4_bf16_scalar_nr1_32_nr0_128_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_down_grouped_ds4<bfloat16_t, false, 128>)
        glm53_moe_q2_down_grouped_ds4<bfloat16_t, false, 128>;
