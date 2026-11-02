using namespace metal;

// DS4 grouped MoE runs gate and up as separate 64x32x32 simdgroup matmuls,
// then applies routed SwiGLU in its own kernel. `IS_UP` selects one MLX
// affine-Q2 matrix from the expert address table.
template <typename M, bool IS_UP, bool VECTOR_LOAD>
[[kernel]] void glm53_moe_q2_mm_grouped_ds4(
    const device ulong* address_table [[buffer(0)]],
    const device float* x [[buffer(1)]],
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

    constexpr int NR0 = 64;
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
    // SIMDgroups 0/1 own routes 0..15 and 2/3 own 16..31. Keep inactive
    // groups in staging and at threadgroup barriers, but skip their MMA work.
    const bool mma_active = 16 * int(sgitg / 2) < valid_routes;

    const size_t expert_base = (size_t)expert * 9u + (IS_UP ? 3u : 0u);
    const device uint8_t* weight = reinterpret_cast<const device uint8_t*>(
        address_table[expert_base]);
    const device M* scales = reinterpret_cast<const device M*>(
        address_table[expert_base + 1u]);
    const device M* biases = reinterpret_cast<const device M*>(
        address_table[expert_base + 2u]);
    const int weight_row_bytes = in_dim / 4;
    const int scale_groups = in_dim / GROUP_SIZE;

    threadgroup half storage[4096];
    threadgroup half* sa = storage;
    threadgroup half* sb = storage + NR0 * NK;
    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < in_dim; loop_k += NK) {
        const int output = int(tiitg) / NL0;
        const int il0 = int(tiitg) % NL0;
        const int sx_base = 2 * il0;
        const int sy = output / 8;
        const int lx = output % 8;
        // Affine groups are 128 values and NK tiles are 32 values, so one
        // scale/bias pair covers this entire K tile.
        const int weight_row = output_row + output;
        const int scale_index = weight_row * scale_groups + loop_k / GROUP_SIZE;
        const float scale = float(scales[scale_index]);
        const float bias = float(biases[scale_index]);
        // Mapped safetensors weight origins may be byte-aligned but not
        // four-byte aligned. Only issue uchar4 loads when the tensor origin
        // itself satisfies the vector type's alignment; otherwise preserve
        // the scalar byte path.
        const bool vector_load_aligned =
            VECTOR_LOAD && (address_table[expert_base] & 3ul) == 0ul;
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
                    sa[64 * block + 8 * ly + lx] = value;
                }
            }
        } else {
            for (short i = 0; i < 16; ++i) {
                const int global_k = loop_k + 16 * il0 + i;
                const uint8_t packed = weight[
                    (size_t)weight_row * (size_t)weight_row_bytes + (size_t)(global_k / 4)];
                const int code = int((packed >> ((global_k & 3) * 2)) & 0x03u);
                const half value = half(float(code) * scale + bias);
                const int sx = sx_base + i / 8;
                const int ly = i % 8;
                const int block = 8 * sx + sy;
                sa[64 * block + 8 * ly + lx] = value;
            }
        }

        const int route = int(tiitg) / NL1;
        const int k_block = int(tiitg) % NL1;
        const int route_block = route / 8;
        const int route_lane = route % 8;
        const int b_block = 4 * k_block + route_block;
        const size_t route_index = (size_t)expert * (size_t)n_tokens +
            (size_t)(route_start + route);
        const int slot = route < valid_routes ? route_ids[route_index] : -1;
        const int token = slot >= 0 ? slot / top_k : 0;
        for (short i = 0; i < 8; ++i) {
            const int global_k = loop_k + 8 * k_block + i;
            const half value = route < valid_routes
                ? half(x[(size_t)token * (size_t)in_dim + (size_t)global_k])
                : half(0.0f);
            sb[64 * b_block + 8 * route_lane + i] = value;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half* lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        if (mma_active) {
            for (short ik = 0; ik < NK / 8; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 4; ++i) {
                    simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 2; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 8; ++i) {
                    simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
                }
                lsma += 8 * 64;
                lsmb += 4 * 64;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* result = reinterpret_cast<threadgroup float*>(storage);
    threadgroup float* temp = result + 32 * (sgitg & 1) +
        (16 * (sgitg >> 1)) * NR0;
    if (mma_active) {
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
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

template [[host_name("glm53_moe_q2_mm_grouped_ds4_f32_gate_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<float, false, true>)
        glm53_moe_q2_mm_grouped_ds4<float, false, true>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_f32_up_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<float, true, true>)
        glm53_moe_q2_mm_grouped_ds4<float, true, true>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_bf16_gate_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<bfloat16_t, false, true>)
        glm53_moe_q2_mm_grouped_ds4<bfloat16_t, false, true>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_bf16_up_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<bfloat16_t, true, true>)
        glm53_moe_q2_mm_grouped_ds4<bfloat16_t, true, true>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_f32_gate_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<float, false, false>)
        glm53_moe_q2_mm_grouped_ds4<float, false, false>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_f32_up_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<float, true, false>)
        glm53_moe_q2_mm_grouped_ds4<float, true, false>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_bf16_gate_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<bfloat16_t, false, false>)
        glm53_moe_q2_mm_grouped_ds4<bfloat16_t, false, false>;
template [[host_name("glm53_moe_q2_mm_grouped_ds4_bf16_up_scalar_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_mm_grouped_ds4<bfloat16_t, true, false>)
        glm53_moe_q2_mm_grouped_ds4<bfloat16_t, true, false>;
