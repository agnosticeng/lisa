using namespace metal;

// DS4's default grouped GLM Q2 route uses the 64-output x 32-route
// simdgroup GEMM, rounds both operands to F16, then writes the routed
// SwiGLU result to an F16 intermediate. The only weight-format adaptation
// here is decoding MLX affine-Q2 rows from U32-packed storage.
template <typename M>
[[kernel]] void glm53_moe_q2_pair_grouped_ds4(
    const device ulong* address_table [[buffer(0)]],
    const device float* x [[buffer(1)]],
    const device int* route_ids [[buffer(2)]],
    const device int* counts [[buffer(3)]],
    const device int* work_count [[buffer(4)]],
    const device int* work_items [[buffer(5)]],
    const device float* route_weights [[buffer(6)]],
    device half* y [[buffer(7)]],
    const constant int& n_tokens [[buffer(8)]],
    const constant int& top_k [[buffer(9)]],
    const constant int& n_experts [[buffer(10)]],
    const constant int& mid_dim [[buffer(11)]],
    const constant int& in_dim [[buffer(12)]],
    const constant float& limit [[buffer(13)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK = 32;
    constexpr int NL0 = NK / 16;
    constexpr int NL1 = NK / 8;
    constexpr int NUM_THREADS = 128;
    constexpr int GROUP_SIZE = 128;

    const uint work_index = tgpig.x;
    if (work_index >= (uint)work_count[0]) return;
    const int expert = work_items[work_index * 2u];
    const int route_start = work_items[work_index * 2u + 1u];
    if (expert < 0 || expert >= n_experts || route_start < 0) return;

    const int route_count = counts[expert];
    if (route_start >= route_count) return;
    const int valid_routes = min(NR1, route_count - route_start);
    const int output_row = int(tgpig.y) * NR0;
    if (output_row >= mid_dim) return;
    const int valid_outputs = min(NR0, mid_dim - output_row);
    // SIMDgroups 0/1 own routes 0..15 and 2/3 own 16..31. Keep inactive
    // groups in staging and at threadgroup barriers, but skip their MMA work.
    const bool mma_active = 16 * int(sgitg / 2) < valid_routes;

    const size_t expert_base = (size_t)expert * 9u;
    const device uint8_t* gate_w = reinterpret_cast<const device uint8_t*>(
        address_table[expert_base + 0u]);
    const device M* gate_s = reinterpret_cast<const device M*>(
        address_table[expert_base + 1u]);
    const device M* gate_b = reinterpret_cast<const device M*>(
        address_table[expert_base + 2u]);
    const device uint8_t* up_w = reinterpret_cast<const device uint8_t*>(
        address_table[expert_base + 3u]);
    const device M* up_s = reinterpret_cast<const device M*>(
        address_table[expert_base + 4u]);
    const device M* up_b = reinterpret_cast<const device M*>(
        address_table[expert_base + 5u]);

    const int row_bytes = in_dim / 4;
    const int scale_groups = in_dim / GROUP_SIZE;

    // The first 10 KiB stage the A/B tiles. After the last K iteration those
    // bytes are reused for the 16 KiB F32 gate/up result tiles, as in DS4.
    threadgroup half storage[8192];
    threadgroup half* sa_gate = storage;
    threadgroup half* sa_up = storage + NR0 * NK;
    threadgroup half* sb = storage + 2 * NR0 * NK;

    simdgroup_half8x8 ma_gate[4];
    simdgroup_half8x8 ma_up[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_gate[8];
    simdgroup_float8x8 mc_up[8];
    for (short i = 0; i < 8; ++i) {
        mc_gate[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_up[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (int loop_k = 0; loop_k < in_dim; loop_k += NK) {
        // DS4's 16-value A fragments are distributed across output rows and
        // K lanes before the simdgroup loads. Keep that shared-memory order.
        const int output = int(tiitg) / NL0;
        const int il0 = int(tiitg) % NL0;
        const int sx_base = 2 * il0;
        const int sy = output / 8;
        const int lx = output % 8;
        // The 128-value affine group contains the complete 32-wide K tile.
        const int weight_row = output_row + output;
        const int scale_index = weight_row * scale_groups + loop_k / GROUP_SIZE;
        const float gate_scale = float(gate_s[scale_index]);
        const float gate_bias = float(gate_b[scale_index]);
        const float up_scale = float(up_s[scale_index]);
        const float up_bias = float(up_b[scale_index]);
        for (short i = 0; i < 16; ++i) {
            const int local_k = 16 * il0 + i;
            const int global_k = loop_k + local_k;
            const uint8_t gate_byte = gate_w[
                (size_t)weight_row * (size_t)row_bytes + (size_t)(global_k / 4)];
            const uint8_t up_byte = up_w[
                (size_t)weight_row * (size_t)row_bytes + (size_t)(global_k / 4)];
            const int shift = (global_k & 3) * 2;
            const float gate_code = float((gate_byte >> shift) & 0x03u);
            const float up_code = float((up_byte >> shift) & 0x03u);
            const half gate_value = half(gate_code * gate_scale + gate_bias);
            const half up_value = half(up_code * up_scale + up_bias);
            const int sx = sx_base + i / 8;
            const int ly = i % 8;
            const int block = 8 * sx + sy;
            const int dest = 64 * block + 8 * ly + lx;
            sa_gate[dest] = gate_value;
            sa_up[dest] = up_value;
        }

        // B is [route, K] in shared memory. DS4 reads F32 activations and
        // narrows each staged value to its F16 GEMM operand.
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
            const float value = route < valid_routes
                ? x[(size_t)token * (size_t)in_dim + (size_t)global_k]
                : 0.0f;
            sb[64 * b_block + 8 * route_lane + i] = half(value);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half* lsma_gate = sa_gate + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsma_up = sa_up + 4 * 64 * (sgitg % 2);
        threadgroup const half* lsmb = sb + 2 * 64 * (sgitg / 2);
        if (mma_active) {
            for (short ik = 0; ik < NK / 8; ++ik) {
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 4; ++i) {
                    simdgroup_load(ma_gate[i], lsma_gate + 64 * i, 8, 0, false);
                    simdgroup_load(ma_up[i], lsma_up + 64 * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 2; ++i) {
                    simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
                }
                simdgroup_barrier(mem_flags::mem_none);
                for (short i = 0; i < 8; ++i) {
                    simdgroup_multiply_accumulate(
                        mc_gate[i], mb[i / 4], ma_gate[i % 4], mc_gate[i]);
                    simdgroup_multiply_accumulate(
                        mc_up[i], mb[i / 4], ma_up[i % 4], mc_up[i]);
                }
                lsma_gate += 8 * 64;
                lsma_up += 8 * 64;
                lsmb += 4 * 64;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float* result_gate = reinterpret_cast<threadgroup float*>(storage);
    threadgroup float* result_up = reinterpret_cast<threadgroup float*>(storage) +
        NR0 * NR1;
    threadgroup float* temp_gate = result_gate + 32 * (sgitg & 1) +
        (16 * (sgitg >> 1)) * NR0;
    threadgroup float* temp_up = result_up + 32 * (sgitg & 1) +
        (16 * (sgitg >> 1)) * NR0;
    if (mma_active) {
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(
                mc_gate[i], temp_gate + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
            simdgroup_store(
                mc_up[i], temp_up + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0, false);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short route = sgitg; route < valid_routes; route += 4) {
        const int slot = route_ids[(size_t)expert * (size_t)n_tokens +
            (size_t)(route_start + route)];
        const float route_weight = route_weights[slot];
        threadgroup const float* gate_row = result_gate + (size_t)route * NR0;
        threadgroup const float* up_row = result_up + (size_t)route * NR0;
        for (int row = tiisg; row < valid_outputs; row += 32) {
            float gate = gate_row[row];
            float up = up_row[row];
            if (limit > 1.0e-6f) {
                gate = min(gate, limit);
                up = clamp(up, -limit, limit);
            }
            const float silu = gate / (1.0f + exp(-gate));
            y[(size_t)slot * (size_t)mid_dim + (size_t)(output_row + row)] =
                half(silu * up * route_weight);
        }
    }
}

template [[host_name("glm53_moe_q2_pair_grouped_ds4_f32_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_pair_grouped_ds4<float>) glm53_moe_q2_pair_grouped_ds4<float>;
template [[host_name("glm53_moe_q2_pair_grouped_ds4_bf16_nr1_32_nr0_64_nk32")]] [[kernel]]
    decltype(glm53_moe_q2_pair_grouped_ds4<bfloat16_t>)
        glm53_moe_q2_pair_grouped_ds4<bfloat16_t>;
