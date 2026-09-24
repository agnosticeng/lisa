// Helpers for the one-token fused expert gate|up reuse kernel.
        template <typename T, int group_size, int bits, int rows>
        METAL_FUNC void qmv_fast_reg_dual(
            const device uint32_t* w0,
            const device T* scales0,
            const device T* biases0,
            const device uint32_t* w1,
            const device T* scales1,
            const device T* biases1,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result0)[rows],
            thread float (&result1)[rows]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws0 = (const device uint8_t*)w0;
          const device uint8_t* ws1 = (const device uint8_t*)w1;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < rows; row++) {
            result0[row] = 0;
            result1[row] = 0;
          }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws0 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          ws1 += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          scales1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases0 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases1 += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < rows; row++) {
              auto wl0 = (const device uint8_t*)(ws0 + row * in_vec_size_w);
              const device T* sl0 = scales0 + row * in_vec_size_g;
              const device T* bl0 = biases0 + row * in_vec_size_g;
              U s0 = sl0[0];
              U b0 = bl0[0];
              result0[row] += qdot<U, values_per_thread, bits>(wl0, x_thread, s0, b0, sum);
              auto wl1 = (const device uint8_t*)(ws1 + row * in_vec_size_w);
              const device T* sl1 = scales1 + row * in_vec_size_g;
              const device T* bl1 = biases1 + row * in_vec_size_g;
              U s1 = sl1[0];
              U b1 = bl1[0];
              result1[row] += qdot<U, values_per_thread, bits>(wl1, x_thread, s1, b1, sum);
            }
            ws0 += block_size * bytes_per_pack / pack_factor;
            ws1 += block_size * bytes_per_pack / pack_factor;
            scales0 += block_size / group_size;
            scales1 += block_size / group_size;
            biases0 += block_size / group_size;
            biases1 += block_size / group_size;
            x += block_size;
          }
          for (int row = 0; row < rows; row++) {
            result0[row] = simd_sum(result0[row]);
            result1[row] = simd_sum(result1[row]);
          }
        }
        