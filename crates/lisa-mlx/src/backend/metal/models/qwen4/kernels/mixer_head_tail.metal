// Mixer helper library: `track_inject_qmv_row`, head/tail GEMV variants.
        // `qmv_impl`'s `out_vec_size < num_simdgroups * results_per_simdgroup`
        // branch with compile-time sizes and the K walk unrolled. Same lanes,
        // same per-lane accumulation order, same simd_sum.
        template <typename T, int group_size, int bits, int in_vec_size, int out_vec_size, int UNR>
        METAL_FUNC void track_inject_qmv(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            device T* y,
            uint simd_gid,
            uint simd_lid) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(out_vec_size < num_simdgroups * results_per_simdgroup, "small-N branch only");
          static_assert(in_vec_size > block_size, "K walk");

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
          const int out_row = simd_gid * results_per_simdgroup;
          if (out_row >= out_vec_size) {
            return;
          }
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          y += out_row;

          // for (k = 0; k < in_vec_size - block_size; k += block_size)
          constexpr int NFULL = (in_vec_size - 1) / block_size;
          // simd_gid 0 only reaches here, so out_row == 0 and the row count is
          // compile-time: same rows, same order, but the compiler can batch
          // the loads (the runtime-bounded loop in `qmv_impl` serializes them).
          constexpr int NR = out_vec_size < results_per_simdgroup ? out_vec_size : results_per_simdgroup;
          #pragma clang loop unroll_count(UNR)
          for (int i = 0; i < NFULL; i++) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          constexpr int k_end = NFULL * block_size;
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k_end - simd_lid * values_per_thread), 0, values_per_thread);
          // OPT-FULLTAIL. in_vec_size is a template constant here, so when it
          // is a multiple of values_per_thread the clamp above can only yield 0 or
          // values_per_thread -- never a partial slice. The _safe helpers then run
          // the SAME arithmetic in the SAME order (their bodies are the plain ones
          // with `N` for `values_per_thread`), but over a RUNTIME trip count, which
          // keeps x_thread dynamically indexed and so pins it in thread-local
          // scratch for the whole function rather than registers. Compile that
          // branch away. Bit-identical by construction.
          if constexpr (in_vec_size % values_per_thread == 0) {
            if (remaining > 0) {
              U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < NR; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device T* sl = scales + row * in_vec_size_g;
                const device T* bl = biases + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
            U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < NR; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[row] = static_cast<T>(result[row]);
            }
          }
        }

        // OPT-INJSPLIT: one original small-N row per simdgroup. ROW changes
        // only the base addresses. Lane-to-K mapping, ascending block updates,
        // qdot/qdot_safe, and simd_sum are verbatim from track_inject_qmv above.
        // UNR is a scheduling hint only; never split or reassociate the sum.
        template <typename T, int group_size, int bits, int in_vec_size, int ROW, int UNR = 8>
        METAL_FUNC void track_inject_qmv_row(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            device T* y,
            uint simd_lid) {
          constexpr int results_per_simdgroup = 1;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(ROW >= 0 && ROW < 4, "inject row");
          static_assert(in_vec_size > block_size, "K walk");

          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          thread U result[results_per_simdgroup] = {0};

          constexpr int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          constexpr int in_vec_size_g = in_vec_size / group_size;
          constexpr int out_row = ROW;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          y += out_row;

          // for (k = 0; k < in_vec_size - block_size; k += block_size)
          constexpr int NFULL = (in_vec_size - 1) / block_size;
          constexpr int NR = 1;
          #pragma clang loop unroll_count(UNR)
          for (int i = 0; i < NFULL; i++) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
            }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
          }
          constexpr int k_end = NFULL * block_size;
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k_end - simd_lid * values_per_thread), 0, values_per_thread);
          // OPT-FULLTAIL. in_vec_size is a template constant here, so when it
          // is a multiple of values_per_thread the clamp above can only yield 0 or
          // values_per_thread -- never a partial slice. The _safe helpers then run
          // the SAME arithmetic in the SAME order (their bodies are the plain ones
          // with `N` for `values_per_thread`), but over a RUNTIME trip count, which
          // keeps x_thread dynamically indexed and so pins it in thread-local
          // scratch for the whole function rather than registers. Compile that
          // branch away. Bit-identical by construction.
          if constexpr (in_vec_size % values_per_thread == 0) {
            if (remaining > 0) {
              U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < NR; row++) {
                auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
                const device T* sl = scales + row * in_vec_size_g;
                const device T* bl = biases + row * in_vec_size_g;
                U s = sl[0];
                U b = bl[0];
                result[row] += qdot<U, values_per_thread, bits>(wl, x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
            U sum = load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < NR; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < NR; row++) {
            result[row] = simd_sum(result[row]);
            if (simd_lid == 0) {
              y[row] = static_cast<T>(result[row]);
            }
          }
        }
        