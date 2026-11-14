// Register-resident GEMV helpers: `qmv_reg`, `qmv_fast_reg`, `qmv_reg_rows`, silu-on-load.
        // qmv_fast_impl with `out_row` given and the row results returned
        // (all lanes hold them after simd_sum). x points at the vector.
        // OPT-MIX2ROW: only the number of independent contiguous rows varies.
        // The default preserves every existing four-row caller's arithmetic.
        template <typename T, int group_size, int bits, int results_per_simdgroup = 4>
        METAL_FUNC void qmv_fast_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[results_per_simdgroup]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          for (int k = 0; k < in_vec_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
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
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // qmv_impl's normal branch (out_vec_size >= 8, full tile) likewise.
        // OPT-FULLTAIL: EXACT_TAIL says the caller's `in_vec_size` is a
        // multiple of values_per_thread. See the tail block below for what that
        // buys and why it stays bit-identical.
        // OPT-DOWNRPS: NR contiguous rows per simdgroup; each row's walk,
        // accumulation order and simd_sum are unchanged for any NR.
        template <typename T, int group_size, int bits, bool EXACT_TAIL = false, int NR = 4>
        METAL_FUNC void qmv_reg(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[NR]) {
          constexpr int results_per_simdgroup = NR;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          const int used_out_row = out_row;
          ws += used_out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += used_out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
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
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          // OPT-FULLTAIL. When in_vec_size is a multiple of
          // values_per_thread, `remaining` is provably 0 or values_per_thread and
          // never a partial slice: in_vec_size - k is a multiple of
          // values_per_thread (k advances by block_size = 32 * values_per_thread)
          // and so is simd_lid * values_per_thread, so their difference is too,
          // and the clamp leaves only the two endpoints. The _safe helpers then
          // run the SAME arithmetic in the SAME order as the plain ones -- their
          // bodies are identical with `N` in place of `values_per_thread` -- but
          // over a RUNTIME trip count. That keeps x_thread dynamically indexed,
          // which pins the array in thread-local scratch for the whole function
          // instead of registers, and costs the main loop as well as the tail.
          // K = 640 (down) is 2.5 blocks, so a fifth of that GEMV's work sits in
          // this branch; K = 2560 puts a tenth there. Both are exact multiples of
          // 8, so EXACT_TAIL erases the runtime-indexed code path entirely.
          // Bit-identical by construction, not by tolerance.
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
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
            for (int row = 0; row < results_per_simdgroup; row++) {
              auto wl = (const device uint8_t*)(ws + row * in_vec_size_w);
              const device T* sl = scales + row * in_vec_size_g;
              const device T* bl = biases + row * in_vec_size_g;
              U s = sl[0];
              U b = bl[0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wl, x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // load_vector / load_vector_safe (4-bit) with `silu` applied to each
        // activation on load: the same bf16 silu the separate launch stored.
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_silu(const device T* x, thread U* x_thread) {
          static_assert(bits == 4, "silu-on-load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < values_per_thread; i += 4) {
            const T a = mlx_silu(x[i]);
            const T b = mlx_silu(x[i + 1]);
            const T c = mlx_silu(x[i + 2]);
            const T d = mlx_silu(x[i + 3]);
            sum += a + b + c + d;
            x_thread[i] = a;
            x_thread[i + 1] = b / 16.0f;
            x_thread[i + 2] = c / 256.0f;
            x_thread[i + 3] = d / 4096.0f;
          }
          return sum;
        }
        template <typename T, typename U, int values_per_thread, int bits>
        inline U load_vector_safe_silu(const device T* x, thread U* x_thread, int N) {
          static_assert(bits == 4, "silu-on-load: 4-bit only");
          U sum = 0;
          for (int i = 0; i < N; i += 4) {
            const T a = mlx_silu(x[i]);
            const T b = mlx_silu(x[i + 1]);
            const T c = mlx_silu(x[i + 2]);
            const T d = mlx_silu(x[i + 3]);
            sum += a + b + c + d;
            x_thread[i] = a;
            x_thread[i + 1] = b / 16.0f;
            x_thread[i + 2] = c / 256.0f;
            x_thread[i + 3] = d / 4096.0f;
          }
          for (int i = N; i < values_per_thread; i++) {
            x_thread[i] = 0;
          }
          return sum;
        }

        // qmv_impl's normal branch over FOUR GIVEN rows (each row's walk is
        // independent of its neighbours), optional silu on the activations.
        // EXACT_TAIL as in qmv_reg above: the caller's in_vec_size is a multiple
        // of values_per_thread, so the runtime-indexed tail can be compiled away.
        template <typename T, int group_size, int bits, bool SILU, bool EXACT_TAIL = false>
        METAL_FUNC void qmv_reg_rows(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const thread int (&rows)[4],
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int results_per_simdgroup = 4;
          constexpr int packs_per_thread = 1;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          thread U x_thread[values_per_thread];
          for (int row = 0; row < results_per_simdgroup; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          const device uint8_t* wr[4];
          const device T* sr[4];
          const device T* br[4];
          for (int row = 0; row < 4; row++) {
            wr[row] = ws + rows[row] * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
            sr[row] = scales + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
            br[row] = biases + rows[row] * in_vec_size_g + simd_lid / scale_step_per_thread;
          }
          x += simd_lid * values_per_thread;
          int k = 0;
          for (; k < in_vec_size - block_size; k += block_size) {
            U sum = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x, x_thread)
                         : load_vector<T, U, values_per_thread, bits>(x, x_thread);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              result[row] += qdot<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum);
            }
            for (int row = 0; row < 4; row++) {
              wr[row] += block_size * bytes_per_pack / pack_factor;
              sr[row] += block_size / group_size;
              br[row] += block_size / group_size;
            }
            x += block_size;
          }
          const int remaining = clamp(
              static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
          // OPT-FULLTAIL, same argument as qmv_reg.
          if constexpr (EXACT_TAIL) {
            if (remaining > 0) {
              U sum = SILU ? load_vector_silu<T, U, values_per_thread, bits>(x, x_thread)
                           : load_vector<T, U, values_per_thread, bits>(x, x_thread);
              for (int row = 0; row < results_per_simdgroup; row++) {
                U s = sr[row][0];
                U b = br[row][0];
                result[row] += qdot<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum);
              }
            }
          } else if (remaining > 0) {
            U sum = SILU ? load_vector_safe_silu<T, U, values_per_thread, bits>(x, x_thread, remaining)
                         : load_vector_safe<T, U, values_per_thread, bits>(x, x_thread, remaining);
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = sr[row][0];
              U b = br[row][0];
              result[row] += qdot_safe<U, values_per_thread, bits>(wr[row], x_thread, s, b, sum, remaining);
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }
        