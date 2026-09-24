// Pipelined GDN/attention helper library shared by the engine kernels.
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg_pf(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int packs_per_thread = bits == 2 ? 1 : 2;
          constexpr int results_per_simdgroup = 4;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(bits == 4 && packs_per_thread == 2, "pipelined path: 4-bit");
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
          // Prefetched raw operands of the NEXT block: the two 32-bit packs, the
          // scale and the bias of each of the 4 rows, and the 16 activations.
          uint32_t pk[4][2];
          T sc[4], bi[4];
          T xr[values_per_thread];
          auto fetch = [&](const device uint8_t* wsb, const device T* scb, const device T* bib, const device T* xb) {
            for (int row = 0; row < 4; row++) {
              const device uint32_t* wl = (const device uint32_t*)(wsb + row * in_vec_size_w);
              pk[row][0] = wl[0]; pk[row][1] = wl[1];
              sc[row] = scb[row * in_vec_size_g];
              bi[row] = bib[row * in_vec_size_g];
            }
            for (int i = 0; i < values_per_thread; i++) { xr[i] = xb[i]; }
          };
          fetch(ws, scales, biases, x);
          for (int k = 0; k < in_vec_size; k += block_size) {
            // hold this block's operands, then issue the next block's loads
            uint32_t cpk[4][2]; T csc[4], cbi[4]; T cx[values_per_thread];
            for (int row = 0; row < 4; row++) { cpk[row][0] = pk[row][0]; cpk[row][1] = pk[row][1]; csc[row] = sc[row]; cbi[row] = bi[row]; }
            for (int i = 0; i < values_per_thread; i++) { cx[i] = xr[i]; }
            ws += block_size * bytes_per_pack / pack_factor;
            scales += block_size / group_size;
            biases += block_size / group_size;
            x += block_size;
            if (k + block_size < in_vec_size) { fetch(ws, scales, biases, x); }
            // load_vector on the held activations (same expression as load_vector)
            U sum = 0;
            for (int i = 0; i < values_per_thread; i += 4) {
              sum += cx[i] + cx[i + 1] + cx[i + 2] + cx[i + 3];
              x_thread[i] = cx[i];
              x_thread[i + 1] = cx[i + 1] / 16.0f;
              x_thread[i + 2] = cx[i + 2] / 256.0f;
              x_thread[i + 3] = cx[i + 3] / 4096.0f;
            }
            for (int row = 0; row < results_per_simdgroup; row++) {
              U s = csc[row];
              U b = cbi[row];
              // qdot over the held packs: same expression as qdot<U, 16, 4>
              U accum = 0;
              const thread uint16_t* wsh = (const thread uint16_t*)&cpk[row][0];
              for (int i = 0; i < (values_per_thread / 4); i++) {
                accum +=
                    (x_thread[4 * i] * (wsh[i] & 0x000f) +
                     x_thread[4 * i + 1] * (wsh[i] & 0x00f0) +
                     x_thread[4 * i + 2] * (wsh[i] & 0x0f00) +
                     x_thread[4 * i + 3] * (wsh[i] & 0xf000));
              }
              result[row] += s * accum + sum * b;
            }
          }
          for (int row = 0; row < results_per_simdgroup; row++) {
            result[row] = simd_sum(result[row]);
          }
        }

        // Ping-pong variant: two register sets, the K loop unrolled by two, no
        // per-block copies. Same arithmetic as qmv_fast (see qmv_fast_reg_pf).
        template <typename T, int group_size, int bits>
        METAL_FUNC void qmv_fast_reg_pf2(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_row,
            uint simd_lid,
            thread float (&result)[4]) {
          constexpr int packs_per_thread = 2;
          constexpr int pack_factor = get_pack_factor<bits, 32>();
          constexpr int bytes_per_pack = get_bytes_per_pack<bits, 32>();
          constexpr int values_per_thread = pack_factor * packs_per_thread;
          constexpr int block_size = values_per_thread * SIMD_SIZE;
          constexpr int scale_step_per_thread = group_size / values_per_thread;
          static_assert(bits == 4, "4-bit");
          const device uint8_t* ws = (const device uint8_t*)w;
          typedef float U;
          for (int row = 0; row < 4; row++) { result[row] = 0; }
          const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
          const int in_vec_size_g = in_vec_size / group_size;
          ws += out_row * in_vec_size_w + simd_lid * packs_per_thread * bytes_per_pack;
          scales += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          biases += out_row * in_vec_size_g + simd_lid / scale_step_per_thread;
          x += simd_lid * values_per_thread;
          constexpr int WSTEP = block_size * bytes_per_pack / pack_factor;
          constexpr int GSTEP = block_size / group_size;
          uint32_t pa[4][2], pb[4][2];
          T sa[4], ba[4], sb[4], bb[4];
          T xa[values_per_thread], xb[values_per_thread];
          #define TRACK_FETCH(PK, SC, BI, XR, OFFB) \
            for (int row = 0; row < 4; row++) { \
              const device uint32_t* wl = (const device uint32_t*)(ws + (OFFB) * WSTEP + row * in_vec_size_w); \
              PK[row][0] = wl[0]; PK[row][1] = wl[1]; \
              SC[row] = scales[(OFFB) * GSTEP + row * in_vec_size_g]; \
              BI[row] = biases[(OFFB) * GSTEP + row * in_vec_size_g]; \
            } \
            for (int i = 0; i < values_per_thread; i++) { XR[i] = x[(OFFB) * block_size + i]; }
          #define TRACK_COMPUTE(PK, SC, BI, XR) { \
            U x_thread[values_per_thread]; \
            U sum = 0; \
            for (int i = 0; i < values_per_thread; i += 4) { \
              sum += XR[i] + XR[i + 1] + XR[i + 2] + XR[i + 3]; \
              x_thread[i] = XR[i]; \
              x_thread[i + 1] = XR[i + 1] / 16.0f; \
              x_thread[i + 2] = XR[i + 2] / 256.0f; \
              x_thread[i + 3] = XR[i + 3] / 4096.0f; \
            } \
            for (int row = 0; row < 4; row++) { \
              U s = SC[row]; U b = BI[row]; U accum = 0; \
              const thread uint16_t* wsh = (const thread uint16_t*)&PK[row][0]; \
              for (int i = 0; i < (values_per_thread / 4); i++) { \
                accum += (x_thread[4 * i] * (wsh[i] & 0x000f) + x_thread[4 * i + 1] * (wsh[i] & 0x00f0) + \
                          x_thread[4 * i + 2] * (wsh[i] & 0x0f00) + x_thread[4 * i + 3] * (wsh[i] & 0xf000)); \
              } \
              result[row] += s * accum + sum * b; \
            } }
          const int nblocks = in_vec_size / block_size;
          TRACK_FETCH(pa, sa, ba, xa, 0)
          int blk = 0;
          for (; blk + 1 < nblocks; blk += 2) {
            TRACK_FETCH(pb, sb, bb, xb, blk + 1)
            TRACK_COMPUTE(pa, sa, ba, xa)
            if (blk + 2 < nblocks) { TRACK_FETCH(pa, sa, ba, xa, blk + 2) }
            TRACK_COMPUTE(pb, sb, bb, xb)
          }
          if (blk < nblocks) { TRACK_COMPUTE(pa, sa, ba, xa) }
          #undef TRACK_FETCH
          #undef TRACK_COMPUTE
          for (int row = 0; row < 4; row++) { result[row] = simd_sum(result[row]); }
        }
        