// Wide (2..8-row) GEMV helpers: `qmv_wide_reg_full` and friends.
        // MLX `qmv_wide_impl` (the M >= 2 quantized GEMV on this GPU generation),
        // verbatim for a FULL 8-row tile (k_folds == 1): same lane -> group
        // assignment (k_lane, stride k_lanes), same per-group decode and
        // element-order accumulation, same shuffle ladder. The row is given by
        // the caller; the vecs_per_tg vectors are x's first rows. The totals
        // are left in `result` on the k_lane == 0 lanes.
        template <typename T> METAL_FUNC vec<T, 4> track_silu4(vec<T, 4> x) {
            return vec<T, 4>(mlx_silu(x.x), mlx_silu(x.y), mlx_silu(x.z), mlx_silu(x.w));
        }
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes, bool SILU>
        METAL_FUNC void qmv_wide_reg_full(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int M,
            const int row,
            uint simd_lid,
            thread float (&result)[vecs_per_tg]) {
          constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)
          typedef float U;
          const short k_lane = simd_lid % k_lanes;
          constexpr int k_folds = 1;
          constexpr int fold = 0;
          constexpr int vec0 = 0;
        
          const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
          const int in_vec_size_g = in_vec_size / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
          }
        
          for (int v = 0; v < vecs_per_tg; v++) { result[v] = 0; }
        
          // Each lane reduces a strided subset of the row's groups: decode the group in
          // 8-value sub-chunks and reuse each chunk across the streamed vectors.
          const int g_stride = k_lanes * k_folds;
          const int g_first = fold < k_folds ? k_lane + fold * k_lanes : in_vec_size_g;
          // Group loop. The 4-bit path is unrolled by g_unroll: the scale, the bias
          // and the packed weights of every group a trip covers are loaded before any
          // of them is decoded, so g_unroll independent memory requests per thread are
          // in flight at once. At the decode output widths only 40-80 threadgroups are
          // resident on the whole GPU, so there is no other thread to hide a load
          // behind and the stock loop paid one memory round trip per group. Every
          // group is decoded by the same expression as before, every vector sums its
          // terms in ascending element order, and the group partials still reach
          // result[v] in ascending group order, so the arithmetic is bit identical.
          if constexpr (bits == 4) {
            constexpr int g_unroll = 3;
            constexpr int packs_per_group = group_size / sub;
            int g = g_first;
            for (; g + (g_unroll - 1) * g_stride < in_vec_size_g;
                 g += g_unroll * g_stride) {
              float su[g_unroll];
              float bu[g_unroll];
              uint32_t wpack[g_unroll][packs_per_group];
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                su[u] = static_cast<float>(srow[gu]);
                bu[u] = static_cast<float>(brow[gu]);
                const device uint32_t* wg =
                    (const device uint32_t*)(wrow + gu * (group_size * bits / 8));
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  wpack[u][sc] = wg[sc];
                }
              }
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                const float s = su[u];
                const float b = bu[u];
                const float s_hi = s / 16.0f;
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  const int k0 = gu * group_size + sc * sub;
                  const uint32_t p = wpack[u][sc];
                  U w_dq[sub];
        #pragma unroll
                  for (int i = 0; i < sub / 2; i++) {
                    const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                    w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                    w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                  }
                  // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                  // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                  // of eight, issued before any product. The element-major order below is
                  // unchanged, so every vector still sums its terms in ascending element
                  // order -- bit identical.
                  vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                    for (int c = 0; c < sub / 4; c++) {
                      xq[v][c] = SILU ? track_silu4<T>(xc4[c]) : xc4[c];
                    }
                  }
                  U accv[vecs_per_tg] = {0};
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                    }
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    result[v] += accv[v];
                  }
                }
              }
            }
            // Groups left over when the row's group count is not a multiple of
            // g_unroll, in the same ascending order.
            for (; g < in_vec_size_g; g += g_stride) {
              const float s = static_cast<float>(srow[g]);
              const float b = static_cast<float>(brow[g]);
              const float s_hi = s / 16.0f;
              const device uint32_t* wg =
                  (const device uint32_t*)(wrow + g * (group_size * bits / 8));
              uint32_t wpack[group_size / sub];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                wpack[sc] = wg[sc];
              }
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const uint32_t p = wpack[sc];
                U w_dq[sub];
        #pragma unroll
                for (int i = 0; i < sub / 2; i++) {
                  const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                  w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                  w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                }
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = SILU ? track_silu4<T>(xc4[c]) : xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          } else {
            for (int g = g_first; g < in_vec_size_g; g += g_stride) {
              U scale = srow[g];
              U bias = brow[g];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const device uint8_t* wc = wrow + k0 * bits / 8;
                U w_dq[sub];
                dequantize<U, sub, bits>(wc, scale, bias, w_dq);
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = SILU ? track_silu4<T>(xc4[c]) : xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          }
          // Reduce each vector's partial over its k_lanes with a shuffle ladder:
          // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
          for (int v = 0; v < vecs_per_tg; v++) {
            if constexpr (k_lanes >= 32) {
              result[v] += simd_shuffle_down(result[v], 16);
            }
            if constexpr (k_lanes >= 16) {
              result[v] += simd_shuffle_down(result[v], 8);
            }
            if constexpr (k_lanes >= 8) {
              result[v] += simd_shuffle_down(result[v], 4);
            }
            if constexpr (k_lanes >= 4) {
              result[v] += simd_shuffle_down(result[v], 2);
            }
            if constexpr (k_lanes >= 2) {
              result[v] += simd_shuffle_down(result[v], 1);
            }
          }
        
        }

        // `qmv_wide_impl` for the ONE tile of a matrix with out_vec_size < 8
        // rows (tile_row0 == 0): the short tile splits K across k_folds slots
        // per row and sums the folds in ascending order, verbatim. Needs the
        // full threadgroup (2 simdgroups) and 8 * vecs_per_tg floats of
        // threadgroup memory. On return, `valid` lanes (k_lane == 0, slot <
        // out_vec_size) hold row `slot`'s totals in `result`.
        template <typename T, int group_size, int bits, int vecs_per_tg, int k_lanes>
        METAL_FUNC void qmv_wide_reg_partial(
            const device uint32_t* w,
            const device T* scales,
            const device T* biases,
            const device T* x,
            const int in_vec_size,
            const int out_vec_size,
            const int M,
            threadgroup float* fold_partials,
            uint simd_gid,
            uint simd_lid,
            thread float (&result)[vecs_per_tg],
            thread bool& valid,
            thread int& row_out) {
          constexpr int num_simdgroups = 2;
          constexpr int results_per_simdgroup = SIMD_SIZE / k_lanes;
          constexpr int sub = 8; // values per sub-chunk (== bits bytes, byte-aligned)
          typedef float U;
          constexpr int rows_per_tg = results_per_simdgroup * num_simdgroups;
          const short k_lane = simd_lid % k_lanes;
          const short sg_row = simd_lid / k_lanes;
          const short slot = simd_gid * results_per_simdgroup + sg_row;
          const int tile_row0 = 0;
          const int tile_rows = min(out_vec_size - tile_row0, rows_per_tg);
          const int vec0 = 0;
          int k_folds = 1;
          int fold = 0;
          int out_row = tile_row0 + slot;
          if (tile_rows < rows_per_tg) {
            while (k_folds * 2 * tile_rows <= rows_per_tg) {
              k_folds *= 2;
            }
            fold = slot / tile_rows;
            out_row = tile_row0 + slot % tile_rows;
          }
          const int row = min(out_row, out_vec_size - 1);
        
          const int in_vec_size_w = in_vec_size * bits / 8; // bytes per weight row
          const int in_vec_size_g = in_vec_size / group_size;
          const device uint8_t* wrow = (const device uint8_t*)w + row * in_vec_size_w;
          const device T* srow = scales + row * in_vec_size_g;
          const device T* brow = biases + row * in_vec_size_g;
        
          const device T* xv[vecs_per_tg];
          for (int v = 0; v < vecs_per_tg; v++) {
            xv[v] = x + min(vec0 + v, M - 1) * in_vec_size;
          }
        
          for (int v = 0; v < vecs_per_tg; v++) { result[v] = 0; }
        
          // Each lane reduces a strided subset of the row's groups: decode the group in
          // 8-value sub-chunks and reuse each chunk across the streamed vectors.
          const int g_stride = k_lanes * k_folds;
          const int g_first = fold < k_folds ? k_lane + fold * k_lanes : in_vec_size_g;
          // Group loop. The 4-bit path is unrolled by g_unroll: the scale, the bias
          // and the packed weights of every group a trip covers are loaded before any
          // of them is decoded, so g_unroll independent memory requests per thread are
          // in flight at once. At the decode output widths only 40-80 threadgroups are
          // resident on the whole GPU, so there is no other thread to hide a load
          // behind and the stock loop paid one memory round trip per group. Every
          // group is decoded by the same expression as before, every vector sums its
          // terms in ascending element order, and the group partials still reach
          // result[v] in ascending group order, so the arithmetic is bit identical.
          if constexpr (bits == 4) {
            constexpr int g_unroll = 3;
            constexpr int packs_per_group = group_size / sub;
            int g = g_first;
            for (; g + (g_unroll - 1) * g_stride < in_vec_size_g;
                 g += g_unroll * g_stride) {
              float su[g_unroll];
              float bu[g_unroll];
              uint32_t wpack[g_unroll][packs_per_group];
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                su[u] = static_cast<float>(srow[gu]);
                bu[u] = static_cast<float>(brow[gu]);
                const device uint32_t* wg =
                    (const device uint32_t*)(wrow + gu * (group_size * bits / 8));
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  wpack[u][sc] = wg[sc];
                }
              }
        #pragma unroll
              for (int u = 0; u < g_unroll; u++) {
                const int gu = g + u * g_stride;
                const float s = su[u];
                const float b = bu[u];
                const float s_hi = s / 16.0f;
        #pragma unroll
                for (int sc = 0; sc < packs_per_group; sc++) {
                  const int k0 = gu * group_size + sc * sub;
                  const uint32_t p = wpack[u][sc];
                  U w_dq[sub];
        #pragma unroll
                  for (int i = 0; i < sub / 2; i++) {
                    const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                    w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                    w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                  }
                  // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                  // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                  // of eight, issued before any product. The element-major order below is
                  // unchanged, so every vector still sums its terms in ascending element
                  // order -- bit identical.
                  vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                    for (int c = 0; c < sub / 4; c++) {
                      xq[v][c] = xc4[c];
                    }
                  }
                  U accv[vecs_per_tg] = {0};
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                    }
        #pragma unroll
                    for (int v = 0; v < vecs_per_tg; v++) {
                      accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                    }
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    result[v] += accv[v];
                  }
                }
              }
            }
            // Groups left over when the row's group count is not a multiple of
            // g_unroll, in the same ascending order.
            for (; g < in_vec_size_g; g += g_stride) {
              const float s = static_cast<float>(srow[g]);
              const float b = static_cast<float>(brow[g]);
              const float s_hi = s / 16.0f;
              const device uint32_t* wg =
                  (const device uint32_t*)(wrow + g * (group_size * bits / 8));
              uint32_t wpack[group_size / sub];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                wpack[sc] = wg[sc];
              }
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const uint32_t p = wpack[sc];
                U w_dq[sub];
        #pragma unroll
                for (int i = 0; i < sub / 2; i++) {
                  const uint32_t wbyte = (p >> (8 * i)) & 0xffu;
                  w_dq[2 * i] = static_cast<U>(s * (wbyte & 0x0fu) + b);
                  w_dq[2 * i + 1] = static_cast<U>(s_hi * (wbyte & 0xf0u) + b);
                }
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          } else {
            for (int g = g_first; g < in_vec_size_g; g += g_stride) {
              U scale = srow[g];
              U bias = brow[g];
        #pragma unroll
              for (int sc = 0; sc < group_size / sub; sc++) {
                const int k0 = g * group_size + sc * sub;
                const device uint8_t* wc = wrow + k0 * bits / 8;
                U w_dq[sub];
                dequantize<U, sub, bits>(wc, scale, bias, w_dq);
                // The sub-chunk is `sub` contiguous activations and `sub` is a multiple
                // of 4, so read them as vec<T, 4>: two loads per streamed vector instead
                // of eight, issued before any product. The element-major order below is
                // unchanged, so every vector still sums its terms in ascending element
                // order -- bit identical.
                vec<T, 4> xq[vecs_per_tg][sub / 4];
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  const device vec<T, 4>* xc4 = (const device vec<T, 4>*)(xv[v] + k0);
        #pragma unroll
                  for (int c = 0; c < sub / 4; c++) {
                    xq[v][c] = xc4[c];
                  }
                }
                U accv[vecs_per_tg] = {0};
        #pragma unroll
                for (int c = 0; c < sub / 4; c++) {
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].x) * w_dq[4 * c + 0];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].y) * w_dq[4 * c + 1];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].z) * w_dq[4 * c + 2];
                  }
        #pragma unroll
                  for (int v = 0; v < vecs_per_tg; v++) {
                    accv[v] += static_cast<U>(xq[v][c].w) * w_dq[4 * c + 3];
                  }
                }
        #pragma unroll
                for (int v = 0; v < vecs_per_tg; v++) {
                  result[v] += accv[v];
                }
              }
            }
          }
          // Reduce each vector's partial over its k_lanes with a shuffle ladder:
          // simd_sum would mix the results_per_simdgroup rows a simdgroup spans.
          for (int v = 0; v < vecs_per_tg; v++) {
            if constexpr (k_lanes >= 32) {
              result[v] += simd_shuffle_down(result[v], 16);
            }
            if constexpr (k_lanes >= 16) {
              result[v] += simd_shuffle_down(result[v], 8);
            }
            if constexpr (k_lanes >= 8) {
              result[v] += simd_shuffle_down(result[v], 4);
            }
            if constexpr (k_lanes >= 4) {
              result[v] += simd_shuffle_down(result[v], 2);
            }
            if constexpr (k_lanes >= 2) {
              result[v] += simd_shuffle_down(result[v], 1);
            }
          }
        
          valid = false;
          row_out = out_row;
          if (k_folds > 1) {
            if (k_lane == 0) {
              for (int v = 0; v < vecs_per_tg; v++) {
                fold_partials[slot * vecs_per_tg + v] = result[v];
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (k_lane == 0 && slot < tile_rows) {
              for (int v = 0; v < vecs_per_tg; v++) {
                U total = fold_partials[slot * vecs_per_tg + v];
                for (int f = 1; f < k_folds; f++) {
                  total += fold_partials[(slot + f * tile_rows) * vecs_per_tg + v];
                }
                result[v] = total;
              }
              valid = true;
            }
            return;
          }
          if (k_lane == 0 && fold == 0 && out_row < out_vec_size) { valid = true; }
        }
        