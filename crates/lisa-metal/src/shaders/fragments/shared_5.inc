    }
  }

  D += params->batch_stride_d * tid.z;

  // Prepare threadgroup memory
  threadgroup T As[gemm_kernel::tgp_mem_size_a];
  threadgroup T Bs[gemm_kernel::tgp_mem_size_b];

  threadgroup_barrier(mem_flags::mem_none);

  // Find block in A, B, C
  const int c_row = tid_y * BM;
  const int c_col = tid_x * BN;
  const size_t c_row_long = size_t(c_row);
  const size_t c_col_long = size_t(c_col);

  A += transpose_a ? c_row_long : c_row_long * params->lda;
  B += transpose_b ? c_col_long * params->ldb : c_col_long;
  D += c_row_long * params->ldd + c_col_long;

  if (use_out_source) {
    C += c_row_long * addmm_params->ldc + c_col_long * addmm_params->fdc;
  }

  // Prepare threadgroup mma operation
  thread mma_t mma_op(simd_group_id, simd_lane_id);

  // Prepare threadgroup loading operations
  thread loader_a_t loader_a(A, params->lda, As, simd_group_id, simd_lane_id);
  thread loader_b_t loader_b(B, params->ldb, Bs, simd_group_id, simd_lane_id);

  // Prepare threadgroup bounds
  const short tgp_bm = align_M ? BM : short(min(BM, params->M - c_row));
  const short tgp_bn = align_N ? BN : short(min(BN, params->N - c_col));

  // Prepare iterations
  int gemm_k_iterations = params->gemm_k_iterations_aligned;

  // Do unaligned K iterations first
  if (!align_K) {
    const int k_last = params->gemm_k_iterations_aligned * BK;
    const int k_remain = params->K - k_last;
    const size_t k_jump_a =
        transpose_a ? params->lda * size_t(k_last) : size_t(k_last);
    const size_t k_jump_b =
        transpose_b ? size_t(k_last) : params->ldb * size_t(k_last);

    // Move loader source ahead to end
    loader_a.src += k_jump_a;
    loader_b.src += k_jump_b;

    // Load tile
    const short2 tile_dims_A =
        transpose_a ? short2(tgp_bm, k_remain) : short2(k_remain, tgp_bm);
    const short2 tile_dims_B =
        transpose_b ? short2(k_remain, tgp_bn) : short2(tgp_bn, k_remain);

    loader_a.load_safe(tile_dims_A);
    loader_b.load_safe(tile_dims_B);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Do matmul
    mma_op.mma(As, Bs);

    // Reset source back to start
    loader_a.src -= k_jump_a;
    loader_b.src -= k_jump_b;
  }

  const TransformAdd<AccumType, AccumType> epilogue_op_add(
      addmm_params->alpha, addmm_params->beta);
  const TransformAxpby<AccumType, AccumType> epilogue_op_axpby(
      addmm_params->alpha, addmm_params->beta);

  ///////////////////////////////////////////////////////////////////////////////
  // MNK aligned loop
  if (align_M && align_N) {
    // Do gemm
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      // Load elements into threadgroup
      loader_a.load_unsafe();
      loader_b.load_unsafe();

      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Multiply and accumulate threadgroup elements
      mma_op.mma(As, Bs);

      // Prepare for next iteration
      loader_a.next();
      loader_b.next();
    }

    threadgroup_barrier(mem_flags::mem_none);

    // Do epilogue
    if (use_out_source) {
      if (do_axpby) {
        mma_op.apply_epilogue(
            C, addmm_params->ldc, addmm_params->fdc, epilogue_op_axpby);
      } else {
        mma_op.apply_epilogue(
            C, addmm_params->ldc, addmm_params->fdc, epilogue_op_add);
      }
    }

    // Store results to device memory
    return mma_op.store_result(D, params->ldd);

  }
  ///////////////////////////////////////////////////////////////////////////////
  // MN unaligned loop
  else { // Loop over K - unaligned case
    const int leftover_bk = 0;

    if ((align_M || tgp_bm == BM) && (align_N || tgp_bn == BN)) {
      // Do gemm
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, true, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue(
              C, addmm_params->ldc, addmm_params->fdc, epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue(
              C, addmm_params->ldc, addmm_params->fdc, epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result(D, params->ldd);

    } else if (align_N || tgp_bn == BN) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, true, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));

    } else if (align_M || tgp_bm == BM) {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<true, false, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));

    } else {
      gemm_kernel::gemm_loop(
          As,
          Bs,
          gemm_k_iterations,
          loader_a,
          loader_b,
          mma_op,
          tgp_bm,
          tgp_bn,
          leftover_bk,
          LoopAlignment<false, false, true>{});

      // Do epilogue
      if (use_out_source) {
        if (do_axpby) {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_axpby);
        } else {
          mma_op.apply_epilogue_safe(
              C,
              addmm_params->ldc,
              addmm_params->fdc,
              short2(tgp_bn, tgp_bm),
              epilogue_op_add);
        }
      }

      // Store results to device memory
      return mma_op.store_result_safe(D, params->ldd, short2(tgp_bn, tgp_bm));
    }
  }
}
