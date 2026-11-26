    STEEL_PRAGMA_UNROLL
    for (short id = 0; id < TDh; id += 2) {
      STEEL_PRAGMA_UNROLL
      for (short ik = 0; ik < TK; ik++) {
        NAXTile<T, 1, 2> Vtile;

        const int V_load_off = ik * kU * int(params->V_strides[2]) + id * kU;

        if (!align_K && is_last_k) {
          Vtile.load_rows(
              V + V_load_off, int(params->V_strides[2]), lim_rows_k - ik * kU);
        } else {
          Vtile.load(V + V_load_off, int(params->V_strides[2]));
        }

        otile_t::NAXFrag_t::mma(
            Otile.frag_at(0, id),
            Otile.frag_at(0, id + 1),
            Stile.frag_at(0, ik),
            metal::false_type{},
            Vtile.frag_at(0, 0),
            Vtile.frag_at(0, 1),
            metal::false_type{});
      }
    }

    // Next block
    K += BK * int(params->K_strides[2]);
    V += BK * int(params->V_strides[2]);
  }

  // Normalize output
  threadgroup_barrier(mem_flags::mem_none);

  metal::vec<AccumType, kRowsPT> rcp;
  STEEL_PRAGMA_UNROLL
  for (short i = 0; i < kRowsPT; ++i) {
    rcp[i] = 1.f / sum_score[i];
  }

