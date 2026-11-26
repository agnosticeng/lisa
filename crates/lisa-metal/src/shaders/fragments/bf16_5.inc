          }
        }
      }
    }

    // Other masking as needed
    if (has_mask) {
      constexpr auto neg_inf = Limits<AccumType>::finite_min;

      const int base_row = tid.x * BQ + tm;
      const int base_col = kb * BK;

      constexpr bool is_bool = is_same_v<MaskType, bool>;
      using melem_t = typename metal::conditional_t<is_bool, bool, AccumType>;
      using mtile_t = NAXTile<melem_t, TQ, TK>;
      using mfrag_t = typename mtile_t::frag_type;

      if (base_row + kU <= params->qL && base_col + BK <= params->kL) {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short ik = 0; ik < TK; ik++) {
          const int row_pos = base_row;
          const int col_pos = base_col + ik * kU;

          mfrag_t mfrag;
          mtile_t::NAXFrag_t::load_safe(
              mfrag,
              mask,
              int64_t(mask_params->M_strides[2]),
              Int<1>{},
              params->qL,
              params->kL,
              row_pos,
              col_pos);

          thread auto& fg = Stile.frag_at(0, ik);

          STEEL_PRAGMA_UNROLL
          for (short jj = 0; jj < mtile_t::kElemsPerFrag; jj++) {
            if constexpr (is_bool) {
              fg[jj] = mfrag[jj] ? fg[jj] : neg_inf;
            } else {
              fg[jj] += M_LOG2E_F * AccumType(mfrag[jj]);
            }
          }
        }
      }
    }

