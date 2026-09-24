// MLX steel GEMM infrastructure plus the engine `track_prefill_indirect` prefill-expert GEMM templates.
#define STEEL_CONST static constant constexpr const
#define STEEL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
#define STEEL_PRAGMA_NO_UNROLL _Pragma("clang loop unroll(disable)")



#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T>
struct is_static : metal::bool_constant<is_empty<remove_cv_t<T>>::value> {};

template <typename T>
struct pointer_element {};

template <typename T>
struct pointer_element<thread T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<device T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<constant T*> {
  using type = remove_cv_t<T>;
};
template <typename T>
struct pointer_element<threadgroup T*> {
  using type = remove_cv_t<T>;
};

template <typename T>
using pointer_element_t = typename pointer_element<remove_cv_t<T>>::type;

} // namespace metal

#pragma METAL internals : disable



#include <metal_stdlib>

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  using type = integral_constant;

  METAL_FUNC constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(__op__, __operator__)          \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto __operator__(                   \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv __op__ uv;                      \
    using res_t = metal::remove_addrspace_t<decltype(res)>; \
    return integral_constant<res_t, res>{};                 \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(true_type, T) {
  return true_type{};
}
template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator||(T, true_type) {
  return true_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(false_type, T) {
  return false_type{};
}

template <typename T, typename = metal::enable_if_t<!is_integral_v<T>>>
METAL_FUNC constexpr auto operator&&(T, false_type) {
  return false_type{};
}

// Dispatch utilities
template <typename F>
void dispatch_bool(bool v, F f) {
  if (v) {
    f(true_type{});
  } else {
    f(false_type{});
  }
}

template <int start, int stop, int step, typename F>
constexpr void const_for_loop(F f) {
  if constexpr (start < stop) {
    constexpr auto idx = Int<start>{};
    f(idx);
    const_for_loop<start + step, stop, step, F>(f);
  }
}

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable



#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// MMA helper
///////////////////////////////////////////////////////////////////////////////

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// NAX Steel with new tiles
///////////////////////////////////////////////////////////////////////////////

struct BaseNAXFrag {
  STEEL_CONST short kFragRows = 16;
  STEEL_CONST short kFragCols = 16;

  STEEL_CONST short kElemsPerFrag = (kFragRows * kFragCols) / 32;

  STEEL_CONST short kElemRows = 2;
  STEEL_CONST short kElemCols = 4;

  STEEL_CONST short kElemRowsJump = 8;

  static_assert(
      kElemRows * kElemCols == kElemsPerFrag,
      "MMAFrag shape is not consistent with MMAFrag size");

  template <typename U>
  using dtype_frag_t = typename metal::vec<U, kElemsPerFrag>;

  METAL_FUNC static short2 get_coord() {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3));
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4;
    return short2{fn, fm};
  }

  METAL_FUNC static short2 get_coord(short idx) {
    const ushort simd_lane_id = __metal_get_thread_index_in_simdgroup(ushort());
    const short qid = simd_lane_id >> 2;
    const short fm = ((qid & 4) | ((simd_lane_id >> 1) & 3)) + (idx >> 2) * 8;
    const short fn = ((qid & 2) | (simd_lane_id & 1)) * 4 + idx % 4;
    return short2{fn, fm};
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + c + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        }
      }
    }
  }

  /// Vector form of `load` for a contiguous, vector-aligned threadgroup stage.
  ///
  /// `load` reads each fragment as `kElemRows` groups of `kElemCols` *scalar*
  /// elements: for the P17 tile that is eight 2-byte loads per fragment, forty
  /// per (A, B0, B1) triple per `kk1` step. The four elements inside one group
  /// are contiguous, and every term of their address is a multiple of four
  /// elements -- `str_x` and `off_y` are compile-time multiples of `kElemCols`,
  /// and `get_coord().x` is `((qid & 2) | (lane & 1)) * 4` -- so the group is
  /// 8-byte aligned and moves as one `vec<T,4>` load. The compiler cannot do
  /// this itself: it cannot prove the alignment of a lane-dependent offset.
  ///
  /// Same elements, same registers, same order: this is `load` with the inner
  /// loop replaced by one vector access.
  template <int str_x, int off_x, int off_y, typename T>
  METAL_FUNC static constexpr void load_vec4(
      thread dtype_frag_t<T>& dst, const threadgroup T* src) {
    static_assert(
        str_x % kElemCols == 0, "P17 row stride is not vector aligned");
    static_assert(
        off_y % kElemCols == 0, "P17 column offset is not vector aligned");
    const short2 sc = get_coord();
    const threadgroup T* base = src + (sc.y + off_x) * str_x + sc.x + off_y;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const metal::vec<T, kElemCols> v =
          *reinterpret_cast<const threadgroup metal::vec<T, kElemCols>*>(
              base + i * kElemRowsJump * str_x);
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        dst[i * kElemCols + j] = v[j];
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_rows(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] = static_cast<T>(src[r * str_x + (c + j)]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[i * kElemCols + j] =
                static_cast<T>(src[r * str_x + (c + j) * str_y]);
          }
        }

      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename SrcPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void load_safe(
      thread dtype_frag_t<T>& dst,
      SrcPtrType src,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    const short2 sc = get_coord();
    src += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if ((r < lx) && ((c + j) < ly)) {
          dst[i * kElemCols + j] =
              static_cast<T>(src[r * str_x + (c + j) * str_y]);
        } else {
          dst[i * kElemCols + j] = T(0);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if constexpr (metal::is_same_v<StrY, Int<1>>) {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
        }
      } else {
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < kElemCols; j++) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  /// Vector form of `store` for a contiguous, vector-aligned device row.
  ///
  /// `store` writes each fragment as `kElemRows` groups of `kElemCols` *scalar*
  /// elements: eight 2-byte device stores per fragment, sixteen per thread in
  /// the gate|up epilogue and thirty-two in the down epilogue. The four
  /// elements inside one group are contiguous and every term of their address
  /// is a multiple of `kElemCols`: `off_y` is `idx_col * kFragCols` (a multiple
  /// of 16, compile-time), `get_coord().x` is `((qid & 2) | (lane & 1)) * 4`,
  /// and the output row stride is the kernel's `N`, carried here as the
  /// compile-time `str_x` so the alignment is a static assertion rather than a
  /// host promise. The compiler will not merge the four stores itself for the
  /// same reason it will not merge the loads: the offset is lane-dependent.
  ///
  /// Same elements, same values, same order as `store`; only the access width
  /// changes.
  template <int str_x, typename U, typename T>
  METAL_FUNC static constexpr void store_vec4(
      const thread dtype_frag_t<T>& src, device U* dst) {
    static_assert(str_x % kElemCols == 0, "P17 output row is not vector aligned");
    const short2 sc = get_coord();
    device U* base = dst + sc.y * str_x + sc.x;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      metal::vec<U, kElemCols> v;
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        v[j] = static_cast<U>(src[i * kElemCols + j]);
      }
      *reinterpret_cast<device metal::vec<U, kElemCols>*>(
          base + i * kElemRowsJump * str_x) = v;
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_rows(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      if (r < lx) {
        if constexpr (metal::is_same_v<StrY, Int<1>>) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + c + j] = static_cast<U>(src[i * kElemCols + j]);
          }
        } else {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < kElemCols; j++) {
            dst[r * str_x + (c + j) * str_y] =
                static_cast<U>(src[i * kElemCols + j]);
          }
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename LimX,
      typename LimY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_safe(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      LimX lim_x,
      LimY lim_y,
      OffX off_x = {},
      OffY off_y = {}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();
    dst += sc.y * str_x + sc.x * str_y;
    auto lx = lim_x - sc.y;
    auto ly = lim_y - sc.x;

    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      const auto r = off_x + i * kElemRowsJump;
      const auto c = off_y;

      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        if (r < lx && (c + j) < ly) {
          dst[r * str_x + (c + j) * str_y] =
              static_cast<U>(src[i * kElemCols + j]);
        }
      }
    }
  }

  template <
      typename T,
      typename DstPtrType,
      typename StrX,
      typename StrY,
      typename StartX,
      typename StopX,
      typename StartY,
      typename StopY,
      typename OffX = Int<0>,
      typename OffY = Int<0>>
  METAL_FUNC static constexpr void store_slice(
      const thread dtype_frag_t<T>& src,
      DstPtrType dst,
      StrX str_x,
      StrY str_y,
      StartX start_x,
      StopX stop_x,
      StartY start_y,
      StopY stop_y,
      OffX off_x = Int<0>{},
      OffY off_y = Int<0>{}) {
    using U = pointer_element_t<DstPtrType>;

    const short2 sc = get_coord();

    const_for_loop<0, kElemRows, 1>([&](auto idx_row) {
      const auto r = off_x + idx_row * Int<kElemRowsJump>{};
      if (r >= stop_x - sc.y || r < start_x - sc.y) {
        return;
      }

      const_for_loop<0, kElemCols, 1>([&](auto idx_col) {
        const auto c = off_y + idx_col;
        if (c >= stop_y - sc.x || c < start_y - sc.x) {
          return;
        }

        const auto src_idx = idx_row * Int<kElemCols>{} + idx_col;
        dst[(r + sc.y) * str_x + (c + sc.x) * str_y] =
            static_cast<U>(src[src_idx]);
      });
    });
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_reduce(
      thread const dtype_frag_t<T>& inp_vals,
      thread T* reduced_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      T thr_reduce = Op::apply(
          Op::apply(inp_vals[i * kElemCols + 0], inp_vals[i * kElemCols + 1]),
          Op::apply(inp_vals[i * kElemCols + 2], inp_vals[i * kElemCols + 3]));

      T qgr_reduce = simd_shuffle_xor(thr_reduce, ushort(1));
      qgr_reduce = Op::apply(thr_reduce, qgr_reduce);

      T sgr_reduce = simd_shuffle_xor(qgr_reduce, ushort(8));
      sgr_reduce = Op::apply(qgr_reduce, sgr_reduce);

      reduced_vals[i] = Op::apply(reduced_vals[i], sgr_reduce);
    }
  }

  template <typename Op, typename T>
  METAL_FUNC static constexpr void row_bin_op(
      thread dtype_frag_t<T>& inp_vals,
      thread T* row_vals) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemRows; i++) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kElemCols; j++) {
        inp_vals[i * kElemCols + j] =
            Op::apply(inp_vals[i * kElemCols + j], row_vals[i]);
      }
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cn0,
      thread dtype_frag_t<CType>& Cn1,
      const thread dtype_frag_t<AType>& A,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& Bn0,
      const thread dtype_frag_t<BType>& Bn1,
      metal::bool_constant<transpose_b>) {
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = A[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = Bn0[i];
      ct_b[kElemsPerFrag + i] = Bn1[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cn0[i];
      ct_c[kElemsPerFrag + i] = Cn1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cn0[i] = ct_c[i];
      Cn1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  template <
      typename CType,
      typename AType,
      typename BType,
      bool transpose_a = false,
      bool transpose_b = false>
  METAL_FUNC static constexpr void mma(
      thread dtype_frag_t<CType>& Cm0,
      thread dtype_frag_t<CType>& Cm1,
      const thread dtype_frag_t<AType>& Am0,
      const thread dtype_frag_t<AType>& Am1,
      metal::bool_constant<transpose_a>,
      const thread dtype_frag_t<BType>& B,
      metal::bool_constant<transpose_b>) {
    // Create Matmul descriptor
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        transpose_a,
        transpose_b,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);

    // Create matmul op
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> gemm_op;

    // Create matmul operands in registers
    auto ct_a =
        gemm_op
            .template get_left_input_cooperative_tensor<AType, BType, CType>();
    auto ct_b =
        gemm_op
            .template get_right_input_cooperative_tensor<AType, BType, CType>();

    // Create matmul output in register
    auto ct_c = gemm_op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ct_a)>,
        metal::remove_addrspace_t<decltype(ct_b)>,
        CType>();

    // Load A in to left operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_a[i] = Am0[i];
      ct_a[kElemsPerFrag + i] = Am1[i];
    }

    // Load B into right operand registers
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_b[i] = B[i];
    }

    // Load C into output registers (op handles accumulation)
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      ct_c[i] = Cm0[i];
      ct_c[kElemsPerFrag + i] = Cm1[i];
    }

    // Do matmul
    gemm_op.run(ct_a, ct_b, ct_c);

    // Copy out results
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kElemsPerFrag; i++) {
      Cm0[i] = ct_c[i];
      Cm1[i] = ct_c[kElemsPerFrag + i];
    }
  }
};

template <
    typename T,
    short kTileRows_,
    short kTileCols_,
    class NAXFrag_ = BaseNAXFrag>
struct NAXTile {
  using NAXFrag_t = NAXFrag_;
  using elem_type = T;

  STEEL_CONST short kFragRows = NAXFrag_t::kFragRows;
  STEEL_CONST short kFragCols = NAXFrag_t::kFragCols;
  STEEL_CONST short kElemsPerFrag = NAXFrag_t::kElemsPerFrag;

  STEEL_CONST short kTileRows = kTileRows_;
  STEEL_CONST short kTileCols = kTileCols_;

  STEEL_CONST short kRows = kTileRows * kFragRows;
  STEEL_CONST short kCols = kTileCols * kFragCols;

  STEEL_CONST short kNumFrags = kTileRows * kTileCols;
  STEEL_CONST short kElemsPerTile = kNumFrags * kElemsPerFrag;

  STEEL_CONST short kFragThrRows = NAXFrag_t::kElemRows;
  STEEL_CONST short kFragThrCols = NAXFrag_t::kElemCols;
  STEEL_CONST short kFragRowsJump = NAXFrag_t::kElemRowsJump;

  STEEL_CONST short kRowsPerThread = kTileRows * NAXFrag_t::kElemRows;
  STEEL_CONST short kColsPerThread = kTileCols * NAXFrag_t::kElemCols;

  typedef typename NAXFrag_t::template dtype_frag_t<T> frag_type;

  frag_type val_frags[kNumFrags]; // = {frag_type(0)};

  METAL_FUNC NAXTile() thread {}

  METAL_FUNC constexpr void clear() thread {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kNumFrags; ++i) {
      val_frags[i] = frag_type(0);
    }
  }

  METAL_FUNC constexpr thread frag_type& frag_at(const short i, const short j)
      thread {
    return val_frags[i * kTileCols + j];
  }

  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j) const thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    return val_frags[i * kTileCols + j];
  }

  template <int i, int j>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    return val_frags[i * kTileCols + j];
  }

  template <bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at(
      const short i,
      const short j,
      metal::bool_constant<transpose>) const thread {
    if constexpr (transpose) {
      return frag_at(j, i);
    } else {
      return frag_at(i, j);
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr thread frag_type& frag_at() thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  template <int i, int j, bool transpose>
  METAL_FUNC constexpr const thread frag_type& frag_at() const thread {
    if constexpr (transpose) {
      return frag_at<j, i>();
    } else {
      return frag_at<i, j>();
    }
  }

  METAL_FUNC thread elem_type* elems() thread {
    return reinterpret_cast<thread elem_type*>(val_frags);
  }

  METAL_FUNC const thread elem_type* elems() const thread {
    return reinterpret_cast<const thread elem_type*>(val_frags);
  }

  template <typename Op>
  METAL_FUNC void row_reduce(
      thread metal::vec<T, kRowsPerThread>& vals) const thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_reduce<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename Op>
  METAL_FUNC void row_bin_op(
      thread metal::vec<T, kRowsPerThread>& vals) thread {
    auto vptr = (thread T*)(&vals);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < kTileRows; ++i) {
      STEEL_PRAGMA_UNROLL
      for (short j = 0; j < kTileCols; ++j) {
        NAXFrag_t::template row_bin_op<Op>(
            frag_at(i, j), &vptr[i * kFragThrRows]);
      }
    }
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void load(const threadgroup U* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  /// `load` restricted to the case its own inner loop is scalar for: the P17
  /// gather's threadgroup stages, whose column offsets and row strides are
  /// vector aligned. See `BaseNAXFrag::load_vec4`.
  template <int str_x>
  METAL_FUNC void loadV(const threadgroup T* src) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::template load_vec4<
            str_x,
            idx_row.value * kFragRows,
            idx_col.value * kFragCols>(
            frag_at<idx_row.value, idx_col.value>(), src);
      });
    });
  }

  template <typename U, int str_x, int str_y>
  METAL_FUNC void store(threadgroup U* dst) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            Int<str_x>{},
            Int<str_y>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load(const device U* src, const int ld) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store(device U* dst, const int ld) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  /// `store` restricted to the case its own inner loop is scalar for: the P17
  /// gather's device epilogues, whose row stride and column offsets are vector
  /// aligned. See `BaseNAXFrag::store_vec4`.
  template <int ld, typename U>
  METAL_FUNC void storeV(device U* dst) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::template store_vec4<ld, U>(
            frag_at<idx_row.value, idx_col.value>(),
            dst + idx_row.value * kFragRows * ld + idx_col.value * kFragCols);
      });
    });
  }

  template <typename U>
  METAL_FUNC void
  load_rows(const device U* src, const int ld, const short n_rows) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_rows(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void load_safe(
      const device U* src,
      const int ld,
      const short2 src_tile_dims) thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::load_safe(
            frag_at<idx_row.value, idx_col.value>(),
            src,
            ld,
            Int<1>{},
            src_tile_dims.y,
            src_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_rows(device U* dst, const int ld, const short n_rows)
      const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_rows(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            n_rows,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_safe(
      device U* dst,
      const int ld,
      const short2 dst_tile_dims) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_safe(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            dst_tile_dims.y,
            dst_tile_dims.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }

  template <typename U>
  METAL_FUNC void store_slice(
      device U* dst,
      const int ld,
      const short2 start,
      const short2 stop) const thread {
    const_for_loop<0, kTileRows, 1>([&](auto idx_row) {
      const_for_loop<0, kTileCols, 1>([&](auto idx_col) {
        NAXFrag_t::store_slice(
            frag_at<idx_row.value, idx_col.value>(),
            dst,
            ld,
            Int<1>{},
            start.y,
            stop.y,
            start.x,
            stop.x,
            idx_row * Int<kFragRows>{},
            idx_col * Int<kFragCols>{});
      });
    });
  }
};

template <
    class CTile,
    class ATile,
    class BTile,
    bool transpose_a,
    bool transpose_b>
METAL_FUNC void tile_matmad_nax(
    thread CTile& C,
    thread ATile& A,
    metal::bool_constant<transpose_a>,
    thread BTile& B,
    metal::bool_constant<transpose_b>) {
  // Static checks
  constexpr short TMa = transpose_a ? ATile::kTileCols : ATile::kTileRows;
  constexpr short TM = CTile::kTileRows;
  static_assert(TMa == TM, "MXU tile matmul: M dimensions do not match");

  constexpr short TNb = transpose_b ? BTile::kTileRows : BTile::kTileCols;
  constexpr short TN = CTile::kTileCols;
  static_assert(TNb == TN, "MXU tile matmul: N dimensions do not match");

  constexpr short TKa = transpose_a ? ATile::kTileRows : ATile::kTileCols;
  constexpr short TK = transpose_b ? BTile::kTileCols : BTile::kTileRows;
  static_assert(TKa == TK, "MXU tile matmul: K dimensions do not match");

  constexpr auto ta = metal::bool_constant<transpose_a>{};
  constexpr auto tb = metal::bool_constant<transpose_b>{};

  if constexpr (TN == 1 && TM % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; mm += 2) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; ++nn) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm + 1, nn),
              A.frag_at(mm, kk, ta),
              A.frag_at(mm + 1, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  } else if constexpr (TN % 2 == 0) {
    STEEL_PRAGMA_UNROLL
    for (short mm = 0; mm < TM; ++mm) {
      STEEL_PRAGMA_UNROLL
      for (short nn = 0; nn < TN; nn += 2) {
        STEEL_PRAGMA_UNROLL
        for (short kk = 0; kk < TK; ++kk) {
          CTile::NAXFrag_t::mma(
              C.frag_at(mm, nn),
              C.frag_at(mm, nn + 1),
              A.frag_at(mm, kk, ta),
              metal::bool_constant<transpose_a>{},
              B.frag_at(kk, nn, tb),
              B.frag_at(kk, nn + 1, tb),
              metal::bool_constant<transpose_b>{});
        }
      }
    }
  }
}

} // namespace steel
} // namespace mlx

using namespace mlx::steel;
#define MLX_MTL_CONST static constant constexpr const
MLX_MTL_CONST int SIMD_SIZE = 32;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}


template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short group_size,
    short bits>
struct QuantizedBlockLoader;

template <
    typename T,
    short BROWS,
    short BCOLS,
    short dst_ld,
    short reduction_dim,
    short tgp_size,
    short bits>
struct QuantizedBlockLoader<
    T,
    BROWS,
    BCOLS,
    dst_ld,
    reduction_dim,
    tgp_size,
    32,
    bits> {
  MLX_MTL_CONST short group_size = 32;

  static_assert(
      BCOLS % group_size == 0,
      "The group size should be divisible by the columns");
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads =
      (BCOLS_PACKED * BROWS < tgp_size) ? 1 : (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_groups = BCOLS / group_size;

  static_assert(
      (BCOLS_PACKED / n_reads) == n_groups,
      "Other configurations are not yet supported");

  const int src_ld;
  const int tile_stride;
  const int group_stride;

  const short thread_idx;
  const short bi;
  const short bj;

  const short group_id;

  threadgroup T* dst;
  const device uint8_t* src;
  const device T* scales;
  const device T* biases;

  QuantizedBlockLoader(
      const device uint8_t* src_,
      const device T* scales_,
      const device T* biases_,
      const int src_ld_,
      threadgroup T* dst_,
      ushort simd_group_id [[simdgroup_index_in_threadgroup]],
      ushort simd_lane_id [[thread_index_in_simdgroup]]) thread
      : src_ld(src_ld_),
        tile_stride(
            reduction_dim ? BCOLS_PACKED* bytes_per_pack
                          : BROWS * src_ld * bytes_per_pack / pack_factor),
        group_stride(BROWS* src_ld / group_size),
        thread_idx(simd_group_id * 32 + simd_lane_id),
        bi(n_reads* thread_idx / BCOLS_PACKED),
        bj((n_reads * thread_idx) % BCOLS_PACKED),
        group_id((bj * pack_factor) / group_size),
        dst(dst_ + bi * dst_ld + bj * pack_factor),
        src(src_ + bi * src_ld * bytes_per_pack / pack_factor +
            bj * bytes_per_pack),
        scales(scales_ + bi * src_ld / group_size + group_id),
        biases(biases_ + bi * src_ld / group_size + group_id) {}

  void next() thread {
    src += tile_stride;
    if (reduction_dim == 1) {
      scales += n_groups;
      biases += n_groups;
    } else {
      scales += group_stride;
      biases += group_stride;
    }
  }
};


struct PackedNAXGroup32 {
  uint4 words;
  bfloat16_t scale;
  bfloat16_t bias;

  template <typename Loader>
  void prefetch(const thread Loader& loader) thread {
    static_assert(Loader::n_reads == 16 && Loader::pack_factor == 2);
    // Four word loads also support batch offsets aligned to 4, not 16, bytes.
    const device uint32_t* src =
        reinterpret_cast<const device uint32_t*>(loader.src);
    words = uint4(src[0], src[1], src[2], src[3]);
    scale = *loader.scales;
    bias = *loader.biases;
  }

  template <typename T>
  void store(threadgroup T* dst) const thread {
    static_assert(metal::is_same_v<T, bfloat16_t>);
    const float s = float(scale);
    const float b = float(bias);
    float sc[2] = {s, s / 16.0f};
    // OPT-WVEC8: the same per-value arithmetic, stored eight values at a
    // time. One `words[j]` holds the four bytes that dequantise into values
    // `8j` through `8j + 7`, so the group is four 16-byte stores instead of
    // eight 8-byte ones. `dst` is 16-byte aligned for every shape this kernel
    // is launched with: the threadgroup stages are declared `alignas(16)`,
    // their rows are 72 or 40 `bfloat16_t` (144 / 80 bytes), and `kk1` is a
    // multiple of 16 elements.
    threadgroup vec<bfloat16_t, 8>* d8 = (threadgroup vec<bfloat16_t, 8>*)dst;
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; j++) {
      const uint8_t w0 = uint8_t(words[j]);
      const uint8_t w1 = uint8_t(words[j] >> 8);
      const uint8_t w2 = uint8_t(words[j] >> 16);
      const uint8_t w3 = uint8_t(words[j] >> 24);
      d8[j] = vec<bfloat16_t, 8>(
          static_cast<bfloat16_t>(sc[0] * (w0 & 0x0f) + b),
          static_cast<bfloat16_t>(sc[1] * (w0 & 0xf0) + b),
          static_cast<bfloat16_t>(sc[0] * (w1 & 0x0f) + b),
          static_cast<bfloat16_t>(sc[1] * (w1 & 0xf0) + b),
          static_cast<bfloat16_t>(sc[0] * (w2 & 0x0f) + b),
          static_cast<bfloat16_t>(sc[1] * (w2 & 0xf0) + b),
          static_cast<bfloat16_t>(sc[0] * (w3 & 0x0f) + b),
          static_cast<bfloat16_t>(sc[1] * (w3 & 0xf0) + b));
    }
  }
};


template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    int NS,
    bool IDENTITY_ROWS = false>
METAL_FUNC void track_prefill_indirect(
    const device T* x,
    const device uint32_t* w,
    const device T* scales,
    const device T* biases,
    const device uint32_t* indices,
    const device uint32_t* token_rows,
    const device uint32_t* tiles,
    device T* y,
    int N,
    int K,
    threadgroup T* Ws,
    threadgroup T* As,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  static_assert(
      transpose && BM == 32 && WM == 2 && WN == 2 &&
          ((BN == 64 && BK == 64) || (BN == 128 && BK == 32)),
      "P17 tile: 32 rows, 2x2 SIMD layout, 64x64 or 128x32 weight block");
  static_assert(
      metal::is_same_v<T, bfloat16_t> && group_size == 32 && bits == 4,
      "P17 requires unchanged bf16 / affine group-32 / 4-bit operands");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BKA_padded = BK_padded;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, transpose, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_col = tid.x * BN;

  auto wl = (const device uint8_t*)w;
  wl += size_t(y_col) * K_w;
  scales += size_t(y_col) * K_g;
  biases += size_t(y_col) * K_g;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  constexpr short BR = TN;
  constexpr short BC = TK;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  using AccumType = float;

  // One tile per threadgroup row. The table (built by track_prefill_tile_table
  // from the sorted ids) holds [begin, end) per tile, aligned to the expert
  // run's start exactly as the former in-kernel scan aligned them; padding
  // slots are [0, 0) and exit at once. Uniform over the whole threadgroup.
  {
    const int tile_begin = int(tiles[2 * tid.y]);
    const int tile_end = int(tiles[2 * tid.y + 1]);
    if (tile_begin == tile_end) {
      return;
    }
    const uint32_t index = indices[tile_begin];
    const short tile_m = short(tile_end - tile_begin);
    const short sgp_sm = short(min(int(SM), max(0, int(tile_m) - int(tm))));
    const bool sg_active = sgp_sm > 0;

    NAXTile<AccumType, TM, TN> Dtile;
    Dtile.clear();

    // OPT-ACC: see `track_prefill_indirect_gu`. The destination cooperative
    // tensor lives across the whole K walk instead of being a per-call
    // temporary.
    constexpr auto acc_desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        false,
        true,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<acc_desc, metal::execution_simdgroup> acc_op;
    auto acc_a =
        acc_op.template get_left_input_cooperative_tensor<T, T, AccumType>();
    auto acc_b =
        acc_op.template get_right_input_cooperative_tensor<T, T, AccumType>();
    using AccAT = metal::remove_addrspace_t<decltype(acc_a)>;
    using AccBT = metal::remove_addrspace_t<decltype(acc_b)>;
    auto acc_c0 =
        acc_op.template get_destination_cooperative_tensor<AccAT, AccBT, AccumType>();
    auto acc_c1 =
        acc_op.template get_destination_cooperative_tensor<AccAT, AccBT, AccumType>();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 2 * NAXTile<AccumType, TM, TN>::kElemsPerFrag; ++e) {
      acc_c0[e] = AccumType(0);
      acc_c1[e] = AccumType(0);
    }

    constexpr short A_PER_THREAD = (BM * BK) / (WM * WN * SIMD_SIZE);  // 16 or 8
    constexpr short A_SPLIT = BK / A_PER_THREAD;                        // threads per row
    const short tgp_thread = short(simd_group_id * SIMD_SIZE + simd_lane_id);
    const short a_row = tgp_thread / A_SPLIT;              // 0..BM-1
    const short a_col = (tgp_thread % A_SPLIT) * A_PER_THREAD;
    threadgroup T* a_dst = As + a_row * BKA_padded + a_col;
    const bool a_live = a_row < tile_m;
    const device T* xb = x;
    if (a_live) {
      if constexpr (IDENTITY_ROWS) {
        xb += size_t(tile_begin + a_row) * K + a_col;
      } else {
        xb += size_t(token_rows[tile_begin + a_row]) * K + a_col;
      }
    }

    thread loader_w_t loader_w(
        wl + index * stride_w,
        scales + index * stride_s,
        biases + index * stride_s,
        K,
        Ws,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(tile_m == BM, [&](auto kAlignedM) {
      // OPT-AVEC: the slice is 16-byte aligned at both ends (K = 2560/640
      // elements, a_col a multiple of 8 elements, As rows 72/40 elements), so
      // it moves as uint4 vectors; the same bytes in the same order.
      constexpr short A_VECS = (A_PER_THREAD * sizeof(T)) / 16;
      uint4 a_buf[A_VECS];
      PackedNAXGroup32 packed_w;
      if (K_it > 0) {
        packed_w.prefetch(loader_w);
        if (a_live) {
          const device uint4* a0 = (const device uint4*)xb;
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < A_VECS; ++v) { a_buf[v] = a0[v]; }
        }
      }
      if (!a_live) {
        // A dead row's slice of the activation stage is zero for every K step:
        // `a_dst` never advances, so the fill is written once here instead of
        // once per step by the else branch that used to sit in the loop.
        threadgroup uint4* d0 = (threadgroup uint4*)a_dst;
        STEEL_PRAGMA_UNROLL
        for (short v = 0; v < A_VECS; ++v) { d0[v] = uint4(0); }
      }
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        packed_w.store(loader_w.dst);
        if (a_live) {
          threadgroup uint4* d4 = (threadgroup uint4*)a_dst;
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < A_VECS; ++v) { d4[v] = a_buf[v]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k + 1 < K_it) {
          loader_w.next();
          packed_w.prefetch(loader_w);
          if (a_live) {
            const device uint4* a_next = (const device uint4*)(xb + BK);
            STEEL_PRAGMA_UNROLL
            for (short v = 0; v < A_VECS; ++v) { a_buf[v] = a_next[v]; }
          }
        }

        STEEL_PRAGMA_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile;

            volatile int compiler_barrier;

            Atile.template loadV<BKA_padded>(
                As + tm * BKA_padded + kk1);

            Btile.template loadV<BK_padded>(Ws + tn * BK_padded + kk1);

            // The same walk `tile_matmad_nax` performs for TN % 2 == 0: `TN / 2`
            // destination pairs, each accumulated by its own cooperative tensor.
            STEEL_PRAGMA_UNROLL
            for (short nn = 0; nn < TN; nn += 2) {
              STEEL_PRAGMA_UNROLL
              for (short kk = 0; kk < TK; ++kk) {
                const thread auto& a_frag = Atile.frag_at(0, kk);
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < NAXTile<T, TM, TK>::kElemsPerFrag; ++e) {
                  acc_a[e] = a_frag[e];
                }
                STEEL_PRAGMA_UNROLL
                for (short e = 0; e < NAXTile<T, BR, BC>::kElemsPerFrag; ++e) {
                  acc_b[e] = Btile.frag_at(nn, kk)[e];
                  acc_b[NAXTile<T, BR, BC>::kElemsPerFrag + e] =
                      Btile.frag_at(nn + 1, kk)[e];
                }
                if (nn == 0) {
                  acc_op.run(acc_a, acc_b, acc_c0);
                } else {
                  acc_op.run(acc_a, acc_b, acc_c1);
                }
              }
            }

            (void)compiler_barrier;
          }
        }

        xb += BK;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < NAXTile<AccumType, TM, TN>::kElemsPerFrag; ++e) {
          Dtile.val_frags[0][e] = acc_c0[e];
          Dtile.val_frags[1][e] =
              acc_c0[NAXTile<AccumType, TM, TN>::kElemsPerFrag + e];
          Dtile.val_frags[2][e] = acc_c1[e];
          Dtile.val_frags[3][e] =
              acc_c1[NAXTile<AccumType, TM, TN>::kElemsPerFrag + e];
        }
        device T* yn = y + size_t(tile_begin + tm) * N + y_col + tn;
        if constexpr (kAlignedM.value) {
          Dtile.template storeV<NS>(yn);
        } else {
          Dtile.store_slice(yn, N, short2(0, 0), short2(SN, sgp_sm));
        }
      }
    });
  }
}


// Gate and up in one launch: one A staging and one pair of barriers per K step
// serve two weight streams; each output is the same MMA sequence as the single-bank kernel.
// MLX's Sigmoid (kernels/unary_ops.h), verbatim, instantiated at bfloat16_t so the
// bf16 math overloads round after every op exactly as the compiled kernel does.
struct P17Sigmoid {
  template <typename T>
  T operator()(T x) thread {
    auto y = 1 / (1 + metal::exp(metal::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};

template <
    typename T,
    int group_size,
    int bits,
    int BM,
    int BN,
    int BK,
    int WM,
    int WN,
    bool transpose,
    bool SILU,
    int NS,
    bool IDENTITY_ROWS = false>
METAL_FUNC void track_prefill_indirect_gu(
    const device T* x,
    const device uint32_t* w0,
    const device T* scales0,
    const device T* biases0,
    const device uint32_t* w1,
    const device T* scales1,
    const device T* biases1,
    const device uint32_t* indices,
    const device uint32_t* token_rows,
    const device uint32_t* tiles,
    device T* y0,
    device T* y1,
    int N,
    int K,
    threadgroup T* Ws0,
    threadgroup T* Ws1,
    threadgroup T* As,
    uint3 tid,
    uint simd_group_id,
    uint simd_lane_id) {
  static_assert(
      transpose && BM == 32 && WM == 2 && WN == 2 &&
          ((BN == 64 && BK == 64) || (BN == 128 && BK == 32)),
      "P17 tile: 32 rows, 2x2 SIMD layout, 64x64 or 128x32 weight block");
  static_assert(
      metal::is_same_v<T, bfloat16_t> && group_size == 32 && bits == 4,
      "P17 requires unchanged bf16 / affine group-32 / 4-bit operands");

  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int BKA_padded = BK_padded;
  using loader_w_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, transpose, WM * WN * SIMD_SIZE, group_size, bits>;

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * K_w;
  const size_t stride_s = size_t(N) * K_g;
  const int y_col = tid.x * BN;

  auto wl0 = (const device uint8_t*)w0 + size_t(y_col) * K_w;
  auto wl1 = (const device uint8_t*)w1 + size_t(y_col) * K_w;
  scales0 += size_t(y_col) * K_g;
  biases0 += size_t(y_col) * K_g;
  scales1 += size_t(y_col) * K_g;
  biases1 += size_t(y_col) * K_g;

  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  constexpr short BR = TN;
  constexpr short BC = TK;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  using AccumType = float;

  // One tile per threadgroup row. The table (built by track_prefill_tile_table
  // from the sorted ids) holds [begin, end) per tile, aligned to the expert
  // run's start exactly as the former in-kernel scan aligned them; padding
  // slots are [0, 0) and exit at once. Uniform over the whole threadgroup.
  {
    const int tile_begin = int(tiles[2 * tid.y]);
    const int tile_end = int(tiles[2 * tid.y + 1]);
    if (tile_begin == tile_end) {
      return;
    }
    const uint32_t index = indices[tile_begin];
    const short tile_m = short(tile_end - tile_begin);
    const short sgp_sm = short(min(int(SM), max(0, int(tile_m) - int(tm))));
    const bool sg_active = sgp_sm > 0;

    NAXTile<AccumType, TM, TN> Dtile0;
    NAXTile<AccumType, TM, TN> Dtile1;
    Dtile0.clear();
    Dtile1.clear();

    // OPT-ACC. The MMA's destination cooperative tensor is the accumulator
    // and stays alive across the whole K walk. The per-call form copies the
    // fragment pair into a fresh `ct_c` and back around every MMA -- eight
    // times per thread per K step, sixteen fragment copies each way; keeping
    // one `ct_c` per stream moves the value out of the accumulator exactly
    // once, at the end. The MMA sequence, its operands and its order are
    // unchanged, so every output element accumulates in the same order.
    constexpr auto acc_desc = mpp::tensor_ops::matmul2d_descriptor(
        16,
        32,
        16,
        false,
        true,
        true,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<acc_desc, metal::execution_simdgroup> acc_op;
    auto acc_a =
        acc_op.template get_left_input_cooperative_tensor<T, T, AccumType>();
    auto acc_b =
        acc_op.template get_right_input_cooperative_tensor<T, T, AccumType>();
    using AccAT = metal::remove_addrspace_t<decltype(acc_a)>;
    using AccBT = metal::remove_addrspace_t<decltype(acc_b)>;
    auto acc_c0 =
        acc_op.template get_destination_cooperative_tensor<AccAT, AccBT, AccumType>();
    auto acc_c1 =
        acc_op.template get_destination_cooperative_tensor<AccAT, AccBT, AccumType>();
    STEEL_PRAGMA_UNROLL
    for (short e = 0; e < 2 * NAXTile<AccumType, TM, TN>::kElemsPerFrag; ++e) {
      acc_c0[e] = AccumType(0);
      acc_c1[e] = AccumType(0);
    }

    constexpr short A_PER_THREAD = (BM * BK) / (WM * WN * SIMD_SIZE);  // 16 or 8
    constexpr short A_SPLIT = BK / A_PER_THREAD;                        // threads per row
    const short tgp_thread = short(simd_group_id * SIMD_SIZE + simd_lane_id);
    const short a_row = tgp_thread / A_SPLIT;              // 0..BM-1
    const short a_col = (tgp_thread % A_SPLIT) * A_PER_THREAD;
    threadgroup T* a_dst = As + a_row * BKA_padded + a_col;
    const bool a_live = a_row < tile_m;
    const device T* xb = x;
    if (a_live) {
      if constexpr (IDENTITY_ROWS) {
        xb += size_t(tile_begin + a_row) * K + a_col;
      } else {
        xb += size_t(token_rows[tile_begin + a_row]) * K + a_col;
      }
    }

    thread loader_w_t loader_w0(
        wl0 + index * stride_w,
        scales0 + index * stride_s,
        biases0 + index * stride_s,
        K,
        Ws0,
        simd_group_id,
        simd_lane_id);
    thread loader_w_t loader_w1(
        wl1 + index * stride_w,
        scales1 + index * stride_s,
        biases1 + index * stride_s,
        K,
        Ws1,
        simd_group_id,
        simd_lane_id);

    dispatch_bool(tile_m == BM, [&](auto kAlignedM) {
      // OPT-AVEC: the slice is 16-byte aligned at both ends (K = 2560/640
      // elements, a_col a multiple of 8 elements, As rows 72/40 elements), so
      // it moves as uint4 vectors; the same bytes in the same order.
      constexpr short A_VECS = (A_PER_THREAD * sizeof(T)) / 16;
      uint4 a_buf[A_VECS];
      PackedNAXGroup32 packed_w0;
      PackedNAXGroup32 packed_w1;
      if (K_it > 0) {
        packed_w0.prefetch(loader_w0);
        packed_w1.prefetch(loader_w1);
        if (a_live) {
          const device uint4* a0 = (const device uint4*)xb;
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < A_VECS; ++v) { a_buf[v] = a0[v]; }
        }
      }
      if (!a_live) {
        // A dead row's slice of the activation stage is zero for every K step:
        // `a_dst` never advances, so the fill is written once here instead of
        // once per step by the else branch that used to sit in the loop.
        threadgroup uint4* d0 = (threadgroup uint4*)a_dst;
        STEEL_PRAGMA_UNROLL
        for (short v = 0; v < A_VECS; ++v) { d0[v] = uint4(0); }
      }
      for (int k = 0; k < K_it; k++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        packed_w0.store(loader_w0.dst);
        packed_w1.store(loader_w1.dst);
        if (a_live) {
          threadgroup uint4* d4 = (threadgroup uint4*)a_dst;
          STEEL_PRAGMA_UNROLL
          for (short v = 0; v < A_VECS; ++v) { d4[v] = a_buf[v]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (k + 1 < K_it) {
          loader_w0.next();
          loader_w1.next();
          packed_w0.prefetch(loader_w0);
        packed_w1.prefetch(loader_w1);
          if (a_live) {
            const device uint4* a_next = (const device uint4*)(xb + BK);
            STEEL_PRAGMA_UNROLL
            for (short v = 0; v < A_VECS; ++v) { a_buf[v] = a_next[v]; }
          }
        }

        STEEL_PRAGMA_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          if (sg_active) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, BR, BC> Btile0;
            NAXTile<T, BR, BC> Btile1;

            volatile int compiler_barrier;

            Atile.template loadV<BKA_padded>(
                As + tm * BKA_padded + kk1);

            Btile0.template loadV<BK_padded>(Ws0 + tn * BK_padded + kk1);
            Btile1.template loadV<BK_padded>(Ws1 + tn * BK_padded + kk1);

            // The same walk `tile_matmad_nax` performs for TN = 2: for each
            // `kk` the A fragment goes to the left operand, the two B
            // fragments to the right one, and the MMA accumulates into the
            // stream's own destination tensor.
            STEEL_PRAGMA_UNROLL
            for (short kk = 0; kk < TK; ++kk) {
              const thread auto& a_frag = Atile.frag_at(0, kk);
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < NAXTile<T, TM, TK>::kElemsPerFrag; ++e) {
                acc_a[e] = a_frag[e];
              }
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < NAXTile<T, BR, BC>::kElemsPerFrag; ++e) {
                acc_b[e] = Btile0.frag_at(0, kk)[e];
                acc_b[NAXTile<T, BR, BC>::kElemsPerFrag + e] =
                    Btile0.frag_at(1, kk)[e];
              }
              acc_op.run(acc_a, acc_b, acc_c0);
              STEEL_PRAGMA_UNROLL
              for (short e = 0; e < NAXTile<T, BR, BC>::kElemsPerFrag; ++e) {
                acc_b[e] = Btile1.frag_at(0, kk)[e];
                acc_b[NAXTile<T, BR, BC>::kElemsPerFrag + e] =
                    Btile1.frag_at(1, kk)[e];
              }
              acc_op.run(acc_a, acc_b, acc_c1);
            }

            (void)compiler_barrier;
          }
        }

        xb += BK;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      if (sg_active) {
        STEEL_PRAGMA_UNROLL
        for (short e = 0; e < NAXTile<AccumType, TM, TN>::kElemsPerFrag; ++e) {
          Dtile0.val_frags[0][e] = acc_c0[e];
          Dtile0.val_frags[1][e] =
              acc_c0[NAXTile<AccumType, TM, TN>::kElemsPerFrag + e];
          Dtile1.val_frags[0][e] = acc_c1[e];
          Dtile1.val_frags[1][e] =
              acc_c1[NAXTile<AccumType, TM, TN>::kElemsPerFrag + e];
        }
        const size_t yoff = size_t(tile_begin + tm) * N + y_col + tn;
        if constexpr (SILU) {
          // silu(gate) * up, op for op as MLX's compiled `silu(gate) * up`: the
          // fp32 accumulators round to bf16 exactly as the two stores would, then
          // Sigmoid, Multiply, Multiply each round to bf16.
          STEEL_PRAGMA_UNROLL
          for (short fi = 0; fi < Dtile0.kNumFrags; fi++) {
            STEEL_PRAGMA_UNROLL
            for (short e = 0; e < Dtile0.kElemsPerFrag; e++) {
              const T g = static_cast<T>(Dtile0.val_frags[fi][e]);
              const T u = static_cast<T>(Dtile1.val_frags[fi][e]);
              const T sg = P17Sigmoid{}(g);
              const T act = (g * sg) * u;
              Dtile0.val_frags[fi][e] = static_cast<float>(act);
            }
          }
          if constexpr (kAlignedM.value) {
            Dtile0.template storeV<NS>(y0 + yoff);
          } else {
            Dtile0.store_slice(y0 + yoff, N, short2(0, 0), short2(SN, sgp_sm));
          }
        } else if constexpr (kAlignedM.value) {
          Dtile0.template storeV<NS>(y0 + yoff);
          Dtile1.template storeV<NS>(y1 + yoff);
        } else {
          Dtile0.store_slice(y0 + yoff, N, short2(0, 0), short2(SN, sgp_sm));
          Dtile1.store_slice(y1 + yoff, N, short2(0, 0), short2(SN, sgp_sm));
        }
      }
    });
  }
}
