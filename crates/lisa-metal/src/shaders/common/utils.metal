// MLX Metal utils: bf16 math, type traits, `instantiate_kernel`, shared defines.
///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/bf16.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/bf16.h"


#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/bf16_math.h"
///////////////////////////////////////////////////////////////////////////////
//@include fragments/bf16_math_1.inc
///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/complex.h"
///////////////////////////////////////////////////////////////////////////////

//@include fragments/complex_1.inc
///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/defines.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/defines.h"


#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/logging.h"
///////////////////////////////////////////////////////////////////////////////

#line 1 "mlx/backend/metal/kernels/logging.h"


#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const thread {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif

///////////////////////////////////////////////////////////////////////////////
// Contents from "mlx/backend/metal/kernels/utils.h"
///////////////////////////////////////////////////////////////////////////////
//@include fragments/utils_5.inc
///////////////////////////////////////////////////////////////////////////////
