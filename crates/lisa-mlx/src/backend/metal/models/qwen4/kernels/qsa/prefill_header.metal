// Header/helpers for the QSA block-sparse prefill kernel.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

constant constexpr short QSA_ELEMS_PER_FRAG = 8;
constant constexpr short QSA_ELEM_COLS = 4;
constant constexpr short QSA_ELEM_ROWS_JUMP = 8;

// NAX fragment lane -> (column, row) mapping used by sdpa_nax_tile.py and the
// proven M5/G17G MetalPerformancePrimitives 16x32x16 descriptor.
inline short2 qsa_nax_coord(ushort lane) {
    short quad = short(lane >> 2);
    short row = ((quad & 4) | ((short(lane) >> 1) & 3));
    short col = ((quad & 2) | (short(lane) & 1)) * 4;
    return short2{col, row};
}
