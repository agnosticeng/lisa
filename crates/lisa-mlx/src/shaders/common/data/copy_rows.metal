// In-place row append: write `src` [.., s, D] into `dst` [.., cap, D] starting at
// row `off` along the second-to-last axis. Only the `s` appended rows are
// touched, so this is O(s) rather than `slice_assign` (O(cap), which
// materialises a full-size padded `src` + mask and runs `where_cond`). Used by
// the KV cache and the indexer tape, whose appends dominate long-context decode.
//
// `dst` is passed as a shape-carrying input AND as the preallocated output, so
// the encoder registers the read-after-write dependency on the cache buffer.
        const int rank = src_ndim;
        const uint D = src_shape[rank - 1];
        const uint s = src_shape[rank - 2];
        const uint cap = dst_shape[rank - 2];
        const uint g = thread_position_in_grid.x;
        const uint d = g % D;
        const uint i = (g / D) % s;
        const uint r = g / (D * s);
        // Strided src (e.g. the transposed v view handed to the KV cache):
        // decompose g through the declared strides instead of assuming a
        // dense layout. Same bytes, same order — bit-exact by construction.
        uint src_idx = 0;
        uint tmp = g;
        for (int a = rank - 1; a >= 0; --a) {
            src_idx += (tmp % (uint)src_shape[a]) * (uint)src_strides[a];
            tmp /= (uint)src_shape[a];
        }
        out[(r * cap + (uint)off + i) * D + d] = src[src_idx];
