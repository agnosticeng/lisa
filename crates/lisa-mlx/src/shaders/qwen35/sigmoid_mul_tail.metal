// Attention-tail sigmoid gate over a NON-transposed sdpa output (specs/27):
// `sigmoid(gate[b,s,h,d]) * up[b,h,s,d]` -> `out[b, s, h*hd]` in one launch.
// Replaces the tail transpose materialization (`[b,h,s,hd]` -> `[b,s,h,hd]`
// copy, ~48/verify at ~75 µs GPU-slot each) + the separate sigmoid_mul.
//
// Parity: identical to `track_sigmoid_mul` — same exact_header per-op bf16
// rounding chain (exp, 1+, 1/, branch, product); only the ADDRESSING differs
// (both operands read through their declared strides, same elements, same
// order of rounding points). Bit-exact by construction; pinned by
// `sigmoid_mul_tail_matches_transposed`.
        const uint i = thread_position_in_grid.x;
        if (i >= (uint)(B * H * S * HD)) return;
        const uint d = i % (uint)HD;
        const uint hidx = (i / (uint)HD) % (uint)H;
        const uint r = (i / ((uint)HD * (uint)H)) % (uint)S;
        const uint b = i / ((uint)HD * (uint)H * (uint)S);
        InT x = gate[b * (uint)gate_strides[0] + r * (uint)gate_strides[1]
                   + hidx * (uint)gate_strides[2] + d * (uint)gate_strides[3]];
        InT ax = static_cast<InT>(metal::abs(static_cast<float>(x)));
        InT e = static_cast<InT>(metal::precise::exp(static_cast<float>(ax)));
        InT t = static_cast<InT>(1.0f + static_cast<float>(e));
        InT y = static_cast<InT>(1.0f / static_cast<float>(t));
        InT s = (x < InT(0)) ? y : static_cast<InT>(1.0f - static_cast<float>(y));
        InT u = up[b * (uint)up_strides[0] + hidx * (uint)up_strides[1]
                 + r * (uint)up_strides[2] + d * (uint)up_strides[3]];
        out[i] = s * u;
