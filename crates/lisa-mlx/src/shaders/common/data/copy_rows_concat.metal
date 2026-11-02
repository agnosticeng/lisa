// Append two same-row tensors side-by-side directly into a wider cache row.
// This is the copy-equivalent of concatenate([first, second], last_axis)
// followed by copy_rows_into, without materializing the concatenated tensor.
// The destination is bound as both an input and preallocated output so the
// command encoder tracks its read-after-write dependency.
        const uint width_a = first_shape[2];
        const uint width_b = second_shape[2];
        const uint width = width_a + width_b;
        const uint rows = first_shape[0];
        const uint steps = first_shape[1];
        const uint cap = dst_shape[1];
        const uint g = thread_position_in_grid.x;
        const uint d = g % width;
        const uint step = (g / width) % steps;
        const uint row = g / (width * steps);
        if (row >= rows) return;
        if (d < width_a) {
            const uint src = row * first_strides[0] +
                step * first_strides[1] + d * first_strides[2];
            out[(row * cap + (uint)off + step) * width + d] = first[src];
        } else {
            const uint col = d - width_a;
            const uint src = row * second_strides[0] +
                step * second_strides[1] + col * second_strides[2];
            out[(row * cap + (uint)off + step) * width + d] = second[src];
        }
