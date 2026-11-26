
constant bool align_M [[function_constant(200)]];
constant bool align_N [[function_constant(201)]];
constant bool align_K [[function_constant(202)]];

using namespace metal;

#define MLX_MTL_CONST static constant constexpr const

MLX_MTL_CONST int SIMD_SIZE = 32;
MLX_MTL_CONST int QUAD_SIZE = 4;

template <int bits, int wsize = 8>
inline constexpr short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector(const device T* x, thread U* x_thread) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < values_per_thread; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < values_per_thread; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  return sum;
}

template <typename T, typename U, int values_per_thread, int bits>
inline U load_vector_safe(const device T* x, thread U* x_thread, int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U sum = 0;

  if (bits == 2) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 4.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 64.0f;
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];

      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 8.0f;
      x_thread[i + 2] = x[i + 2] / 64.0f;
      x_thread[i + 3] = x[i + 3] / 2.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 128.0f;
      x_thread[i + 6] = x[i + 6] / 4.0f;
      x_thread[i + 7] = x[i + 7] / 32.0f;
    }
  }

  else if (bits == 4) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 16.0f;
      x_thread[i + 2] = x[i + 2] / 256.0f;
      x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < N; i += 8) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3] + x[i + 4] + x[i + 5] +
          x[i + 6] + x[i + 7];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 32.0f;
      x_thread[i + 2] = x[i + 2] / 4.0f;
      x_thread[i + 3] = x[i + 3] / 128.0f;
      x_thread[i + 4] = x[i + 4] / 16.0f;
      x_thread[i + 5] = x[i + 5] / 2.0f;
      x_thread[i + 6] = x[i + 6] / 64.0f;
      x_thread[i + 7] = x[i + 7] / 8.0f;
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < N; i += 4) {
      sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
      x_thread[i] = x[i];
      x_thread[i + 1] = x[i + 1] / 64.0f;
      x_thread[i + 2] = x[i + 2] / 16.0f;
      x_thread[i + 3] = x[i + 3] / 4.0f;
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      sum += x[i];
      x_thread[i] = x[i];
    }
  }

  for (int i = N; i < values_per_thread; i++) {
    x_thread[i] = 0;
  }

  return sum;
}

template <typename U, int values_per_thread, int bits>
inline U qdot(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (values_per_thread / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline U qdot_safe(
    const device uint8_t* w,
    const thread U* x_thread,
    U scale,
    U bias,
    U sum,
    int N) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  U accum = 0;

  if (bits == 2) {
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (w[i] & 0x03) +
           x_thread[4 * i + 1] * (w[i] & 0x0c) +
           x_thread[4 * i + 2] * (w[i] & 0x30) +
           x_thread[4 * i + 3] * (w[i] & 0xc0));
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 3 * i;

      accum += (w[0] & 0x07) * x_thread[0];
      accum += (w[0] & 0x38) * x_thread[1];
      accum += (w[0] & 0xc0) * x_thread[2];
      accum += (w[1] & 0x01) * (x_thread[2] * 256.0f);

      accum += (w[1] & 0x0e) * x_thread[3];
      accum += (w[1] & 0x70) * x_thread[4];
      accum += (w[1] & 0x80) * x_thread[5];
      accum += (w[2] & 0x03) * (x_thread[5] * 256.0f);

      accum += (w[2] & 0x1c) * x_thread[6];
      accum += (w[2] & 0xe0) * x_thread[7];
    }
  }

  else if (bits == 4) {
    const device uint16_t* ws = (const device uint16_t*)w;
    for (int i = 0; i < (N / 4); i++) {
      accum +=
          (x_thread[4 * i] * (ws[i] & 0x000f) +
           x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
           x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
           x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (N / 8); i++) {
      x_thread += 8 * i;
      w += 5 * i;

      accum += (w[0] & 0x1f) * x_thread[0];
      accum += (w[0] & 0xe0) * x_thread[1];
      accum += (w[1] & 0x3) * (x_thread[1] * 256.0f);
      accum += (w[1] & 0x7c) * x_thread[2];
      accum += (w[1] & 0x80) * x_thread[3];
      accum += (w[2] & 0xf) * (x_thread[3] * 256.0f);
      accum += (w[2] & 0xf0) * x_thread[4];
      accum += (w[3] & 0x1) * (x_thread[4] * 256.0f);
      accum += (w[3] & 0x3e) * x_thread[5];
      accum += (w[3] & 0xc0) * x_thread[6];
      accum += (w[4] & 0x7) * (x_thread[6] * 256.0f);
      accum += (w[4] & 0xf8) * x_thread[7];
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (N / 4); i++) {
      x_thread += 4 * i;
      w += 3 * i;

      accum += (w[0] & 0x3f) * x_thread[0];

      accum += (w[0] & 0xc0) * x_thread[1];
      accum += (w[1] & 0x0f) * (x_thread[1] * 256.0f);

      accum += (w[1] & 0xf0) * x_thread[2];
      accum += (w[2] & 0x03) * (x_thread[2] * 256.0f);

      accum += (w[2] & 0xfc) * x_thread[3];
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < N; i++) {
      accum += x_thread[i] * w[i];
    }
  }

  return scale * accum + sum * bias;
}

template <typename U, int values_per_thread, int bits>
inline void
qouter(const thread uint8_t* w, U x, U scale, U bias, thread U* result) {
  static_assert(
      bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 ||
          bits == 8,
      "Template undefined for bits not in {2, 3, 4, 5, 6, 8}");

  if (bits == 2) {
    U s[4] = {scale, scale / 4.0f, scale / 16.0f, scale / 64.0f};
    for (int i = 0; i < (values_per_thread / 4); i++) {
      result[4 * i] += x * (s[0] * (w[i] & 0x03) + bias);
      result[4 * i + 1] += x * (s[1] * (w[i] & 0x0c) + bias);
      result[4 * i + 2] += x * (s[2] * (w[i] & 0x30) + bias);
      result[4 * i + 3] += x * (s[3] * (w[i] & 0xc0) + bias);
    }
  }

  else if (bits == 3) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[8 * i] += x * ((w0 & 0x7) * scale + bias);
      result[8 * i + 1] += x * (((w0 & 0x38) >> 3) * scale + bias);
      result[8 * i + 2] +=
          x * ((((w0 & 0xc0) >> 6) + ((w1 & 0x1) << 2)) * scale + bias);
      result[8 * i + 3] += x * (((w1 & 0xe) >> 1) * scale + bias);
      result[8 * i + 4] += x * (((w1 & 0x70) >> 4) * scale + bias);
      result[8 * i + 5] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0x3) << 1)) * scale + bias);
      result[8 * i + 6] += x * (((w2 & 0x1c) >> 2) * scale + bias);
      result[8 * i + 7] += x * (((w2 & 0xe0) >> 5) * scale + bias);
    }
  }

  else if (bits == 4) {
    U s[2] = {scale, scale / 16.0f};
    for (int i = 0; i < (values_per_thread / 2); i++) {
      result[2 * i] += x * (s[0] * (w[i] & 0x0f) + bias);
      result[2 * i + 1] += x * (s[1] * (w[i] & 0xf0) + bias);
    }
  }

  else if (bits == 5) {
    for (int i = 0; i < (values_per_thread / 8); i++) {
      uint8_t w0 = w[5 * i];
      uint8_t w1 = w[5 * i + 1];
      uint8_t w2 = w[5 * i + 2];
      uint8_t w3 = w[5 * i + 3];
      uint8_t w4 = w[5 * i + 4];
      result[8 * i] += x * ((w0 & 0x1f) * scale + bias);
      result[8 * i + 1] +=
          x * ((((w0 & 0xe0) >> 5) + ((w1 & 0x3) << 3)) * scale + bias);
      result[8 * i + 2] += x * (((w1 & 0x7c) >> 2) * scale + bias);
      result[8 * i + 3] +=
          x * ((((w1 & 0x80) >> 7) + ((w2 & 0xf) << 1)) * scale + bias);
      result[8 * i + 4] +=
          x * ((((w2 & 0xf0) >> 4) + ((w3 & 0x1) << 4)) * scale + bias);
      result[8 * i + 5] += x * (((w3 & 0x3e) >> 1) * scale + bias);
      result[8 * i + 6] +=
          x * ((((w3 & 0xc0) >> 6) + ((w4 & 0x7) << 2)) * scale + bias);
      result[8 * i + 7] += x * (((w4 & 0xf8) >> 3) * scale + bias);
    }
  }

  else if (bits == 6) {
    for (int i = 0; i < (values_per_thread / 4); i++) {
      uint8_t w0 = w[3 * i];
      uint8_t w1 = w[3 * i + 1];
      uint8_t w2 = w[3 * i + 2];

      result[4 * i] += x * ((w0 & 0x3f) * scale + bias);
      result[4 * i + 1] +=
          x * ((((w0 >> 6) & 0x03) + ((w1 & 0x0f) << 2)) * scale + bias);
      result[4 * i + 2] +=
          x * ((((w1 >> 4) & 0x0f) + ((w2 & 0x03) << 4)) * scale + bias);
      result[4 * i + 3] += x * (((w2 >> 2) & 0x3f) * scale + bias);
    }
  }

  else if (bits == 8) {
    for (int i = 0; i < values_per_thread; i++) {
      result[i] += x * (scale * w[i] + bias);
    }
  }
}

