// Original packed NVFP4 storage; GGML format attribution and license are in
// crates/lisa-gguf/src/nvfp4_provenance.json and its adjacent LICENSE-GGML.
// Each lane accumulates columns lane + 32*k using explicit FMA, followed by
// SIMD reduction. Dense, grouped and expert calls share this arithmetic law.
#include <metal_stdlib>
using namespace metal;
struct ds4_metal_args_mul_mv {
    int ne00;
    int ne01;
    int ne02;
    ulong nb00;
    ulong nb01;
    ulong nb02;
    ulong nb03;
    int ne10;
    int ne11;
    int ne12;
    ulong nb10;
    ulong nb11;
    ulong nb12;
    ulong nb13;
    int ne0;
    int ne1;
    int nr0;
    short r2;
    short r3;
};
struct ds4_metal_args_mul_mv_id {
    int32_t  nei0;
    int32_t  nei1;
    uint64_t nbi1;
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t  ne0;
    int32_t  ne1;
    uint64_t nb1;
    int32_t  nr0;
    /* Tensor-parallel expert ownership: when tp_world > 1, routed-MoE
     * kernels skip experts outside this rank's contiguous id range so this
     * rank never touches their weights. Zero-initialized (tp_world 0) means
     * no split; only the routed id kernels read these. */
    int32_t  tp_rank;
    int32_t  tp_world;
    /* When set, the sum6 kernels add this per-row vector (the shared-expert
     * partial) into their accumulation, folding the TP local-add. */
    int32_t  tp_addend;
    /* First expert id present at the bound blob base (TP mappings bind only
     * the owned range; kernels rebase before pointer math). */
    int32_t  tp_expert_base;
};

constant float nv_values[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};
// Every native E4M3 scale is exactly representable in F16. A 512-byte table
// avoids rebuilding its exponent in every dot-product lane. Preserve all 256
// byte patterns, including the native zero treatment of 0x7f. Products and FMA
// accumulators remain F32. ordered_nvfp4_scale_bits_cover_every_encoding pins
// the complete scale/nibble domain against the independent CPU expression.
constant ushort nv_scale_bits[256] = {
    0x0000, 0x1400, 0x1800, 0x1a00, 0x1c00, 0x1d00, 0x1e00, 0x1f00,
    0x2000, 0x2080, 0x2100, 0x2180, 0x2200, 0x2280, 0x2300, 0x2380,
    0x2400, 0x2480, 0x2500, 0x2580, 0x2600, 0x2680, 0x2700, 0x2780,
    0x2800, 0x2880, 0x2900, 0x2980, 0x2a00, 0x2a80, 0x2b00, 0x2b80,
    0x2c00, 0x2c80, 0x2d00, 0x2d80, 0x2e00, 0x2e80, 0x2f00, 0x2f80,
    0x3000, 0x3080, 0x3100, 0x3180, 0x3200, 0x3280, 0x3300, 0x3380,
    0x3400, 0x3480, 0x3500, 0x3580, 0x3600, 0x3680, 0x3700, 0x3780,
    0x3800, 0x3880, 0x3900, 0x3980, 0x3a00, 0x3a80, 0x3b00, 0x3b80,
    0x3c00, 0x3c80, 0x3d00, 0x3d80, 0x3e00, 0x3e80, 0x3f00, 0x3f80,
    0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380,
    0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780,
    0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80,
    0x4c00, 0x4c80, 0x4d00, 0x4d80, 0x4e00, 0x4e80, 0x4f00, 0x4f80,
    0x5000, 0x5080, 0x5100, 0x5180, 0x5200, 0x5280, 0x5300, 0x5380,
    0x5400, 0x5480, 0x5500, 0x5580, 0x5600, 0x5680, 0x5700, 0x5780,
    0x5800, 0x5880, 0x5900, 0x5980, 0x5a00, 0x5a80, 0x5b00, 0x0000,
    0x0000, 0x1400, 0x1800, 0x1a00, 0x1c00, 0x1d00, 0x1e00, 0x1f00,
    0x2000, 0x2080, 0x2100, 0x2180, 0x2200, 0x2280, 0x2300, 0x2380,
    0x2400, 0x2480, 0x2500, 0x2580, 0x2600, 0x2680, 0x2700, 0x2780,
    0x2800, 0x2880, 0x2900, 0x2980, 0x2a00, 0x2a80, 0x2b00, 0x2b80,
    0x2c00, 0x2c80, 0x2d00, 0x2d80, 0x2e00, 0x2e80, 0x2f00, 0x2f80,
    0x3000, 0x3080, 0x3100, 0x3180, 0x3200, 0x3280, 0x3300, 0x3380,
    0x3400, 0x3480, 0x3500, 0x3580, 0x3600, 0x3680, 0x3700, 0x3780,
    0x3800, 0x3880, 0x3900, 0x3980, 0x3a00, 0x3a80, 0x3b00, 0x3b80,
    0x3c00, 0x3c80, 0x3d00, 0x3d80, 0x3e00, 0x3e80, 0x3f00, 0x3f80,
    0x4000, 0x4080, 0x4100, 0x4180, 0x4200, 0x4280, 0x4300, 0x4380,
    0x4400, 0x4480, 0x4500, 0x4580, 0x4600, 0x4680, 0x4700, 0x4780,
    0x4800, 0x4880, 0x4900, 0x4980, 0x4a00, 0x4a80, 0x4b00, 0x4b80,
    0x4c00, 0x4c80, 0x4d00, 0x4d80, 0x4e00, 0x4e80, 0x4f00, 0x4f80,
    0x5000, 0x5080, 0x5100, 0x5180, 0x5200, 0x5280, 0x5300, 0x5380,
    0x5400, 0x5480, 0x5500, 0x5580, 0x5600, 0x5680, 0x5700, 0x5780,
    0x5800, 0x5880, 0x5900, 0x5980, 0x5a00, 0x5a80, 0x5b00, 0x5b80,
};
inline float nv_value(device const uchar * row, uint col) {
    device const uchar * b = row + (col / 64) * 36;
    uint sub = (col % 64) / 16, j = col % 16;
    uchar s = b[sub];
    float d = float(as_type<half>(nv_scale_bits[s]));
    uint q = (b[4 + sub*8 + j%8] >> (j < 8 ? 0 : 4)) & 15;
    return nv_values[q] * d;
}
// Keep each lane's original column/FMA sequence. Four independent rows are
// interleaved to reuse each activation load and hide packed-load latency.
inline float4 nv_dot4(device const uchar * row, ulong stride, device const float * x, uint cols, uint lane) {
    float4 a = 0.0f;
    for (uint c = lane; c < cols; c += 32) {
        float v = x[c];
        float4 w = float4(nv_value(row,c), nv_value(row+stride,c),
                          nv_value(row+2*stride,c), nv_value(row+3*stride,c));
        a = fma(w,float4(v),a);
    }
    return simd_sum(a);
}
kernel void lisa_nvfp4_mv(
    constant ds4_metal_args_mul_mv & a [[buffer(0)]],
    device const uchar * w [[buffer(1)]], device const float * x [[buffer(2)]],
    device float * y [[buffer(3)]], uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]]) {
    uint first = group.x*8 + simd*4;
    device const float * input = x + (group.z*a.ne11 + group.y)*a.ne00;
    float4 v = nv_dot4(w + group.z*a.nb02 + first*a.nb01, a.nb01, input, a.ne00, lane);
    if (lane == 0) for (uint r=0;r<4;++r) y[(group.z*a.ne11+group.y)*a.ne01+first+r]=v[r];
}
kernel void lisa_nvfp4_id(
    constant ds4_metal_args_mul_mv_id & a [[buffer(0)]],
    device const uchar * w [[buffer(1)]], device const float * x [[buffer(2)]],
    device float * y [[buffer(3)]], device const int * ids [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    uint token = group.z/a.nei0, slot = group.z%a.nei0, first = group.x*8 + simd*4;
    int expert = ids[group.z];
    device const float * input = x + (token*a.ne11 + slot%a.ne11)*a.ne00;
    float4 v = 0.0f;
    if (expert >= 0 && expert < a.ne02)
        v = nv_dot4(w + ulong(expert)*a.nb02 + first*a.nb01, a.nb01, input, a.ne00, lane);
    if (lane == 0) for (uint r=0;r<4;++r) y[group.z*a.ne01+first+r]=v[r];
}

// Stable GPU-only routing tiles. A tile contains four independent token/routes
// selecting the same expert. Invalid IDs form a separate zero-writing bucket.
kernel void lisa_nvfp4_route_tiles(
    constant uint4 & a [[buffer(0)]], device const int * ids [[buffer(1)]],
    device int * tiles [[buffer(2)]], uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint prefix[512];
    uint count = 0;
    if (tid <= a.y) for (uint r=0; r<a.x; ++r) {
        int id = ids[r];
        uint bucket = id >= 0 && uint(id) < a.y ? uint(id) : a.y;
        count += bucket == tid;
    }
    uint runs = (count + 3) / 4;
    prefix[tid] = runs;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step=1; step<a.w; step*=2) {
        uint left = tid >= step ? prefix[tid-step] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[tid] += left;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i=tid; i<a.z*5; i+=a.w) tiles[i] = -1;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid > a.y || count == 0) return;
    uint begin = prefix[tid] - runs, ordinal = 0;
    for (uint r=0; r<a.x; ++r) {
        int id = ids[r];
        uint bucket = id >= 0 && uint(id) < a.y ? uint(id) : a.y;
        if (bucket == tid) {
            uint at = (begin + ordinal/4)*5;
            tiles[at] = int(tid);
            tiles[at + 1 + ordinal%4] = int(r);
            ++ordinal;
        }
    }
}

// Decode each packed value once for up to four routed inputs. Each accumulator
// retains nv_dot4's lane + 32*k FMA sequence and the same SIMD reduction.
kernel void lisa_nvfp4_id_tiles(
    constant ds4_metal_args_mul_mv_id & a [[buffer(0)]],
    device const uchar * w [[buffer(1)]], device const float * x [[buffer(2)]],
    device float * y [[buffer(3)]], device const int * tiles [[buffer(4)]],
    uint3 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]]) {
    device const int * tile = tiles + group.z*5;
    if (tile[1] < 0) return;
    uint first = group.x*8 + simd*4;
    float4 acc[4] = {float4(0),float4(0),float4(0),float4(0)};
    if (tile[0] < a.ne02) {
        device const uchar * row = w + ulong(tile[0])*a.nb02 + first*a.nb01;
        ulong offsets[4];
        for(uint t=0;t<4;++t) {
            uint route = uint(max(tile[t+1],0));
            offsets[t] = (ulong(route/a.nei0)*a.ne11 + (route%a.nei0)%a.ne11)*a.ne00;
        }
        for(uint c=lane;c<uint(a.ne00);c+=32) {
            float4 weights = float4(nv_value(row,c),nv_value(row+a.nb01,c),
                nv_value(row+2*a.nb01,c),nv_value(row+3*a.nb01,c));
            for(uint t=0;t<4;++t) if(tile[t+1]>=0) acc[t] = fma(weights,float4(x[offsets[t]+c]),acc[t]);
        }
    }
    for(uint t=0;t<4;++t) {
        float4 sum = simd_sum(acc[t]);
        if(lane==0 && tile[t+1]>=0) for(uint c=0;c<4;++c)
            y[ulong(tile[t+1])*a.ne01+first+c]=sum[c];
    }
}
