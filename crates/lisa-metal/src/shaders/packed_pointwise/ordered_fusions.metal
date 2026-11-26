// Strided F32 concatenation as one full writer; integer copies preserve
// every F32 bit.
#include <metal_stdlib>
using namespace metal;

struct ConcatArgs { uint rank, axis, inputs, count; uint shape[3], ends[8]; ulong strides[24]; };
kernel void lisa_concat_f32(
    constant ConcatArgs & a [[buffer(0)]],
    device const float * x0 [[buffer(1)]], device const float * x1 [[buffer(2)]],
    device const float * x2 [[buffer(3)]], device const float * x3 [[buffer(4)]],
    device const float * x4 [[buffer(5)]], device const float * x5 [[buffer(6)]],
    device const float * x6 [[buffer(7)]], device const float * x7 [[buffer(8)]],
    device float * y [[buffer(9)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    uint coord[3] = {0,0,0}, rem = i;
    for (int d=int(a.rank)-1;d>=0;--d) {coord[d]=rem%a.shape[d]; rem/=a.shape[d];}
    uint input=0, start=0;
    while (coord[a.axis]>=a.ends[input]) {start=a.ends[input]; ++input;}
    coord[a.axis]-=start;
    ulong offset=0;
    for (uint d=0;d<a.rank;++d) offset+=coord[d]*a.strides[input*3+d];
    device const float * sources[8]={x0,x1,x2,x3,x4,x5,x6,x7};
    // Integer copy preserves every F32 bit, including signed zero and NaNs.
    ((device uint *)y)[i]=((device const uint *)sources[input])[offset];
}
