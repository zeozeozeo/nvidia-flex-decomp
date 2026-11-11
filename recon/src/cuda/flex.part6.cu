// FleX 1.2.0 CUDA solver — device kernels, slice 2 of the flex.cu split.
//
// Reconstruction evidence for every function: the shipped device PTX
// (work/binary/gpu/dbg/nvflex.flex/sm_30.ptx), the DWARF signature and the
// decompiled host launch site (work/decomp/nvflex.flex/cudaflex.c).  Each
// function carries the original flex.cu line recorded by DWARF.
//
// recovered.h redeclares CUDA's own dim3/uint3 and the cuda*_t typedefs, so
// CUDA is included first and the recovered duplicates are renamed/suppressed
// here instead of in the shared header.
#include <cuda_runtime.h>
#define FLEXDECOMP_DEF_CUcontext
#define FLEXDECOMP_DEF_cudaArray_t
#define FLEXDECOMP_DEF_cudaEvent_t
#define FLEXDECOMP_DEF_cudaGraphicsResource_t
#define FLEXDECOMP_DEF_cudaTextureObject_t
#define FLEXDECOMP_DEF_uint32_t
#define dim3 flex_dim3_recovered
#define uint3 flex_uint3_recovered
#include "nvflex/all.h"
#undef dim3
#undef uint3

// ---------------------------------------------------------------------------
// small device helpers shared by the kernels below
// ---------------------------------------------------------------------------

__device__ __forceinline__ int flexThreadIndex()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ float4 loadVec4(const float* p, int i)
{
    return *reinterpret_cast<const float4*>(p + 4 * i);
}

__device__ __forceinline__ void storeVec4(float* p, int i, float4 v)
{
    *reinterpret_cast<float4*>(p + 4 * i) = v;
}

// v = a +/- b component-wise (w included)
__device__ __forceinline__ float4 sub4(float4 a, float4 b)
{
    float4 r;
    r.x = a.x - b.x;
    r.y = a.y - b.y;
    r.z = a.z - b.z;
    r.w = a.w - b.w;
    return r;
}

__device__ __forceinline__ float4 scale4(float4 a, float s)
{
    float4 r;
    r.x = a.x * s;
    r.y = a.y * s;
    r.z = a.z * s;
    r.w = a.w * s;
    return r;
}

// ---------------------------------------------------------------------------
// kernels, in original flex.cu order
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/flex.cu:2075 _Z16UpdateVelocitiesi8Vec3IterS_S_f
// i = blockIdx.x*blockDim.x + threadIdx.x; if (i < numParticles)
//   p3[i] = (p2[i] - p1[i]) * s                    (PTX 2900..2949)
__global__ void UpdateVelocities(int numParticles, Vec3Iter p1, Vec3Iter p2, Vec3Iter p3, float s)
{
    int i = flexThreadIndex();
    if (i < numParticles) {
        float4 a = loadVec4(p1.x, i);
        float4 b = loadVec4(p2.x, i);
        storeVec4(p3.x, i, scale4(sub4(b, a), s));
    }
}

// flexdecomp: src/cuda/flex.cu:3008 _Z25ClampDiffuseParticleCountPiS_i
// *p0 = min(*p0, limit); *p1 = 0                    (PTX 7114..7137)
__global__ void ClampDiffuseParticleCount(int* count, int* overflow, int limit)
{
    *count = min(*count, limit);
    *overflow = 0;
}

// flexdecomp: src/cuda/flex.cu:3014 _Z23CompactDiffuseParticlesPK8XVector4IfES2_PS0_S3_PKiPi
// if (i < *numParticles && p0[i].w > 0)
//   { n = atomicAdd(p5, 1); p2[n] = p0[i]; p3[n] = p1[i]; }   (PTX 7138..7206)
__global__ void CompactDiffuseParticles(const Vec4* inPositions, const Vec4* inVelocities,
                                        Vec4* outPositions, Vec4* outVelocities,
                                        const int* numParticles, int* outCount)
{
    int i = flexThreadIndex();
    if (i >= *numParticles)
        return;

    float4 p = loadVec4(reinterpret_cast<const float*>(inPositions), i);
    if (!(p.w > 0.0f))
        return;

    int n = atomicAdd(outCount, 1);
    float4 v = loadVec4(reinterpret_cast<const float*>(inVelocities), i);
    storeVec4(reinterpret_cast<float*>(outPositions), n, p);
    storeVec4(reinterpret_cast<float*>(outVelocities), n, v);
}
