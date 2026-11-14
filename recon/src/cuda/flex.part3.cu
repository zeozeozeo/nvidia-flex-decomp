// nvflex.flex host slice 3: public solver API (flex.cu:4865-5790).
//
// Reconstructed from:
//   work/decomp/nvflex.flex/cudaflex.c   (Ghidra pseudocode, authoritative)
//   work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt
//   work/dwarf/nvflex.flex.json          (line numbers, signatures)
//
// Private counterparts of the anonymous-namespace buffer helpers live in this
// file because the originals have internal linkage in flex.cu.  The original
// helper units stay owned by the flex-host-1 slice; these copies exist only so
// this translation unit can call them.

#include "nvflex/all.h"
#include "NvFlex.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <vector>

// ---------------------------------------------------------------------------
// Backend helpers defined in util.cpp / sort.cu / bvh.cu / flex.cu.
// DWARF decl sites: src/cuda/util.h, src/cuda/sort.h.
// ---------------------------------------------------------------------------
void FlexLogFunc(NvFlexErrorCallback func, NvFlexErrorSeverity severity,
                 const char* msg, const char* file, int line, ...);
void* FlexAlloc(NvFlexLibrary* lib, int size);
void FlexDeviceMallocInternal(void** ptr, size_t size, const char* name,
                              const char* file, int line);
void FlexDeviceFreeInternal(void* ptr);
cudaTextureObject_t CreateTexture(const Vec4* data, int size);
cudaTextureObject_t CreateTexture(const int* data, int size);
void DestroyTexture(cudaTextureObject_t tex);
void SortReset(NvFlexLibrary* lib);
void FreeBVH(BVH& bvh);
void NvFlexSetFence(NvFlexLibrary* lib);

// __global__ launch defined in flex.cu:680; nvcc emits its host stub with the
// same mangled name, so a device launch here reproduces the original
// __cudaPushCallConfiguration + _Z17InitActiveIndicesiPi call pair.
__global__ void InitActiveIndices(int n, int* indices);

// cudaflex.o references seed1/seed2 as undefined data (cudaflex.nm.txt).  They
// are the module's PRNG state globals; the reconstruction owns their single
// definition (the parity link supplied them from data_stubs.c).
unsigned int seed1 = 0;
unsigned int seed2 = 0;

namespace {

const char* const kFlexFile = "../../../src/cuda/flex.cu";

// Recovered layouts keep these members opaque; the accessors restore the
// DWARF-recorded types without changing the byte layout.
inline NvFlexErrorCallback LibErrorFunc(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlexErrorCallback*>(lib->mErrorFunc);
}

inline NvFlexSolverDesc* SolverDesc(NvFlexSolver* s)
{
    return reinterpret_cast<NvFlexSolverDesc*>(s->mDesc);
}

inline NvFlexParams* SolverParams(NvFlexSolver* s)
{
    return reinterpret_cast<NvFlexParams*>(s->mParams);
}

typedef std::vector<NvFlexSolver*> SolverVector;

inline SolverVector& SolverList(NvFlexLibrary* lib)
{
    return *reinterpret_cast<SolverVector*>(lib->mResourceTracker.mSolver);
}

inline void* ByteOffset(void* ptr, int byteOffset)
{
    return static_cast<char*>(ptr) + byteOffset;
}

inline const void* ByteOffset(const void* ptr, int byteOffset)
{
    return static_cast<const char*>(ptr) + byteOffset;
}

// cudaMemcpyKind for device-side source buffers: host-mappable buffers are the
// only kind that maps to a host direction; type 3 is a graphics resource
// (flex.cu:5210-5290).
inline cudaMemcpyKind CopyKind(NvFlexBuffer* buffer, bool toDevice)
{
    if (buffer->type == 0)
        return toDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    return cudaMemcpyDeviceToDevice;
}

// Original anonymous-namespace spelling, flex.cu:5210 (ReadData, 5 args).
#if !defined(_MSC_VER)
__attribute__((noinline))
#endif
void ReadData(NvFlexBuffer* dstBuffer, int dstByteOffset, const void* srcPtr,
              int srcByteOffset, int byteSize)
{
    NvFlexLibrary* lib = dstBuffer->lib;

    if (dstBuffer->mapped)
    {
        if (LibErrorFunc(lib))
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError,
                        "Reading from a mapped buffer, buffers must be unmapped before being passed to flexGet*() methods",
                        kFlexFile, 5210, "");
        return;
    }

    cudaMemcpyKind kind = CopyKind(dstBuffer, false);
    if (dstBuffer->type == 3)
    {
        size_t s;
        cudaError_t err = cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&dstBuffer->graphicsResource), 0);
        if (err != cudaSuccess && LibErrorFunc(lib))
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexFile, 5226, (unsigned long)err,
                        "cudaGraphicsMapResources(1, &dstBuffer->graphicsResource)");
        err = cudaGraphicsResourceGetMappedPointer(&dstBuffer->ptr, &s, reinterpret_cast<cudaGraphicsResource_t>(dstBuffer->graphicsResource));
        if (err != cudaSuccess && LibErrorFunc(lib))
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexFile, 5227, (unsigned long)err,
                        "cudaGraphicsResourceGetMappedPointer(&dstBuffer->ptr, &s, reinterpret_cast<cudaGraphicsResource_t>(dstBuffer->graphicsResource))");
        kind = cudaMemcpyDeviceToDevice;
    }

    cudaError_t err = cudaMemcpyAsync(ByteOffset(dstBuffer->ptr, dstByteOffset),
                                      ByteOffset(srcPtr, srcByteOffset),
                                      byteSize, kind, 0);
    if (err != cudaSuccess && LibErrorFunc(lib))
        FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                    kFlexFile, 5234, (unsigned long)err,
                    "cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset), ByteOffset(srcPtr, srcByteOffset), byteSize, kind)");

    if (dstBuffer->type == 3)
    {
        err = cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&dstBuffer->graphicsResource), 0);
        if (err != cudaSuccess && LibErrorFunc(lib))
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexFile, 5239, (unsigned long)err,
                        "cudaGraphicsUnmapResources(1, &dstBuffer->graphicsResource)");
    }

    NvFlexSetFence(lib);
}

// Original anonymous-namespace spelling, flex.cu:5251 (WriteData, 5 args).
#if !defined(_MSC_VER)
__attribute__((noinline))
#endif
void WriteData(void* dstPtr, int dstByteOffset, NvFlexBuffer* srcBuffer,
               int srcByteOffset, int byteSize)
{
    NvFlexLibrary* lib = srcBuffer->lib;

    if (!srcBuffer->mapped)
    {
        cudaError_t err = cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset),
                                          ByteOffset(srcBuffer->ptr, srcByteOffset),
                                          byteSize, CopyKind(srcBuffer, true), 0);
        if (err != cudaSuccess && LibErrorFunc(lib))
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexFile, 5270, (unsigned long)err,
                        "cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset), ByteOffset(srcPtr, srcByteOffset), byteSize, kind, 0)");
        NvFlexSetFence(lib);
        return;
    }

    if (LibErrorFunc(lib))
        FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError,
                    "Writing to a mapped buffer, buffers must be unmapped before being passed to flexSet*() methods",
                    kFlexFile, 5251, "");
}

// flex.cu:5297 (ReadData, 6 args).
#if !defined(_MSC_VER)
__attribute__((noinline))
#endif
void ReadData(NvFlexErrorCallback errorFunc, NvFlexBuffer* dstBuffer, const void* srcPtr,
              int srcLength, const NvFlexCopyDesc* desc, int elementWidth)
{
    if (desc == 0)
    {
        int n = dstBuffer->elementCount;
        if (srcLength <= dstBuffer->elementCount)
            n = srcLength;
        ReadData(dstBuffer, 0, srcPtr, 0, n * elementWidth);
        return;
    }

    const int n = desc->elementCount;
    if (dstBuffer->elementCount < desc->dstOffset + n)
    {
        if (errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexGet*() method with destination range outside of destination buffer",
                        kFlexFile, 5303, "");
        return;
    }

    if (n + desc->srcOffset <= srcLength)
    {
        ReadData(dstBuffer, desc->dstOffset * elementWidth, srcPtr,
                 desc->srcOffset * elementWidth, n * elementWidth);
        return;
    }

    if (errorFunc)
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexGet*() method with source range outside of source buffer",
                    kFlexFile, 5309, "");
}

// flex.cu:5338 (WriteData, 6 args).
#if !defined(_MSC_VER)
__attribute__((noinline))
#endif
void WriteData(NvFlexErrorCallback errorFunc, void* dstPtr, int destLength,
               NvFlexBuffer* srcBuffer, const NvFlexCopyDesc* desc, int elementWidth)
{
    if (srcBuffer == 0)
        return;

    if (desc == 0)
    {
        int n = srcBuffer->elementCount;
        if (destLength <= srcBuffer->elementCount)
            n = destLength;
        WriteData(dstPtr, 0, srcBuffer, 0, n * elementWidth);
        return;
    }

    const int n = desc->elementCount;
    if (destLength < desc->dstOffset + n)
    {
        if (errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexSet*() method with destination range outside of destination buffer",
                        kFlexFile, 5334, "");
        return;
    }

    if (n + desc->srcOffset <= srcBuffer->elementCount)
    {
        WriteData(dstPtr, desc->dstOffset * elementWidth, srcBuffer,
                  desc->srcOffset * elementWidth, n * elementWidth);
        return;
    }

    if (errorFunc)
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexSet*() method with source range outside of source buffer",
                    kFlexFile, 5340, "");
}

// reports one CUDA call; the expression is evaluated exactly once and is
// printed verbatim in the log, matching the shipped assert text
#define FLEX_CHECK_CUDA(call, line, text)                                       \
    do {                                                                        \
        cudaError_t _fd_err = (call);                                           \
        if (_fd_err != cudaSuccess && LibErrorFunc(lib))                        \
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", \
                        kFlexFile, (line), (unsigned long)_fd_err, (text));     \
    } while (0)

// same check for an already evaluated cudaError_t
#define FLEX_CHECK_CUDA_ERR(err, line, text)                                    \
    do {                                                                        \
        cudaError_t _fd_err = (err);                                            \
        if (_fd_err != cudaSuccess && LibErrorFunc(lib))                        \
            FlexLogFunc(LibErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", \
                        kFlexFile, (line), (unsigned long)_fd_err, (text));     \
    } while (0)

}  // namespace

// flexdecomp: src/cuda/flex.cu:4865 NvFlexSetSolverDescDefaults
void NvFlexSetSolverDescDefaults(NvFlexSolverDesc* desc)
{
    // default maximums observed in .LC18 of cudaflex.o
    desc->featureMode = eNvFlexFeatureModeDefault;
    desc->maxParticles = 16384;
    desc->maxDiffuseParticles = 0;
    desc->maxNeighborsPerParticle = 96;
    desc->maxContactsPerParticle = 6;
}

// flexdecomp: src/cuda/flex.cu:4874 NvFlexCreateSolver
NvFlexSolver* NvFlexCreateSolver(NvFlexLibrary* lib, const NvFlexSolverDesc* desc)
{
    NvFlexSolver* s = new NvFlexSolver();

    const int maxParticles = desc->maxParticles;
    const int maxDiffuseParticles = desc->maxDiffuseParticles;
    const int maxNeighborsPerParticle = desc->maxNeighborsPerParticle;
    const int maxContactsPerParticle = desc->maxContactsPerParticle;

    s->mLib = lib;
    s->mDebug = false;
    s->numParticles = 0;
    s->mMaxNeighborsPerParticle = maxNeighborsPerParticle;
    s->mMaxContactsPerParticle = maxContactsPerParticle;
    s->mMaxParticles = maxParticles;
    *SolverDesc(s) = *desc;

    const size_t vec4Bytes = (size_t)maxParticles * sizeof(Vec4);
    const size_t vec3Bytes = (size_t)maxParticles * sizeof(Vec3);
    const size_t intBytes = (size_t)maxParticles * sizeof(int);

    FlexDeviceMallocInternal((void**)&s->mActiveIndices, intBytes,
                             "&s->mActiveIndices", kFlexFile, 4895);

    {
        const int numThreads = 1024;
        const int numBlocks = (maxParticles + numThreads - 1) / numThreads;
        InitActiveIndices<<<numBlocks, numThreads>>>(maxParticles, s->mActiveIndices);
    }

    FlexDeviceMallocInternal((void**)&s->mPositions, vec4Bytes, "&s->mPositions", kFlexFile, 4898);
    FLEX_CHECK_CUDA(cudaMemset(s->mPositions, 0, vec4Bytes), 4899,
                    "cudaMemset(s->mPositions, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mNewPositions, vec4Bytes, "&s->mNewPositions", kFlexFile, 4901);
    FLEX_CHECK_CUDA(cudaMemset(s->mNewPositions, 0, vec4Bytes), 4902,
                    "cudaMemset(s->mNewPositions, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mRestPositions, vec4Bytes, "&s->mRestPositions", kFlexFile, 4904);
    FLEX_CHECK_CUDA(cudaMemset(s->mRestPositions, 0, vec4Bytes), 4905,
                    "cudaMemset(s->mRestPositions, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mVelocities, vec3Bytes, "&s->mVelocities", kFlexFile, 4907);
    FLEX_CHECK_CUDA(cudaMemset(s->mVelocities, 0, vec3Bytes), 4908,
                    "cudaMemset(s->mVelocities, 0, maxParticles*sizeof(Vec3))");

    FlexDeviceMallocInternal((void**)&s->mNormals, vec4Bytes, "&s->mNormals", kFlexFile, 4910);
    FLEX_CHECK_CUDA(cudaMemset(s->mNormals, 0, vec4Bytes), 4911,
                    "cudaMemset(s->mNormals, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mDeltas, vec4Bytes, "&s->mDeltas", kFlexFile, 4913);
    FLEX_CHECK_CUDA(cudaMemset(s->mDeltas, 0, vec4Bytes), 4914,
                    "cudaMemset(s->mDeltas, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mParticleNeighbors,
                             (size_t)(maxNeighborsPerParticle * maxParticles) * sizeof(int),
                             "&s->mParticleNeighbors", kFlexFile, 4916);

    FlexDeviceMallocInternal((void**)&s->mSortedPositions, vec4Bytes,
                             "&s->mSortedPositions", kFlexFile, 4918);
    FlexDeviceMallocInternal((void**)&s->mSortedNewPositions, vec4Bytes,
                             "&s->mSortedNewPositions", kFlexFile, 4919);
    FlexDeviceMallocInternal((void**)&s->mSortedNewVelocities, vec4Bytes,
                             "&s->mSortedNewVelocities", kFlexFile, 4920);

    const size_t contactBytes = (size_t)(maxContactsPerParticle * maxParticles) * sizeof(Vec4);
    FlexDeviceMallocInternal((void**)&s->mStaticContactPlanes, contactBytes,
                             "&s->mStaticContactPlanes", kFlexFile, 4922);
    FlexDeviceMallocInternal((void**)&s->mStaticContactVelocities, contactBytes,
                             "&s->mStaticContactVelocities", kFlexFile, 4923);

    FlexDeviceMallocInternal((void**)&s->mStaticContactCounts, intBytes,
                             "&s->mStaticContactCounts", kFlexFile, 4924);
    FLEX_CHECK_CUDA(cudaMemset(s->mStaticContactCounts, 0, intBytes), 4925,
                    "cudaMemset(s->mStaticContactCounts, 0, maxParticles*sizeof(int))");

    FlexDeviceMallocInternal((void**)&s->mDensities, intBytes, "&s->mDensities", kFlexFile, 4927);

    FlexDeviceMallocInternal((void**)&s->mPhases, intBytes, "&s->mPhases", kFlexFile, 4929);
    FLEX_CHECK_CUDA(cudaMemset(s->mPhases, 0, intBytes), 4930,
                    "cudaMemset(s->mPhases, 0, maxParticles*sizeof(int))");

    FlexDeviceMallocInternal((void**)&s->mSortedPhases, intBytes, "&s->mSortedPhases", kFlexFile, 4932);
    FLEX_CHECK_CUDA(cudaMemset(s->mSortedPhases, 0, intBytes), 4933,
                    "cudaMemset(s->mSortedPhases, 0, maxParticles*sizeof(int))");

    FlexDeviceMallocInternal((void**)&s->mPotentials, intBytes, "&s->mPotentials", kFlexFile, 4935);
    FLEX_CHECK_CUDA(cudaMemset(s->mPotentials, 0, intBytes), 4936,
                    "cudaMemset(s->mPotentials, 0, maxParticles*sizeof(float))");

    FlexDeviceMallocInternal((void**)&s->mCurl, vec4Bytes, "&s->mCurl", kFlexFile, 4938);
    FLEX_CHECK_CUDA(cudaMemset(s->mCurl, 0, vec4Bytes), 4939,
                    "cudaMemset(s->mCurl, 0, maxParticles*sizeof(Vec4))");

    FlexDeviceMallocInternal((void**)&s->mAnisotropy1, vec4Bytes, "&s->mAnisotropy1", kFlexFile, 4941);
    FlexDeviceMallocInternal((void**)&s->mAnisotropy2, vec4Bytes, "&s->mAnisotropy2", kFlexFile, 4942);
    FlexDeviceMallocInternal((void**)&s->mAnisotropy3, vec4Bytes, "&s->mAnisotropy3", kFlexFile, 4943);

    // the density array aliases the first anisotropy vector (cudaflex.c:4944)
    s->mSortedDensities = reinterpret_cast<float*>(s->mAnisotropy1);

    std::vector<Vec4> a1, a2, a3;
    if (maxParticles)
    {
        // default anisotropy frames: unit axes scaled by 0.1 (.LC161-163)
        a1.resize(maxParticles, Vec4{1.0f, 0.0f, 0.0f, 0.1f});
        a2.resize(maxParticles, Vec4{0.0f, 1.0f, 0.0f, 0.1f});
        a3.resize(maxParticles, Vec4{0.0f, 0.0f, 1.0f, 0.1f});
    }

    cudaError_t err = cudaMemcpy(s->mAnisotropy1, a1.empty() ? 0 : &a1[0], vec4Bytes,
                                 cudaMemcpyHostToDevice);
    FLEX_CHECK_CUDA_ERR(err, 4953,
                    "cudaMemcpy(s->mAnisotropy1, &a1[0], maxParticles*sizeof(Vec4), cudaMemcpyHostToDevice)");
    err = cudaMemcpy(s->mAnisotropy2, a2.empty() ? 0 : &a2[0], vec4Bytes, cudaMemcpyHostToDevice);
    FLEX_CHECK_CUDA_ERR(err, 4954,
                    "cudaMemcpy(s->mAnisotropy2, &a2[0], maxParticles*sizeof(Vec4), cudaMemcpyHostToDevice)");
    err = cudaMemcpy(s->mAnisotropy3, a3.empty() ? 0 : &a3[0], vec4Bytes, cudaMemcpyHostToDevice);
    FLEX_CHECK_CUDA_ERR(err, 4955,
                    "cudaMemcpy(s->mAnisotropy3, &a3[0], maxParticles*sizeof(Vec4), cudaMemcpyHostToDevice)");

    FlexDeviceMallocInternal((void**)&s->mParticleLower, sizeof(Vec3), "&s->mParticleLower", kFlexFile, 4957);
    FLEX_CHECK_CUDA(cudaMemset(s->mParticleLower, 0, sizeof(Vec3)), 4958,
                    "cudaMemset(s->mParticleLower, 0, sizeof(Vec3))");

    FlexDeviceMallocInternal((void**)&s->mParticleUpper, sizeof(Vec3), "&s->mParticleUpper", kFlexFile, 4960);
    FLEX_CHECK_CUDA(cudaMemset(s->mParticleUpper, 0, sizeof(Vec3)), 4961,
                    "cudaMemset(s->mParticleUpper, 0, sizeof(Vec3))");

    FlexDeviceMallocInternal((void**)&s->mCellIds, (size_t)maxParticles * 8, "&s->mCellIds", kFlexFile, 4963);
    FlexDeviceMallocInternal((void**)&s->mIndices, (size_t)maxParticles * 8, "&s->mIndices", kFlexFile, 4964);

    FlexDeviceMallocInternal((void**)&s->mReverseLookup, intBytes, "&s->mReverseLookup", kFlexFile, 4966);
    FLEX_CHECK_CUDA(cudaMemset(s->mReverseLookup, 0, intBytes), 4967,
                    "cudaMemset(s->mReverseLookup, 0, maxParticles*sizeof(int))");

    FlexDeviceMallocInternal((void**)&s->mParticleNeighborCounts, intBytes,
                             "&s->mParticleNeighborCounts", kFlexFile, 4969);
    FlexDeviceMallocInternal((void**)&s->mPlanes, 128, "&s->mPlanes", kFlexFile, 4971);
    FlexDeviceMallocInternal((void**)&s->mRandom, 1024, "&s->mRandom", kFlexFile, 4974);

    {
        const int kRandomTableSize = 256;
        std::vector<float> random(kRandomTableSize);

        unsigned int s1 = seed1;
        unsigned int s2 = seed2;
        for (int i = 0; i < kRandomTableSize; ++i)
        {
            s1 = (s1 >> 27 | s1 << 5) ^ s1 * s2 ^ s2;
            s2 = (s2 >> 20 | s2 << 12) ^ s1;
            // 2 * (uint32 / 2^32) - 1
            float t = (float)s1 * 2.3283064e-10f;
            random[i] = t + t - 1.0f;
        }
        seed1 = s1;
        seed2 = s2;

        err = cudaMemcpy(s->mRandom, &random[0], sizeof(float) * kRandomTableSize,
                         cudaMemcpyHostToDevice);
        FLEX_CHECK_CUDA_ERR(err, 4978,
                        "cudaMemcpy(s->mRandom, &random[0], sizeof(float)*kRandomTableSize, cudaMemcpyHostToDevice)");
    }

    if (maxDiffuseParticles != 0)
    {
        const size_t diffuseBytes = (size_t)maxDiffuseParticles * sizeof(Vec4);

        FlexDeviceMallocInternal((void**)&s->mDiffusePositions, diffuseBytes,
                                 "&s->mDiffusePositions", kFlexFile, 4983);
        FLEX_CHECK_CUDA(cudaMemset(s->mDiffusePositions, 0, diffuseBytes), 4984,
                        "cudaMemset(s->mDiffusePositions, 0, maxDiffuseParticles*sizeof(Vec4))");

        FlexDeviceMallocInternal((void**)&s->mDiffuseVelocities, diffuseBytes,
                                 "&s->mDiffuseVelocities", kFlexFile, 4986);
        FLEX_CHECK_CUDA(cudaMemset(s->mDiffuseVelocities, 0, diffuseBytes), 4987,
                        "cudaMemset(s->mDiffuseVelocities, 0, maxDiffuseParticles*sizeof(Vec4))");

        FlexDeviceMallocInternal((void**)&s->mDiffusePositionsNew, diffuseBytes,
                                 "&s->mDiffusePositionsNew", kFlexFile, 4989);
        FLEX_CHECK_CUDA(cudaMemset(s->mDiffusePositionsNew, 0, diffuseBytes), 4990,
                        "cudaMemset(s->mDiffusePositionsNew, 0, maxDiffuseParticles*sizeof(Vec4))");

        FlexDeviceMallocInternal((void**)&s->mDiffuseVelocitiesNew, diffuseBytes,
                                 "&s->mDiffuseVelocitiesNew", kFlexFile, 4992);
        FLEX_CHECK_CUDA(cudaMemset(s->mDiffuseVelocitiesNew, 0, diffuseBytes), 4993,
                        "cudaMemset(s->mDiffuseVelocitiesNew, 0, maxDiffuseParticles*sizeof(Vec4))");

        FlexDeviceMallocInternal((void**)&s->mNumDiffuseParticles, 4, "&s->mNumDiffuseParticles", kFlexFile, 4995);
        FlexDeviceMallocInternal((void**)&s->mNumDiffuseParticlesNew, 4, "&s->mNumDiffuseParticlesNew", kFlexFile, 4996);
        FLEX_CHECK_CUDA(cudaMemset(s->mNumDiffuseParticles, 0, 4), 4997,
                        "cudaMemset(s->mNumDiffuseParticles, 0, sizeof(int))");

        s->mNumActiveDiffuseParticlesHost = (int*)FlexAlloc(s->mLib, 4);
        *s->mNumActiveDiffuseParticlesHost = 0;

        s->mDiffuseRate = 1.0f / 60.0f;
        s->mDiffuseTimer = 0.0f;
        s->mMaxDiffuseParticles = maxDiffuseParticles;
    }

    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mStartDetailEvent);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5009, "cudaEventCreate(&s->mStartDetailEvent)");
    }
    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mStopDetailEvent);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5010, "cudaEventCreate(&s->mStopDetailEvent)");
    }
    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mStartTotalEvent);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5012, "cudaEventCreate(&s->mStartTotalEvent)");
    }
    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mStopTotalEvent);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5013, "cudaEventCreate(&s->mStopTotalEvent)");
    }
    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mLatencyTimerBegin);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5015, "cudaEventCreate(&s->mLatencyTimerBegin)");
    }
    {
        cudaEvent_t* event = reinterpret_cast<cudaEvent_t*>(&s->mLatencyTimerEnd);
        FLEX_CHECK_CUDA(cudaEventCreate(event), 5016, "cudaEventCreate(&s->mLatencyTimerEnd)");
    }

    s->mLatencyTimerBeginRecorded = false;
    s->mLatencyTimerEndRecorded = false;

    if (maxParticles != 0)
    {
        const int particleBytes = maxParticles << 4;

        s->mTextures.sortedPositionsTex = CreateTexture(s->mSortedPositions, particleBytes);
        s->mTextures.sortedNewPositionsTex = CreateTexture(s->mSortedNewPositions, particleBytes);
        s->mTextures.sortedNewVelocitiesTex = CreateTexture(s->mSortedNewVelocities, particleBytes);
        s->mTextures.restPositionsTex = CreateTexture(s->mRestPositions, particleBytes);
        s->mTextures.normalsTex = CreateTexture(s->mNormals, particleBytes);
        s->mTextures.curlTex = CreateTexture(s->mCurl, particleBytes);
        s->mTextures.cellStartsTex = CreateTexture(s->mLib->mCellStarts, 0x800000);
        s->mTextures.cellEndsTex = CreateTexture(s->mLib->mCellEnds, 0x800000);
        s->mTextures.contactsTex = CreateTexture(s->mParticleNeighbors,
                                                 maxNeighborsPerParticle * maxParticles * 4);
    }

    NvFlexParams* params = SolverParams(s);
    *params = NvFlexParams();

    params->relaxationMode = eNvFlexRelaxationLocal;
    params->relaxationFactor = 1.0f;
    params->maxSpeed = 3.4028235e+38f;
    params->maxAcceleration = 3.4028235e+38f;
    params->numIterations = 1;

    SolverList(lib).push_back(s);

    return s;
}

// flexdecomp: src/cuda/flex.cu:5049 NvFlexDestroySolver
void NvFlexDestroySolver(NvFlexSolver* s)
{
    NvFlexLibrary* lib = s->mLib;

    FlexDeviceFreeInternal(s->mActiveIndices);
    FlexDeviceFreeInternal(s->mPositions);
    FlexDeviceFreeInternal(s->mNewPositions);
    FlexDeviceFreeInternal(s->mRestPositions);
    FlexDeviceFreeInternal(s->mVelocities);
    FlexDeviceFreeInternal(s->mDeltas);
    FlexDeviceFreeInternal(s->mSortedPositions);
    FlexDeviceFreeInternal(s->mSortedNewPositions);
    FlexDeviceFreeInternal(s->mSortedNewVelocities);
    FlexDeviceFreeInternal(s->mSortedPhases);
    FlexDeviceFreeInternal(s->mParticleLower);
    FlexDeviceFreeInternal(s->mParticleUpper);
    FlexDeviceFreeInternal(s->mStaticContactPlanes);
    FlexDeviceFreeInternal(s->mStaticContactVelocities);
    FlexDeviceFreeInternal(s->mStaticContactCounts);
    FlexDeviceFreeInternal(s->mSpringIndices);
    FlexDeviceFreeInternal(s->mSpringLengths);
    FlexDeviceFreeInternal(s->mSpringStiffness);
    FlexDeviceFreeInternal(s->mRigidOffsets);
    FlexDeviceFreeInternal(s->mRigidIndices);
    FlexDeviceFreeInternal(s->mRigidLocalPositions);
    FlexDeviceFreeInternal(s->mRigidLocalNormals);
    FlexDeviceFreeInternal(s->mRigidRotations);
    FlexDeviceFreeInternal(s->mRigidTranslations);
    FlexDeviceFreeInternal(s->mRigidCoefficients);
    FlexDeviceFreeInternal(s->mRigidPlasticThresholds);
    FlexDeviceFreeInternal(s->mRigidPlasticCreeps);
    FlexDeviceFreeInternal(s->mDensities);
    FlexDeviceFreeInternal(s->mPhases);
    FlexDeviceFreeInternal(s->mPotentials);
    FlexDeviceFreeInternal(s->mCurl);
    FlexDeviceFreeInternal(s->mNormals);
    FlexDeviceFreeInternal(s->mParticleNeighbors);
    FlexDeviceFreeInternal(s->mParticleNeighborCounts);
    FlexDeviceFreeInternal(s->mPlanes);
    FlexDeviceFreeInternal(s->mCellIds);
    FlexDeviceFreeInternal(s->mIndices);
    FlexDeviceFreeInternal(s->mReverseLookup);
    FlexDeviceFreeInternal(s->mAnisotropy1);
    FlexDeviceFreeInternal(s->mAnisotropy2);
    FlexDeviceFreeInternal(s->mAnisotropy3);

    if (s->mMaxDiffuseParticles != 0)
    {
        FlexDeviceFreeInternal(s->mDiffusePositions);
        FlexDeviceFreeInternal(s->mDiffuseVelocities);
        FlexDeviceFreeInternal(s->mDiffusePositionsNew);
        FlexDeviceFreeInternal(s->mDiffuseVelocitiesNew);
        FlexDeviceFreeInternal(s->mNumDiffuseParticles);
        FlexDeviceFreeInternal(s->mNumDiffuseParticlesNew);

        cudaError_t err = cudaFreeHost(s->mNumActiveDiffuseParticlesHost);
        FLEX_CHECK_CUDA_ERR(err, 5117, "cudaFreeHost(s->mNumActiveDiffuseParticlesHost)");
    }

    FreeBVH(s->mShapes.mBVH);
    FlexDeviceFreeInternal(s->mShapes.mAabbMin);
    FlexDeviceFreeInternal(s->mShapes.mAabbMax);
    FlexDeviceFreeInternal(s->mShapes.mGeometry);
    FlexDeviceFreeInternal(s->mShapes.mPositions);
    FlexDeviceFreeInternal(s->mShapes.mRotations);
    FlexDeviceFreeInternal(s->mShapes.mPrevPositions);
    FlexDeviceFreeInternal(s->mShapes.mPrevRotations);
    FlexDeviceFreeInternal(s->mShapes.mFlags);

    FlexDeviceFreeInternal(s->mDynamicTriangles);
    FlexDeviceFreeInternal(s->mDynamicTriangleNormals);
    FlexDeviceFreeInternal(s->mRandom);

    {
        cudaEvent_t event = reinterpret_cast<cudaEvent_t>(s->mStartDetailEvent);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5140, "cudaEventDestroy(s->mStartDetailEvent)");
        event = reinterpret_cast<cudaEvent_t>(s->mStopDetailEvent);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5141, "cudaEventDestroy(s->mStopDetailEvent)");
        event = reinterpret_cast<cudaEvent_t>(s->mStartTotalEvent);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5143, "cudaEventDestroy(s->mStartTotalEvent)");
        event = reinterpret_cast<cudaEvent_t>(s->mStopTotalEvent);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5144, "cudaEventDestroy(s->mStopTotalEvent)");
        event = reinterpret_cast<cudaEvent_t>(s->mLatencyTimerBegin);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5146, "cudaEventDestroy(s->mLatencyTimerBegin)");
        event = reinterpret_cast<cudaEvent_t>(s->mLatencyTimerEnd);
        FLEX_CHECK_CUDA(cudaEventDestroy(event), 5147, "cudaEventDestroy(s->mLatencyTimerEnd)");
    }
    FlexDeviceFreeInternal(s->mInflatables);
    FlexDeviceFreeInternal(s->mPressures);

    SortReset(s->mLib);

    DestroyTexture(s->mTextures.sortedPositionsTex);
    DestroyTexture(s->mTextures.sortedNewPositionsTex);
    DestroyTexture(s->mTextures.sortedNewVelocitiesTex);
    DestroyTexture(s->mTextures.restPositionsTex);
    DestroyTexture(s->mTextures.normalsTex);
    DestroyTexture(s->mTextures.curlTex);
    DestroyTexture(s->mTextures.cellStartsTex);
    DestroyTexture(s->mTextures.cellEndsTex);
    DestroyTexture(s->mTextures.contactsTex);
    DestroyTexture(s->mTextures.shapeGeometryTex);
    DestroyTexture(s->mTextures.shapePositionsTex);
    DestroyTexture(s->mTextures.shapeRotationsTex);
    DestroyTexture(s->mTextures.shapePrevPositionsTex);
    DestroyTexture(s->mTextures.shapePrevRotationsTex);

    NvFlexLibrary* owner = NvFlexGetSolverLibrary(s);
    SolverVector& solvers = SolverList(owner);
    SolverVector::iterator it = std::find(solvers.begin(), solvers.end(), s);
    if (it != solvers.end())
        solvers.erase(it);

    delete s;
}

// flexdecomp: src/cuda/flex.cu:5177 NvFlexGetSolvers
int NvFlexGetSolvers(NvFlexLibrary* lib, NvFlexSolver** solvers, int n)
{
    SolverVector& list = SolverList(lib);
    const unsigned long long count = list.size();
    unsigned long long num = (unsigned long long)(long)n;
    if (count <= num)
        num = count;

    memcpy(solvers, list.data(), num * 8);
    return (int)count;
}

// flexdecomp: src/cuda/flex.cu:5182 NvFlexGetSolverLibrary
NvFlexLibrary* NvFlexGetSolverLibrary(NvFlexSolver* solver)
{
    return solver->mLib;
}

// flexdecomp: src/cuda/flex.cu:5187 NvFlexGetSolverDesc
void NvFlexGetSolverDesc(NvFlexSolver* solver, NvFlexSolverDesc* desc)
{
    *desc = *SolverDesc(solver);
}

// flexdecomp: src/cuda/flex.cu:5192 NvFlexGetDeviceAndContext
void NvFlexGetDeviceAndContext(NvFlexLibrary* lib, void** device, void** context)
{
    (void)device;
    *context = lib->mCudaContext;
}

// flexdecomp: src/cuda/flex.cu:5356 NvFlexSetActive
void NvFlexSetActive(NvFlexSolver* s, NvFlexBuffer* indices, const NvFlexCopyDesc* desc)
{
    if (indices != 0)
        WriteData(LibErrorFunc(s->mLib), s->mActiveIndices, s->mMaxParticles, indices, desc, sizeof(int));
}

// flexdecomp: src/cuda/flex.cu:5366 NvFlexGetActive
void NvFlexGetActive(NvFlexSolver* s, NvFlexBuffer* indices, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), indices, s->mActiveIndices, s->mMaxParticles, desc, sizeof(int));
}

// flexdecomp: src/cuda/flex.cu:5373 NvFlexSetActiveCount
void NvFlexSetActiveCount(NvFlexSolver* s, int n)
{
    s->numParticles = n;
}

// flexdecomp: src/cuda/flex.cu:5378 NvFlexGetActiveCount
int NvFlexGetActiveCount(NvFlexSolver* s)
{
    return s->numParticles;
}

// flexdecomp: src/cuda/flex.cu:5383 NvFlexSetPhases
void NvFlexSetPhases(NvFlexSolver* s, NvFlexBuffer* phases, const NvFlexCopyDesc* desc)
{
    WriteData(LibErrorFunc(s->mLib), s->mPhases, s->mMaxParticles, phases, desc, sizeof(int));
}

// flexdecomp: src/cuda/flex.cu:5390 NvFlexGetPhases
void NvFlexGetPhases(NvFlexSolver* s, NvFlexBuffer* phases, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), phases, s->mPhases, s->mMaxParticles, desc, sizeof(int));
}

// flexdecomp: src/cuda/flex.cu:5398 NvFlexSetVelocities
void NvFlexSetVelocities(NvFlexSolver* s, NvFlexBuffer* v, const NvFlexCopyDesc* desc)
{
    WriteData(LibErrorFunc(s->mLib), s->mVelocities, s->mMaxParticles, v, desc, sizeof(Vec3));
}

// flexdecomp: src/cuda/flex.cu:5405 NvFlexGetParticles
void NvFlexGetParticles(NvFlexSolver* s, NvFlexBuffer* p, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), p, s->mPositions, s->mMaxParticles, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5412 NvFlexSetParticles
void NvFlexSetParticles(NvFlexSolver* s, NvFlexBuffer* p, const NvFlexCopyDesc* desc)
{
    WriteData(LibErrorFunc(s->mLib), s->mPositions, s->mMaxParticles, p, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5419 NvFlexSetRestParticles
void NvFlexSetRestParticles(NvFlexSolver* s, NvFlexBuffer* p, const NvFlexCopyDesc* desc)
{
    WriteData(LibErrorFunc(s->mLib), s->mRestPositions, s->mMaxParticles, p, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5426 NvFlexGetRestParticles
void NvFlexGetRestParticles(NvFlexSolver* s, NvFlexBuffer* p, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), p, s->mRestPositions, s->mMaxParticles, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5433 NvFlexGetSmoothParticles
void NvFlexGetSmoothParticles(NvFlexSolver* s, NvFlexBuffer* p, const NvFlexCopyDesc* desc)
{
    if (s->mStepCounter == 0)
        ReadData(LibErrorFunc(s->mLib), p, s->mPositions, s->mMaxParticles, desc, sizeof(Vec4));
    else
        ReadData(LibErrorFunc(s->mLib), p, s->mDeltas, s->mMaxParticles, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5450 NvFlexGetDensities
void NvFlexGetDensities(NvFlexSolver* s, NvFlexBuffer* d, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), d, s->mDensities, s->mMaxParticles, desc, sizeof(float));
}

// flexdecomp: src/cuda/flex.cu:5457 NvFlexSetNormals
void NvFlexSetNormals(NvFlexSolver* s, NvFlexBuffer* normals, const NvFlexCopyDesc* desc)
{
    WriteData(LibErrorFunc(s->mLib), s->mNormals, s->mMaxParticles, normals, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5464 NvFlexGetNormals
void NvFlexGetNormals(NvFlexSolver* s, NvFlexBuffer* normals, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), normals, s->mNormals, s->mMaxParticles, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5471 NvFlexGetVelocities
void NvFlexGetVelocities(NvFlexSolver* s, NvFlexBuffer* v, const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), v, s->mVelocities, s->mMaxParticles, desc, sizeof(Vec3));
}

// flexdecomp: src/cuda/flex.cu:5478 NvFlexGetAnisotropy
void NvFlexGetAnisotropy(NvFlexSolver* s, NvFlexBuffer* q1, NvFlexBuffer* q2, NvFlexBuffer* q3,
                         const NvFlexCopyDesc* desc)
{
    ReadData(LibErrorFunc(s->mLib), q1, s->mAnisotropy1, s->mMaxParticles, desc, sizeof(Vec4));
    ReadData(LibErrorFunc(s->mLib), q2, s->mAnisotropy2, s->mMaxParticles, desc, sizeof(Vec4));
    ReadData(LibErrorFunc(s->mLib), q3, s->mAnisotropy3, s->mMaxParticles, desc, sizeof(Vec4));
}

// flexdecomp: src/cuda/flex.cu:5487 NvFlexGetDiffuseParticles
void NvFlexGetDiffuseParticles(NvFlexSolver* s, NvFlexBuffer* p, NvFlexBuffer* v, NvFlexBuffer* count)
{
    if (s->mMaxDiffuseParticles != 0)
    {
        if (p != 0)
            ReadData(p, 0, s->mDiffusePositions, 0, s->mMaxDiffuseParticles * sizeof(Vec4));
        if (v != 0)
            ReadData(v, 0, s->mDiffuseVelocities, 0, s->mMaxDiffuseParticles * sizeof(Vec4));
        if (count != 0)
            ReadData(count, 0, s->mNumDiffuseParticles, 0, sizeof(int));
    }
}

// flexdecomp: src/cuda/flex.cu:5504 NvFlexSetDiffuseParticles
void NvFlexSetDiffuseParticles(NvFlexSolver* s, NvFlexBuffer* p, NvFlexBuffer* v, int n)
{
    NvFlexLibrary* lib = s->mLib;
    int count = n;

    if (p != 0)
        WriteData(s->mDiffusePositions, 0, p, 0, n * sizeof(Vec4));
    if (v != 0)
        WriteData(s->mDiffuseVelocities, 0, v, 0, count * sizeof(Vec4));

    cudaError_t err = cudaMemcpyAsync(s->mNumDiffuseParticles, &count, sizeof(int),
                                      cudaMemcpyHostToDevice, 0);
    FLEX_CHECK_CUDA_ERR(err, 5516,
                    "cudaMemcpyAsync(s->mNumDiffuseParticles, &n, sizeof(int), cudaMemcpyHostToDevice)");

    *s->mNumActiveDiffuseParticlesHost = count;
}

// flexdecomp: src/cuda/flex.cu:5521 NvFlexSetRigids
void NvFlexSetRigids(NvFlexSolver* s, NvFlexBuffer* offsets, NvFlexBuffer* indices,
                     NvFlexBuffer* rest, NvFlexBuffer* normals, NvFlexBuffer* stiffness,
                     NvFlexBuffer* thresholds, NvFlexBuffer* creeps, NvFlexBuffer* rotations,
                     NvFlexBuffer* translations, int numRigids, int numIndices)
{
    NvFlexLibrary* lib = s->mLib;

    if (numRigids == 0)
    {
        s->mNumRigids = 0;
        s->mNumRigidIndices = 0;
        return;
    }

    if (s->mMaxRigids < numRigids)
    {
        const int numToAlloc = (numRigids * 3) / 2;

        FlexDeviceFreeInternal(s->mRigidOffsets);
        FlexDeviceFreeInternal(s->mRigidRotations);
        FlexDeviceFreeInternal(s->mRigidTranslations);
        FlexDeviceFreeInternal(s->mRigidCoefficients);

        FlexDeviceMallocInternal((void**)&s->mRigidOffsets, (size_t)(numToAlloc + 1) * sizeof(int),
                                 "&s->mRigidOffsets", kFlexFile, 5543);
        FlexDeviceMallocInternal((void**)&s->mRigidRotations, (size_t)numToAlloc * sizeof(Vec4),
                                 "&s->mRigidRotations", kFlexFile, 5544);
        FlexDeviceMallocInternal((void**)&s->mRigidTranslations, (size_t)numToAlloc * sizeof(Vec3),
                                 "&s->mRigidTranslations", kFlexFile, 5545);
        FlexDeviceMallocInternal((void**)&s->mRigidCoefficients, (size_t)numToAlloc * sizeof(float),
                                 "&s->mRigidCoefficients", kFlexFile, 5546);

        s->mMaxRigids = numToAlloc;
    }

    if (s->mMaxRigidIndices < numIndices)
    {
        const int numToAlloc = (numIndices * 3) / 2;

        FlexDeviceFreeInternal(s->mRigidIndices);
        FlexDeviceFreeInternal(s->mRigidLocalPositions);
        FlexDeviceFreeInternal(s->mRigidLocalNormals);

        FlexDeviceMallocInternal((void**)&s->mRigidIndices, (size_t)numToAlloc * sizeof(int),
                                 "&s->mRigidIndices", kFlexFile, 5560);
        FlexDeviceMallocInternal((void**)&s->mRigidLocalPositions, (size_t)numToAlloc * sizeof(Vec3),
                                 "&s->mRigidLocalPositions", kFlexFile, 5561);
        FlexDeviceMallocInternal((void**)&s->mRigidLocalNormals, (size_t)numToAlloc * sizeof(Vec4),
                                 "&s->mRigidLocalNormals", kFlexFile, 5562);

        s->mMaxRigidIndices = numToAlloc;
    }

    if (rotations != 0)
        WriteData(s->mRigidRotations, 0, rotations, 0, numRigids * sizeof(Vec4));
    if (translations != 0)
        WriteData(s->mRigidTranslations, 0, translations, 0, numRigids * sizeof(Vec3));
    if (offsets != 0)
        WriteData(s->mRigidOffsets, 0, offsets, 0, numRigids * sizeof(int) + sizeof(int));
    if (indices != 0)
        WriteData(s->mRigidIndices, 0, indices, 0, numIndices * sizeof(int));
    if (stiffness != 0)
        WriteData(s->mRigidCoefficients, 0, stiffness, 0, numRigids * sizeof(float));

    s->mNumRigids = numRigids;
    s->mNumRigidIndices = numIndices;

    if (rest != 0)
        WriteData(s->mRigidLocalPositions, 0, rest, 0, numIndices * sizeof(Vec3));

    if (normals == 0)
    {
        cudaError_t err = cudaMemsetAsync(s->mRigidLocalNormals, 0, (size_t)numIndices * sizeof(Vec4), 0);
        FLEX_CHECK_CUDA_ERR(err, 5590, "cudaMemsetAsync(s->mRigidLocalNormals, 0, sizeof(Vec4)*numIndices)");
    }
    else
    {
        WriteData(s->mRigidLocalNormals, 0, normals, 0, numIndices * sizeof(Vec4));
    }

    if (thresholds != 0 && creeps != 0)
    {
        int num = s->mNumRigids;
        if (s->mMaxPlasticEntries < num)
        {
            FlexDeviceFreeInternal(s->mRigidPlasticThresholds);
            FlexDeviceFreeInternal(s->mRigidPlasticCreeps);
            FlexDeviceMallocInternal((void**)&s->mRigidPlasticThresholds,
                                     (size_t)s->mMaxRigids * sizeof(float),
                                     "&s->mRigidPlasticThresholds", kFlexFile, 5599);
            FlexDeviceMallocInternal((void**)&s->mRigidPlasticCreeps,
                                     (size_t)s->mMaxRigids * sizeof(float),
                                     "&s->mRigidPlasticCreeps", kFlexFile, 5600);
            num = s->mNumRigids;
            s->mMaxPlasticEntries = s->mMaxRigids;
        }

        WriteData(s->mRigidPlasticThresholds, 0, thresholds, 0, num * sizeof(float));
        WriteData(s->mRigidPlasticCreeps, 0, creeps, 0, s->mNumRigids * sizeof(float));
        return;
    }

    if (s->mRigidPlasticThresholds != 0)
        FlexDeviceFreeInternal(s->mRigidPlasticThresholds);
    if (s->mRigidPlasticCreeps != 0)
        FlexDeviceFreeInternal(s->mRigidPlasticCreeps);

    s->mRigidPlasticThresholds = 0;
    s->mRigidPlasticCreeps = 0;
    s->mMaxPlasticEntries = 0;
}

// flexdecomp: src/cuda/flex.cu:5620 NvFlexGetRigids
void NvFlexGetRigids(NvFlexSolver* s, NvFlexBuffer* offsets, NvFlexBuffer* indices,
                     NvFlexBuffer* restPositions, NvFlexBuffer* restNormals,
                     NvFlexBuffer* stiffness, NvFlexBuffer* thresholds, NvFlexBuffer* creeps,
                     NvFlexBuffer* rotations, NvFlexBuffer* translations)
{
    const int numRigids = s->mNumRigids;
    const int numIndices = s->mNumRigidIndices;

    if (numRigids == 0)
        return;

    if (rotations != 0)
        ReadData(rotations, 0, s->mRigidRotations, 0, numRigids * sizeof(Vec4));
    if (translations != 0)
        ReadData(translations, 0, s->mRigidTranslations, 0, numRigids * sizeof(Vec3));
    if (offsets != 0)
        ReadData(offsets, 0, s->mRigidOffsets, 0, numRigids * sizeof(int) + sizeof(int));
    if (indices != 0)
        ReadData(indices, 0, s->mRigidIndices, 0, numIndices * sizeof(int));
    if (stiffness != 0)
        ReadData(stiffness, 0, s->mRigidCoefficients, 0, numRigids * sizeof(float));
    if (restPositions != 0)
        ReadData(restPositions, 0, s->mRigidLocalPositions, 0, numIndices * sizeof(Vec3));
    if (restNormals != 0)
        ReadData(restNormals, 0, s->mRigidLocalNormals, 0, numIndices * sizeof(Vec4));

    if (thresholds != 0 && s->mRigidPlasticThresholds != 0)
        ReadData(thresholds, 0, s->mRigidPlasticThresholds, 0, numRigids * sizeof(float));

    if (creeps != 0 && s->mRigidPlasticCreeps != 0)
        ReadData(creeps, 0, s->mRigidPlasticCreeps, 0, numRigids * sizeof(float));
}

// flexdecomp: src/cuda/flex.cu:5662 NvFlexSetSprings
void NvFlexSetSprings(NvFlexSolver* s, NvFlexBuffer* springIndices, NvFlexBuffer* springLengths,
                      NvFlexBuffer* springStiffness, int numSprings)
{
    if (s->mMaxSprings < numSprings)
    {
        FlexDeviceFreeInternal(s->mSpringIndices);

        const int numToAlloc = (numSprings * 3) / 2;

        FlexDeviceFreeInternal(s->mSpringLengths);
        FlexDeviceFreeInternal(s->mSpringStiffness);

        FlexDeviceMallocInternal((void**)&s->mSpringIndices, (size_t)numToAlloc * 2 * sizeof(int),
                                 "&s->mSpringIndices", kFlexFile, 5674);
        FlexDeviceMallocInternal((void**)&s->mSpringLengths, (size_t)numToAlloc * sizeof(float),
                                 "&s->mSpringLengths", kFlexFile, 5675);
        FlexDeviceMallocInternal((void**)&s->mSpringStiffness, (size_t)numToAlloc * sizeof(float),
                                 "&s->mSpringStiffness", kFlexFile, 5676);

        s->mMaxSprings = numToAlloc;
    }

    if (numSprings != 0)
    {
        if (springIndices != 0)
            WriteData(s->mSpringIndices, 0, springIndices, 0, numSprings * 2 * sizeof(int));
        if (springLengths != 0)
            WriteData(s->mSpringLengths, 0, springLengths, 0, numSprings * sizeof(float));
        if (springStiffness != 0)
            WriteData(s->mSpringStiffness, 0, springStiffness, 0, numSprings * sizeof(float));
    }

    s->mNumSprings = numSprings;
}

// flexdecomp: src/cuda/flex.cu:5696 NvFlexGetSprings
void NvFlexGetSprings(NvFlexSolver* s, NvFlexBuffer* springIndices, NvFlexBuffer* springLengths,
                      NvFlexBuffer* springStiffness, int numSprings)
{
    int count = s->mNumSprings;
    if (numSprings <= s->mNumSprings)
        count = numSprings;

    if (count <= 0)
        return;

    if (springIndices != 0)
        ReadData(springIndices, 0, s->mSpringIndices, 0, count * 2 * sizeof(int));
    if (springLengths != 0)
        ReadData(springLengths, 0, s->mSpringLengths, 0, count * sizeof(float));
    if (springStiffness != 0)
        ReadData(springStiffness, 0, s->mSpringStiffness, 0, count * sizeof(float));
}

// flexdecomp: src/cuda/flex.cu:5715 NvFlexGetNeighbors
void NvFlexGetNeighbors(NvFlexSolver* s, NvFlexBuffer* neighbors, NvFlexBuffer* counts,
                        NvFlexBuffer* apiToInternal, NvFlexBuffer* internalToApi)
{
    ReadData(neighbors, 0, s->mParticleNeighbors, 0,
             s->mMaxParticles * s->mMaxNeighborsPerParticle * (int)sizeof(int));
    ReadData(counts, 0, s->mParticleNeighborCounts, 0, s->mMaxParticles * (int)sizeof(int));
    ReadData(apiToInternal, 0, s->mReverseLookup, 0, s->mMaxParticles * (int)sizeof(int));
    ReadData(internalToApi, 0, s->mIndices, 0, s->mMaxParticles * (int)sizeof(int));
}

// flexdecomp: src/cuda/flex.cu:5726 NvFlexSetInflatables
// the shipped object was compiled without loop vectorisation; keep the simple
// scalar struct-fill loop (GCC 4.8 codegen)
#if !defined(_MSC_VER)
__attribute__((optimize("no-tree-vectorize")))
#endif
void NvFlexSetInflatables(NvFlexSolver* s, NvFlexBuffer* startTris, NvFlexBuffer* numTris,
                          NvFlexBuffer* restVolumes, NvFlexBuffer* overPressures,
                          NvFlexBuffer* constraintScales, int n)
{
    NvFlexLibrary* lib = s->mLib;

    if (s->mMaxInflatables < n)
    {
        const int numToAlloc = (n * 3) / 2;

        FlexDeviceFreeInternal(s->mInflatables);
        FlexDeviceMallocInternal((void**)&s->mInflatables,
                                 (size_t)numToAlloc * sizeof(FlexInflatable),
                                 "&s->mInflatables", kFlexFile, 5737);

        FlexDeviceFreeInternal(s->mPressures);
        FlexDeviceMallocInternal((void**)&s->mPressures, (size_t)numToAlloc * sizeof(float),
                                 "&s->mPressures", kFlexFile, 5741);

        s->mMaxInflatables = numToAlloc;
    }

    if (n > 0)
    {
        // original allocates with scalar operator new and zero-fills
        // (cudaflex.c:5750); a std::vector would add libstdc++ EH paths
        FlexInflatable* inflatables =
            static_cast<FlexInflatable*>(::operator new((size_t)n * sizeof(FlexInflatable)));
        memset(inflatables, 0, (size_t)n * sizeof(FlexInflatable));

        const int* __restrict__ startTriData = (const int*)startTris->ptr;
        const int* __restrict__ numTriData = (const int*)numTris->ptr;
        const float* __restrict__ restVolumeData = (const float*)restVolumes->ptr;
        const float* __restrict__ overPressureData = (const float*)overPressures->ptr;
        const float* __restrict__ constraintScaleData = (const float*)constraintScales->ptr;

        for (int i = 0; i < n; ++i)
        {
            FlexInflatable& inf = inflatables[i];
            inf.mStartTri = startTriData[i];
            inf.mNumTris = numTriData[i];
            inf.mRestVolume = restVolumeData[i] * overPressureData[i];
            inf.mConstraintScale = constraintScaleData[i];
        }

        cudaError_t err = cudaMemcpyAsync(s->mInflatables, inflatables,
                                          (size_t)n * sizeof(FlexInflatable),
                                          cudaMemcpyHostToDevice, 0);
        FLEX_CHECK_CUDA_ERR(err, 5761,
                        "cudaMemcpyAsync(s->mInflatables, &inflatables[0], sizeof(FlexInflatable)*n, cudaMemcpyHostToDevice)");

        ::operator delete(inflatables);
    }

    s->mNumInflatables = n;
}
