// Reconstructed FleX 1.2.0 CUDA solver - host layer, part 4.
//
// This translation unit holds the public API tail of the shipped cudaflex.o
// (source order: flex.cu:5769 .. flex.cu:6608): the NvFlex* buffer, parameter,
// distance-field, shape, convex-mesh and triangle-mesh entry points, the
// update/latency/timer queries and the copy helpers.
//
// Evidence used for every unit:
//   work/decomp/nvflex.flex/cudaflex.c            Ghidra pseudocode, by address
//   work/dwarf/nvflex.flex.json                   DIEs: decl file/line, params
//   work/binary/obj/dbg/NvFlexDebugCUDA_x64/
//       dump/cudaflex.disasm.txt                  objdump -drw
//   recon/include/nvflex/recovered.h              exact recovered layouts
//   FleX/bin/linux64/NvFlexDemoDebugCUDA_x64      exact assert/pretty-function text
//
// The anonymous-namespace transfer helpers (ReadData/WriteData/writeData3D) are
// defined once in flex.part1.cu under the shipped wrapper-namespace name; this
// file only declares and calls them so both translation units share one copy.
// The hash below is the *release* object's (`_00000f25_`); the debug object
// carries a different hash for the same namespace (`_00000d59_`).

#include <cuda_runtime.h>

#include "nvflex/all.h"
#include "NvFlex.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- helpers defined in other translation units ---------------------------

// nvflex.util (util.cpp)
void FlexDeviceMallocInternal(void** ptr, size_t size, const char* label,
                              const char* file, int line);
void FlexDeviceFreeInternal(void* ptr);
unsigned long long CreateTexture(const Vec4* buffer, int sizeInBytes);
unsigned long long CreateTexture(const int* buffer, int sizeInBytes);
void DestroyTexture(cudaTextureObject_t tex);
void FlexLogFunc(NvFlexErrorCallback callback, NvFlexErrorSeverity type,
                 const char* msg, const char* file, int line, ...);

// nvflex.bvh (bvh.cu)
void FreeBVH(BVH& bvh);

// linear BVH builder member (bvh.cu); declared as a free function because the
// recovered type carries data members only.
void LinearBVHBuilderGPUBuild(const LinearBVHBuilderGPU* self, NvFlexLibrary* lib,
                              BVH& bvh, const Vec4* itemLowers, const Vec4* itemUppers,
                              const int* itemPriorities, int numItems, Bounds* totalBounds)
    __asm__("_ZN19LinearBVHBuilderGPU5BuildEP13NvFlexLibraryR3BVHPK8XVector4IfES7_PKiiP6Bounds");

// this module (flex.part1.cu / flex.part3.cu)
void NvFlexSetFence(NvFlexLibrary* lib);
void NvFlexWaitFence(NvFlexLibrary* lib);
void InitParams(NvFlexSolver* s);
void Update(NvFlexSolver* s, float dt, int substep, int substeps);

namespace _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207 {

void ReadData(NvFlexBuffer* dstBuffer, int dstByteOffset, const void* srcPtr,
              int srcByteOffset, int byteSize);
void WriteData(void* dstPtr, int dstByteOffset, NvFlexBuffer* srcBuffer,
               int srcByteOffset, int byteSize);
void writeData3D(cudaArray_t dst, NvFlexBuffer* src, int pitch, int width, int height);

}  // namespace _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207

// device kernels of the solver (flex-kernels translation units)
__global__ void TransformShapeBounds(Vec4* aabbMin, Vec4* aabbMax, const Vec4* positions,
                                     const Quat* rotations, const Vec4* prevPositions,
                                     const Quat* prevRotations,
                                     const NvFlexCollisionGeometry* geometry,
                                     const int* flags, int numShapes,
                                     const FlexConvexMeshInternal* convexes,
                                     const FlexTriangleMeshInternal* meshes);
__global__ void ComputeTriangleBounds(const Vec4* vertices, const int* indices, int numTris,
                                      Vec4* lowers, Vec4* uppers);

// ---- local view of the shared literals ------------------------------------
// The shipped literal is the build-relative path of the original source;
// __FILE__ would not reproduce it and the static gate compares literal sets.
static const char* const kFlexSourceFile = "../../../src/cuda/flex.cu";
static const char* const kUtilSourceFile = "../../../src/cuda/util.h";

// NvFlexLibrary::mErrorFunc is recorded as an opaque 8-byte member by the
// DWARF recovery (it is an NvFlexErrorCallback); read it through memcpy.
// XVector3<float> is a plain aggregate in the recovered layout
static Vec3 makeVec3(const float* v)
{
    Vec3 r;
    r.x = v[0];
    r.y = v[1];
    r.z = v[2];
    return r;
}

static NvFlexErrorCallback libErrorFunc(NvFlexLibrary* lib)
{
    void* fn = 0;
    std::memcpy(&fn, lib->mErrorFunc, sizeof(fn));
    return reinterpret_cast<NvFlexErrorCallback>(fn);
}

namespace {

// ---- standard-library objects hidden behind opaque recovered members ------

typedef std::vector<CudaRangeTimer> CudaRangeTimerVector;

// ltstr is the comparator declared for NameToTimerMap by the recovered header
typedef std::map<const char*, int, ltstr> CudaTimerNameMap;

inline CudaRangeTimerVector& timerPoolElems(CudaRangeTimerPool* pool)
{
    return *reinterpret_cast<CudaRangeTimerVector*>(pool->mElem);
}

typedef std::pair<CudaTimerNameMap::iterator, CudaTimerNameMap::iterator> CudaTimerNameRange;

// The shipped getTimer is a 30-instruction wrapper: the tree search is out of
// line (_Rb_tree::equal_range). GCC 15 inlines the header template into the
// caller, so it is kept out of line here to preserve the same shape.
__attribute__((noinline))
CudaTimerNameRange timerNameRange(CudaTimerNameMap& map, const char* name)
{
    return map.equal_range(name);
}

inline CudaTimerNameMap& timerNameMap(CudaTimers* timers)
{
    return *reinterpret_cast<CudaTimerNameMap*>(timers->mNameToTimerMap);
}

// DeviceBuffer<XVector4<float> >: recovered as an opaque 24-byte member of
// FlexTriangleMeshInternal; same layout as DeviceBuffer<int> (DWARF).
struct DeviceBufferVec4
{
    Vec4* mPtr;
    int mCapacity;
    unsigned char _pad[4];
    unsigned long long mTexture;
};

inline DeviceBufferVec4& verticesBuffer(FlexTriangleMeshInternal* mesh)
{
    return *reinterpret_cast<DeviceBufferVec4*>(mesh->mVertices);
}

inline DeviceBufferVec4& staticTriBuffer(unsigned char* storage)
{
    return *reinterpret_cast<DeviceBufferVec4*>(storage);
}

__attribute__((noinline))
void pushRegisteredId(std::vector<unsigned int>& ids, unsigned int id)
{
    ids.push_back(id);
}

inline std::vector<unsigned int>& idVector(unsigned char* storage)
{
    return *reinterpret_cast<std::vector<unsigned int>*>(storage);
}

}  // namespace

// flexdecomp: flex.cu:5769 NvFlexSetDynamicTriangles
void NvFlexSetDynamicTriangles(NvFlexSolver* s, NvFlexBuffer* triangles, NvFlexBuffer* normals,
                               int numTris)
{
    if (numTris != 0)
    {
        if (s->mDynamicMaxTris < numTris)
        {
            const int numToAlloc = (numTris * 3) / 2;

            FlexDeviceFreeInternal(s->mDynamicTriangles);
            FlexDeviceFreeInternal(s->mDynamicTriangleNormals);

            const size_t numBytes = size_t(numToAlloc) * 12;

            FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mDynamicTriangles), numBytes,
                                     "&s->mDynamicTriangles", kFlexSourceFile, 0x169a);
            FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mDynamicTriangleNormals), numBytes,
                                     "&s->mDynamicTriangleNormals", kFlexSourceFile, 0x169b);

            s->mDynamicMaxTris = numToAlloc;
        }

        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mDynamicTriangles, 0, triangles, 0, numTris * 12);

        if (normals)
        {
            _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
                s->mDynamicTriangleNormals, 0, normals, 0, numTris * 12);
        }

        s->mDynamicNumTris = numTris;
        return;
    }

    s->mDynamicNumTris = 0;
}

// flexdecomp: flex.cu:5802 NvFlexGetDynamicTriangles
void NvFlexGetDynamicTriangles(NvFlexSolver* s, NvFlexBuffer* triangles, NvFlexBuffer* normals,
                               int numTris)
{
    assert(numTris <= s->mDynamicNumTris);

    if (triangles)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            triangles, 0, s->mDynamicTriangles, 0, numTris * 12);
    }

    if (normals)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            normals, 0, s->mDynamicTriangleNormals, 0, numTris * 12);
    }
}

// flexdecomp: flex.cu:5814 _Z24NvFlexGetRigidTransformsP12NvFlexSolverP12NvFlexBufferS2_
void NvFlexGetRigidTransforms(NvFlexSolver* s, NvFlexBuffer* rotations,
                              NvFlexBuffer* translations);

void NvFlexGetRigidTransforms(NvFlexSolver* s, NvFlexBuffer* rotations, NvFlexBuffer* translations)
{
    if (rotations)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            rotations, 0, s->mRigidRotations, 0, s->mNumRigids << 4);
    }

    if (translations)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            translations, 0, s->mRigidTranslations, 0, s->mNumRigids * 12);
    }
}

// flexdecomp: flex.cu:5826 NvFlexSetParams
void NvFlexSetParams(NvFlexSolver* s, const NvFlexParams* params)
{
    // The random seed globals live in the demo/application layer of the SDK
    // (undefined data symbols in cudaflex.o); the sampler below reseeds them.
    extern int seed1;
    extern int seed2;

    const float fluidRestDistance = params->fluidRestDistance;

    if (fluidRestDistance > 0.0f)
    {
        const float radius = params->radius;

        const NvFlexParams* storedParams =
            reinterpret_cast<const NvFlexParams*>(s->mParams);

        if (radius != storedParams->radius ||
            fluidRestDistance != storedParams->fluidRestDistance)
        {
            // Sample a lattice of interaction kernels and integrate the poly6
            // density and its gradient to obtain the rest density and the
            // constraint normalisation (flex.cu:5834..5851).
            const int kMaxSamples = 2048;
            const float kSqrt3Over2 = 0.8660253882408142f;  // rodata .LC52
            const float kHalf = 0.5f;                       // rodata .LC53
            const float kPi = 3.1415927410125732f;          // rodata .LC6
            const float kPoly6 = 15.0f;                     // rodata .LC7

            Vec3* samples =
                reinterpret_cast<Vec3*>(operator new(size_t(kMaxSamples) * sizeof(Vec3)));
            for (int i = 0; i < kMaxSamples; ++i)
            {
                samples[i].x = 0.0f;
                samples[i].y = 0.0f;
                samples[i].z = 0.0f;
            }

            seed1 = 0x258f4aa;
            seed2 = 0x7583f;

            const int dim = int(ceilf(radius / fluidRestDistance));

            int n = 0;
            float surfaceDensity = 0.0f;
            float surfaceVolume = 0.0f;

            for (int z = -dim; z <= dim; ++z)
            {
                const float zpos = float(z) * kSqrt3Over2 * fluidRestDistance;

                for (int y = -dim; y <= dim; ++y)
                {
                    const float ypos = float(y) * kSqrt3Over2 * fluidRestDistance;

                    if (((y + z) & 1) == 0)
                    {
                        for (int x = -dim; x <= dim; ++x)
                        {
                            const float xpos = float(x) * fluidRestDistance + 0.0f;
                            const float lSq = xpos * xpos + ypos * ypos + zpos * zpos;

                            if (lSq != 0.0f && n < kMaxSamples)
                            {
                                const float d = sqrtf(lSq);

                                if (d <= radius)
                                {
                                    samples[n].x = xpos;
                                    samples[n].y = ypos;
                                    samples[n].z = zpos;
                                    ++n;
                                }
                            }
                        }
                    }
                    else
                    {
                        for (int x = -dim; x <= dim; ++x)
                        {
                            const float xpos = float(x) * fluidRestDistance + fluidRestDistance * kHalf;
                            const float lSq = xpos * xpos + ypos * ypos + zpos * zpos;

                            if (lSq != 0.0f && n < kMaxSamples)
                            {
                                const float d = sqrtf(lSq);

                                if (d <= radius)
                                {
                                    samples[n].x = xpos;
                                    samples[n].y = ypos;
                                    samples[n].z = zpos;
                                    ++n;
                                }
                            }
                        }
                    }
                }
            }

            s->mFluidRestDensity = 0.0f;
            s->mFluidDensityConstraintScale = 0.0f;

            if (n != 0)
            {
                for (int i = 0; i < n; ++i)
                {
                    const Vec3& p = samples[i];
                    const float lSq = p.x * p.x + p.y * p.y + p.z * p.z;
                    const float d = sqrtf(lSq);
                    const float k = kPoly6 / (radius * kPi * radius * radius);
                    const float w = 1.0f - d / radius;
                    const float kNeg = -k;

                    s->mFluidRestDensity += w * w * k;

                    const float grad = ((kNeg + kNeg) * w) / radius;
                    s->mFluidDensityConstraintScale += grad * grad;

                    // only the lower hemisphere contributes to the surface term
                    if (p.y <= 0.0f)
                    {
                        const float cosTheta = p.y / d;
                        surfaceDensity += grad * cosTheta;
                        surfaceVolume -= d * cosTheta;
                    }
                }
            }

            s->mFluidSurfaceConstraintScale = surfaceDensity / surfaceVolume;

            operator delete(samples);
        }
    }
    else
    {
        s->mFluidRestDensity = 0.0f;
        s->mFluidDensityConstraintScale = 0.0f;
        s->mFluidSurfaceConstraintScale = 0.0f;
    }

    std::memcpy(s->mParams, params, sizeof(NvFlexParams));
}

// flexdecomp: flex.cu:5854 NvFlexGetParams
void NvFlexGetParams(NvFlexSolver* s, NvFlexParams* params)
{
    // The shipped copy shape (flex.cu:5856): NvFlexParams is 4-byte aligned and
    // 308 bytes long, so a leading int is copied when the destination is not
    // 8-byte aligned and the remainder is copied as 38 eight-byte words.
    unsigned char* dst = reinterpret_cast<unsigned char*>(params);
    const unsigned char* src = s->mParams;

    size_t numBytes = sizeof(NvFlexParams);

    if ((reinterpret_cast<uintptr_t>(dst) & 4) != 0)
    {
        *reinterpret_cast<int*>(dst) = *reinterpret_cast<const int*>(src);
        dst += 4;
        src += 4;
        numBytes -= 4;
    }

    for (size_t i = 0; i < numBytes; i += 8)
    {
        *reinterpret_cast<unsigned long long*>(dst + i) =
            *reinterpret_cast<const unsigned long long*>(src + i);
    }
}

// flexdecomp: flex.cu:5861 NvFlexCreateDistanceField
NvFlexDistanceFieldId NvFlexCreateDistanceField(NvFlexLibrary* lib)
{
    NvFlexDistanceFieldId newIndex = 0;

    int index = lib->mSDFData.mNumSDFs;

    if (index < lib->mSDFData.mMaxSDFs)
    {
        lib->mSDFData.mNumSDFs = index + 1;

        index = lib->mSDFData.mFreeList[index];
        newIndex = NvFlexDistanceFieldId(index + 1);

        FlexSDFInternal& sdf = lib->mSDFData.mSDFTableHost[index];
        sdf.mDim[0] = 0;
        sdf.mDim[1] = 0;
        sdf.mDim[2] = 0;
        sdf.mInvDim[0] = 0.0f;
        sdf.mInvDim[1] = 0.0f;
        sdf.mInvDim[2] = 0.0f;
        sdf.mFieldTex = 0;
        sdf.mFieldArray = 0;
    }

    pushRegisteredId(idVector(lib->mResourceTracker.mSDF), newIndex);

    return newIndex;
}

// flexdecomp: flex.cu:5871 NvFlexDestroyDistanceField
void NvFlexDestroyDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdfIndex)
{
    if (sdfIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid signed distance field", kFlexSourceFile,
                        0x16f5);
        }
        return;
    }

    std::vector<unsigned int>& sdfs = idVector(lib->mResourceTracker.mSDF);
    sdfs.erase(std::remove(sdfs.begin(), sdfs.end(), (unsigned long)sdfIndex), sdfs.end());

    FlexSDFInternal* sdf = lib->mSDFData.mSDFTableHost + (sdfIndex - 1);

    if (sdf->mFieldArray != 0)
    {
        cudaFreeArray(reinterpret_cast<cudaArray_t>(sdf->mFieldArray));
        sdf->mFieldArray = 0;
    }

    if (sdf->mFieldTex != 0)
    {
        cudaDestroyTextureObject(cudaTextureObject_t(sdf->mFieldTex));
        sdf->mFieldTex = 0;
    }

    const int freeIndex = --lib->mSDFData.mNumSDFs;
    lib->mSDFData.mFreeList[freeIndex] = int(sdfIndex - 1);
}

// flexdecomp: flex.cu:5885 NvFlexGetDistanceFields
int NvFlexGetDistanceFields(NvFlexLibrary* lib, NvFlexDistanceFieldId* sdfs, int n)
{
    const std::vector<unsigned int>& registered = idVector(lib->mResourceTracker.mSDF);

    const unsigned long count = (unsigned long)registered.size();
    const unsigned long limit = (unsigned long)(long)n;

    const unsigned long numToCopy = count <= limit ? count : limit;

    std::memcpy(sdfs, registered.data(), numToCopy * sizeof(unsigned int));

    return int((unsigned long)registered.size());
}

// flexdecomp: flex.cu:5890 NvFlexUpdateDistanceField
void NvFlexUpdateDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId sdfIndex, int dimx,
                               int dimy, int dimz, NvFlexBuffer* field)
{
    if (sdfIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid signed distance field", kFlexSourceFile,
                        0x1708);
        }
        return;
    }

    const int index = int(sdfIndex - 1);
    FlexSDFInternal* sdfHost = lib->mSDFData.mSDFTableHost + index;

    if (sdfHost->mDim[0] != dimx || sdfHost->mDim[1] != dimy || sdfHost->mDim[2] != dimz)
    {
        const cudaChannelFormatDesc channelDesc =
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        const cudaExtent extents = make_cudaExtent(dimx, dimy, dimz);

        cudaArray_t fieldArray = reinterpret_cast<cudaArray_t>(sdfHost->mFieldArray);

        cudaError_t err = cudaFreeArray(fieldArray);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1710, err, "cudaFreeArray(sdfHost->mFieldArray)");
        }

        err = cudaDestroyTextureObject(cudaTextureObject_t(sdfHost->mFieldTex));
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1711,
                        err, "cudaDestroyTextureObject(sdfHost->mFieldTex)");
        }

        err = cudaFreeArray(fieldArray);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1716, err, "cudaFreeArray(sdfHost->mFieldArray)");
        }

        fieldArray = 0;
        err = cudaMalloc3DArray(&fieldArray, &channelDesc, extents, 0);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1717, err,
                        "cudaMalloc3DArray(&sdfHost->mFieldArray, &channelDesc, extents)");
        }

        sdfHost->mFieldArray = fieldArray;

        cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = fieldArray;

        // binary evidence: addressMode[0..2]=Clamp, filterMode=Linear, the rest
        // (readMode=ElementType, normalizedCoords=false) stays zero
        cudaTextureDesc texDesc = {};
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.addressMode[2] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;

        cudaTextureObject_t tex = 0;
        err = cudaCreateTextureObject(&tex, &resDesc, &texDesc, 0);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1729, err,
                        "cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL)");
        }

        sdfHost->mFieldTex = tex;
    }

    _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::writeData3D(
        reinterpret_cast<cudaArray_t>(sdfHost->mFieldArray), field, dimx << 2, dimy, dimz);

    sdfHost->mDim[0] = dimx;
    sdfHost->mDim[1] = dimy;
    sdfHost->mDim[2] = dimz;

    const float invx = 1.0f / float(dimx);
    const float invy = 1.0f / float(dimy);
    const float invz = 1.0f / float(dimz);

    sdfHost->mInvDim[0] = invx;
    sdfHost->mInvDim[1] = invy;
    sdfHost->mInvDim[2] = invz;

    // refresh the host mirror (the shipped code copies the entry onto itself)
    lib->mSDFData.mSDFTableHost[index] = *sdfHost;

    cudaMemcpyAsync(lib->mSDFData.mSDFTableDevice + index, sdfHost,
                    sizeof(FlexSDFInternal), cudaMemcpyHostToDevice, 0);
}

// flexdecomp: flex.cu:5948 NvFlexUpdateSolver
void NvFlexUpdateSolver(NvFlexSolver* s, float dt, int substeps, bool enableTimers)
{
    s->mLib->mTimers.mEnabled = enableTimers;

    cudaEventRecord(reinterpret_cast<cudaEvent_t>(s->mLatencyTimerBegin), 0);
    s->mLatencyTimerBeginRecorded = true;

    if (s->numParticles != 0)
    {
        InitParams(s);

        for (int substep = 0; substep < substeps; ++substep)
        {
            Update(s, dt / float(substeps), substep, substeps);
        }

        s->mLatencyTimerEndRecorded = true;
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(s->mLatencyTimerEnd), 0);
        return;
    }

    cudaEventRecord(reinterpret_cast<cudaEvent_t>(s->mLatencyTimerEnd), 0);
    s->mLatencyTimerEndRecorded = true;
}

// flexdecomp: flex.cu:5984 NvFlexGetBounds
void NvFlexGetBounds(NvFlexSolver* s, NvFlexBuffer* lower, NvFlexBuffer* upper)
{
    _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
        lower, 0, s->mParticleLower, 0, 12);
    _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
        upper, 0, s->mParticleUpper, 0, 12);
}

// flexdecomp: flex.cu:5992 NvFlexRegisterSolverCallback
NvFlexSolverCallback NvFlexRegisterSolverCallback(NvFlexSolver* solver,
                                                  NvFlexSolverCallback callback,
                                                  NvFlexSolverCallbackStage stage)
{
    NvFlexSolverCallback* callbacks =
        reinterpret_cast<NvFlexSolverCallback*>(solver->mCallbacks);

    const NvFlexSolverCallback previous = callbacks[int(stage)];

    callbacks[int(stage)].function = callback.function;
    callbacks[int(stage)].userData = callback.userData;

    return previous;
}

// flexdecomp: flex.cu:6003 NvFlexSetDebug
void NvFlexSetDebug(NvFlexSolver* s, bool enable)
{
    s->mDebug = enable;
}

// flexdecomp: flex.cu:6075 NvFlexSetShapes
void NvFlexSetShapes(NvFlexSolver* s, NvFlexBuffer* geometry, NvFlexBuffer* shapePositions,
                     NvFlexBuffer* shapeRotations, NvFlexBuffer* prevShapePositions,
                     NvFlexBuffer* prevShapeRotations, NvFlexBuffer* shapeFlags, int numShapes)
{
    const int kNumThreadsPerBlock = 256;

    if (numShapes == 0)
    {
        s->mShapes.mNumShapes = 0;
        return;
    }

    if (s->mShapes.mMaxShapes < (unsigned int)numShapes)
    {
        const unsigned int numToAlloc = (static_cast<unsigned int>(numShapes) * 3) / 2;

        FlexDeviceFreeInternal(s->mShapes.mGeometry);
        FlexDeviceFreeInternal(s->mShapes.mAabbMin);
        FlexDeviceFreeInternal(s->mShapes.mAabbMax);
        FlexDeviceFreeInternal(s->mShapes.mPositions);
        FlexDeviceFreeInternal(s->mShapes.mRotations);
        FlexDeviceFreeInternal(s->mShapes.mPrevPositions);
        FlexDeviceFreeInternal(s->mShapes.mPrevRotations);
        FlexDeviceFreeInternal(s->mShapes.mFlags);

        const size_t numBytes = size_t(int(numToAlloc)) << 4;

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mGeometry), numBytes,
                                 "&s->mShapes.mGeometry", kFlexSourceFile, 0x17d4);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mAabbMin), numBytes,
                                 "&s->mShapes.mAabbMin", kFlexSourceFile, 0x17d5);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mAabbMax), numBytes,
                                 "&s->mShapes.mAabbMax", kFlexSourceFile, 0x17d6);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mPositions), numBytes,
                                 "&s->mShapes.mPositions", kFlexSourceFile, 0x17d7);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mRotations), numBytes,
                                 "&s->mShapes.mRotations", kFlexSourceFile, 0x17d8);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mPrevPositions), numBytes,
                                 "&s->mShapes.mPrevPositions", kFlexSourceFile, 0x17d9);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mPrevRotations), numBytes,
                                 "&s->mShapes.mPrevRotations", kFlexSourceFile, 0x17da);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&s->mShapes.mFlags),
                                 size_t(int(numToAlloc)) * 4, "&s->mShapes.mFlags",
                                 kFlexSourceFile, 0x17db);

        DestroyTexture(s->mTextures.shapeGeometryTex);
        DestroyTexture(s->mTextures.shapePositionsTex);
        DestroyTexture(s->mTextures.shapeRotationsTex);
        DestroyTexture(s->mTextures.shapePrevPositionsTex);
        DestroyTexture(s->mTextures.shapePrevRotationsTex);

        const int textureBytes = int(numToAlloc) << 4;

        s->mTextures.shapeGeometryTex =
            CreateTexture(reinterpret_cast<const Vec4*>(s->mShapes.mGeometry), textureBytes);
        s->mTextures.shapePositionsTex =
            CreateTexture(s->mShapes.mPositions, textureBytes);
        s->mTextures.shapeRotationsTex =
            CreateTexture(s->mShapes.mRotations, textureBytes);
        s->mTextures.shapePrevPositionsTex =
            CreateTexture(s->mShapes.mPrevPositions, textureBytes);
        s->mTextures.shapePrevRotationsTex =
            CreateTexture(s->mShapes.mPrevRotations, textureBytes);

        s->mShapes.mMaxShapes = numToAlloc;
    }

    if (geometry)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mGeometry, 0, geometry, 0, numShapes << 4);
    }

    if (shapePositions)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mPositions, 0, shapePositions, 0, numShapes << 4);
    }

    if (shapeRotations)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mRotations, 0, shapeRotations, 0, numShapes << 4);
    }

    if (prevShapePositions == 0)
    {
        prevShapePositions = shapePositions;
    }

    if (prevShapeRotations == 0)
    {
        prevShapeRotations = shapeRotations;
    }

    if (prevShapePositions)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mPrevPositions, 0, prevShapePositions, 0, numShapes << 4);
    }

    if (prevShapeRotations)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mPrevRotations, 0, prevShapeRotations, 0, numShapes << 4);
    }

    if (shapeFlags)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
            s->mShapes.mFlags, 0, shapeFlags, 0, numShapes * 4);
    }

    s->mShapes.mNumShapes = numShapes;

    const int kNumBlocks = (numShapes + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock;

    TransformShapeBounds<<<kNumBlocks, kNumThreadsPerBlock>>>(
        reinterpret_cast<Vec4*>(s->mShapes.mAabbMin),
        reinterpret_cast<Vec4*>(s->mShapes.mAabbMax),
        s->mShapes.mPositions,
        reinterpret_cast<const Quat*>(s->mShapes.mRotations),
        s->mShapes.mPrevPositions,
        reinterpret_cast<const Quat*>(s->mShapes.mPrevRotations),
        reinterpret_cast<const NvFlexCollisionGeometry*>(s->mShapes.mGeometry),
        s->mShapes.mFlags,
        numShapes,
        s->mLib->mConvexMeshData.mConvexes,
        s->mLib->mTriangleMeshData.mMeshTableDevice);

    LinearBVHBuilderGPUBuild(s->mLib->mBVHBuilder, s->mLib, s->mShapes.mBVH,
                             reinterpret_cast<const Vec4*>(s->mShapes.mAabbMin),
                             reinterpret_cast<const Vec4*>(s->mShapes.mAabbMax),
                             s->mShapes.mFlags, numShapes, 0);
}

// flexdecomp: flex.cu:6169 NvFlexGetContacts
void NvFlexGetContacts(NvFlexSolver* s, NvFlexBuffer* planes, NvFlexBuffer* velocities,
                       NvFlexBuffer* indices, NvFlexBuffer* counts)
{
    if (planes)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            planes, 0, s->mStaticContactPlanes, 0,
            s->mMaxParticles * s->mMaxContactsPerParticle * 16);
    }

    if (velocities)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            velocities, 0, s->mStaticContactVelocities, 0,
            s->mMaxParticles * s->mMaxContactsPerParticle * 16);
    }

    if (indices)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            indices, 0, s->mReverseLookup, 0, s->mMaxParticles << 2);
    }

    if (counts)
    {
        _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::ReadData(
            counts, 0, s->mStaticContactCounts, 0, s->mMaxParticles << 2);
    }
}

// flexdecomp: flex.cu:6206 NvFlexCreateConvexMesh
NvFlexConvexMeshId NvFlexCreateConvexMesh(NvFlexLibrary* lib)
{
    NvFlexConvexMeshId newIndex = 0;

    const int index = lib->mConvexMeshData.mNumConvexes;

    if (index < lib->mConvexMeshData.mMaxConvexes)
    {
        lib->mConvexMeshData.mNumConvexes = index + 1;

        newIndex = NvFlexConvexMeshId(lib->mConvexMeshData.mFreeList[index] + 1);
    }

    pushRegisteredId(idVector(lib->mResourceTracker.mConvexMesh), newIndex);

    return newIndex;
}

// flexdecomp: flex.cu:6216 NvFlexDestroyConvexMesh
void NvFlexDestroyConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convexIndex)
{
    if (convexIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid convex mesh", kFlexSourceFile, 0x184e);
        }
        return;
    }

    std::vector<unsigned int>& convexMeshes = idVector(lib->mResourceTracker.mConvexMesh);
    convexMeshes.erase(std::remove(convexMeshes.begin(), convexMeshes.end(),
                                   (unsigned long)convexIndex),
                       convexMeshes.end());

    const int freeIndex = --lib->mConvexMeshData.mNumConvexes;
    lib->mConvexMeshData.mFreeList[freeIndex] = int(convexIndex - 1);
}

// flexdecomp: flex.cu:6230 NvFlexGetConvexMeshes
int NvFlexGetConvexMeshes(NvFlexLibrary* lib, NvFlexConvexMeshId* meshes, int n)
{
    const std::vector<unsigned int>& registered = idVector(lib->mResourceTracker.mConvexMesh);

    const unsigned long count = (unsigned long)registered.size();
    const unsigned long limit = (unsigned long)(long)n;

    const unsigned long numToCopy = count <= limit ? count : limit;

    std::memcpy(meshes, registered.data(), numToCopy * sizeof(unsigned int));

    return int((unsigned long)registered.size());
}

// flexdecomp: flex.cu:6235 NvFlexUpdateConvexMesh
void NvFlexUpdateConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId convexIndex,
                            NvFlexBuffer* planes, int numPlanes, const float* lower,
                            const float* upper)
{
    const int kMaxPlanesPerConvex = 32;

    if (convexIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to update an invalid convex mesh", kFlexSourceFile, 0x1861);
        }
        return;
    }

    if (numPlanes < kMaxPlanesPerConvex + 1)
    {
        FlexConvexMeshInternal convex;

        convex.mPlaneCount = numPlanes;
        convex.mPlaneOffset = lib->mConvexMeshData.mMaxPlanesPerConvex * int(convexIndex - 1);
        convex.mLower = makeVec3(lower);
        convex.mUpper = makeVec3(upper);

        cudaMemcpyAsync(lib->mConvexMeshData.mPlanes + convex.mPlaneOffset, planes->ptr,
                        size_t(numPlanes) << 4, cudaMemcpyHostToDevice, 0);
        cudaMemcpyAsync(lib->mConvexMeshData.mConvexes + int(convexIndex - 1), &convex,
                        sizeof(FlexConvexMeshInternal), cudaMemcpyHostToDevice, 0);

        NvFlexSetFence(lib);
        return;
    }

    if (libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogWarning,
                    "Trying to update a convex mesh with too many planes %d > %d", kFlexSourceFile,
                    0x1867, int(numPlanes), kMaxPlanesPerConvex);
    }
}

// flexdecomp: flex.cu:6257 NvFlexGetConvexMeshBounds
void NvFlexGetConvexMeshBounds(NvFlexLibrary* lib, const NvFlexConvexMeshId meshIndex,
                               float* lower, float* upper)
{
    if (meshIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to query bounds for an invalid convex mesh", kFlexSourceFile,
                        0x1877);
        }
        return;
    }

    cudaError_t err = cudaMemcpy(lower,
                                 &lib->mConvexMeshData.mConvexes[meshIndex - 1].mLower,
                                 sizeof(Vec3), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess && libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x187c, err,
                    "cudaMemcpy(lower, &lib->mConvexMeshData.mConvexes[meshIndex-1].mLower.x, sizeof(Vec3), cudaMemcpyDeviceToHost)");
    }

    err = cudaMemcpy(upper, &lib->mConvexMeshData.mConvexes[meshIndex - 1].mUpper, sizeof(Vec3),
                     cudaMemcpyDeviceToHost);
    if (err != cudaSuccess && libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x187d, err,
                    "cudaMemcpy(upper, &lib->mConvexMeshData.mConvexes[meshIndex-1].mUpper.x, sizeof(Vec3), cudaMemcpyDeviceToHost)");
    }
}

// flexdecomp: flex.cu:6272 NvFlexCreateTriangleMesh
NvFlexTriangleMeshId NvFlexCreateTriangleMesh(NvFlexLibrary* lib)
{
    NvFlexTriangleMeshId newIndex = 0;

    int index = lib->mTriangleMeshData.mNumMeshes;

    if (index < lib->mTriangleMeshData.mMaxMeshes)
    {
        lib->mTriangleMeshData.mNumMeshes = index + 1;

        index = lib->mTriangleMeshData.mFreeList[index];
        newIndex = NvFlexTriangleMeshId(index + 1);

        FlexTriangleMeshInternal* mesh = lib->mTriangleMeshData.mMeshTableHost + index;
        std::memset(mesh, 0, sizeof(FlexTriangleMeshInternal));
    }

    pushRegisteredId(idVector(lib->mResourceTracker.mTriangleMesh), newIndex);

    return newIndex;
}

// flexdecomp: flex.cu:6282 NvFlexDestroyTriangleMesh
void NvFlexDestroyTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId meshIndex)
{
    if (meshIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid triangle mesh", kFlexSourceFile, 0x1890);
        }
        return;
    }

    std::vector<unsigned int>& triangleMeshes = idVector(lib->mResourceTracker.mTriangleMesh);
    triangleMeshes.erase(std::remove(triangleMeshes.begin(), triangleMeshes.end(),
                                     (unsigned long)meshIndex),
                         triangleMeshes.end());

    const int index = int(meshIndex - 1);
    FlexTriangleMeshInternal* mesh = lib->mTriangleMeshData.mMeshTableHost + index;

    FreeBVH(mesh->mBVH);

    DeviceBufferVec4& vertices = verticesBuffer(mesh);
    DestroyTexture(cudaTextureObject_t(vertices.mTexture));
    vertices.mTexture = 0;
    FlexDeviceFreeInternal(vertices.mPtr);
    vertices.mPtr = 0;
    vertices.mCapacity = 0;

    DestroyTexture(cudaTextureObject_t(mesh->mIndices.mTexture));
    mesh->mIndices.mTexture = 0;
    FlexDeviceFreeInternal(mesh->mIndices.mPtr);
    mesh->mIndices.mPtr = 0;
    mesh->mIndices.mCapacity = 0;

    std::memset(mesh, 0, sizeof(FlexTriangleMeshInternal));

    const int freeIndex = --lib->mTriangleMeshData.mNumMeshes;
    lib->mTriangleMeshData.mFreeList[freeIndex] = index;
}

// flexdecomp: flex.cu:6296 NvFlexGetTriangleMeshes
int NvFlexGetTriangleMeshes(NvFlexLibrary* lib, NvFlexTriangleMeshId* meshes, int n)
{
    const std::vector<unsigned int>& registered = idVector(lib->mResourceTracker.mTriangleMesh);

    const unsigned long count = (unsigned long)registered.size();
    const unsigned long limit = (unsigned long)(long)n;

    const unsigned long numToCopy = count <= limit ? count : limit;

    std::memcpy(meshes, registered.data(), numToCopy * sizeof(unsigned int));

    return int((unsigned long)registered.size());
}

// flexdecomp: flex.cu:6301 NvFlexUpdateTriangleMesh
void NvFlexUpdateTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId meshIndex,
                              NvFlexBuffer* positions, NvFlexBuffer* indices, int numPositions,
                              int numTris, const float* lower, const float* upper)
{
    const int kNumThreadsPerBlock = 256;

    if (meshIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid triangle mesh", kFlexSourceFile, 0x18a3);
        }
        return;
    }

    if (lower == 0 || upper == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Spatial bounds of mesh cannot be null", kFlexSourceFile, 0x18a9);
        }
        return;
    }

    const int index = int(meshIndex - 1);
    FlexTriangleMeshInternal* mesh = lib->mTriangleMeshData.mMeshTableHost + index;

    if (numTris == 0 || numPositions == 0)
    {
        mesh->mNumTris = 0;
        mesh->mNumVertices = 0;
        return;
    }

    mesh->mNumTris = numTris;
    mesh->mNumVertices = numPositions;

    DeviceBufferVec4& vertices = verticesBuffer(mesh);

    if (mesh->mIndices.mCapacity < numTris * 3)
    {
        DestroyTexture(cudaTextureObject_t(mesh->mIndices.mTexture));
        mesh->mIndices.mTexture = 0;
        FlexDeviceFreeInternal(mesh->mIndices.mPtr);
        mesh->mIndices.mPtr = 0;
        mesh->mIndices.mCapacity = 0;

        const int capacity = (numTris * 9) / 2;

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&mesh->mIndices.mPtr),
                                 size_t(capacity) << 2, "&mPtr", kUtilSourceFile, 0x77);

        mesh->mIndices.mTexture = CreateTexture(mesh->mIndices.mPtr,
                                                mesh->mIndices.mCapacity * 4);
        mesh->mIndices.mCapacity = capacity;
    }

    // DeviceBuffer<XVector4<float> >::resize(numPositions), inlined by the
    // shipped build (flex.cu:6316 / util.h:159)
    if (vertices.mCapacity < numPositions)
    {
        DestroyTexture(cudaTextureObject_t(vertices.mTexture));
        vertices.mTexture = 0;
        FlexDeviceFreeInternal(vertices.mPtr);
        vertices.mPtr = 0;
        vertices.mCapacity = 0;

        const int capacity = (numPositions * 3) / 2;

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&vertices.mPtr), size_t(capacity) << 4,
                                 "&mPtr", kUtilSourceFile, 0x77);

        vertices.mTexture = CreateTexture(vertices.mPtr, vertices.mCapacity * 16);
        vertices.mCapacity = capacity;
    }

    _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
        vertices.mPtr, 0, positions, 0, numPositions << 4);
    _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207::WriteData(
        mesh->mIndices.mPtr, 0, indices, 0, numTris * 12);

    Bounds totalBounds;
    totalBounds.lower = makeVec3(lower);
    totalBounds.upper = makeVec3(upper);

    const int numBounds = numTris << 4;

    DeviceBufferVec4& triLowers = staticTriBuffer(lib->mStaticTriLowers);
    if (triLowers.mCapacity < numBounds)
    {
        DestroyTexture(cudaTextureObject_t(triLowers.mTexture));
        triLowers.mTexture = 0;
        FlexDeviceFreeInternal(triLowers.mPtr);
        triLowers.mPtr = 0;
        triLowers.mCapacity = 0;

        const int capacity = (numBounds * 3) / 2;

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&triLowers.mPtr), size_t(capacity) << 4,
                                 "&mPtr", kUtilSourceFile, 0x77);

        triLowers.mTexture = CreateTexture(triLowers.mPtr, triLowers.mCapacity * 16);
        triLowers.mCapacity = capacity;
    }

    DeviceBufferVec4& triUppers = staticTriBuffer(lib->mStaticTriUppers);
    if (triUppers.mCapacity < numBounds)
    {
        DestroyTexture(cudaTextureObject_t(triUppers.mTexture));
        triUppers.mTexture = 0;
        FlexDeviceFreeInternal(triUppers.mPtr);
        triUppers.mPtr = 0;
        triUppers.mCapacity = 0;

        const int capacity = (numBounds * 3) / 2;

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&triUppers.mPtr), size_t(capacity) << 4,
                                 "&mPtr", kUtilSourceFile, 0x77);

        triUppers.mTexture = CreateTexture(triUppers.mPtr, triUppers.mCapacity * 16);
        triUppers.mCapacity = capacity;
    }

    const int numBlocks = (mesh->mNumTris + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock;

    ComputeTriangleBounds<<<numBlocks, kNumThreadsPerBlock>>>(
        vertices.mPtr, mesh->mIndices.mPtr, mesh->mNumTris, triLowers.mPtr, triUppers.mPtr);

    LinearBVHBuilderGPUBuild(lib->mBVHBuilder, lib, mesh->mBVH, triLowers.mPtr, triUppers.mPtr, 0,
                             numTris, &totalBounds);

    mesh->mLower = totalBounds.lower;
    mesh->mUpper = totalBounds.upper;

    cudaMemcpyAsync(lib->mTriangleMeshData.mMeshTableDevice + index, mesh,
                    sizeof(FlexTriangleMeshInternal), cudaMemcpyHostToDevice, 0);
}

// flexdecomp: flex.cu:6359 NvFlexGetTriangleMeshBounds
void NvFlexGetTriangleMeshBounds(NvFlexLibrary* lib, const NvFlexTriangleMeshId meshIndex,
                                 float* lower, float* upper)
{
    if (meshIndex == 0)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Trying to destroy an invalid triangle mesh", kFlexSourceFile, 0x18dd);
        }
        return;
    }

    const FlexTriangleMeshInternal* mesh = lib->mTriangleMeshData.mMeshTableHost + (meshIndex - 1);

    lower[0] = mesh->mLower.x;
    lower[1] = mesh->mLower.y;
    lower[2] = mesh->mLower.z;

    upper[0] = mesh->mUpper.x;
    upper[1] = mesh->mUpper.y;
    upper[2] = mesh->mUpper.z;
}

// flexdecomp: flex.cu:6375 NvFlexGetDeviceName
const char* NvFlexGetDeviceName(NvFlexLibrary* lib)
{
    return lib->mDeviceName;
}

// flexdecomp: flex.cu:6380 NvFlexGetShapeBVH
void NvFlexGetShapeBVH(NvFlexSolver* s, void* bvh)
{
    const BVH& shapeBVH = s->mShapes.mBVH;

    BVH* dst = reinterpret_cast<BVH*>(bvh);
    dst->mNodeLowersTex = shapeBVH.mNodeLowersTex;
    dst->mNodeUppersTex = shapeBVH.mNodeUppersTex;
    dst->mRootNode = shapeBVH.mRootNode;
    dst->mNodeLowers = shapeBVH.mNodeLowers;
    dst->mNodeUppers = shapeBVH.mNodeUppers;
    dst->mNumNodes = shapeBVH.mNumNodes;
    dst->mMaxNodes = shapeBVH.mMaxNodes;
    dst->mMaxDepth = shapeBVH.mMaxDepth;
}

// flexdecomp: flex.cu:6389 NvFlexGetDeviceLatency
float NvFlexGetDeviceLatency(NvFlexSolver* solver, unsigned long long* begin,
                             unsigned long long* end, unsigned long long* frequency)
{
    NvFlexLibrary* lib = solver->mLib;

    if (solver->mLatencyTimerBeginRecorded && solver->mLatencyTimerEndRecorded)
    {
        float elapsedTime = 0.0f;

        cudaError_t err = cudaEventSynchronize(
            reinterpret_cast<cudaEvent_t>(solver->mLatencyTimerEnd));
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x18ff, err,
                        "cudaEventSynchronize(solver->mLatencyTimerEnd)");
        }

        err = cudaEventElapsedTime(&elapsedTime,
                                   reinterpret_cast<cudaEvent_t>(solver->mLatencyTimerBegin),
                                   reinterpret_cast<cudaEvent_t>(solver->mLatencyTimerEnd));
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1902, err,
                        "cudaEventElapsedTime(&elapsedTime, solver->mLatencyTimerBegin, solver->mLatencyTimerEnd)");
        }

        solver->mLatencyTimerBeginRecorded = false;
        solver->mLatencyTimerEndRecorded = false;

        return elapsedTime;
    }

    return 0.0f;
}

// flexdecomp: flex.cu:6414 _Z8getTimerR10CudaTimersPKc
float getTimer(CudaTimers& timers, const char* name)
{
    CudaTimerNameMap& map = timerNameMap(&timers);

    const CudaTimerNameRange range = timerNameRange(map, name);

    CudaRangeTimerVector& elems = timerPoolElems(timers.mTimerPool);

    float retval = 0.0f;

    for (CudaTimerNameMap::iterator it = range.first; it != range.second; ++it)
    {
        retval += elems[it->second].mTime;
    }

    return retval;
}

// flexdecomp: flex.cu:6430 NvFlexGetTimers
void NvFlexGetTimers(NvFlexSolver* solver, NvFlexTimers* timers)
{
    NvFlexLibrary* lib = solver->mLib;

    if (lib->mTimers.mEnabled)
    {
        CudaTimerNameMap& map = timerNameMap(&lib->mTimers);
        CudaRangeTimerVector& elems = timerPoolElems(lib->mTimers.mTimerPool);

        for (CudaTimerNameMap::iterator it = map.begin(); it != map.end(); ++it)
        {
            CudaRangeTimer* timer = &elems[it->second];

            cudaError_t err = cudaEventSynchronize(
                reinterpret_cast<cudaEvent_t>(timer->mEnd));
            if (err != cudaSuccess && libErrorFunc(lib))
            {
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x1927, err, "cudaEventSynchronize(timer.mEnd)");
            }

            float elapsedTime = 0.0f;
            err = cudaEventElapsedTime(&elapsedTime, reinterpret_cast<cudaEvent_t>(timer->mBegin),
                                       reinterpret_cast<cudaEvent_t>(timer->mEnd));
            if (err != cudaSuccess && libErrorFunc(lib))
            {
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x1929, err,
                            "cudaEventElapsedTime(&elapsedTime, timer.mBegin, timer.mEnd)");
            }

            timer->mTime = elapsedTime;
        }

        CudaTimers& ctimers = lib->mTimers;

        timers->predict = getTimer(ctimers, "Predict");
        timers->createCellIndices = getTimer(ctimers, "CreateCellIndices");
        timers->sortCellIndices = getTimer(ctimers, "SortCellIndices");
        timers->createGrid = getTimer(ctimers, "Create Grid");
        timers->reorder = getTimer(ctimers, "ReorderParticles");
        timers->collideParticles = getTimer(ctimers, "CollideParticles");
        timers->collideShapes = getTimer(ctimers, "Collide Shapes");
        timers->collideTriangles = getTimer(ctimers, "Collide Triangles");
        timers->collideFields = getTimer(ctimers, "CollideFields");
        timers->calculateDensity = getTimer(ctimers, "Calculate Density");
        timers->solveDensities = getTimer(ctimers, "Solve Densities");
        timers->solveVelocities = getTimer(ctimers, "Solve Velocities");
        timers->solveShapes = getTimer(ctimers, "Solve Shapes");
        timers->solveSprings = getTimer(ctimers, "Solve Springs");
        timers->solveContacts = getTimer(ctimers, "Solve Static Contacts");
        timers->solveInflatables = getTimer(ctimers, "Solve Inflatables");
        timers->applyDeltas = getTimer(ctimers, "Apply Deltas");
        timers->calculateAnisotropy = getTimer(ctimers, "Calc Anisotropy");
        timers->updateDiffuse = getTimer(ctimers, "Update Diffuse");
        timers->updateTriangles = getTimer(ctimers, "Update Triangles");
        timers->updateNormals = getTimer(ctimers, "Update Normals");
        timers->finalize = getTimer(ctimers, "Finalize");

        const float updateBounds = getTimer(ctimers, "UpdateBounds");
        timers->updateBounds = updateBounds;

        timers->total = timers->predict + timers->createCellIndices + timers->sortCellIndices +
                        timers->createGrid + timers->reorder + timers->collideParticles +
                        timers->collideShapes + timers->collideTriangles + timers->collideFields +
                        timers->calculateDensity + timers->solveDensities + timers->solveVelocities +
                        timers->solveShapes + timers->solveSprings + timers->solveContacts +
                        timers->solveInflatables + timers->applyDeltas +
                        timers->calculateAnisotropy + timers->updateDiffuse +
                        timers->updateTriangles + timers->updateNormals + timers->finalize +
                        updateBounds;
    }

    // the shipped code resets the pool and drops every registered timer
    lib->mTimers.mTimerPool->mUsedElemCount = 0;
    timerNameMap(&lib->mTimers).clear();
}

// flexdecomp: flex.cu:6502 NvFlexGetDetailTimers
int NvFlexGetDetailTimers(NvFlexSolver* solver, NvFlexDetailTimer** timers)
{
    int totalTimers = 0;

    NvFlexLibrary* lib = solver->mLib;

    if (lib->mTimers.mEnabled)
    {
        std::vector<std::string> names;
        std::vector<float> times;

        CudaTimerNameMap& map = timerNameMap(&lib->mTimers);
        CudaRangeTimerVector& elems = timerPoolElems(lib->mTimers.mTimerPool);

        for (CudaTimerNameMap::iterator it = map.begin(); it != map.end(); ++it)
        {
            CudaRangeTimer* timer = &elems[it->second];

            cudaError_t err = cudaEventSynchronize(
                reinterpret_cast<cudaEvent_t>(timer->mEnd));
            if (err != cudaSuccess && libErrorFunc(lib))
            {
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x1974, err, "cudaEventSynchronize(timer.mEnd)");
            }

            float elapsedTime = 0.0f;
            err = cudaEventElapsedTime(&elapsedTime, reinterpret_cast<cudaEvent_t>(timer->mBegin),
                                       reinterpret_cast<cudaEvent_t>(timer->mEnd));
            if (err != cudaSuccess && libErrorFunc(lib))
            {
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x1976, err,
                            "cudaEventElapsedTime(&elapsedTime, timer.mBegin, timer.mEnd)");
            }

            timer->mTime = elapsedTime;
        }

        CudaTimers& ctimers = lib->mTimers;

        names.push_back("Predict");
        times.push_back(getTimer(ctimers, "Predict"));
        names.push_back("Create Cell Indices");
        times.push_back(getTimer(ctimers, "CreateCellIndices"));
        names.push_back("Sort Cell Indices");
        times.push_back(getTimer(ctimers, "SortCellIndices"));
        names.push_back("Create Grid");
        times.push_back(getTimer(ctimers, "Create Grid"));
        names.push_back("Reorder Particles");
        times.push_back(getTimer(ctimers, "ReorderParticles"));
        names.push_back("Collide Particles");
        times.push_back(getTimer(ctimers, "CollideParticles"));
        names.push_back("Collide Shapes");
        times.push_back(getTimer(ctimers, "Collide Shapes"));
        names.push_back("Collide Triangles");
        times.push_back(getTimer(ctimers, "Collide Triangles"));
        names.push_back("Collide Fields");
        times.push_back(getTimer(ctimers, "CollideFields"));
        names.push_back("Calculate Density");
        times.push_back(getTimer(ctimers, "Calculate Density"));
        names.push_back("Solve Densities");
        times.push_back(getTimer(ctimers, "Solve Densities"));
        names.push_back("Solve Velocities");
        times.push_back(getTimer(ctimers, "Solve Velocities"));
        names.push_back("Solve Shapes");
        times.push_back(getTimer(ctimers, "Solve Shapes"));
        names.push_back("Solve Springs");
        times.push_back(getTimer(ctimers, "Solve Springs"));
        names.push_back("Solve Static Contacts");
        times.push_back(getTimer(ctimers, "Solve Static Contacts"));
        names.push_back("Solve Inflatables");
        times.push_back(getTimer(ctimers, "Solve Inflatables"));
        names.push_back("Apply Deltas");
        times.push_back(getTimer(ctimers, "Apply Deltas"));
        names.push_back("Calc Anisotropy");
        times.push_back(getTimer(ctimers, "Calc Anisotropy"));
        names.push_back("Update Diffuse");
        times.push_back(getTimer(ctimers, "Update Diffuse"));
        names.push_back("Update Triangles");
        times.push_back(getTimer(ctimers, "Update Triangles"));
        names.push_back("Update Normals");
        times.push_back(getTimer(ctimers, "Update Normals"));
        names.push_back("Finalize");
        times.push_back(getTimer(ctimers, "Finalize"));
        names.push_back("Update Bounds");
        times.push_back(getTimer(ctimers, "UpdateBounds"));

        float total = 0.0f;

        if (solver->mLatencyTimerBeginRecorded && solver->mLatencyTimerEndRecorded)
        {
            cudaError_t err = cudaEventElapsedTime(
                &total, reinterpret_cast<cudaEvent_t>(solver->mLatencyTimerBegin),
                reinterpret_cast<cudaEvent_t>(solver->mLatencyTimerEnd));
            if (err != cudaSuccess && libErrorFunc(lib))
            {
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x19ae, err,
                            "cudaEventElapsedTime(&total, solver->mLatencyTimerBegin, solver->mLatencyTimerEnd)");
            }
        }

        names.push_back("Total");
        times.push_back(total);

        totalTimers = int(times.size());

        NvFlexDetailTimer* out =
            static_cast<NvFlexDetailTimer*>(malloc(size_t(names.size()) * sizeof(NvFlexDetailTimer)));
        *timers = out;

        for (size_t i = 0; i < names.size(); ++i)
        {
            out[i].name = strdup(names[i].c_str());
            out[i].time = times[i];
        }
    }

    lib->mTimers.mTimerPool->mUsedElemCount = 0;
    timerNameMap(&lib->mTimers).clear();

    return totalTimers;
}

// flexdecomp: flex.cu:6597 NvFlexCopyDeviceToHost
void NvFlexCopyDeviceToHost(NvFlexSolver* solver, NvFlexBuffer* pDevice, void* pHost, int size,
                            int stride)
{
    if (size > 0)
    {
        NvFlexWaitFence(solver->mLib);
        cudaMemcpy(pHost, pDevice->ptr, size_t(size * stride), cudaMemcpyDeviceToHost);
    }
}

// flexdecomp: flex.cu:6608 NvFlexComputeWaitForGraphics
void NvFlexComputeWaitForGraphics(NvFlexLibrary* lib)
{
    (void)lib;
}
