// Reconstructed FleX 1.2.0 CUDA solver - host layer, part 2.
//
// Slice: units/_slices/flex-host-2.txt (src/cuda/flex.cu:468 .. 4860).
//
// Evidence used for every unit:
//   work/decomp/nvflex.flex/cudaflex.c   Ghidra pseudocode, by address
//   work/dwarf/nvflex.flex.json          DIEs: decl file/line, locals, offsets
//   work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudaflex.disasm.txt
//   work/binary/gpu/dbg/nvflex.flex/sm_30.ptx   (kernel signatures / launch)
//   recon/include/nvflex/recovered.h     exact recovered layouts
//
// The shipped object's .text is identical in the debug and release archives, so
// the -O3 debug-object line table is also the release baseline.

// SolveShapes<true/false> are blocked kernel units (no definition in the
// reconstructed library yet); see the comment at their launch site.
#ifndef FLEXDECOMP_HAVE_SOLVESHAPES
#define FLEXDECOMP_HAVE_SOLVESHAPES 0
#endif
#include "nvflex/all.h"
#include "NvFlex.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <cmath>
#include <cstring>
#include <map>
#include <new>
#include <vector>

// ---------------------------------------------------------------------------
// module-wide symbols owned by other translation units
// ---------------------------------------------------------------------------
// Constant buffers: defined once in flex.cu (integrator-owned).
extern __constant__ KernelParams gParams;
extern __constant__ KernelBounds gBounds;

// Defined here: the shipped object's unmangled .bss slot used by the
// NvFlexAcquireContext/NvFlexRestoreContext pair (nm: B g_savedContext).
CUcontext g_savedContext;

// nvflex.util
extern void FlexLogFunc(NvFlexErrorCallback func, NvFlexErrorSeverity type, const char* msg,
                        const char* file, int line, ...);
bool CudaSafeDelayLoad(NvFlexErrorCallback errorFunc, int deviceOrdinal, CUcontext* outContext);
CUcontext CudaGetContext();
void CudaSetContext(CUcontext ctx);
void FlexDeviceMallocInternal(void** ptr, size_t size, const char* label, const char* file,
                              int line);
void FlexDeviceFreeInternal(void* ptr);
cudaTextureObject_t CreateTexture(const Vec4* buffer, int sizeInBytes);
void DestroyTexture(cudaTextureObject_t tex);

// nvflex.bvh (LinearBVHBuilderGPU ctor/dtor carry the shipped C1/D1 mangling)
void LinearBVHBuilderGPUConstruct(LinearBVHBuilderGPU* self) __asm__("_ZN19LinearBVHBuilderGPUC1Ev");
void LinearBVHBuilderGPUDestroy(LinearBVHBuilderGPU* self) __asm__("_ZN19LinearBVHBuilderGPUD1Ev");
void FreeBVH(BVH& bvh);

// util.h inline members emitted out of line in this module
void cudaRangeTimerPoolCtor(CudaRangeTimerPool* self) __asm__("_ZN18CudaRangeTimerPoolC1Ev");
void cudaTimerCtor(CudaTimer* self, const char* name, CudaTimers& timers)
    __asm__("_ZN9CudaTimerC1EPKcR10CudaTimers");
void cudaTimerDtor(CudaTimer* self) __asm__("_ZN9CudaTimerD1Ev");
void NvFlexBuffer_construct(NvFlexBuffer* self, int elementCount, int elementByteStride, int type,
                            NvFlexLibrary* lib) __asm__("_ZN12NvFlexBufferC1EiiiP13NvFlexLibrary");
void NvFlexBuffer_destroy(NvFlexBuffer* self) __asm__("_ZN12NvFlexBufferD1Ev");
void* NvFlexBuffer_map(NvFlexBuffer* self) __asm__("_ZN12NvFlexBuffer3mapEv");
void NvFlexBuffer_unmap(NvFlexBuffer* self) __asm__("_ZN12NvFlexBuffer5unmapEv");

// nvflex.sort
void SortCellIndices(NvFlexLibrary* lib, int* cellIds, int* indices, int numParticles,
                     int maxCells);

// host API implemented in flex.part3.cu
void NvFlexDestroySolver(NvFlexSolver* solver);
void NvFlexDestroyTriangleMesh(NvFlexLibrary* lib, NvFlexTriangleMeshId mesh);
void NvFlexDestroyDistanceField(NvFlexLibrary* lib, NvFlexDistanceFieldId field);
void NvFlexDestroyConvexMesh(NvFlexLibrary* lib, NvFlexConvexMeshId mesh);

// flex.part1.cu (anonymous namespace of the original CUDA-generated wrapper)
namespace _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207 {
void ExecuteCallback(NvFlexSolver* solver, NvFlexSolverCallbackStage stage, float dt);
}

// ---------------------------------------------------------------------------
// device kernels (definitions live in the kernel slices)
// ---------------------------------------------------------------------------
extern __global__ void Predict(int numParticles, const int* indices, const Vec4* positions,
                               const int* phases, Vec3 velocity, float dt, Vec4* newPositions,
                               Vec3* outVelocities, float maxSpeed);
extern __global__ void CalculateBounds(int numParticles, const int* indices,
                                       const Vec4* newPositions, Vec3* lower, Vec3* upper);
extern __global__ void CreateCellIndices(int numParticles, const int* indices,
                                         const Vec4* positions, int* cellIds, int* indicesOut,
                                         float invCellSize);
extern __global__ void CreateGrid(int numParticles, const int* cellIds, int* cellStarts,
                                  int* cellEnds);
extern __global__ void ReorderParticles(int numParticles, const Vec4* positions,
                                        const Vec4* newPositions, const int* phases,
                                        const int* indices, int* reverseLookup,
                                        Vec4* sortedPositions, Vec4* sortedNewPositions,
                                        int* sortedPhases);
extern __global__ void CollideParticles(int numParticles, const int* cellStarts,
                                        const int* cellEnds, const int* indices,
                                        Vec3Iter sortedPositions, const Vec4* restPositions,
                                        const int* sortedPhases, float radiusSq, float invRadius,
                                        int maxNeighbors, int* neighbors, int* neighborCounts);
extern __global__ void CollideTriangles(FlexCollisionShapeData shapes, Vec3Iter sortedPositions,
                                        const int* sortedPhases, int numParticles, int* counts,
                                        Vec4* planes, Vec4* velocities, float startTime,
                                        float endTime);
extern __global__ void CollideShapes(FlexCollisionShapeData shapes, Vec3Iter sortedPositions,
                                     const int* sortedPhases, int numParticles, int* counts,
                                     Vec4* planes, Vec4* velocities, float startTime,
                                     float endTime);
extern __global__ void ContinuousShockPropagation(int numParticles, Vec4* sortedNewPositions,
                                                  int* neighborCounts);
template <bool T>
extern __global__ void CalculateDensity(int numParticles, Vec3Iter sortedNewPositions,
                                        const int* sortedPhases, const int* particleNeighbors,
                                        const int* particleNeighborCounts, const int* indices,
                                        const Vec4* staticContactPlanes, float* densities,
                                        Vec4* normals, const int* staticContactCounts);
template <bool T>
extern __global__ void SolveDensities(int numParticles, Vec3Iter sortedPositions,
                                      Vec3Iter sortedNewPositions, const int* sortedPhases,
                                      const float* densities, const Vec4* normals,
                                      const int* particleNeighbors,
                                      const int* particleNeighborCounts, const int* indices,
                                      float dt, Vec4* deltas);
extern __global__ void SolveDensitiesSimpleSolids(
    int numParticles, Vec3Iter sortedPositions, Vec3Iter sortedNewPositions,
    const int* sortedPhases, const float* densities, const Vec4* normals,
    const int* particleNeighbors, const int* particleNeighborCounts, const int* indices, float dt,
    Vec4* deltas);
extern __global__ void SolveDensitiesSimpleFluids(
    int numParticles, Vec3Iter sortedPositions, Vec3Iter sortedNewPositions,
    const int* sortedPhases, const float* densities, const Vec4* normals,
    const int* particleNeighbors, const int* particleNeighborCounts, const int* indices, float dt,
    Vec4* deltas);
extern __global__ void SolveSprings(int* springIndices, float* springLengths,
                                    float* springStiffness, const int* reverseLookup,
                                    int numSprings, Vec3Iter deltas);
template <int T>
extern __global__ void CalculateInflatableVolume(const FlexInflatable* inflatables,
                                                 Vec3Iter sortedNewPositions,
                                                 const int* reverseLookup,
                                                 const int* dynamicTriangles, float* pressures);
extern __global__ void SolveInflatableVolume(const FlexInflatable* inflatables,
                                             Vec3Iter sortedNewPositions,
                                             const int* reverseLookup, const int* dynamicTriangles,
                                             const float* pressures, Vec3Iter deltas, float maxSpeed);
template <bool T>
extern __global__ void SolveShapes(const int* rigidOffsets, const int* rigidIndices,
                                   const float* rigidCoefficients,
                                   const float* rigidPlasticThresholds,
                                   const float* rigidPlasticCreeps, const int* indices,
                                   Vec3* rigidLocalPositions, Vec4* rigidLocalNormals,
                                   Vec3Iter sortedPositions, Vec3Iter sortedNewPositions,
                                   Quat* rigidRotations, Vec3* rigidTranslations, Vec4* normals,
                                   Vec3Iter deltas, int numRigids, float unusedScale);
template <bool T>
extern __global__ void ApplyDeltas(int numParticles, Vec3Iter sortedPositions,
                                   Vec3Iter sortedNewPositions, const Vec4* deltas,
                                   const Vec4* planes, float relaxation, float damp);
template <bool T>
extern __global__ void SolveContacts(Vec3Iter sortedPositions, Vec3Iter sortedNewPositions,
                                     int numParticles, const int* counts, const Vec4* planes,
                                     const Vec4* velocities, const float* densities,
                                     Vec4* deltas);
extern __global__ void UpdateVelocities(int numParticles, Vec3Iter sortedPositions,
                                        Vec3Iter sortedNewPositions, Vec3Iter sortedNewVelocities,
                                        float invDt);
extern __global__ void CalculateVorticity(int numParticles, Vec3Iter sortedNewPositions,
                                          const Vec3* velocities, Vec3Iter sortedNewVelocities,
                                          int* sortedPhases, const int* particleNeighbors,
                                          const int* particleNeighborCounts,
                                          const Vec4* staticContactPlanes,
                                          const int* staticContactCounts, Vec4* curl);
template <bool T>
extern __global__ void SolveVelocities(int numParticles, Vec3Iter sortedNewPositions,
                                       const Vec3* velocities, Vec3Iter sortedNewVelocities,
                                       const Vec4* curl, const int* sortedPhases,
                                       const int* particleNeighbors,
                                       const int* particleNeighborCounts,
                                       const Vec4* staticContactPlanes,
                                       const int* staticContactCounts, const int* indices,
                                       float dt, Vec4* deltas, float* sortedDensities,
                                       float* potentials);
extern __global__ void UpdateTriangles(int numDynamicTris, Vec3Iter sortedNewPositions,
                                       Vec3Iter sortedNewVelocities, const int* reverseLookup,
                                       const int* dynamicTriangles, Vec3* dynamicTriangleNormals,
                                       Vec3 wind, Vec3Iter deltas, float dt);
extern __global__ void UpdateVertexNormals(int numDynamicTris, const int* reverseLookup,
                                           const int* dynamicTriangles,
                                           const Vec3* dynamicTriangleNormals,
                                           Vec3Iter tempNormals);
extern __global__ void NormalizeVertexNormals(int numParticles, const int* indices,
                                              const Vec4* tempNormals, Vec4* normals);
extern __global__ void Finalize(int numParticles, const Vec4* sortedNewPositions,
                                const Vec4* sortedNewVelocities, const float* sortedDensities,
                                const Vec4* deltas, const int* indices, Vec4* positions,
                                Vec3* velocities, float* densities, float maxSpeed,
                                float maxAccelerationDt);
extern __global__ void CalculateAnisotropy(int numParticles, const int* particleNeighbors,
                                           const int* particleNeighborCounts, const int* indices,
                                           const int* sortedPhases, const Vec4* sortedPositions,
                                           Vec4* anisotropy1, Vec4* anisotropy2,
                                           Vec4* anisotropy3);
extern __global__ void SmoothPositions(int numParticles, const Vec4* sortedNewPositions,
                                       const int* sortedPhases, const int* particleNeighbors,
                                       const int* particleNeighborCounts, const int* indices,
                                       Vec4* sortedPositions, Vec4* deltas);
extern __global__ void CreateDiffuseParticles(int numParticles, Vec3Iter sortedNewPositions,
                                              const float* potentials, Vec4* diffusePositions,
                                              Vec4* diffuseVelocities, int* numDiffuseParticles,
                                              int maxDiffuseParticles, const float* random,
                                              float diffuseDt);
extern __global__ void UpdateDiffuseParticles(
    Vec4* diffusePositions, Vec4* diffuseVelocities, const int* sortedPhases,
    const int* numDiffuseParticles, const int* cellStarts, const int* cellEnds, const int* indices,
    const int* particleNeighborCounts, Vec3Iter sortedNewVelocities, float radiusSq,
    float invRadius, float diffuseDt, float maxLifetime);
extern __global__ void ClampDiffuseParticleCount(int* numDiffuseParticles,
                                                 int* numDiffuseParticlesNew,
                                                 int maxDiffuseParticles);
extern __global__ void CompactDiffuseParticles(const Vec4* diffusePositions,
                                               const Vec4* diffuseVelocities,
                                               Vec4* diffusePositionsNew,
                                               Vec4* diffuseVelocitiesNew,
                                               const int* numDiffuseParticles,
                                               int* numDiffuseParticlesNew);

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
namespace {

const char* const kFlexSourceFile = "../../../src/cuda/flex.cu";
const char* const kLibrarySourceFile = "../../../src/cuda/library.h";

const int kGridDim = 128;          // gParams memset size 0x800000 = 128^3 * sizeof(int)
const int kNumThreads = 256;       // 0x100 in every shipped launch configuration

NvFlexErrorCallback libErrorFunc(const NvFlexLibrary* lib)
{
    void* fn = 0;
    std::memcpy(&fn, lib->mErrorFunc, sizeof(fn));
    return reinterpret_cast<NvFlexErrorCallback>(fn);
}

// CudaDetailTimer's shipped spelling: RAII scope around one stage.
struct DetailTimer {
    CudaTimer timer;
    DetailTimer(const char* name, CudaTimers& timers) { cudaTimerCtor(&timer, name, timers); }
    ~DetailTimer() { cudaTimerDtor(&timer); }
};

// DeviceBuffer<T> as recovered (24 B): used for the members the DWARF recovery
// keeps opaque (FlexTriangleMeshInternal::mVertices, NvFlexLibrary::mStaticTri*).
template <typename T>
struct DeviceBufferLayout {
    T* mPtr;
    int mCapacity;
    unsigned char _pad[4];
    unsigned long long mTexture;
};
static_assert(sizeof(DeviceBufferLayout<Vec4>) == 24, "DeviceBuffer<Vec4> layout");

typedef std::vector<NvFlexSolver*> SolverVector;
typedef std::vector<unsigned int> IdVector;
typedef std::vector<CudaRangeTimer> CudaRangeTimerVector;
typedef std::map<const char*, int, ltstr> NameToTimerMap;

SolverVector& solverVector(NvFlexLibrary* lib)
{
    return *reinterpret_cast<SolverVector*>(lib->mResourceTracker.mSolver);
}
IdVector& triangleMeshVector(NvFlexLibrary* lib)
{
    return *reinterpret_cast<IdVector*>(lib->mResourceTracker.mTriangleMesh);
}
IdVector& sdfVector(NvFlexLibrary* lib)
{
    return *reinterpret_cast<IdVector*>(lib->mResourceTracker.mSDF);
}
IdVector& convexMeshVector(NvFlexLibrary* lib)
{
    return *reinterpret_cast<IdVector*>(lib->mResourceTracker.mConvexMesh);
}
NameToTimerMap& timerNameMap(CudaTimers* timers)
{
    return *reinterpret_cast<NameToTimerMap*>(timers->mNameToTimerMap);
}
CudaRangeTimerVector& timerPoolElems(CudaRangeTimerPool* pool)
{
    return *reinterpret_cast<CudaRangeTimerVector*>(pool->mElem);
}

struct Vec4Buffer {
    Vec4* mPtr;
    int mCapacity;
    unsigned char _pad[4];
    unsigned long long mTexture;
};

// The shipped source allocates buffers with `new NvFlexBuffer(...)`; the class
// constructor is declared in util.h and is not part of the DWARF recovery, so a
// layout-compatible wrapper restores the new-expression (operator new + ctor +
// the compiler's delete-on-throw landing pad).
struct NvFlexBufferAlloc {
    bool mapped;
    int type;
    void* graphicsResource;
    NvFlexLibrary* lib;
    void* ptr;
    int elementCount;
    int elementByteStride;

    NvFlexBufferAlloc(int count, int stride, int type_, NvFlexLibrary* library)
    {
        NvFlexBuffer_construct(reinterpret_cast<NvFlexBuffer*>(this), count, stride, type_,
                               library);
    }
};
static_assert(sizeof(NvFlexBufferAlloc) == sizeof(NvFlexBuffer), "NvFlexBuffer layout");

int divUp(int value, int divisor)
{
    // shipped idiom: (value + divisor - 1) with a negative correction
    return value > 0 ? (value + divisor - 1) / divisor : (value + 2 * divisor - 2) / divisor;
}

}  // namespace

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:468 _Z10InitParamsP12NvFlexSolver
// KernelParams template parameters; 568 bytes are staged on the stack and
// copied to the gParams constant buffer at the end.
// ---------------------------------------------------------------------------
void InitParams(NvFlexSolver* s)
{
    KernelParams p;
    NvFlexParams& params = *reinterpret_cast<NvFlexParams*>(s->mParams);

    std::memcpy(p.kPlanes, params.planes, sizeof(p.kPlanes));
    p.kNumPlanes = params.numPlanes;
    p.kMaxContactsPerParticle = s->mMaxContactsPerParticle;

    // 469-499
    p.kRadius = params.radius;
    p.kRadiusSq = p.kRadius * p.kRadius;
    p.kInvRadius = 1.0f / p.kRadius;

    p.kRestDensity = s->mFluidRestDensity;
    p.kInvRestDensity = 1.0f / s->mFluidRestDensity;
    p.kLambdaScale = 1.0f / s->mFluidDensityConstraintScale;
    p.kFluidRestDistance = params.fluidRestDistance;
    p.kSolidRestDistance = params.solidRestDistance;
    p.kVorticityConfinement = params.vorticityConfinement;
    p.kSolidPressure = params.solidPressure;
    p.kBuoyancy = params.buoyancy;
    p.kViscosity = params.viscosity * p.kInvRestDensity;
    p.kFreeSurfaceDrag = params.freeSurfaceDrag;

    // Poly6 / spiky kernel normalisation; the shipped object uses the decimal
    // literal 315/(201.0619354*r^9) where 201.0619354f == 64*pi (LC4 = 0x43490fdb).
    const float r = p.kRadius;
    const float r3 = r * r * r;
    p.kPoly6 = 315.0f / (201.0619354248047f * r3 * r3 * r3);
    p.kSpiky1 = 15.0f / (3.1415927410125732f * r3);
    p.kSpiky2 = 30.0f / (3.1415927410125732f * r3 * r);

    // 502-507: cohesion spline coefficients in terms of h = fluidRestDistance/radius
    const float h = p.kFluidRestDistance / p.kRadius;
    const float h2 = h * h;
    p.kCohesion1 = -(h + 1.0f) / h2;
    p.kCohesion2 = (h2 + h + 1.0f) / h2;
    p.kSurfaceTension =
        (p.kInvRestDensity * params.surfaceTension) / s->mFluidSurfaceConstraintScale;
    p.kCohesion = params.cohesion * p.kRadius;
    p.kAdhesion = params.adhesion;

    // 510-516
    p.kGravity[0] = params.gravity[0];
    p.kGravity[1] = params.gravity[1];
    p.kGravity[2] = params.gravity[2];
    p.kDynamicFriction = params.dynamicFriction;
    p.kStaticFriction = params.staticFriction;
    p.kParticleFriction = params.particleFriction;

    // 518-522
    p.kDiffuseThreshold = s->mMaxDiffuseParticles != 0 ? params.diffuseThreshold : 0.0f;
    p.kDiffuseBuoyancy = params.diffuseBuoyancy;
    p.kDiffuseDrag = params.diffuseDrag;
    p.kDiffuseBallistic = params.diffuseBallistic;
    p.kDiffuseLifetime = params.diffuseLifetime;

    // 524-534
    p.kAnisotropy = params.anisotropyScale;
    p.kAnisotropyMin = params.anisotropyMin * p.kRadius;
    p.kAnisotropyMax = params.anisotropyMax * p.kRadius;
    p.kSmoothing = params.smoothing;
    p.kDissipation = params.dissipation;
    p.kDamping = params.damping;
    p.kDrag = params.drag * 60.0f;
    p.kLift = params.lift * 60.0f;

    // 536-548
    p.kRestitution = params.restitution;
    p.kCollisionDistance = params.collisionDistance;
    p.kCollisionThreshold = params.shapeCollisionMargin;
    p.kCollisionMargin = params.collisionDistance + params.shapeCollisionMargin;
    p.kCollisionMarginSq = p.kCollisionMargin * p.kCollisionMargin;
    p.kSleepThreshold = params.sleepThreshold;
    p.kSleepThresholdSq = p.kSleepThreshold * p.kSleepThreshold;
    p.kShockPropagation = params.shockPropagation;
    p.kSOR = 1.0f / (1.0f + params.relaxationFactor);

    // 547: textures
    p.kTextures = s->mTextures;

    // 550: shared mesh tables
    p.kConvexMeshData = s->mLib->mConvexMeshData;
    p.kTriangleMeshData = s->mLib->mTriangleMeshData;
    p.kSDFData = s->mLib->mSDFData;

    cudaMemcpyToSymbolAsync(gParams, &p, sizeof(KernelParams), 0, cudaMemcpyHostToDevice, 0);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:3934 _Z6UpdateP12NvFlexSolverfii
//
// One substep of the solver.  The substep loop itself lives in
// NvFlexUpdateSolver; `frame` is the shipped static step counter.
// ---------------------------------------------------------------------------
void Update(NvFlexSolver* s, float dt, int substep, int numSubsteps)
{
    const int n = s->numParticles;
    NvFlexLibrary* lib = s->mLib;
    const int kNumBlocks = divUp(n, kNumThreads);

    NvFlexParams& params = *reinterpret_cast<NvFlexParams*>(s->mParams);
    NvFlexSolverDesc& desc = *reinterpret_cast<NvFlexSolverDesc*>(s->mDesc);

    Vec3Iter sortedPositions;
    sortedPositions.x = &s->mSortedPositions->x;
    Vec3Iter sortedNewPositions;
    sortedNewPositions.x = &s->mSortedNewPositions->x;
    Vec3Iter sortedNewVelocities;
    sortedNewVelocities.x = &s->mSortedNewVelocities->x;
    Vec3Iter deltas;
    deltas.x = &s->mDeltas->x;
    Vec3Iter curl;
    curl.x = &s->mCurl->x;

    static int frame = 0;

    // 3948-3950 Predict
    {
        DetailTimer timer("Predict", lib->mTimers);
        Vec3 gravity = {params.gravity[0], params.gravity[1], params.gravity[2]};
        Predict<<<kNumBlocks, kNumThreads>>>(n, s->mActiveIndices, s->mPositions, s->mPhases,
                                             gravity, dt, s->mNewPositions, s->mVelocities,
                                             params.maxSpeed);
    }

    // 3962-3979 UpdateBounds
    {
        DetailTimer timer("UpdateBounds", lib->mTimers);
        static const Vec3 upper = {-3.4028235e+38f, -3.4028235e+38f, -3.4028235e+38f};
        static const Vec3 lower = {3.4028235e+38f, 3.4028235e+38f, 3.4028235e+38f};

        cudaError_t err = cudaMemcpyAsync(s->mParticleLower, &lower, sizeof(lower),
                                          cudaMemcpyHostToDevice, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xf7f, (unsigned long)err,
                        "cudaMemcpyAsync(s->mParticleLower, &lower, sizeof(lower), "
                        "cudaMemcpyHostToDevice)");
        err = cudaMemcpyAsync(s->mParticleUpper, &upper, sizeof(upper), cudaMemcpyHostToDevice, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xf80, (unsigned long)err,
                        "cudaMemcpyAsync(s->mParticleUpper, &upper, sizeof(upper), "
                        "cudaMemcpyHostToDevice)");

        CalculateBounds<<<kNumBlocks, kNumThreads>>>(n, s->mActiveIndices, s->mNewPositions,
                                                     s->mParticleLower, s->mParticleUpper);

        err = cudaMemcpyToSymbolAsync(gBounds, s->mParticleLower, sizeof(Vec3), 0,
                                      cudaMemcpyDeviceToDevice, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xf8a, (unsigned long)err,
                        "cudaMemcpyToSymbolAsync(gBounds, s->mParticleLower, sizeof(Vec3), 0, "
                        "cudaMemcpyDeviceToDevice)");
        err = cudaMemcpyToSymbolAsync(gBounds, s->mParticleUpper, sizeof(Vec3), sizeof(Vec3),
                                      cudaMemcpyDeviceToDevice, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xf8b, (unsigned long)err,
                        "cudaMemcpyToSymbolAsync(gBounds, s->mParticleUpper, sizeof(Vec3), "
                        "sizeof(Vec3), cudaMemcpyDeviceToDevice)");
    }

    // 3983-3985 CreateCellIndices
    {
        DetailTimer timer("CreateCellIndices", lib->mTimers);
        CreateCellIndices<<<kNumBlocks, kNumThreads>>>(
            n, s->mActiveIndices, s->mNewPositions, s->mCellIds, s->mIndices,
            1.0f / (params.radius + params.particleCollisionMargin));
    }

    // 3994-3997 SortCellIndices
    {
        DetailTimer timer("SortCellIndices", lib->mTimers);
        if (s->numParticles != 0)
            SortCellIndices(lib, s->mCellIds, s->mIndices, s->numParticles, 0x15);
    }

    // 4000-4005 Create Grid
    {
        DetailTimer timer("Create Grid", lib->mTimers);
        cudaError_t err = cudaMemsetAsync(lib->mCellStarts, 0,
                                          sizeof(int) * kGridDim * kGridDim * kGridDim, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xfa2, (unsigned long)err,
                        "cudaMemsetAsync(s->mLib->mCellStarts, 0, "
                        "sizeof(int)*kGridDim*kGridDim*kGridDim)");
        err = cudaMemsetAsync(lib->mCellEnds, 0, sizeof(int) * kGridDim * kGridDim * kGridDim, 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0xfa3, (unsigned long)err,
                        "cudaMemsetAsync(s->mLib->mCellEnds, 0, "
                        "sizeof(int)*kGridDim*kGridDim*kGridDim)");

        CreateGrid<<<kNumBlocks, kNumThreads>>>(n, s->mCellIds, lib->mCellStarts, lib->mCellEnds);
    }

    // 4013-4016 ReorderParticles
    {
        DetailTimer timer("ReorderParticles", lib->mTimers);
        ReorderParticles<<<kNumBlocks, kNumThreads>>>(
            n, s->mPositions, s->mNewPositions, s->mPhases, s->mIndices, s->mReverseLookup,
            s->mSortedPositions, s->mSortedNewPositions, s->mSortedPhases);
    }

    // 4042-4044 CollideParticles
    {
        DetailTimer timer("CollideParticles", lib->mTimers);
        const float radius = params.radius + params.particleCollisionMargin;
        CollideParticles<<<kNumBlocks, kNumThreads>>>(
            n, lib->mCellStarts, lib->mCellEnds, s->mIndices, sortedNewPositions,
            s->mRestPositions, s->mSortedPhases, radius * radius, 1.0f / radius,
            s->mMaxNeighborsPerParticle, s->mParticleNeighbors, s->mParticleNeighborCounts);
    }

    // 4060
    _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207::ExecuteCallback(
        s, eNvFlexStageSubstepBegin, dt);

    // 4113-4118 Collide Triangles
    {
        DetailTimer timer("Collide Triangles", lib->mTimers);
        cudaError_t err = cudaMemsetAsync(s->mStaticContactCounts, 0, n * sizeof(int), 0);
        if (err && libErrorFunc(lib))
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1014, (unsigned long)err,
                        "cudaMemsetAsync(s->mStaticContactCounts, 0, n*sizeof(int))");

        FlexCollisionShapeData shapes = s->mShapes;
        CollideTriangles<<<kNumBlocks, kNumThreads>>>(
            shapes, sortedPositions, s->mSortedPhases, n, s->mStaticContactCounts,
            s->mStaticContactPlanes, s->mStaticContactVelocities,
            (float)substep / (float)numSubsteps, (float)(substep + 1) / (float)numSubsteps);
    }

    // 4130-4133 Collide Shapes
    {
        DetailTimer timer("Collide Shapes", lib->mTimers);
        FlexCollisionShapeData shapes = s->mShapes;
        CollideShapes<<<kNumBlocks, kNumThreads>>>(
            shapes, sortedPositions, s->mSortedPhases, n, s->mStaticContactCounts,
            s->mStaticContactPlanes, s->mStaticContactVelocities,
            (float)substep / (float)numSubsteps, (float)(substep + 1) / (float)numSubsteps);
    }

    // 4145-4150
    frame++;
    if (params.shockPropagation > 0.0f)
        ContinuousShockPropagation<<<kNumBlocks, kNumThreads>>>(n, s->mSortedNewPositions,
                                                                s->mParticleNeighborCounts);

    // 4153: constraint iterations
    for (int i = 0; i < params.numIterations; ++i)
    {
        _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207::ExecuteCallback(
            s, eNvFlexStageIterationStart, dt);

        // 4158-4178 Calculate Density
        {
            DetailTimer timer("Calculate Density", lib->mTimers);
            if (desc.featureMode != eNvFlexFeatureModeSimpleSolids && params.fluidRestDistance > 0.0f)
            {
                if (params.surfaceTension <= 0.0f)
                    CalculateDensity<false><<<kNumBlocks, kNumThreads>>>(
                        n, sortedNewPositions, s->mSortedPhases, s->mParticleNeighbors,
                        s->mParticleNeighborCounts, s->mIndices, s->mStaticContactPlanes,
                        s->mDensities, s->mNormals, s->mStaticContactCounts);
                else
                    CalculateDensity<true><<<kNumBlocks, kNumThreads>>>(
                        n, sortedNewPositions, s->mSortedPhases, s->mParticleNeighbors,
                        s->mParticleNeighborCounts, s->mIndices, s->mStaticContactPlanes,
                        s->mDensities, s->mNormals, s->mStaticContactCounts);
            }
        }

        // 4193-4213 Solve Densities
        {
            DetailTimer timer("Solve Densities", lib->mTimers);
            if (desc.featureMode == eNvFlexFeatureModeSimpleSolids)
            {
                SolveDensitiesSimpleSolids<<<kNumBlocks, kNumThreads>>>(
                    n, sortedPositions, sortedNewPositions, s->mSortedPhases, s->mDensities,
                    s->mNormals, s->mParticleNeighbors, s->mParticleNeighborCounts, s->mIndices, dt,
                    s->mDeltas);
            }
            else if (desc.featureMode == eNvFlexFeatureModeSimpleFluids)
            {
                SolveDensitiesSimpleFluids<<<kNumBlocks, kNumThreads>>>(
                    n, sortedPositions, sortedNewPositions, s->mSortedPhases, s->mDensities,
                    s->mNormals, s->mParticleNeighbors, s->mParticleNeighborCounts, s->mIndices, dt,
                    s->mDeltas);
            }
            else
            {
                if (params.surfaceTension <= 0.0f)
                    SolveDensities<false><<<kNumBlocks, kNumThreads>>>(
                        n, sortedPositions, sortedNewPositions, s->mSortedPhases, s->mDensities,
                        s->mNormals, s->mParticleNeighbors, s->mParticleNeighborCounts,
                        s->mIndices, dt, s->mDeltas);
                else
                    SolveDensities<true><<<kNumBlocks, kNumThreads>>>(
                        n, sortedPositions, sortedNewPositions, s->mSortedPhases, s->mDensities,
                        s->mNormals, s->mParticleNeighbors, s->mParticleNeighborCounts,
                        s->mIndices, dt, s->mDeltas);
            }
        }

        // 4220-4227 Solve Springs
        {
            DetailTimer timer("Solve Springs", lib->mTimers);
            if (s->mNumSprings != 0)
            {
                const int kNumSpringBlocks = divUp(s->mNumSprings, kNumThreads);
                SolveSprings<<<kNumSpringBlocks, kNumThreads>>>(
                    s->mSpringIndices, s->mSpringLengths, s->mSpringStiffness, s->mReverseLookup,
                    s->mNumSprings, deltas);
            }
        }

        // 4230-4257 Solve Inflatables
        {
            DetailTimer timer("Solve Inflatables", lib->mTimers);
            if (s->mNumInflatables != 0)
            {
                cudaError_t err =
                    cudaMemsetAsync(s->mPressures, 0, s->mNumInflatables * sizeof(float), 0);
                if (err && libErrorFunc(lib))
                    FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                                kFlexSourceFile, 0x108a, (unsigned long)err,
                                "cudaMemsetAsync(s->mPressures, 0, "
                                "sizeof(float)*s->mNumInflatables)");

                CalculateInflatableVolume<256><<<s->mNumInflatables, kNumThreads>>>(
                    s->mInflatables, sortedNewPositions, s->mReverseLookup, s->mDynamicTriangles,
                    s->mPressures);

                SolveInflatableVolume<<<s->mNumInflatables, kNumThreads>>>(
                    s->mInflatables, sortedNewPositions, s->mReverseLookup, s->mDynamicTriangles,
                    s->mPressures, deltas, dt * params.maxSpeed);
            }
        }

        // 4260-4312 Solve Shapes
        //
        // The two SolveShapes<true/false> device kernels are blocked in the
        // ledger (PTX 11675-19602 decoded only partially by their owner), so no
        // definition exists in the reconstructed library.  Referencing them
        // would make libnvflex.so fail to link for every module consumer; the
        // launch is therefore compiled out until the kernels land.  Flip
        // FLEXDECOMP_HAVE_SOLVESHAPES to 1 (or drop the guard) then.
        {
            DetailTimer timer("Solve Shapes", lib->mTimers);
#if FLEXDECOMP_HAVE_SOLVESHAPES
            if (s->mNumRigids != 0)
            {
                if (s->mRigidPlasticThresholds != 0 && s->mRigidPlasticCreeps != 0)
                    SolveShapes<true><<<s->mNumRigids, 64>>>(
                        s->mRigidOffsets, s->mRigidIndices, s->mRigidCoefficients,
                        s->mRigidPlasticThresholds, s->mRigidPlasticCreeps, s->mIndices,
                        s->mRigidLocalPositions, s->mRigidLocalNormals, sortedPositions,
                        sortedNewPositions, s->mRigidRotations, s->mRigidTranslations, s->mNormals,
                        deltas, s->mNumRigids, 0.0f);
                else
                    SolveShapes<false><<<s->mNumRigids, 64>>>(
                        s->mRigidOffsets, s->mRigidIndices, s->mRigidCoefficients,
                        s->mRigidPlasticThresholds, s->mRigidPlasticCreeps, s->mIndices,
                        s->mRigidLocalPositions, s->mRigidLocalNormals, sortedPositions,
                        sortedNewPositions, s->mRigidRotations, s->mRigidTranslations, s->mNormals,
                        deltas, s->mNumRigids, 0.0f);
            }
#endif
        }

        // 4320-4332 Apply Deltas
        {
            DetailTimer timer("Apply Deltas", lib->mTimers);
            if (params.relaxationMode == eNvFlexRelaxationLocal)
                ApplyDeltas<true><<<kNumBlocks, kNumThreads>>>(
                    n, sortedPositions, sortedNewPositions, s->mDeltas, 0, 1.0f,
                    1.0f / (1.0f + params.relaxationFactor));
            else
                ApplyDeltas<false><<<kNumBlocks, kNumThreads>>>(
                    n, sortedPositions, sortedNewPositions, s->mDeltas, 0, 1.0f,
                    params.relaxationFactor);
        }

        // 4341-4343 Solve Static Contacts
        {
            DetailTimer timer("Solve Static Contacts", lib->mTimers);
            SolveContacts<false><<<kNumBlocks, kNumThreads>>>(
                sortedPositions, sortedNewPositions, n, s->mStaticContactCounts,
                s->mStaticContactPlanes, s->mStaticContactVelocities, s->mDensities, 0);
        }

        _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207::ExecuteCallback(
            s, eNvFlexStageIterationEnd, dt);
    }

    // 4350-4422 Solve Velocities
    {
        DetailTimer timer("Solve Velocities", lib->mTimers);
        UpdateVelocities<<<kNumBlocks, kNumThreads>>>(n, sortedPositions, sortedNewPositions,
                                                      sortedNewVelocities, 1.0f / dt);

        _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207::ExecuteCallback(
            s, eNvFlexStageSubstepEnd, dt);
        if (substep == numSubsteps - 1)
            _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207::ExecuteCallback(
                s, eNvFlexStageUpdateEnd, (float)numSubsteps * dt);

        if (params.vorticityConfinement <= 0.0f)
        {
            SolveVelocities<false><<<kNumBlocks, kNumThreads>>>(
                n, sortedNewPositions, s->mVelocities, sortedNewVelocities, s->mCurl,
                s->mSortedPhases, s->mParticleNeighbors, s->mParticleNeighborCounts,
                s->mStaticContactPlanes, s->mStaticContactCounts, s->mIndices, dt, s->mDeltas,
                s->mSortedDensities, s->mPotentials);
        }
        else
        {
            CalculateVorticity<<<kNumBlocks, kNumThreads>>>(
                n, sortedNewPositions, s->mVelocities, sortedNewVelocities, s->mSortedPhases,
                s->mParticleNeighbors, s->mParticleNeighborCounts, s->mStaticContactPlanes,
                s->mStaticContactCounts, s->mCurl);
            SolveVelocities<true><<<kNumBlocks, kNumThreads>>>(
                n, sortedNewPositions, s->mVelocities, sortedNewVelocities, s->mCurl,
                s->mSortedPhases, s->mParticleNeighbors, s->mParticleNeighborCounts,
                s->mStaticContactPlanes, s->mStaticContactCounts, s->mIndices, dt, s->mDeltas,
                s->mSortedDensities, s->mPotentials);
        }
    }

    // 4427-4457 Update Triangles / Update Normals
    if (s->mDynamicNumTris != 0)
    {
        const int kNumTriBlocks = divUp(s->mDynamicNumTris, kNumThreads);
        {
            DetailTimer timer("Update Triangles", lib->mTimers);
            Vec3 wind = {params.wind[0], params.wind[1], params.wind[2]};
            UpdateTriangles<<<kNumTriBlocks, kNumThreads>>>(
                s->mDynamicNumTris, sortedNewPositions, sortedNewVelocities, s->mReverseLookup,
                s->mDynamicTriangles, s->mDynamicTriangleNormals, wind, deltas, dt);
        }

        if (substep == numSubsteps - 1)
        {
            DetailTimer timer("Update Normals", lib->mTimers);
            Vec4* tempNormals = s->mAnisotropy2;
            Vec3Iter tempNormalsIter;
            tempNormalsIter.x = &tempNormals->x;

            cudaError_t err = cudaMemsetAsync(tempNormals, 0, s->mMaxParticles * sizeof(Vec4), 0);
            if (err && libErrorFunc(lib))
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x115d, (unsigned long)err,
                            "cudaMemsetAsync(tempNormals, 0, sizeof(Vec4)*s->mMaxParticles)");

            UpdateVertexNormals<<<kNumTriBlocks, kNumThreads>>>(
                s->mDynamicNumTris, s->mReverseLookup, s->mDynamicTriangles,
                s->mDynamicTriangleNormals, tempNormalsIter);
            NormalizeVertexNormals<<<kNumBlocks, kNumThreads>>>(n, s->mIndices, tempNormals,
                                                                s->mNormals);
        }
    }

    // 4461-4463 Finalize
    {
        DetailTimer timer("Finalize", lib->mTimers);
        Finalize<<<kNumBlocks, kNumThreads>>>(n, s->mSortedNewPositions, s->mSortedNewVelocities,
                                              s->mSortedDensities, s->mDeltas, s->mIndices,
                                              s->mPositions, s->mVelocities, s->mDensities,
                                              params.maxSpeed, dt * params.maxAcceleration);
    }

    // 4484-4550 Update Diffuse
    s->mDiffuseTimer = dt + s->mDiffuseTimer;
    if (substep == numSubsteps - 1 && s->mMaxDiffuseParticles != 0)
    {
        DetailTimer timer("Update Diffuse", lib->mTimers);
        const float diffuseDt = (float)(substep + 1) * dt;

        if (s->mDiffuseRate <= s->mDiffuseTimer)
        {
            s->mDiffuseTimer = std::fmod(s->mDiffuseTimer, s->mDiffuseRate);
            CreateDiffuseParticles<<<kNumBlocks, kNumThreads>>>(
                n, sortedNewPositions, s->mPotentials, s->mDiffusePositions,
                s->mDiffuseVelocities, s->mNumDiffuseParticles, s->mMaxDiffuseParticles,
                s->mRandom, diffuseDt);
        }

        const int kNumDiffuseBlocks = divUp(s->mMaxDiffuseParticles, kNumThreads);
        const float radius = params.radius;
        UpdateDiffuseParticles<<<kNumDiffuseBlocks, kNumThreads>>>(
            s->mDiffusePositions, s->mDiffuseVelocities, s->mSortedPhases,
            s->mNumDiffuseParticles, s->mLib->mCellStarts, s->mLib->mCellEnds, s->mIndices,
            s->mParticleNeighborCounts, sortedNewVelocities, radius * radius, 1.0f / radius,
            diffuseDt, 3.4028234663852886e+38f);

        ClampDiffuseParticleCount<<<1, 1>>>(s->mNumDiffuseParticles, s->mNumDiffuseParticlesNew,
                                            s->mMaxDiffuseParticles);

        CompactDiffuseParticles<<<kNumDiffuseBlocks, kNumThreads>>>(
            s->mDiffusePositions, s->mDiffuseVelocities, s->mDiffusePositionsNew,
            s->mDiffuseVelocitiesNew, s->mNumDiffuseParticles, s->mNumDiffuseParticlesNew);

        Vec4* tmpPositions = s->mDiffusePositions;
        s->mDiffusePositions = s->mDiffusePositionsNew;
        s->mDiffusePositionsNew = tmpPositions;

        Vec4* tmpVelocities = s->mDiffuseVelocities;
        s->mDiffuseVelocities = s->mDiffuseVelocitiesNew;
        s->mDiffuseVelocitiesNew = tmpVelocities;

        int* tmpCount = s->mNumDiffuseParticles;
        s->mNumDiffuseParticles = s->mNumDiffuseParticlesNew;
        s->mNumDiffuseParticlesNew = tmpCount;
    }

    // 4553-4572 Calc Anisotropy
    {
        DetailTimer timer("Calc Anisotropy", lib->mTimers);
        if (params.anisotropyScale > 0.0f || params.smoothing > 0.0f)
            SmoothPositions<<<kNumBlocks, kNumThreads>>>(
                n, s->mSortedNewPositions, s->mSortedPhases, s->mParticleNeighbors,
                s->mParticleNeighborCounts, s->mIndices, s->mSortedPositions, s->mDeltas);

        if (params.anisotropyScale > 0.0f)
            CalculateAnisotropy<<<kNumBlocks, kNumThreads>>>(
                n, s->mParticleNeighbors, s->mParticleNeighborCounts, s->mIndices,
                s->mSortedPhases, s->mSortedPositions, s->mAnisotropy1, s->mAnisotropy2,
                s->mAnisotropy3);
    }

    s->mStepCounter = s->mStepCounter + 1;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4590 NvFlexInit
// ---------------------------------------------------------------------------
NvFlexLibrary* NvFlexInit(int version, NvFlexErrorCallback errorFunc, NvFlexInitDesc* desc)
{
    NvFlexLibrary* lib = 0;

    if (version == NV_FLEX_VERSION)
    {
        if (desc == 0 || desc->computeType == eNvFlexCUDA)
        {
            lib = static_cast<NvFlexLibrary*>(::operator new(sizeof(NvFlexLibrary)));

            lib->mCellStarts = 0;
            lib->mCellEnds = 0;
            lib->mBVHBuilder = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriLowers)->mPtr = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriLowers)->mCapacity = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriLowers)->mTexture = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriUppers)->mPtr = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriUppers)->mCapacity = 0;
            reinterpret_cast<Vec4Buffer*>(lib->mStaticTriUppers)->mTexture = 0;
            lib->mCudaContext = 0;
            lib->mCudaOwner = false;

            // timers.mNameToTimerMap is an empty std::map<const char*,int,ltstr>:
            // a self-referential _Rb_tree header (colour red, parent/left/right =
            // &header, node count 0).
            CudaTimers& timers = lib->mTimers;
            std::memset(timers.mNameToTimerMap, 0, sizeof(timers.mNameToTimerMap));
            timers.mNameToTimerMap[0] = 1;  // _S_red
            void* header = timers.mNameToTimerMap;
            std::memcpy(timers.mNameToTimerMap + 16, &header, sizeof(header));
            std::memcpy(timers.mNameToTimerMap + 24, &header, sizeof(header));
            timers.mTimerPool = 0;

            lib->mSortTemp = 0;
            lib->mSortMaxTempSize = 0;

            std::memset(&lib->mResourceTracker, 0, sizeof(lib->mResourceTracker));

            std::memset(lib->mErrorFunc, 0, sizeof(lib->mErrorFunc));
            std::memcpy(lib->mErrorFunc, &errorFunc, sizeof(errorFunc));

            int deviceOrdinal = -1;
            if (desc != 0)
                deviceOrdinal = desc->deviceIndex;

            bool ok = CudaSafeDelayLoad(errorFunc, deviceOrdinal,
                                                reinterpret_cast<CUcontext*>(&lib->mCudaContext));
            if (!ok)
            {
                if (lib->mCudaContext == 0)
                {
                    if (libErrorFunc(lib))
                    {
                        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                    "Could not initialize CUDA driver", kFlexSourceFile, 0x1210,
                                    "");
                        return 0;
                    }
                    return 0;
                }
                lib->mCudaOwner = false;
            }
            else
            {
                lib->mCudaOwner = true;
            }

            int currentDevice = 0;
            cudaError_t err = cudaGetDevice(&currentDevice);
            if (err == cudaSuccess)
            {
                cudaDeviceProp props;
                err = cudaGetDeviceProperties(&props, currentDevice);
                if (err == cudaSuccess)
                {
                    if (props.major > 2)
                    {
                        std::memcpy(lib->mDeviceName, props.name, sizeof(lib->mDeviceName));

                        FlexDeviceMallocInternal(reinterpret_cast<void**>(&lib->mCellStarts),
                                                 kGridDim * kGridDim * kGridDim * sizeof(int),
                                                 "&lib->mCellStarts", kFlexSourceFile, 0x123b);
                        FlexDeviceMallocInternal(reinterpret_cast<void**>(&lib->mCellEnds),
                                                 kGridDim * kGridDim * kGridDim * sizeof(int),
                                                 "&lib->mCellEnds", kFlexSourceFile, 0x123c);

                        err = cudaMemset(lib->mCellStarts, 0,
                                         kGridDim * kGridDim * kGridDim * sizeof(int));
                        if (err && libErrorFunc(lib))
                            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                        "Cuda error: %d in %s", kFlexSourceFile, 0x123d,
                                        (unsigned long)err,
                                        "cudaMemset(lib->mCellStarts, 0, "
                                        "kGridDim*kGridDim*kGridDim*sizeof(int))");
                        err = cudaMemset(lib->mCellEnds, 0,
                                         kGridDim * kGridDim * kGridDim * sizeof(int));
                        if (err && libErrorFunc(lib))
                            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                        "Cuda error: %d in %s", kFlexSourceFile, 0x123e,
                                        (unsigned long)err,
                                        "cudaMemset(lib->mCellEnds, 0, "
                                        "kGridDim*kGridDim*kGridDim*sizeof(int))");

                        lib->mBVHBuilder = static_cast<LinearBVHBuilderGPU*>(
                            ::operator new(sizeof(LinearBVHBuilderGPU)));
                        LinearBVHBuilderGPUConstruct(lib->mBVHBuilder);

                        err = cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&lib->mMemoryFence));
                        if (err && libErrorFunc(lib))
                            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                        "Cuda error: %d in %s", kFlexSourceFile, 0x1245,
                                        (unsigned long)err,
                                        "cudaEventCreate(&lib->mMemoryFence)");

                        CudaRangeTimerPool* timerPool = static_cast<CudaRangeTimerPool*>(
                            ::operator new(sizeof(CudaRangeTimerPool)));
                        cudaRangeTimerPoolCtor(timerPool);
                        lib->mTimers.mTimerPool = timerPool;
                        lib->mTimers.mEnabled = false;

                        // 0x1000 convexes, 32 planes each, 0x1000 meshes, 0x400 SDFs
                        FlexConvexMeshData& convex = lib->mConvexMeshData;
                        convex.mNumConvexes = 0;
                        convex.mMaxConvexes = 0x1000;
                        convex.mMaxPlanesPerConvex = 0x20;
                        FlexDeviceMallocInternal(reinterpret_cast<void**>(&convex.mPlanes),
                                                 0x200000, "&mPlanes", kLibrarySourceFile, 0x23);
                        FlexDeviceMallocInternal(
                            reinterpret_cast<void**>(&convex.mConvexes),
                            (size_t)convex.mMaxConvexes * 32, "&mConvexes", kLibrarySourceFile,
                            0x24);
                        convex.mPlanesTex = CreateTexture(
                            convex.mPlanes,
                            convex.mMaxPlanesPerConvex * convex.mMaxConvexes * 16);
                        convex.mFreeList = new int[convex.mMaxConvexes];
                        for (int i = 0; i < convex.mMaxConvexes; ++i)
                            convex.mFreeList[i] = i;

                        FlexTriangleMeshData& meshes = lib->mTriangleMeshData;
                        meshes.mNumMeshes = 0;
                        meshes.mMaxMeshes = 0x1000;
                        FlexDeviceMallocInternal(
                            reinterpret_cast<void**>(&meshes.mMeshTableDevice), 0x90000,
                            "&mMeshTableDevice", kLibrarySourceFile, 0x82);
                        meshes.mMeshTableHost =
                            new FlexTriangleMeshInternal[meshes.mMaxMeshes]();
                        meshes.mFreeList = new int[meshes.mMaxMeshes];
                        for (int i = 0; i < meshes.mMaxMeshes; ++i)
                            meshes.mFreeList[i] = i;

                        FlexSDFData& sdf = lib->mSDFData;
                        sdf.mNumSDFs = 0;
                        sdf.mMaxSDFs = 0x400;
                        FlexDeviceMallocInternal(reinterpret_cast<void**>(&sdf.mSDFTableDevice),
                                                 0xa000, "&mSDFTableDevice", kLibrarySourceFile,
                                                 0xe5);
                        sdf.mSDFTableHost = new FlexSDFInternal[sdf.mMaxSDFs]();
                        sdf.mFreeList = new int[sdf.mMaxSDFs];
                        for (int i = 0; i < sdf.mMaxSDFs; ++i)
                            sdf.mFreeList[i] = i;

                        return lib;
                    }

                    if (libErrorFunc(lib))
                        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                    "Could not initialize Flex, selected device is not SM3.0 or "
                                    "higher.",
                                    kFlexSourceFile, 0x122f, "");
                    return 0;
                }

                if (libErrorFunc(lib))
                    FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                                "Could not initialize Flex, error querying devices.",
                                kFlexSourceFile, 0x1228, "");
                return 0;
            }

            if (libErrorFunc(lib))
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                            "Could not initialize Flex, error querying devices.", kFlexSourceFile,
                            0x1220, "");
            return 0;
        }

        if (errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Trying to initialize CUDA Flex with a D3D compute type.",
                        kFlexSourceFile, 0x11f9, "");
        return 0;
    }

    if (errorFunc)
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Could not initialize Flex, header/library mismatch, provided: %d, expected: "
                    "%d",
                    kFlexSourceFile, 0x11f3, version, (int)NV_FLEX_VERSION, "");
    return 0;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4689 NvFlexShutdown
// ---------------------------------------------------------------------------
void NvFlexShutdown(NvFlexLibrary* lib)
{
    SolverVector& solvers = solverVector(lib);
    while (!solvers.empty())
        NvFlexDestroySolver(solvers[0]);

    IdVector& triangleMeshes = triangleMeshVector(lib);
    while (!triangleMeshes.empty())
        NvFlexDestroyTriangleMesh(lib, triangleMeshes[0]);

    IdVector& sdfs = sdfVector(lib);
    while (!sdfs.empty())
        NvFlexDestroyDistanceField(lib, sdfs[0]);

    IdVector& convexMeshes = convexMeshVector(lib);
    while (!convexMeshes.empty())
        NvFlexDestroyConvexMesh(lib, convexMeshes[0]);

    DestroyTexture(lib->mConvexMeshData.mPlanesTex);
    FlexDeviceFreeInternal(lib->mConvexMeshData.mPlanes);
    FlexDeviceFreeInternal(lib->mConvexMeshData.mConvexes);
    delete[] lib->mConvexMeshData.mFreeList;
    lib->mConvexMeshData.mNumConvexes = 0;

    FlexDeviceFreeInternal(lib->mTriangleMeshData.mMeshTableDevice);
    for (int i = 0; i < lib->mTriangleMeshData.mMaxMeshes; ++i)
    {
        FlexTriangleMeshInternal* mesh = &lib->mTriangleMeshData.mMeshTableHost[i];
        Vec4Buffer* vertices = reinterpret_cast<Vec4Buffer*>(mesh->mVertices);
        FreeBVH(mesh->mBVH);
        DestroyTexture(vertices->mTexture);
        vertices->mTexture = 0;
        FlexDeviceFreeInternal(vertices->mPtr);
        vertices->mPtr = 0;
        vertices->mCapacity = 0;
        DestroyTexture(mesh->mIndices.mTexture);
        mesh->mIndices.mTexture = 0;
        FlexDeviceFreeInternal(mesh->mIndices.mPtr);
        mesh->mIndices.mPtr = 0;
        mesh->mIndices.mCapacity = 0;
    }
    delete[] lib->mTriangleMeshData.mMeshTableHost;
    delete[] lib->mTriangleMeshData.mFreeList;

    FlexDeviceFreeInternal(lib->mSDFData.mSDFTableDevice);
    for (int i = 0; i < lib->mSDFData.mMaxSDFs; ++i)
    {
        FlexSDFInternal* sdf = &lib->mSDFData.mSDFTableHost[i];
        if (sdf->mFieldArray != 0)
        {
            cudaFreeArray(reinterpret_cast<cudaArray_t>(sdf->mFieldArray));
            sdf->mFieldArray = 0;
        }
        if (sdf->mFieldTex != 0)
        {
            cudaDestroyTextureObject(sdf->mFieldTex);
            sdf->mFieldTex = 0;
        }
    }
    delete[] lib->mSDFData.mSDFTableHost;
    delete[] lib->mSDFData.mFreeList;

    {
        Vec4Buffer* lowers = reinterpret_cast<Vec4Buffer*>(lib->mStaticTriLowers);
        DestroyTexture(lowers->mTexture);
        lowers->mTexture = 0;
        FlexDeviceFreeInternal(lowers->mPtr);
        lowers->mPtr = 0;
        lowers->mCapacity = 0;

        Vec4Buffer* uppers = reinterpret_cast<Vec4Buffer*>(lib->mStaticTriUppers);
        DestroyTexture(uppers->mTexture);
        uppers->mTexture = 0;
        FlexDeviceFreeInternal(uppers->mPtr);
        uppers->mPtr = 0;
        uppers->mCapacity = 0;
    }

    if (lib->mBVHBuilder != 0)
    {
        LinearBVHBuilderGPUDestroy(lib->mBVHBuilder);
        ::operator delete(lib->mBVHBuilder);
    }

    FlexDeviceFreeInternal(lib->mCellStarts);
    FlexDeviceFreeInternal(lib->mCellEnds);

    cudaError_t err = cudaEventDestroy(reinterpret_cast<cudaEvent_t>(lib->mMemoryFence));
    if (err && libErrorFunc(lib))
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x1263, (unsigned long)err, "cudaEventDestroy(lib->mMemoryFence)");

    if (lib->mTimers.mTimerPool != 0)
    {
        CudaRangeTimerVector& elems = timerPoolElems(lib->mTimers.mTimerPool);
        for (size_t i = 0; i < elems.size(); ++i)
        {
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(elems[i].mBegin));
            cudaEventDestroy(reinterpret_cast<cudaEvent_t>(elems[i].mEnd));
        }
        if (!elems.empty())
            ::operator delete(elems.data());
        ::operator delete(lib->mTimers.mTimerPool);
    }

    timerNameMap(&lib->mTimers).clear();

    if (solvers.data())
        ::operator delete(solvers.data());
    if (triangleMeshes.data())
        ::operator delete(triangleMeshes.data());
    if (sdfs.data())
        ::operator delete(sdfs.data());
    if (convexMeshes.data())
        ::operator delete(convexMeshes.data());

    ::operator delete(lib);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4722 _Z14NvFlexSetFenceP13NvFlexLibrary
// ---------------------------------------------------------------------------
void NvFlexSetFence(NvFlexLibrary* lib)
{
    cudaError_t err = cudaEventRecord(reinterpret_cast<cudaEvent_t>(lib->mMemoryFence), 0);
    if (err && libErrorFunc(lib))
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x1276, (unsigned long)err, "cudaEventRecord(lib->mMemoryFence, 0)");
    cudaStreamQuery(0);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4735 _Z15NvFlexWaitFenceP13NvFlexLibrary
// ---------------------------------------------------------------------------
void NvFlexWaitFence(NvFlexLibrary* lib)
{
    cudaError_t err = cudaEventSynchronize(reinterpret_cast<cudaEvent_t>(lib->mMemoryFence));
    if (err && libErrorFunc(lib))
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x1283, (unsigned long)err, "cudaEventSynchronize(lib->mMemoryFence)");
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4742 NvFlexFlush
// ---------------------------------------------------------------------------
void NvFlexFlush(NvFlexLibrary* lib)
{
    (void)lib;
    cudaStreamQuery(0);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4750 NvFlexMap
// ---------------------------------------------------------------------------
void* NvFlexMap(NvFlexBuffer* buffer, int flags)
{
    if (buffer->type == eNvFlexBufferHost)
    {
        if ((flags & eNvFlexMapDoNotWait) == 0)
        {
            NvFlexWaitFence(buffer->lib);
        }
        else
        {
            cudaError_t err =
                cudaEventQuery(reinterpret_cast<cudaEvent_t>(buffer->lib->mMemoryFence));
            if (err == cudaErrorNotReady)
                return 0;
        }
    }

    return NvFlexBuffer_map(buffer);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4773 NvFlexUnmap
// ---------------------------------------------------------------------------
void NvFlexUnmap(NvFlexBuffer* buffer)
{
    NvFlexBuffer_unmap(buffer);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4778 NvFlexAllocBuffer
// ---------------------------------------------------------------------------
NvFlexBuffer* NvFlexAllocBuffer(NvFlexLibrary* lib, int elementCount, int elementByteStride,
                                NvFlexBufferType type)
{
    NvFlexBufferAlloc* buf = new NvFlexBufferAlloc(elementCount, elementByteStride, type, lib);
    return reinterpret_cast<NvFlexBuffer*>(buf);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4784 NvFlexFreeBuffer
// ---------------------------------------------------------------------------
void NvFlexFreeBuffer(NvFlexBuffer* buf)
{
    if (buf != 0)
    {
        NvFlexBuffer_destroy(buf);
        ::operator delete(buf);
    }
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4789 NvFlexRegisterOGLBuffer
// ---------------------------------------------------------------------------
NvFlexBuffer* NvFlexRegisterOGLBuffer(NvFlexLibrary* lib, int glbuf, int elementCount,
                                      int elementByteStride)
{
    cudaGraphicsResource_t res = 0;

    cudaError_t err =
        cudaGraphicsGLRegisterBuffer(&res, glbuf, cudaGraphicsRegisterFlagsWriteDiscard);
    if (err && libErrorFunc(lib))
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x12b8, (unsigned long)err,
                    "cudaGraphicsGLRegisterBuffer(&res, glbuf, "
                    "cudaGraphicsRegisterFlagsWriteDiscard)");

    err = cudaGraphicsResourceSetMapFlags(res, cudaGraphicsMapFlagsWriteDiscard);
    if (err && libErrorFunc(lib))
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s", kFlexSourceFile,
                    0x12b9, (unsigned long)err,
                    "cudaGraphicsResourceSetMapFlags(res, cudaGraphicsMapFlagsWriteDiscard)");

    NvFlexBufferAlloc* buf = new NvFlexBufferAlloc(elementCount, elementByteStride, 3, lib);
    buf->graphicsResource = res;
    return reinterpret_cast<NvFlexBuffer*>(buf);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4801 NvFlexUnregisterOGLBuffer
// ---------------------------------------------------------------------------
void NvFlexUnregisterOGLBuffer(NvFlexBuffer* buf)
{
    if (buf != 0)
    {
        NvFlexLibrary* lib = buf->lib;
        if (buf->graphicsResource != 0)
        {
            cudaError_t err = cudaGraphicsUnregisterResource(
                reinterpret_cast<cudaGraphicsResource_t>(buf->graphicsResource));
            if (err && libErrorFunc(lib))
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x12ca, (unsigned long)err,
                            "cudaGraphicsUnregisterResource(buf->graphicsResource)");
        }

        NvFlexBuffer_destroy(buf);
        ::operator delete(buf);
    }
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4816 NvFlexRegisterD3DBuffer
// ---------------------------------------------------------------------------
NvFlexBuffer* NvFlexRegisterD3DBuffer(NvFlexLibrary* lib, void* dxbuf, int elementCount,
                                      int elementByteStride)
{
    (void)lib;
    (void)dxbuf;
    (void)elementCount;
    (void)elementByteStride;
    return 0;
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4832 NvFlexUnregisterD3DBuffer
// ---------------------------------------------------------------------------
void NvFlexUnregisterD3DBuffer(NvFlexBuffer* buf)
{
    if (buf != 0)
    {
        NvFlexLibrary* lib = buf->lib;
        if (buf->graphicsResource != 0)
        {
            cudaError_t err = cudaGraphicsUnregisterResource(
                reinterpret_cast<cudaGraphicsResource_t>(buf->graphicsResource));
            if (err && libErrorFunc(lib))
                FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                            kFlexSourceFile, 0x12e9, (unsigned long)err,
                            "cudaGraphicsUnregisterResource(buf->graphicsResource)");
        }

        NvFlexBuffer_destroy(buf);
        ::operator delete(buf);
    }
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4849 NvFlexAcquireContext
// ---------------------------------------------------------------------------
void NvFlexAcquireContext(NvFlexLibrary* lib)
{
    g_savedContext = CudaGetContext();
    CudaSetContext(reinterpret_cast<CUcontext>(lib->mCudaContext));
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4855 NvFlexRestoreContext
// ---------------------------------------------------------------------------
void NvFlexRestoreContext(NvFlexLibrary* lib)
{
    (void)lib;
    CudaSetContext(g_savedContext);
}

// ---------------------------------------------------------------------------
// flexdecomp: src/cuda/flex.cu:4860 NvFlexGetVersion
// ---------------------------------------------------------------------------
int NvFlexGetVersion()
{
    return 0x78;
}
