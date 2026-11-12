// flexdecomp: reconstructed D3D backend — flex.obj, solver object.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  Each function
// names its `// ---- <Name> @ <rva>` block and byte size.
//
// The solver owns ~40 D3D buffers plus a radix sort, a timer pool, a dense set
// of shader handles and the host mirror of the constant buffer
// (`mKernelParamsHost`).  Every buffer is created through `DxBuffer::CreateZero`
// with `bufferType = eBuffer | eUAV_SRV | eStructured` (0x13), which is what the
// shipped code spells out inline for each of them.
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "FlexObjects.h"
#include "FlexPipeline.h"
#include "generated/shaders_flex.h"

#include <windows.h>

namespace
{

// DxBuffer::CreateRawZero (flex.obj @0x2e940, 246 bytes): a structured buffer
// with a fixed 16-byte stride and the RAW view bit, initialised from zeroed
// host memory.
void CreateRawZero(DxBufferExt& buffer, NvFlex::ContextEx* context, unsigned dim)
{
    const unsigned bytes = dim << 4;
    void* zeros = std::malloc(bytes);
    if (zeros != 0)
        std::memset(zeros, 0, bytes);

    buffer.Release();

    NvFlex::BufferDesc desc;
    desc.format = NvFlexFormat(1);   // eNvFlexFormat_r32_typeless
    desc.dim = dim;
    desc.stride = 0x10;
    desc.bufferType = NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eRaw;   // 0xb
    desc.data = zeros;

    buffer.mBufferDesc = desc;
    buffer.mBuffer = context->createBuffer(&desc, 0);

    std::free(zeros);
}

// DxBuffer::CreateDispatchArgs (flex.obj @0x2ea40, 205 bytes): three uint
// arguments for an indirect compute dispatch.
void CreateDispatchArgs(DxBufferExt& buffer, NvFlex::ContextEx* context,
                        unsigned dim)
{
    buffer.Release();

    int args[3] = {1, 1, 1};

    NvFlex::BufferDesc desc;
    desc.format = NvFlexFormat(8);   // eNvFlexFormat_r32_uint
    desc.dim = 3;
    desc.stride = 4;
    desc.bufferType = NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eIndirect;   // 0x23
    desc.data = args;

    buffer.mBufferDesc = desc;
    buffer.mBuffer = context->createBuffer(&desc, 0);
    (void)dim;
}

} // namespace

// ---------------------------------------------------------------------------
// NvFlexSolver::NvFlexSolver — @0x3a7e0 (669 bytes)
// ---------------------------------------------------------------------------
// The shipped body default-initialises the D3D wrappers and the POD members;
// everything else is assigned by NvFlexCreateSolver.
NvFlexSolver::NvFlexSolver()
{
    std::memset(&mKernelParamsHost, 0, sizeof(mKernelParamsHost));

    DxConstantBuffer* constantBuffers[] =
    {
        &mKernelParams, &mHalfSpringConstantBuffer, &mSubstepParams, &mNumShapes
    };
    for (unsigned i = 0; i < sizeof(constantBuffers) / sizeof(constantBuffers[0]); ++i)
    {
        constantBuffers[i]->mBufferDesc.stride = 0;
        constantBuffers[i]->mBufferDesc.dim = 0;
        constantBuffers[i]->mBufferDesc.uploadAccess = false;
        constantBuffers[i]->mBuffer = 0;
    }

    DxBuffer* buffers[] =
    {
        &mPositions, &mNewPositions, &mVelocities, &mRestPositions, &mDeltas, &mPhases,
        &mActiveIndices, &mDensities, &mPotentials, &mCurl, &mNormals, &mSortedNormals,
        &mNormalsTemp, &mAnisotropy1, &mAnisotropy2, &mAnisotropy3, &mSortedPositions,
        &mSortedVelocities, &mSortedNewPositions, &mSortedNewVelocities, &mSortedPhases,
        &mSortedDensities, &mStaticContactPlanes, &mStaticContactVelocities,
        &mStaticContactCounts, &mParticleNeighbors, &mParticleNeighborCounts, &mPlanes,
        &mSpringIndices, &mSpringLengths, &mSpringStiffness, &mHalfSpringIndices,
        &mHalfSpringOpposites, &mHalfSpringLengths, &mHalfSpringStiffness,
        &mHalfSpringParticleBegin, &mHalfSpringParticleEnd, &mRigidOffsets, &mRigidIndices,
        &mRigidLocalPositions, &mRigidLocalNormals, &mRigidCoefficients,
        &mRigidPlasticThresholds, &mRigidPlasticCreeps, &mRigidRotations,
        &mRigidTranslations, &mDynamicTriangles, &mDynamicTriangleNormals,
        &mDynamicTriangleEdges, &mDynamicNumTriEdges, &mPressures, &mInflatables,
        &mDiffusePositions, &mDiffuseVelocities, &mDiffusePositionsNew,
        &mDiffuseVelocitiesNew, &mNumDiffuseParticles, &mNumDiffuseParticlesNew,
        &mDiffuseSortSize, &mDiffuseLaunchArg, &mCellIds, &mReverseLookup,
        &mParticleBounds, &mBoundsLower, &mBoundsUpper, &mGroupBoundsLower,
        &mGroupBoundsUpper, &mSDFShapeDX, &mRandom, &mSmoothPositionsOriginal
    };
    for (unsigned i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i)
        buffers[i]->mBuffer = 0;

    mMaxNeighborsPerParticle = 0;
    mMaxContactsPerParticle = 0;
    mNumSprings = 0;
    mMaxSprings = 0;
    mMaxHalfSprings = 0;
    mNumRigids = 0;
    mMaxRigids = 0;
    mNumRigidIndices = 0;
    mMaxRigidIndices = 0;
    mMaxPlasticEntries = 0;
    mDynamicNumTris = 0;
    mDynamicMaxTris = 0;
    mNumInflatables = 0;
    mMaxInflatables = 0;
    mMaxDiffuseParticles = 0;
    mNumActiveDiffuseParticles = 0;
    mDiffuseRate = 0.0f;
    mDiffuseTimer = 0.0f;
    mNumParticles = 0;
    mNumParticlesAligned = 0;
    mMaxParticles = 0;
    mDebug = false;
    mFluidRestDensity = 0.0f;
    mFluidDensityConstraintScale = 0.0f;
    mFluidSurfaceConstraintScale = 0.0f;
    mStepCounter = 0;
    std::memset(&mCallbacks, 0, sizeof(mCallbacks));
    mRadixSortSize = 0;
    mRadixSort = 0;
    mTimers = 0;
    mNumDetailTimers = 0;
    mDetailTimers = 0;
    mLatencyTimerBeginRecorded = false;
    mLatencyTimerEndRecorded = false;
    mLatencyTimerEnable = false;
    mLatencyTimerIndex = 0;
    mLib = 0;

    std::memset(&mDesc, 0, sizeof(mDesc));
    std::memset(&mParams, 0, sizeof(mParams));
    mShapes.mNumShapes = 0;
}

// ---------------------------------------------------------------------------
// ~NvFlexSolver — @0x3c1f0 (55 bytes)
// ---------------------------------------------------------------------------
NvFlexSolver::~NvFlexSolver()
{
    mSDFDeviceShapes.clear();
}

// ---------------------------------------------------------------------------
// NvFlexSetSolverDescDefaults — @0x37f60 (27 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetSolverDescDefaults(NvFlexSolverDesc* desc)
{
    desc->maxParticles = 0x4000;
    desc->maxDiffuseParticles = 0;
    desc->maxNeighborsPerParticle = 0x60;
    desc->featureMode = eNvFlexFeatureModeDefault;
    desc->maxContactsPerParticle = 6;
}

// ---------------------------------------------------------------------------
// NvFlexGetSolverDesc — @0x3c2a0 (20 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetSolverDesc(NvFlexSolver* solver, NvFlexSolverDesc* desc)
{
    *desc = solver->mDesc;
}

// ---------------------------------------------------------------------------
// NvFlexCreateSolver — @0x37f80 (10326 bytes)
// ---------------------------------------------------------------------------
extern "C" NvFlexSolver* NvFlexCreateSolver(NvFlexLibrary* lib, const NvFlexSolverDesc* desc)
{
    if (lib == 0 || desc == 0)
        return 0;

    void* memory = NvFlex::g_FlexMalloc(sizeof(NvFlexSolver));
    if (memory == 0)
        return 0;
    NvFlexSolver* solver = new (memory) NvFlexSolver();
    std::memset(solver, 0, sizeof(NvFlexSolver));
    new (solver) NvFlexSolver();

    NvFlex::ContextEx* context = lib->mContext;

    const unsigned maxParticles = (unsigned)desc->maxParticles;
    const int maxDiffuse = desc->maxDiffuseParticles;
    const unsigned alignedParticles = (maxParticles + 0xff) & ~0xffu;

    solver->mLib = lib;

    // KernelParams constant buffer: one element of 512 bytes, CPU writable.
    solver->mKernelParams.Create(context, 1, sizeof(KernelParams));

    // Identity active-index list: the initial active set is 0..maxParticles-1.
    {
        unsigned* indices = (unsigned*)std::malloc(maxParticles * 4);
        for (unsigned i = 0; i < maxParticles; ++i)
            indices[i] = i;
        solver->mActiveIndices.Create(context, maxParticles, 4, indices);
        std::free(indices);
    }

    // Particle attribute buffers.
    solver->mPositions.CreateZero(context, maxParticles, 0x10);
    solver->mNewPositions.CreateZero(context, maxParticles, 0x10);
    solver->mRestPositions.CreateZero(context, maxParticles, 0x10);
    solver->mVelocities.CreateZero(context, maxParticles, 0xc);
    solver->mNormals.CreateZero(context, maxParticles, 0x10);
    solver->mSortedNormals.CreateZero(context, maxParticles, 0x10);
    CreateRawZero(solver->mNormalsTemp, context, maxParticles);
    CreateRawZero(solver->mDeltas, context, maxParticles);

    // Neighbour lists: maxNeighborsPerParticle entries per aligned particle.
    const unsigned neighbors = (unsigned)desc->maxNeighborsPerParticle * alignedParticles;
    solver->mParticleNeighbors.CreateZero(context, neighbors, 4);
    solver->mParticleNeighborCounts.CreateZero(context, maxParticles, 4);

    solver->mSortedPositions.CreateZero(context, maxParticles, 0x10);
    solver->mSortedNewPositions.CreateZero(context, maxParticles, 0x10);
    solver->mSortedNewVelocities.CreateZero(context, maxParticles, 0x10);
    solver->mSortedVelocities.CreateZero(context, maxParticles, 0x10);

    // Static contacts: maxContactsPerParticle per particle.
    const unsigned contacts = (unsigned)desc->maxContactsPerParticle * maxParticles;
    solver->mStaticContactPlanes.CreateZero(context, contacts, 0x10);
    solver->mStaticContactVelocities.CreateZero(context, contacts, 0x10);
    solver->mStaticContactCounts.CreateZero(context, maxParticles, 4);

    solver->mDensities.CreateZero(context, maxParticles, 4);
    solver->mPhases.CreateZero(context, maxParticles, 4);
    solver->mSortedPhases.CreateZero(context, maxParticles, 4);
    solver->mPotentials.CreateZero(context, maxParticles, 4);
    solver->mCurl.CreateZero(context, maxParticles, 0x10);
    solver->mAnisotropy1.CreateZero(context, maxParticles, 0x10);
    solver->mAnisotropy2.CreateZero(context, maxParticles, 0x10);
    solver->mAnisotropy3.CreateZero(context, maxParticles, 0x10);
    solver->mSortedDensities.CreateZero(context, maxParticles, 4);
    solver->mParticleBounds.CreateZero(context, 2, 0xc);

    // Bounds reduction: one entry per 128 particles, then per 128 groups.
    const unsigned boundsGroups = (maxParticles + 0x7f) >> 7;
    const unsigned boundsSuperGroups = (boundsGroups + 0x7f) >> 7;
    solver->mBoundsLower.CreateZero(context, boundsGroups, 0x10);
    solver->mBoundsUpper.CreateZero(context, boundsGroups, 0x10);
    solver->mGroupBoundsLower.CreateZero(context, boundsGroups, 0x10);
    solver->mGroupBoundsUpper.CreateZero(context, boundsGroups, 0x10);
    (void)boundsSuperGroups;

    // Radix sort over maxSortBlocks = ceil(maxParticles / 1024) blocks.
    {
        NvFlex::RadixSortDesc sortDesc;
        sortDesc.maxSortBlocks = (maxParticles + 0x3ff) >> 10;
        solver->mRadixSort = static_cast<NvFlex::RadixSortImpl*>(
            NvFlex::createRadixSort(context, &sortDesc));
        solver->mRadixSortSize = (int)maxParticles;
    }

    solver->mReverseLookup.CreateZero(context, maxParticles, 4);
    solver->mCellIds.CreateZero(context, maxParticles, 4);

    // 256 pseudo-random floats, generated by the shipped xorshift so the
    // diffuse-particle jitter matches bit for bit.
    {
        float* random = (float*)std::malloc(0x100 * sizeof(float));
        unsigned seed1 = 0, seed2 = 0;
        for (int i = 0; i < 0x100; ++i)
        {
            seed1 = (seed1 << 5 | seed1 >> 0x1b) ^ seed2 * seed1 ^ seed2;
            seed2 = (seed2 << 0xc | seed2 >> 0x14) ^ seed1;
            random[i] = (float)seed1 * 5.4210108624275222e-20f * 2.0f - 1.0f;
        }
        solver->mRandom.Create(context, 0x100, 4, random);
        std::free(random);
    }

    if (maxDiffuse != 0)
    {
        solver->mDiffusePositions.CreateZero(context, (unsigned)maxDiffuse, 0x10);
        solver->mDiffuseVelocities.CreateZero(context, (unsigned)maxDiffuse, 0x10);
        solver->mDiffusePositionsNew.CreateZero(context, (unsigned)maxDiffuse, 0x10);
        solver->mDiffuseVelocitiesNew.CreateZero(context, (unsigned)maxDiffuse, 0x10);
        solver->mDiffuseSortSize.CreateZero(context, 1, 4);
        CreateDispatchArgs(solver->mDiffuseLaunchArg, context, 0);
        solver->mNumDiffuseParticles.CreateZero(context, 1, 4);
        solver->mNumDiffuseParticlesNew.CreateZero(context, 1, 4);
        solver->mMaxDiffuseParticles = maxDiffuse;
        solver->mDiffuseRate = 0.016666668f;
        solver->mDiffuseTimer = 0.0f;
    }

    // Shipped parameter defaults (flex.obj @0x3a8xx).
    solver->mParams.relaxationMode = eNvFlexRelaxationLocal;
    solver->mParams.relaxationFactor = 1.0f;
    solver->mParams.maxSpeed = 3.4028235e+38f;
    solver->mParams.maxAcceleration = 3.4028235e+38f;
    solver->mParams.numIterations = 1;
    solver->mShapes.mNumShapes = 0;

    solver->mSortedVelocities.CreateZero(context, maxParticles, 0x10);
    solver->mSmoothPositionsOriginal.CreateZero(context, maxParticles, 0x10);

    // The two per-substep constant buffers.
    solver->mSubstepParams.Create(context, 1, 8);
    solver->mNumShapes.Create(context, 1, 4);

    // Timers: one DxTimers (disabled) with a pool, plus the detail-timer table.
    {
        void* timersMemory = NvFlex::g_FlexMalloc(sizeof(DxTimers));
        if (timersMemory != 0)
        {
            DxTimers* timers = new (timersMemory) DxTimers();
            timers->mEnabled = false;
            timers->mContext = context;
            timers->mTimerPool = context->createTimerPool(200);
            solver->mTimers = timers;
        }
    }

    solver->mLatencyTimerBeginRecorded = false;
    solver->mLatencyTimerEndRecorded = false;
    solver->mLatencyTimerEnable = false;
    // Each detail-timer record owns a 0x100-byte name buffer (the shipped
    // ctor allocates one per entry and the dtor frees it).  NvFlexDetailTimer's
    // `name` is a char*, so the record array alone is not enough.
    solver->mNumDetailTimers = 0x100;
    solver->mDetailTimers = new NvFlexDetailTimer[0x100];
    for (int i = 0; i < solver->mNumDetailTimers; ++i)
    {
        solver->mDetailTimers[i].name = new char[0x100];
        solver->mDetailTimers[i].name[0] = '\0';
        solver->mDetailTimers[i].time = 0.0f;
    }

    solver->mMaxParticles = (int)maxParticles;
    solver->mMaxNeighborsPerParticle = desc->maxNeighborsPerParticle;
    solver->mMaxContactsPerParticle = desc->maxContactsPerParticle;
    solver->mNumParticles = 0;
    solver->mNumParticlesAligned = 0;
    solver->mDesc = *desc;

    lib->mResourceTracker.mSolver.push_back(solver);
    return solver;
}

// ---------------------------------------------------------------------------
// NvFlexDestroySolver — @0x3ab30 (5737 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexDestroySolver(NvFlexSolver* solver)
{
    if (solver == 0)
        return;

    NvFlexLibrary* lib = solver->mLib;

    DxBufferExt* buffers[] =
    {
        &solver->mActiveIndices, &solver->mPositions, &solver->mNewPositions,
        &solver->mRestPositions, &solver->mVelocities, &solver->mNormals,
        &solver->mSortedNormals, &solver->mNormalsTemp, &solver->mDeltas,
        &solver->mParticleNeighbors, &solver->mParticleNeighborCounts,
        &solver->mSortedPositions, &solver->mSortedNewPositions,
        &solver->mSortedNewVelocities, &solver->mSortedVelocities,
        &solver->mStaticContactPlanes, &solver->mStaticContactVelocities,
        &solver->mStaticContactCounts, &solver->mDensities, &solver->mPhases,
        &solver->mSortedPhases, &solver->mPotentials, &solver->mCurl,
        &solver->mAnisotropy1, &solver->mAnisotropy2, &solver->mAnisotropy3,
        &solver->mSortedDensities, &solver->mParticleBounds, &solver->mBoundsLower,
        &solver->mBoundsUpper, &solver->mGroupBoundsLower, &solver->mGroupBoundsUpper,
        &solver->mReverseLookup, &solver->mCellIds, &solver->mRandom,
        &solver->mSmoothPositionsOriginal, &solver->mPlanes, &solver->mSpringIndices,
        &solver->mSpringLengths, &solver->mSpringStiffness, &solver->mHalfSpringIndices,
        &solver->mHalfSpringOpposites, &solver->mHalfSpringLengths,
        &solver->mHalfSpringStiffness, &solver->mHalfSpringParticleBegin,
        &solver->mHalfSpringParticleEnd, &solver->mRigidOffsets, &solver->mRigidIndices,
        &solver->mRigidLocalPositions, &solver->mRigidLocalNormals,
        &solver->mRigidCoefficients, &solver->mRigidPlasticThresholds,
        &solver->mRigidPlasticCreeps, &solver->mRigidRotations, &solver->mRigidTranslations,
        &solver->mDynamicTriangles, &solver->mDynamicTriangleNormals,
        &solver->mDynamicTriangleEdges, &solver->mDynamicNumTriEdges, &solver->mPressures,
        &solver->mInflatables, &solver->mDiffusePositions, &solver->mDiffuseVelocities,
        &solver->mDiffusePositionsNew, &solver->mDiffuseVelocitiesNew,
        &solver->mNumDiffuseParticles, &solver->mNumDiffuseParticlesNew,
        &solver->mDiffuseSortSize, &solver->mDiffuseLaunchArg, &solver->mSDFShapeDX
    };
    for (unsigned i = 0; i < sizeof(buffers) / sizeof(buffers[0]); ++i)
        buffers[i]->Release();

    solver->mKernelParams.Release();
    solver->mHalfSpringConstantBuffer.Release();
    solver->mSubstepParams.Release();
    solver->mNumShapes.Release();

    if (solver->mRadixSort != 0)
        solver->mRadixSort->release();
    solver->mRadixSort = 0;

    if (solver->mTimers != 0)
    {
        if (solver->mTimers->mTimerPool != 0)
            solver->mTimers->mTimerPool->release();
        solver->mTimers->~DxTimers();
        NvFlex::g_FlexFree(solver->mTimers);
        solver->mTimers = 0;
    }

    for (int i = 0; i < solver->mNumDetailTimers; ++i)
        delete[] solver->mDetailTimers[i].name;
    delete[] solver->mDetailTimers;
    solver->mDetailTimers = 0;
    solver->mNumDetailTimers = 0;

    solver->mShapes.mAabbMin.Release();
    solver->mShapes.mAabbMax.Release();
    solver->mShapes.mGeometry.Release();
    solver->mShapes.mPositions.Release();
    solver->mShapes.mRotations.Release();
    solver->mShapes.mPrevPositions.Release();
    solver->mShapes.mPrevRotations.Release();
    solver->mShapes.mFlags.Release();
    solver->mShapes.mBVH.mNodeLowers.Release();
    solver->mShapes.mBVH.mNodeUppers.Release();
    solver->mShapes.mBVH.mRootNode.Release();

    if (lib != 0)
    {
        std::vector<NvFlexSolver*>& solvers = lib->mResourceTracker.mSolver;
        for (unsigned i = 0; i < solvers.size(); ++i)
        {
            if (solvers[i] == solver)
            {
                solvers.erase(solvers.begin() + i);
                break;
            }
        }
    }

    solver->~NvFlexSolver();
    NvFlex::g_FlexFree(solver);
}

// ---------------------------------------------------------------------------
// NvFlexGetSolvers — @0x3c230 (102 bytes)
// ---------------------------------------------------------------------------
extern "C" int NvFlexGetSolvers(NvFlexLibrary* lib, NvFlexSolver** solvers, int maxSolvers)
{
    const std::vector<NvFlexSolver*>& registered = lib->mResourceTracker.mSolver;
    const int count = (int)registered.size();
    const int n = count < maxSolvers ? count : maxSolvers;
    for (int i = 0; i < n; ++i)
        solvers[i] = registered[i];
    return count;
}

// ---------------------------------------------------------------------------
// NvFlexUpdateSolver — @0x3c2d0 (1157 bytes)
// ---------------------------------------------------------------------------
// The public step entry point: enable timers, rebuild the flattened triangle
// data and the shape BVH when they are dirty, then run `numSubsteps` substeps of
// `Update` with a fixed `dt / numSubsteps`.  Timer pools are inert in this
// reconstruction (`DxTimers::mEnabled` only turns on through NvFlexSetDebug),
// so the timer scopes keep their shape without affecting behaviour.
extern "C" void NvFlexUpdateSolver(NvFlexSolver* solver, float dt, int numSubsteps,
                                   bool enableTimers)
{
    solver->mTimers->mEnabled = enableTimers;
    solver->mTimers->mTimerPool->clear();

    if (solver->mTimers->mEnabled)
        solver->mTimers->mTimerPool->reserveTimers(
            (unsigned)((solver->mParams.numIterations + 5) * numSubsteps * 10));

    if (solver->mLatencyTimerEnable)
    {
        solver->mLatencyTimerIndex = solver->mTimers->mTimerPool->begin();
        solver->mLatencyTimerBeginRecorded = true;
    }

    if (solver->mLib->mRebuildTriangleData)
    {
        DxTimer timer(solver->mTimers, L"FlattenTriangleMeshes");
        FlattenTriangleMeshes(solver->mLib);
        solver->mLib->mRebuildTriangleData = false;
    }

    if (solver->mShapes.mRebuildShapeData)
    {
        NvFlex::ContextEx* context = solver->mLib->mContext;

        // Publish the shape count before the transform pass reads it.
        int* numShapes = (int*)context->map(solver->mNumShapes.mBuffer);
        *numShapes = solver->mShapes.mNumShapes;
        context->unmap(solver->mNumShapes.mBuffer);

        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        // Shader 047 TransformShapeBounds: u0 lower bounds, u1 upper bounds;
        // t0 positions, t1 rotations, t2 previous positions, t3 previous
        // rotations, t4 flags, t5 geometry, t6 convex meshes, t7 triangle-mesh
        // table.  The shape count arrives through the second constant buffer.
        params.shader = solver->mLib->mShaderTransformShapeBounds;
        params.readWrite[0] = solver->mShapes.mAabbMin.mBuffer->getResourceRW();
        params.readWrite[1] = solver->mShapes.mAabbMax.mBuffer->getResourceRW();
        params.readOnly[0] = solver->mShapes.mPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = solver->mShapes.mRotations.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] =
            solver->mShapes.mPrevPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] =
            solver->mShapes.mPrevRotations.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[4] = solver->mShapes.mFlags.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[5] = solver->mShapes.mGeometry.mBuffer->getResource(NvFlex::eBuffer);
        if (solver->mLib->mConvexMeshData.mConvexes.mBuffer != 0)
            params.readOnly[6] =
                solver->mLib->mConvexMeshData.mConvexes.mBuffer->getResource(NvFlex::eBuffer);
        if (solver->mLib->mTriangleMeshData.mTriMeshTableDevice.mBuffer != 0)
            params.readOnly[7] = solver->mLib->mTriangleMeshData.mTriMeshTableDevice.mBuffer
                                     ->getResource(NvFlex::eBuffer);
        params.secondConstantBuffer = solver->mNumShapes.mBuffer;
        params.gridDim.x = (unsigned)((solver->mShapes.mNumShapes + 0xff) >> 8);
        params.gridDim.y = 1;
        params.gridDim.z = 1;

        {
            DxTimer timer(solver->mTimers, params.shader->desc().label);
            nvpipe::Dispatch(solver, params);
        }

        solver->mLib->mBVHBuilder->Build(&solver->mShapes.mBVH, &solver->mShapes.mAabbMin,
                                         &solver->mShapes.mAabbMax, 0,
                                         solver->mShapes.mNumShapes, 0, solver->mTimers);
        solver->mShapes.mRebuildShapeData = false;
    }

    if (solver->mNumParticles != 0)
    {
        const float subDt = dt / (float)numSubsteps;
        InitParams(solver, subDt, numSubsteps);
        for (int i = 0; i < numSubsteps; ++i)
            Update(solver, subDt, i, numSubsteps);
    }

    if (solver->mLatencyTimerEnable)
    {
        solver->mTimers->mTimerPool->end(solver->mLatencyTimerIndex);
        solver->mLatencyTimerEndRecorded = true;
    }

    solver->mTimers->mTimerPool->resolve();
}

// ===========================================================================
// Appended section — flex.obj's host-side NvFlex* entry points that only touch
// the solver's own buffers / plain fields (no compute dispatch of their own).
//
// Every function below is a 1:1 transcription of its
// `// ---- <Name>  @ <rva>  (<bytes>)` block in
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  The shipped bodies spell
// the same buffer-growth block out inline for every attribute:
//
//   if (capacity < needed)
//       create/replace with { format = eNvFlexFormat_unknown, dim, stride,
//                             bufferType = 0x13, data = <host image|null> }
//
// which is what `EnsureSolverBuffer` below does.  The user-buffer copies use
// `Context::copyToDevice` with a destination kind chosen from the user
// buffer's type (host -> 8, native pointer -> 16, device -> 2), exactly the
// three-case dispatch the pseudocode shows.
//
// Note: the shipped `NvFlexGetSprings` body is empty (28 bytes of NVTX range
// push/pop) and stays empty here.
// ===========================================================================
namespace
{

// Buffer type every solver attribute buffer is created with:
// eBuffer | eUAV_SRV | eStructured == 0x13.
const unsigned kSolverBufferType =
    (unsigned)(NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eStructured);
// eBuffer | eUAV_SRV | eStage | eStructured == 0x17 (the inflatable staging
// buffer).
const unsigned kStageBufferType =
    (unsigned)(NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eStage |
               NvFlex::eStructured);

// The inline growth block of every NvFlexSet*() entry point: keep the existing
// buffer when its descriptor already covers the request, otherwise release it
// and recreate it with the shipped descriptor.
void EnsureSolverBuffer(DxBuffer& buffer, NvFlex::ContextEx* context,
                        unsigned dim, unsigned stride, unsigned bufferType,
                        void* data = 0)
{
    if (buffer.mBuffer != 0 && buffer.mBufferDesc.dim >= dim &&
        buffer.mBufferDesc.stride >= stride)
        return;

    buffer.Release();

    buffer.mBufferDesc.format = NvFlexFormat(0);   // eNvFlexFormat_unknown
    buffer.mBufferDesc.dim = dim;
    buffer.mBufferDesc.stride = stride;
    buffer.mBufferDesc.bufferType = bufferType;
    buffer.mBufferDesc.data = data;
    buffer.mBuffer = context->createBuffer(&buffer.mBufferDesc, 0);
}

// User buffer -> solver buffer.  The shipped code dispatches on the user
// buffer's `type`: host memory through the buffer's staging view, a native
// D3D resource the user registered (type 3) straight from its pointer, and a
// solver-owned device buffer through the default view.  In every case `dst` is
// the solver-side buffer and the offsets are zero.
void CopyToSolverBuffer(NvFlex::ContextEx* context, DxBuffer& destination,
                        NvFlexBuffer* source, unsigned size)
{
    if (source == 0 || size == 0)
        return;

    switch ((int)source->type)
    {
    case eNvFlexBufferHost:
        context->copyToDevice(destination.mBuffer, 0, source->mBuffer, 0, size,
                              NvFlex::eDownloadBuffer);
        return;
    case 3:
        context->copyToDevice(destination.mBuffer, 0,
                              static_cast<NvFlex::Buffer*>(source->ptr), 0, size,
                              NvFlex::eNativeBuffer);
        return;
    case eNvFlexBufferDevice:
        context->copyToDevice(destination.mBuffer, 0, source->mBuffer, 0, size,
                              NvFlex::BufferResourceTypes(kDefaultBuffer));
        return;
    default:
        return;
    }
}

// ---------------------------------------------------------------------------
// utils.obj `anonymous namespace'::copy(DxBuffer*, DxBuffer*) @0x4e140
// (200 bytes) and its constant-buffer sibling @0x4e210 (177 bytes).
//
// flex.obj emits its own private copies of these (the PDB records them inside
// flex.obj's anonymous namespace); utils.cpp's copies have internal linkage,
// so NvFlexCopySolver carries the same two bodies here.
// ---------------------------------------------------------------------------
void CopyDxBuffer(NvFlex::ContextEx* dstContext, DxBuffer* dst,
                  NvFlex::ContextEx* srcContext, DxBuffer* src)
{
    if (dst->mBufferDesc.dim * dst->mBufferDesc.stride <
        src->mBufferDesc.stride * src->mBufferDesc.dim)
    {
        dst->CreateZero(dstContext, src->mBufferDesc.dim, dst->mBufferDesc.stride, true);
    }

    void* srcPtr = srcContext->map(src->mBuffer, NvFlex::eMapReadWrite, true);
    void* dstPtr = dstContext->map(dst->mBuffer, NvFlex::eMapReadWrite, true);
    std::memcpy(dstPtr, srcPtr, src->mBufferDesc.stride * src->mBufferDesc.dim);
    dstContext->unmap(dst->mBuffer);
    srcContext->unmap(src->mBuffer);
}

void CopyConstantBuffer(NvFlex::ContextEx* dstContext, DxConstantBuffer* dst,
                        NvFlex::ContextEx* srcContext, DxConstantBuffer* src)
{
    if (dst->mBufferDesc.dim * dst->mBufferDesc.stride <
        src->mBufferDesc.dim * src->mBufferDesc.stride)
    {
        dst->Create(dstContext, src->mBufferDesc.dim, src->mBufferDesc.stride);
    }

    void* srcPtr = srcContext->map(src->mBuffer);
    void* dstPtr = dstContext->map(dst->mBuffer);
    std::memcpy(dstPtr, srcPtr, src->mBufferDesc.dim * src->mBufferDesc.stride);
    dstContext->unmap(dst->mBuffer);
    srcContext->unmap(src->mBuffer);
}

// ---------------------------------------------------------------------------
// getTimer — flex.obj @0x429f0 (224 bytes) and @0x42ad0 (224 bytes)
// ---------------------------------------------------------------------------
// Sums the elapsed time of every timer-pool slot registered under `name`.  The
// shipped bodies walk the name->index map's equal_range; the two overloads
// differ only in which map they read (narrow / wide names).
float getTimer(DxTimers* timers, const char* name)
{
    float total = 0.0f;
    std::pair<std::map<const char*, int, NvFlex::ltstr>::iterator,
              std::map<const char*, int, NvFlex::ltstr>::iterator>
        range = timers->mNameToTimerMap.equal_range(name);
    for (std::map<const char*, int, NvFlex::ltstr>::iterator it = range.first;
         it != range.second; ++it)
    {
        total += timers->mTimerPool->get(it->second);
    }
    return total;
}

float getTimer(DxTimers* timers, const wchar_t* name)
{
    float total = 0.0f;
    std::pair<std::map<const wchar_t*, int, NvFlex::ltwstr>::iterator,
              std::map<const wchar_t*, int, NvFlex::ltwstr>::iterator>
        range = timers->mNameToTimerMapW.equal_range(name);
    for (std::map<const wchar_t*, int, NvFlex::ltwstr>::iterator it = range.first;
         it != range.second; ++it)
    {
        total += timers->mTimerPool->get(it->second);
    }
    return total;
}

// FlexInflatable is 16 bytes { int mStartTri; int mNumTris; float mRestVolume;
// float mConstraintScale; } (PDB field list 0x2642).  It has no definition in
// FlexInternal.h, so the staging image is built with this local record; only
// its bytes reach the GPU.
struct InflatableHost
{
    int   mStartTri;
    int   mNumTris;
    float mRestVolume;
    float mConstraintScale;
};
static_assert(sizeof(InflatableHost) == 16, "InflatableHost");

// ---------------------------------------------------------------------------
// CalculateRestDensity — flex.obj @0x34ba0 (914 bytes)
// ---------------------------------------------------------------------------
// Not declared anywhere else in this reconstruction (it is a file-static helper
// of flex.obj).  It samples the rest-density kernel over a packed 3D lattice of
// neighbour offsets produced by `TightPack3D` (450 bytes, flex_objects.cpp) and
// returns the fluid rest density, the density-constraint scale and the surface
// constraint scale.
void CalculateRestDensity(float fluidRestDistance, float radius, float* restDensity,
                          float* densityConstraintScale, float* surfaceConstraintScale)
{
    // The shipped body build-constructs a 2048-element vector (0x6000 bytes /
    // 12) and passes its size as TightPack3D's capacity.
    std::vector<XVector3<float> > points(2048u);

    const int count = NvFlex::flexobj::TightPack3D(radius, fluidRestDistance,
                                                   points.empty() ? 0 : &points[0],
                                                   (int)points.size());

    *restDensity = 0.0f;
    *densityConstraintScale = 0.0f;

    const float minusTwo = -2.0f;
    const float scale = 1.0f / radius;
    // 15 / (radius * pi * radius^2)  (0x41700000 == 15.0f, 0x40490fdb == pi)
    const float weight = 15.0f / (radius * 3.14159274101257324219f * radius * radius);
    const float weightDerivative = weight * minusTwo;

    float sumConstraint = 0.0f;      // accumulated density-gradient term
    float sumSurface = 0.0f;         // accumulated normal-projection term
    int i = 0;

    for (i = 0; i < count; ++i)
    {
        const XVector3<float>& p = points[(size_t)i];

        float length = p.x * p.x + p.y * p.y + p.z * p.z;
        if (length == 0.0f)
            length = 0.0f;
        else
            length = std::sqrt(length);

        const float w = 1.0f - scale * length;
        const float dw = weightDerivative * w * scale;
        *restDensity += w * w * weight;
        *densityConstraintScale += dw * dw;

        if (p.y <= 0.0f)
        {
            float normal = p.x * p.x + p.y * p.y + p.z * p.z;
            if (normal == 0.0f)
                normal = 0.0f;
            else
                normal = std::sqrt(normal);
            normal = p.y / normal;

            sumConstraint += weight * minusTwo * w * scale * normal;

            float distance = p.x * p.x + p.y * p.y + p.z * p.z;
            if (distance == 0.0f)
                distance = 0.0f;
            else
                distance = std::sqrt(distance);
            sumSurface -= distance * normal;
        }
    }

    *surfaceConstraintScale = sumConstraint / sumSurface;
}

} // namespace

// ---------------------------------------------------------------------------
// NvFlexSetParams — @0x3f980 (255 bytes)
// ---------------------------------------------------------------------------
// Recomputes the fluid rest-density constants only when the fluid rest
// distance / radius actually change, then copies the whole parameter block.
extern "C" void NvFlexSetParams(NvFlexSolver* solver, const NvFlexParams* params)
{
    const float fluidRestDistance = params->fluidRestDistance;

    if (!(fluidRestDistance > 0.0f))
    {
        solver->mFluidRestDensity = 0.0f;
        solver->mFluidDensityConstraintScale = 0.0f;
        solver->mFluidSurfaceConstraintScale = 0.0f;
    }
    else if (params->radius != solver->mParams.radius ||
             fluidRestDistance != solver->mParams.fluidRestDistance)
    {
        CalculateRestDensity(fluidRestDistance, params->radius,
                             &solver->mFluidRestDensity,
                             &solver->mFluidDensityConstraintScale,
                             &solver->mFluidSurfaceConstraintScale);
    }

    solver->mParams = *params;
}

// ---------------------------------------------------------------------------
// NvFlexSetShapes — @0x40360 (2407 bytes)
// ---------------------------------------------------------------------------
// Shapes: geometry@0x10 stride, positions/rotations/previous poses@0x10, flags
// 4 bytes; all sized to slack = numShapes * 3 / 2.
extern "C" void NvFlexSetShapes(NvFlexSolver* solver, NvFlexBuffer* geometry,
                                NvFlexBuffer* shapePositions, NvFlexBuffer* shapeRotations,
                                NvFlexBuffer* shapePrevPositions,
                                NvFlexBuffer* shapePrevRotations, NvFlexBuffer* shapeFlags,
                                int numShapes)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;
    FlexShapeData& shapes = solver->mShapes;

    if (numShapes == 0)
    {
        shapes.mNumShapes = 0;
        return;
    }

    if (shapes.mMaxShapes < numShapes)
    {
        const unsigned slack = (unsigned)((numShapes * 3) / 2);

        EnsureSolverBuffer(shapes.mAabbMin, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mAabbMax, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mPositions, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mRotations, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mPrevPositions, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mPrevRotations, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(shapes.mFlags, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(shapes.mGeometry, context, slack, 0x10, kSolverBufferType);

        shapes.mMaxShapes = (int)slack;
    }

    const unsigned vectorBytes = (unsigned)numShapes << 4;
    const unsigned flagBytes = (unsigned)numShapes << 2;

    CopyToSolverBuffer(context, shapes.mGeometry, geometry, vectorBytes);
    CopyToSolverBuffer(context, shapes.mPositions, shapePositions, vectorBytes);
    CopyToSolverBuffer(context, shapes.mRotations, shapeRotations, vectorBytes);
    CopyToSolverBuffer(context, shapes.mPrevPositions,
                       shapePrevPositions != 0 ? shapePrevPositions : shapePositions,
                       vectorBytes);
    CopyToSolverBuffer(context, shapes.mPrevRotations,
                       shapePrevRotations != 0 ? shapePrevRotations : shapeRotations,
                       vectorBytes);
    CopyToSolverBuffer(context, shapes.mFlags, shapeFlags, flagBytes);

    shapes.mNumShapes = numShapes;
    shapes.mRebuildShapeData = true;
}

// ---------------------------------------------------------------------------
// NvFlexSetRigids — @0x3d350 (3183 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetRigids(NvFlexSolver* solver, NvFlexBuffer* offsets,
                                NvFlexBuffer* indices, NvFlexBuffer* restPositions,
                                NvFlexBuffer* restNormals, NvFlexBuffer* stiffness,
                                NvFlexBuffer* thresholds, NvFlexBuffer* creeps,
                                NvFlexBuffer* rotations, NvFlexBuffer* translations,
                                int numRigids, int numIndices)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    if (numRigids == 0)
    {
        solver->mNumRigids = 0;
        solver->mNumRigidIndices = 0;
        return;
    }

    if (solver->mMaxRigids < numRigids)
    {
        const unsigned slack = (unsigned)((numRigids * 3) / 2);

        // The offset array carries one extra trailing sentinel entry.
        EnsureSolverBuffer(solver->mRigidOffsets, context, slack + 1, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidRotations, context, slack, 0x10, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidTranslations, context, slack, 0xc, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidCoefficients, context, slack, 4, kSolverBufferType);

        solver->mMaxRigids = (int)slack;
    }

    if (solver->mMaxRigidIndices < numIndices)
    {
        const unsigned slack = (unsigned)((numIndices * 3) / 2);

        EnsureSolverBuffer(solver->mRigidIndices, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidLocalPositions, context, slack, 0xc, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidLocalNormals, context, slack, 0x10, kSolverBufferType);

        solver->mMaxRigidIndices = (int)slack;
    }

    CopyToSolverBuffer(context, solver->mRigidRotations, rotations,
                       (unsigned)numRigids << 4);
    CopyToSolverBuffer(context, solver->mRigidTranslations, translations,
                       (unsigned)numRigids * 0xc);
    CopyToSolverBuffer(context, solver->mRigidOffsets, offsets,
                       (unsigned)numRigids * 4 + 4);
    CopyToSolverBuffer(context, solver->mRigidIndices, indices,
                       (unsigned)numIndices * 4);
    CopyToSolverBuffer(context, solver->mRigidCoefficients, stiffness,
                       (unsigned)numRigids * 4);
    CopyToSolverBuffer(context, solver->mRigidLocalPositions, restPositions,
                       (unsigned)numIndices * 0xc);

    if (restNormals == 0)
    {
        XVector4<float> zero(0.0f, 0.0f, 0.0f, 0.0f);
        ClearBufferFloat4(solver->mLib, solver->mRigidLocalNormals.mBuffer,
                          numIndices << 4, &zero);
    }
    else
    {
        CopyToSolverBuffer(context, solver->mRigidLocalNormals, restNormals,
                           (unsigned)numIndices << 4);
    }

    solver->mNumRigidIndices = numIndices;
    solver->mNumRigids = numRigids;

    if (thresholds == 0 || creeps == 0)
    {
        solver->mRigidPlasticThresholds.Release();
        solver->mRigidPlasticCreeps.Release();
        solver->mMaxPlasticEntries = 0;
        return;
    }

    if (solver->mMaxPlasticEntries < numRigids)
    {
        EnsureSolverBuffer(solver->mRigidPlasticThresholds, context,
                           (unsigned)solver->mMaxRigids, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mRigidPlasticCreeps, context,
                           (unsigned)solver->mMaxRigids, 4, kSolverBufferType);

        solver->mMaxPlasticEntries = solver->mMaxRigids;
    }

    NvFlex::flexobj::WriteData(context, solver->mRigidPlasticThresholds.mBuffer, 0,
                               thresholds, 0, (unsigned)numRigids * 4);
    NvFlex::flexobj::WriteData(context, solver->mRigidPlasticCreeps.mBuffer, 0,
                               creeps, 0, (unsigned)numRigids * 4);
}

// ---------------------------------------------------------------------------
// NvFlexGetRigids — @0x3dfc0 (1269 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetRigids(NvFlexSolver* solver, NvFlexBuffer* offsets,
                                NvFlexBuffer* indices, NvFlexBuffer* restPositions,
                                NvFlexBuffer* restNormals, NvFlexBuffer* stiffness,
                                NvFlexBuffer* thresholds, NvFlexBuffer* creeps,
                                NvFlexBuffer* rotations, NvFlexBuffer* translations)
{
    if (solver->mNumRigids == 0)
        return;

    const unsigned numRigids = (unsigned)solver->mNumRigids;
    const unsigned numIndices = (unsigned)solver->mNumRigidIndices;
    NvFlex::ContextEx* context = solver->mLib->mContext;

    NvFlex::flexobj::ReadData(context, rotations, 0, solver->mRigidRotations.mBuffer, 0,
                              numRigids << 4);
    NvFlex::flexobj::ReadData(context, translations, 0, solver->mRigidTranslations.mBuffer, 0,
                              numRigids * 0xc);
    NvFlex::flexobj::ReadData(context, offsets, 0, solver->mRigidOffsets.mBuffer, 0,
                              numRigids * 4 + 4);
    NvFlex::flexobj::ReadData(context, indices, 0, solver->mRigidIndices.mBuffer, 0,
                              numIndices * 4);
    NvFlex::flexobj::ReadData(context, stiffness, 0, solver->mRigidCoefficients.mBuffer, 0,
                              numRigids * 4);
    NvFlex::flexobj::ReadData(context, restPositions, 0,
                              solver->mRigidLocalPositions.mBuffer, 0, numIndices * 0xc);
    NvFlex::flexobj::ReadData(context, restNormals, 0, solver->mRigidLocalNormals.mBuffer, 0,
                              numIndices << 4);

    if (solver->mRigidPlasticThresholds.mBuffer != 0)
        NvFlex::flexobj::ReadData(context, thresholds, 0,
                                  solver->mRigidPlasticThresholds.mBuffer, 0,
                                  numRigids * 4);
    if (solver->mRigidPlasticCreeps.mBuffer != 0)
        NvFlex::flexobj::ReadData(context, creeps, 0, solver->mRigidPlasticCreeps.mBuffer, 0,
                                  numRigids * 4);
}

// ---------------------------------------------------------------------------
// NvFlexSetInflatables — @0x3e4c0 (1395 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetInflatables(NvFlexSolver* solver, NvFlexBuffer* startTris,
                                     NvFlexBuffer* numTris, NvFlexBuffer* restVolumes,
                                     NvFlexBuffer* overPressures,
                                     NvFlexBuffer* constraintScales, int numInflatables)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    if (numInflatables > solver->mMaxInflatables)
    {
        const unsigned slack = (unsigned)((numInflatables * 3) / 2);

        EnsureSolverBuffer(solver->mInflatables, context, slack, 0x10, kStageBufferType);

        // The pressure array is created from a zero-filled host image.
        void* zeros = std::malloc(slack * 4);
        if (zeros != 0)
            std::memset(zeros, 0, slack * 4);

        solver->mPressures.Release();
        EnsureSolverBuffer(solver->mPressures, context, slack, 4, kSolverBufferType, zeros);
        std::free(zeros);

        solver->mMaxInflatables = (int)slack;
    }

    if (numInflatables > 0)
    {
        std::vector<InflatableHost> inflatables((size_t)numInflatables);

        for (int i = 0; i < numInflatables; ++i)
        {
            const int* start = static_cast<const int*>(startTris->map(0));
            const int* triCount = static_cast<const int*>(numTris->map(0));
            const float* rest = static_cast<const float*>(restVolumes->map(0));
            const float* over = static_cast<const float*>(overPressures->map(0));
            const float* scale = static_cast<const float*>(constraintScales->map(0));

            InflatableHost& entry = inflatables[(size_t)i];
            entry.mStartTri = start[i];
            entry.mNumTris = triCount[i];
            entry.mRestVolume = over[i] * rest[i];
            entry.mConstraintScale = scale[i];

            startTris->unmap();
            numTris->unmap();
            restVolumes->unmap();
            overPressures->unmap();
            constraintScales->unmap();
        }

        const unsigned bytes = (unsigned)numInflatables << 4;
        void* mapped = context->map(solver->mInflatables.mBuffer, NvFlex::eMapWrite, true);
        std::memcpy(mapped, &inflatables[0], bytes);
        context->unmap(solver->mInflatables.mBuffer);
        context->upload(solver->mInflatables.mBuffer, 0, bytes);
    }

    solver->mNumInflatables = numInflatables;
}

// ---------------------------------------------------------------------------
// NvFlexSetSprings — @0x3c760 (3003 bytes)
// ---------------------------------------------------------------------------
// Spring arrays are replicated for both half-edges: the index buffer is stored
// twice (numSprings * 2 ints), the half-spring arrays get 3/2 slack of
// numSprings * 6 entries and the per-particle range arrays are mMaxParticles
// long.  With springs present, the body builds a radix sort over the
// half-spring pairs (the HalfSpring dedup the shipped code performs through
// SpringsGenerateIndices / SpringsReorderData, both at @0x325b0/@0x32770).
extern "C" void NvFlexSetSprings(NvFlexSolver* solver, NvFlexBuffer* indices,
                                 NvFlexBuffer* restLengths, NvFlexBuffer* stiffness,
                                 int numSprings)
{
    NvFlexLibrary* lib = solver->mLib;
    NvFlex::ContextEx* context = lib->mContext;

    if (solver->mMaxSprings < numSprings)
    {
        const unsigned slack = (unsigned)((numSprings * 3) / 2);

        EnsureSolverBuffer(solver->mSpringIndices, context, slack * 2, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mSpringLengths, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mSpringStiffness, context, slack, 4, kSolverBufferType);

        solver->mMaxSprings = (int)slack;
    }

    if (numSprings != 0)
    {
        CopyToSolverBuffer(context, solver->mSpringIndices, indices,
                           (unsigned)numSprings * 8);
        CopyToSolverBuffer(context, solver->mSpringLengths, restLengths,
                           (unsigned)numSprings * 4);
        CopyToSolverBuffer(context, solver->mSpringStiffness, stiffness,
                           (unsigned)numSprings * 4);
    }

    const int numHalfSprings = numSprings * 2;

    if (numHalfSprings > solver->mMaxHalfSprings)
    {
        const unsigned slack = (unsigned)((numSprings * 6) / 2);

        EnsureSolverBuffer(solver->mHalfSpringIndices, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mHalfSpringOpposites, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mHalfSpringLengths, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mHalfSpringStiffness, context, slack, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mHalfSpringParticleBegin, context,
                           (unsigned)solver->mMaxParticles, 4, kSolverBufferType);
        EnsureSolverBuffer(solver->mHalfSpringParticleEnd, context,
                           (unsigned)solver->mMaxParticles, 4, kSolverBufferType);

        if (solver->mHalfSpringConstantBuffer.mBuffer == 0 ||
            solver->mHalfSpringConstantBuffer.mBufferDesc.dim == 0 ||
            solver->mHalfSpringConstantBuffer.mBufferDesc.stride < 4)
        {
            solver->mHalfSpringConstantBuffer.Create(context, 1, 4);
        }

        solver->mMaxHalfSprings = (int)slack;
    }

    if (numSprings != 0)
    {
        // Radix sort over the half-edge index pairs; rebuilt when the current
        // one covers fewer keys than numSprings * 2.
        if (solver->mRadixSortSize < numHalfSprings)
        {
            solver->mRadixSort->release();

            NvFlex::RadixSortDesc sortDesc;
            sortDesc.maxSortBlocks = (unsigned)(numHalfSprings + 0x3ff) >> 10;
            solver->mRadixSort = static_cast<NvFlex::RadixSortImpl*>(
                NvFlex::createRadixSort(context, &sortDesc));
            solver->mRadixSortSize = numHalfSprings;
        }

        // Publish the spring count to the shared half-spring constant buffer.
        int* springCount =
            static_cast<int*>(context->map(solver->mHalfSpringConstantBuffer.mBuffer));
        *springCount = numSprings;
        context->unmap(solver->mHalfSpringConstantBuffer.mBuffer);

        NvFlex::RadixSortBuffer input = solver->mRadixSort->getInputBuffer();
        context->copyToDevice(input.key, 0, indices->mBuffer, 0,
                              (unsigned)numSprings * 8, NvFlex::eDownloadBuffer);

        SpringsGenerateIndices(lib, input.val, solver->mHalfSpringConstantBuffer.mBuffer,
                               numSprings);

        NvFlex::RadixSortParams sortParams;
        sortParams.numKeys = (unsigned)numHalfSprings;
        sortParams.numSortBlocks = (unsigned)(numHalfSprings + 0x3ff) >> 10;
        sortParams.bits = 0x20;
        solver->mRadixSort->sort(context, &sortParams, solver->mTimers);

        NvFlex::RadixSortBuffer output = solver->mRadixSort->getOutputBuffer();

        // The shipped 4th argument of ClearBufferInt is the DxTimers pointer,
        // but the shipped body stores a literal 0 in the constant buffer, so 0
        // is the behaviour-preserving value here.
        ClearBufferInt(lib, solver->mHalfSpringParticleBegin.mBuffer,
                       solver->mMaxParticles << 2, 0);
        ClearBufferInt(lib, solver->mHalfSpringParticleEnd.mBuffer,
                       solver->mMaxParticles << 2, 0);

        SpringsFindParticleRange(lib, output.key, solver->mHalfSpringParticleBegin.mBuffer,
                                 solver->mHalfSpringParticleEnd.mBuffer,
                                 solver->mHalfSpringConstantBuffer.mBuffer, numSprings);

        SpringsReorderData(lib, output.val, solver->mSpringIndices.mBuffer,
                           solver->mSpringLengths.mBuffer, solver->mSpringStiffness.mBuffer,
                           solver->mHalfSpringOpposites.mBuffer,
                           solver->mHalfSpringLengths.mBuffer,
                           solver->mHalfSpringStiffness.mBuffer,
                           solver->mHalfSpringConstantBuffer.mBuffer, numSprings);
    }

    solver->mNumSprings = numSprings;
}

// ---------------------------------------------------------------------------
// NvFlexGetSprings — @0x3d330 (28 bytes)
// ---------------------------------------------------------------------------
// The shipped body is empty: NVTX range push, range pop.
extern "C" void NvFlexGetSprings(NvFlexSolver*, NvFlexBuffer*, NvFlexBuffer*,
                                 NvFlexBuffer*, int)
{
}

// ---------------------------------------------------------------------------
// NvFlexSetDynamicTriangles — @0x3f4f0 (783 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetDynamicTriangles(NvFlexSolver* solver, NvFlexBuffer* indices,
                                          NvFlexBuffer* normals, int numTris)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    if (numTris == 0)
    {
        solver->mDynamicNumTris = 0;
        return;
    }

    if (solver->mDynamicMaxTris < numTris)
    {
        const unsigned slack = (unsigned)((numTris * 3) / 2);

        EnsureSolverBuffer(solver->mDynamicTriangles, context, slack * 3, 4,
                           kSolverBufferType);
        EnsureSolverBuffer(solver->mDynamicTriangleNormals, context, slack, 0xc,
                           kSolverBufferType);

        solver->mDynamicMaxTris = (int)slack;
    }

    CopyToSolverBuffer(context, solver->mDynamicTriangles, indices,
                       (unsigned)numTris * 0xc);
    CopyToSolverBuffer(context, solver->mDynamicTriangleNormals, normals,
                       (unsigned)numTris * 0xc);

    solver->mDynamicNumTris = numTris;
}

// ---------------------------------------------------------------------------
// NvFlexGetDynamicTriangles — @0x3f800 (371 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetDynamicTriangles(NvFlexSolver* solver, NvFlexBuffer* indices,
                                          NvFlexBuffer* normals, int numTris)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;
    const unsigned bytes = (unsigned)numTris * 0xc;

    NvFlex::flexobj::ReadData(context, indices, 0, solver->mDynamicTriangles.mBuffer, 0,
                              bytes);
    NvFlex::flexobj::ReadData(context, normals, 0,
                              solver->mDynamicTriangleNormals.mBuffer, 0, bytes);
}

// ---------------------------------------------------------------------------
// NvFlexGetShapeBVH — @0x42290 (3 bytes)
// ---------------------------------------------------------------------------
// The shipped body is empty (the shape BVH is fetched through
// NvFlexGetTriangleMeshes / NvFlexGetShapeBVH is a no-op in this build).
extern "C" void NvFlexGetShapeBVH(NvFlexSolver*, void*)
{
}

// ---------------------------------------------------------------------------
// NvFlexGetBounds — @0x3ff00 (355 bytes)
// ---------------------------------------------------------------------------
// mParticleBounds holds { lower.xyz, upper.xyz } (two 12-byte records).
extern "C" void NvFlexGetBounds(NvFlexSolver* solver, NvFlexBuffer* lower,
                                NvFlexBuffer* upper)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    NvFlex::flexobj::ReadData(context, lower, 0, solver->mParticleBounds.mBuffer, 0, 0xc);
    NvFlex::flexobj::ReadData(context, upper, 0, solver->mParticleBounds.mBuffer, 0xc, 0xc);
}

// ---------------------------------------------------------------------------
// NvFlexGetContacts — @0x40cd0 (715 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetContacts(NvFlexSolver* solver, NvFlexBuffer* planes,
                                  NvFlexBuffer* velocities, NvFlexBuffer* indices,
                                  NvFlexBuffer* counts)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    const unsigned contactCount =
        (unsigned)(solver->mMaxContactsPerParticle * solver->mMaxParticles);
    const unsigned contactBytes = contactCount * 0x10;
    const unsigned particleBytes = (unsigned)solver->mMaxParticles << 2;

    NvFlex::flexobj::ReadData(context, planes, 0, solver->mStaticContactPlanes.mBuffer, 0,
                              contactBytes);
    NvFlex::flexobj::ReadData(context, velocities, 0,
                              solver->mStaticContactVelocities.mBuffer, 0, contactBytes);
    NvFlex::flexobj::ReadData(context, indices, 0, solver->mReverseLookup.mBuffer, 0,
                              particleBytes);
    NvFlex::flexobj::ReadData(context, counts, 0, solver->mStaticContactCounts.mBuffer, 0,
                              particleBytes);
}

// ---------------------------------------------------------------------------
// NvFlexGetNeighbors — @0x40fa0 (741 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetNeighbors(NvFlexSolver* solver, NvFlexBuffer* neighbors,
                                   NvFlexBuffer* counts, NvFlexBuffer* apiToInternal,
                                   NvFlexBuffer* internalToApi)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    const unsigned neighborCount =
        (unsigned)(solver->mMaxNeighborsPerParticle * solver->mMaxParticles);
    const unsigned particleBytes = (unsigned)solver->mMaxParticles << 2;

    NvFlex::flexobj::ReadData(context, neighbors, 0, solver->mParticleNeighbors.mBuffer, 0,
                              neighborCount * 4);
    NvFlex::flexobj::ReadData(context, counts, 0, solver->mParticleNeighborCounts.mBuffer, 0,
                              particleBytes);
    NvFlex::flexobj::ReadData(context, apiToInternal, 0, solver->mReverseLookup.mBuffer, 0,
                              particleBytes);

    // The final lookup table is the radix sort's sorted payload.
    NvFlex::RadixSortBuffer output = solver->mRadixSort->getOutputBuffer();
    NvFlex::flexobj::ReadData(context, internalToApi, 0, output.val, 0, particleBytes);
}

// ---------------------------------------------------------------------------
// NvFlexGetAnisotropy — @0x3f0b0 (160 bytes)
// ---------------------------------------------------------------------------
// Library-level copies (with the stride/range validation in flex_copy.cpp).
extern "C" void NvFlexGetAnisotropy(NvFlexSolver* solver, NvFlexBuffer* q1,
                                    NvFlexBuffer* q2, NvFlexBuffer* q3,
                                    const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, q1, solver->mAnisotropy1.mBuffer, desc);
    NvFlex::flexobj::ReadData(solver->mLib, q2, solver->mAnisotropy2.mBuffer, desc);
    NvFlex::flexobj::ReadData(solver->mLib, q3, solver->mAnisotropy3.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// NvFlexGetDiffuseParticles — @0x3f150 (537 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetDiffuseParticles(NvFlexSolver* solver, NvFlexBuffer* p,
                                          NvFlexBuffer* v, NvFlexBuffer* count)
{
    if (solver->mMaxDiffuseParticles == 0)
        return;

    NvFlex::ContextEx* context = solver->mLib->mContext;
    const unsigned bytes = (unsigned)solver->mMaxDiffuseParticles << 4;

    NvFlex::flexobj::ReadData(context, p, 0, solver->mDiffusePositions.mBuffer, 0, bytes);
    NvFlex::flexobj::ReadData(context, v, 0, solver->mDiffuseVelocities.mBuffer, 0, bytes);
    NvFlex::flexobj::ReadData(context, count, 0, solver->mNumDiffuseParticles.mBuffer, 0, 4);
}

// ---------------------------------------------------------------------------
// NvFlexSetDiffuseParticles — @0x3f370 (382 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetDiffuseParticles(NvFlexSolver* solver, NvFlexBuffer* p,
                                          NvFlexBuffer* v, int n)
{
    NvFlex::ContextEx* context = solver->mLib->mContext;

    CopyToSolverBuffer(context, solver->mDiffusePositions, p, (unsigned)n << 4);
    CopyToSolverBuffer(context, solver->mDiffuseVelocities, v, (unsigned)n << 4);

    int* count = static_cast<int*>(
        context->map(solver->mNumDiffuseParticles.mBuffer, NvFlex::eMapWrite, true));
    *count = n;
    context->unmap(solver->mNumDiffuseParticles.mBuffer);
    context->upload(solver->mNumDiffuseParticles.mBuffer, 0, 4);
}

// ---------------------------------------------------------------------------
// NvFlexGetTimers — @0x42bb0 (1510 bytes)
// ---------------------------------------------------------------------------
// Timers are inert in this reconstruction (mEnabled is false unless
// NvFlexSetDebug turned it on), but the loops are transcribed so the returned
// array is byte-identical to the shipped one.
extern "C" void NvFlexGetTimers(NvFlexSolver* solver, NvFlexTimers* timers)
{
    NvFlexLibrary* lib = solver->mLib;

    if (solver->mTimers->mEnabled == false)
        return;

    solver->mTimers->mTimerPool->sync();

    timers->predict = getTimer(solver->mTimers, lib->mShaderPredict->desc().label);
    timers->createCellIndices =
        getTimer(solver->mTimers, lib->mShaderCalculateParticleHash->desc().label);
    timers->sortCellIndices = getTimer(solver->mTimers, "SortCellIndices");
    timers->createGrid = getTimer(solver->mTimers, lib->mShaderCreateGrid->desc().label);
    timers->reorder =
        getTimer(solver->mTimers, lib->mShaderReorderParticles->desc().label);
    timers->collideParticles =
        getTimer(solver->mTimers, lib->mShaderCollideParticles->desc().label);
    timers->collideShapes =
        getTimer(solver->mTimers, lib->mShaderCollideShapes->desc().label);
    timers->collideTriangles =
        getTimer(solver->mTimers, lib->mShaderCollideTriangles->desc().label);
    timers->collideFields = getTimer(solver->mTimers, "CollideFields");

    timers->calculateDensity =
        getTimer(solver->mTimers,
                 lib->mShaderCalculateDensitySurfaceTension->desc().label) +
        getTimer(solver->mTimers, lib->mShaderCalculateDensity->desc().label);

    timers->solveDensities =
        getTimer(solver->mTimers,
                 lib->mShaderSolveDensitiesSurfaceTension->desc().label) +
        getTimer(solver->mTimers, lib->mShaderSolveDensities->desc().label) +
        getTimer(solver->mTimers, lib->mShaderSolveDensitiesNonFluid->desc().label);

    timers->solveVelocities =
        getTimer(solver->mTimers, lib->mShaderSolveVelocities->desc().label) +
        getTimer(solver->mTimers, lib->mShaderCalculateVorticity->desc().label) +
        getTimer(solver->mTimers, lib->mShaderUpdateVelocities->desc().label);

    const bool fastShapes = lib->mIsSHFLSupported != false &&
                            lib->mIsFP32ATOMICSupported != false;
    const bool plastic = !(solver->mRigidPlasticThresholds.mBuffer == 0 ||
                           solver->mRigidPlasticCreeps.mBuffer == 0);

    if (!plastic)
    {
        if (fastShapes)
        {
            timers->solveShapes =
                getTimer(solver->mTimers, lib->mShaderSolveShapes32->desc().label) +
                getTimer(solver->mTimers, lib->mShaderSolveShapes->desc().label) +
                getTimer(solver->mTimers, lib->mShaderSolveShapes128->desc().label);
        }
        else
        {
            timers->solveShapes =
                getTimer(solver->mTimers, lib->mShaderSolveShapes->desc().label);
        }
    }
    else
    {
        if (fastShapes)
        {
            timers->solveShapes =
                getTimer(solver->mTimers,
                         lib->mShaderSolveShapesPlasticDeformation32->desc().label) +
                getTimer(solver->mTimers,
                         lib->mShaderSolveShapesPlasticDeformation->desc().label) +
                getTimer(solver->mTimers,
                         lib->mShaderSolveShapesPlasticDeformation128->desc().label);
        }
        else
        {
            timers->solveShapes =
                getTimer(solver->mTimers,
                         lib->mShaderSolveShapesPlasticDeformation->desc().label);
        }
    }

    timers->solveSprings =
        getTimer(solver->mTimers, lib->mShaderSolveSprings->desc().label);
    timers->solveContacts =
        getTimer(solver->mTimers, lib->mShaderSolveContactsSequential->desc().label);
    timers->solveInflatables =
        getTimer(solver->mTimers, lib->mShaderSolveInflatableVolume->desc().label) +
        getTimer(solver->mTimers,
                 lib->mShaderCalculateInflatableVolume->desc().label);
    timers->applyDeltas =
        getTimer(solver->mTimers, lib->mShaderApplyDeltas->desc().label);

    timers->calculateAnisotropy =
        getTimer(solver->mTimers, lib->mShaderCalculateAnisotropy->desc().label) +
        getTimer(solver->mTimers, lib->mShaderSmoothPositions->desc().label);

    timers->updateDiffuse =
        getTimer(solver->mTimers,
                 lib->mShaderCompactDiffuseParticles->desc().label) +
        getTimer(solver->mTimers,
                 lib->mShaderClampDiffuseParticleCount->desc().label) +
        getTimer(solver->mTimers,
                 lib->mShaderUpdateDiffuseParticles->desc().label) +
        getTimer(solver->mTimers,
                 lib->mShaderCreateDiffuseParticles->desc().label);

    timers->updateTriangles =
        getTimer(solver->mTimers, lib->mShaderUpdateTrianglesInit->desc().label) +
        getTimer(solver->mTimers, lib->mShaderUpdateTriangles->desc().label);

    timers->updateNormals =
        getTimer(solver->mTimers, lib->mShaderNormalizeVertexNormals->desc().label) +
        getTimer(solver->mTimers, lib->mShaderUpdateVertexNormals->desc().label) +
        getTimer(solver->mTimers, lib->mShaderUpdateVertexNormalsInit->desc().label);

    timers->finalize = getTimer(solver->mTimers, lib->mShaderFinalize->desc().label);

    timers->updateBounds =
        getTimer(solver->mTimers, lib->mShaderCalculateBoundsGroup->desc().label) +
        getTimer(solver->mTimers, lib->mShaderCalculateBounds->desc().label) +
        getTimer(solver->mTimers,
                 lib->mShaderCalculateBoundsFinalize->desc().label);

    timers->total = timers->predict + timers->createCellIndices + timers->sortCellIndices +
                    timers->createGrid + timers->reorder + timers->collideParticles +
                    timers->collideShapes + timers->collideTriangles +
                    timers->collideFields + timers->calculateDensity +
                    timers->solveDensities + timers->solveVelocities + timers->solveShapes +
                    timers->solveSprings + timers->solveContacts +
                    timers->solveInflatables + timers->applyDeltas +
                    timers->calculateAnisotropy + timers->updateDiffuse +
                    timers->updateTriangles + timers->updateNormals + timers->finalize +
                    timers->updateBounds;
}

// ---------------------------------------------------------------------------
// NvFlexGetDetailTimers — @0x431a0 (2358 bytes)
// ---------------------------------------------------------------------------
// Collects one entry per registered wide-name timer, appends the synthetic
// "Total" entry (filled from the latency timer when one is armed) and copies
// the names into the solver's detail-timer table.
extern "C" int NvFlexGetDetailTimers(NvFlexSolver* solver, NvFlexDetailTimer** timers)
{
    int count = 0;

    if (solver->mTimers->mEnabled != false)
    {
        std::vector<std::wstring> names;
        std::vector<float> values;

        for (std::map<const wchar_t*, int, NvFlex::ltwstr>::iterator it =
                 solver->mTimers->mNameToTimerMapW.begin();
             it != solver->mTimers->mNameToTimerMapW.end(); ++it)
        {
            names.push_back(std::wstring(it->first));
            values.push_back(getTimer(solver->mTimers, it->first));
        }

        names.push_back(std::wstring(L"Total"));
        values.push_back(0.0f);

        if (solver->mLatencyTimerEnable == false)
        {
            solver->mLatencyTimerEnable = true;
        }
        else if (solver->mLatencyTimerBeginRecorded != false &&
                 solver->mLatencyTimerEndRecorded != false)
        {
            solver->mTimers->mTimerPool->sync();
            values.back() =
                solver->mTimers->mTimerPool->get(solver->mLatencyTimerIndex);
        }

        count = (int)names.size();
        for (int i = 0; i < count; ++i)
        {
            NvFlexDetailTimer& timer = solver->mDetailTimers[i];

            const wchar_t* wide = names[(size_t)i].c_str();
            const int wideLength = (int)names[(size_t)i].size();
            const int narrowLength = WideCharToMultiByte(0, 0, wide, wideLength, 0, 0, 0, 0);

            std::string narrow((size_t)(narrowLength > 0 ? narrowLength : 0), '\0');
            if (narrowLength > 0)
                WideCharToMultiByte(0, 0, wide, wideLength, &narrow[0], narrowLength, 0, 0);

            std::memset(timer.name, 0, 0x100);
            std::strncpy(timer.name, narrow.c_str(), narrow.size());
            timer.time = values[(size_t)i];
        }
    }

    *timers = solver->mDetailTimers;
    return count;
}

// ---------------------------------------------------------------------------
// NvFlexCopySolver — @0x42310 (1603 bytes)
// ---------------------------------------------------------------------------
// Copies the simulation state of `src` into `dst` buffer by buffer (through
// host mappings) and then the plain scalar members.  Order and selection are
// exactly the shipped body's.
extern "C" void NvFlexCopySolver(NvFlexSolver* dst, NvFlexSolver* src)
{
    NvFlex::ContextEx* dstContext = dst->mLib->mContext;
    NvFlex::ContextEx* srcContext = src->mLib->mContext;

    CopyConstantBuffer(dstContext, &dst->mKernelParams, srcContext, &src->mKernelParams);

    CopyDxBuffer(dstContext, &dst->mPositions, srcContext, &src->mPositions);
    CopyDxBuffer(dstContext, &dst->mVelocities, srcContext, &src->mVelocities);
    CopyDxBuffer(dstContext, &dst->mRestPositions, srcContext, &src->mRestPositions);
    CopyDxBuffer(dstContext, &dst->mPhases, srcContext, &src->mPhases);
    CopyDxBuffer(dstContext, &dst->mActiveIndices, srcContext, &src->mActiveIndices);
    CopyDxBuffer(dstContext, &dst->mDensities, srcContext, &src->mDensities);
    CopyDxBuffer(dstContext, &dst->mCurl, srcContext, &src->mCurl);
    CopyDxBuffer(dstContext, &dst->mNormals, srcContext, &src->mNormals);
    CopyDxBuffer(dstContext, &dst->mAnisotropy1, srcContext, &src->mAnisotropy1);
    CopyDxBuffer(dstContext, &dst->mAnisotropy2, srcContext, &src->mAnisotropy2);
    CopyDxBuffer(dstContext, &dst->mAnisotropy3, srcContext, &src->mAnisotropy3);
    CopyDxBuffer(dstContext, &dst->mStaticContactPlanes, srcContext,
                 &src->mStaticContactPlanes);
    CopyDxBuffer(dstContext, &dst->mStaticContactVelocities, srcContext,
                 &src->mStaticContactVelocities);
    CopyDxBuffer(dstContext, &dst->mStaticContactCounts, srcContext,
                 &src->mStaticContactCounts);

    dst->mMaxNeighborsPerParticle = src->mMaxNeighborsPerParticle;

    CopyDxBuffer(dstContext, &dst->mRigidOffsets, srcContext, &src->mRigidOffsets);
    CopyDxBuffer(dstContext, &dst->mRigidIndices, srcContext, &src->mRigidIndices);
    CopyDxBuffer(dstContext, &dst->mRigidLocalPositions, srcContext,
                 &src->mRigidLocalPositions);
    CopyDxBuffer(dstContext, &dst->mRigidLocalNormals, srcContext, &src->mRigidLocalNormals);
    CopyDxBuffer(dstContext, &dst->mRigidCoefficients, srcContext,
                 &src->mRigidCoefficients);
    CopyDxBuffer(dstContext, &dst->mRigidPlasticThresholds, srcContext,
                 &src->mRigidPlasticThresholds);
    CopyDxBuffer(dstContext, &dst->mRigidPlasticCreeps, srcContext, &src->mRigidPlasticCreeps);
    CopyDxBuffer(dstContext, &dst->mRigidRotations, srcContext, &src->mRigidRotations);
    CopyDxBuffer(dstContext, &dst->mRigidTranslations, srcContext, &src->mRigidTranslations);

    dst->mNumRigids = src->mNumRigids;
    dst->mMaxRigids = src->mMaxRigids;
    dst->mNumRigidIndices = src->mNumRigidIndices;
    dst->mMaxRigidIndices = src->mMaxRigidIndices;
    dst->mMaxPlasticEntries = src->mMaxPlasticEntries;

    CopyDxBuffer(dstContext, &dst->mDynamicTriangles, srcContext, &src->mDynamicTriangles);
    CopyDxBuffer(dstContext, &dst->mDynamicTriangleNormals, srcContext,
                 &src->mDynamicTriangleNormals);

    dst->mDynamicNumTris = src->mDynamicNumTris;
    dst->mDynamicMaxTris = src->mDynamicMaxTris;

    CopyDxBuffer(dstContext, &dst->mInflatables, srcContext, &src->mInflatables);

    dst->mNumInflatables = src->mNumInflatables;
    dst->mMaxInflatables = src->mMaxInflatables;

    CopyDxBuffer(dstContext, &dst->mDiffusePositions, srcContext, &src->mDiffusePositions);
    CopyDxBuffer(dstContext, &dst->mDiffuseVelocities, srcContext, &src->mDiffuseVelocities);
    CopyDxBuffer(dstContext, &dst->mNumDiffuseParticles, srcContext,
                 &src->mNumDiffuseParticles);

    dst->mMaxDiffuseParticles = src->mMaxDiffuseParticles;
    dst->mDiffuseRate = src->mDiffuseRate;
    dst->mDiffuseTimer = src->mDiffuseTimer;
    dst->mNumParticles = src->mNumParticles;
    dst->mNumParticlesAligned = src->mNumParticlesAligned;
    dst->mMaxParticles = src->mMaxParticles;

    CopyDxBuffer(dstContext, &dst->mParticleBounds, srcContext, &src->mParticleBounds);

    dst->mParams = src->mParams;

    // mShapes: the first call copies mAabbMin through the DxBuffer view, then
    // each remaining GPU array, then the plain members.
    CopyDxBuffer(dstContext, reinterpret_cast<DxBuffer*>(&dst->mShapes), srcContext,
                 reinterpret_cast<DxBuffer*>(&src->mShapes));
    CopyDxBuffer(dstContext, &dst->mShapes.mAabbMax, srcContext, &src->mShapes.mAabbMax);
    CopyDxBuffer(dstContext, &dst->mShapes.mGeometry, srcContext, &src->mShapes.mGeometry);
    CopyDxBuffer(dstContext, &dst->mShapes.mPositions, srcContext, &src->mShapes.mPositions);
    CopyDxBuffer(dstContext, &dst->mShapes.mRotations, srcContext, &src->mShapes.mRotations);
    CopyDxBuffer(dstContext, &dst->mShapes.mPrevPositions, srcContext,
                 &src->mShapes.mPrevPositions);
    CopyDxBuffer(dstContext, &dst->mShapes.mPrevRotations, srcContext,
                 &src->mShapes.mPrevRotations);
    CopyDxBuffer(dstContext, &dst->mShapes.mFlags, srcContext, &src->mShapes.mFlags);

    dst->mShapes.mNumShapes = src->mShapes.mNumShapes;
    dst->mShapes.mMaxShapes = src->mShapes.mMaxShapes;
    dst->mShapes.mLower = src->mShapes.mLower;
    dst->mShapes.mUpper = src->mShapes.mUpper;

    dst->mFluidRestDensity = src->mFluidRestDensity;
    dst->mFluidDensityConstraintScale = src->mFluidDensityConstraintScale;
    dst->mFluidSurfaceConstraintScale = src->mFluidSurfaceConstraintScale;
    dst->mStepCounter = src->mStepCounter;

    for (int i = 0; i < 5; ++i)
    {
        dst->mCallbacks[i].userData = src->mCallbacks[i].userData;
        dst->mCallbacks[i].function = src->mCallbacks[i].function;
    }

    dst->mRadixSortSize = src->mRadixSortSize;

    CopyDxBuffer(dstContext, &dst->mSmoothPositionsOriginal, srcContext,
                 &src->mSmoothPositionsOriginal);

    CopyConstantBuffer(dstContext, &dst->mSubstepParams, srcContext, &src->mSubstepParams);
}
