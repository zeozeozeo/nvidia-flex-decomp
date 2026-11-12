// flexdecomp: reconstructed D3D backend — stage group "collide".
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c; the shader
// interfaces are quoted from work/dxbc/interfaces.txt (index in the block
// header of each function).  Every stage follows the shape established by
// recon/src/dx/flex_pipeline.cpp: zeroed DispatchParams, `readOnly[k]` = t<k>,
// `readWrite[k]` = u<k> in the order the shipped body binds them, grid from the
// stage's group count, b0 = the solver's KernelParams, dispatch inside a
// DxTimer scope.  There is no null-shader guard in the shipped code and none is
// added here.
//
//   ComputeBounds           @ 0x30fb0 (747 bytes)   3 dispatches
//   CalculateParticleHash   @ 0x312a0 (377 bytes)
//   SortParticleHash        @ 0x31420 (330 bytes)   radix sort only
//   ReorderParticles        @ 0x316a0 (491 bytes)
//   CollideParticles        @ 0x31890 (433 bytes)
//   CollideShapes           @ 0x31a50 (834 bytes)
//   CollideTriangles        @ 0x31da0 (829 bytes)
//   LazyClearGrid           @ 0x34630 (321 bytes)
#include "FlexPipeline.h"

// ---------------------------------------------------------------------------
// ComputeBounds — @0x30fb0 (747 bytes)
// ---------------------------------------------------------------------------
// Three sequential reductions, all with b0 = KernelParams:
//   dispatch 1  shader 021 CalculateBounds
//               t0 activeIndices, t1 newPositions;
//               u0 boundsLower, u1 boundsUpper, u2 particleBounds
//               (the generic shader declares only u0/u1; the shipped body
//               binds particleBounds as u2 regardless and the NV/AMD variants
//               consume it).
//   dispatch 2  shader 079 CalculateBoundsGroup
//               t2 boundsLower, t3 boundsUpper (t0/t1 are explicitly null);
//               u0 groupBoundsLower, u1 groupBoundsUpper, u2 particleBounds
//   dispatch 3  shader 039 CalculateBoundsFinalize
//               t2 groupBoundsLower, t3 groupBoundsUpper (t0/t1 null);
//               u2 particleBounds (u0/u1 null)
void ComputeBounds(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlexLibrary* lib = solver->mLib;

    {
        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = lib->mShaderCalculateBounds;
        params.readWrite[0] = solver->mBoundsLower.mBuffer->getResourceRW();
        params.readWrite[1] = solver->mBoundsUpper.mBuffer->getResourceRW();
        params.readWrite[2] = solver->mParticleBounds.mBuffer->getResourceRW();
        params.readOnly[0] = solver->mActiveIndices.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = solver->mNewPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = numBlocks;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    {
        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = lib->mShaderCalculateBoundsGroup;
        params.readWrite[0] = solver->mGroupBoundsLower.mBuffer->getResourceRW();
        params.readWrite[1] = solver->mGroupBoundsUpper.mBuffer->getResourceRW();
        params.readWrite[2] = solver->mParticleBounds.mBuffer->getResourceRW();
        params.readOnly[2] = solver->mBoundsLower.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] = solver->mBoundsUpper.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = (numBlocks + 0x7f) >> 7;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    {
        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = lib->mShaderCalculateBoundsFinalize;
        params.readWrite[2] = solver->mParticleBounds.mBuffer->getResourceRW();
        params.readOnly[2] = solver->mGroupBoundsLower.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] = solver->mGroupBoundsUpper.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = 1;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CalculateParticleHash — @0x312a0 (377 bytes)
// ---------------------------------------------------------------------------
// Shader 064: t0 activeIndices, t1 newPositions, t2 particleBounds;
// u0/u1 are the radix sort's *input* side (keys and payload), obtained through
// `getInputBuffer()` before the sort.  Grid: ceil(GetSortSize(numParticles)/256).
void CalculateParticleHash(NvFlexSolver* solver, int substep, int numSubsteps,
                           unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    NvFlex::RadixSortBuffer sortBuffers = solver->mRadixSort->getInputBuffer();

    const unsigned numParticles = (unsigned)solver->mNumParticles;
    const unsigned groups = ((numParticles >> 10) + (unsigned)((numParticles & 0x3ffu) != 0)) *
                                0x400u +
                            0xffu;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderCalculateParticleHash;
    params.readWrite[0] = sortBuffers.key->getResourceRW();
    params.readWrite[1] = sortBuffers.val->getResourceRW();
    params.readOnly[0] = solver->mActiveIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mParticleBounds.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = groups >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// SortParticleHash — @0x31420 (330 bytes)
// ---------------------------------------------------------------------------
// No dispatch of its own: one radix-sort invocation with 24 significand bits
// and ceil(numParticles/1024) sort blocks, inside the "SortCellIndices" timer
// scope the shipped body opens by hand (DxTimer is the same push/begin pair).
void SortParticleHash(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    NvFlex::RadixSortParams sortParams;
    sortParams.numKeys = (unsigned)solver->mNumParticles;
    sortParams.numSortBlocks = ((unsigned)solver->mNumParticles + 0x3ffu) >> 10;
    sortParams.bits = 0x18;

    DxTimer timer(solver->mTimers, L"SortCellIndices");
    solver->mRadixSort->sort(solver->mLib->mContext, &sortParams, solver->mTimers);
}

// ---------------------------------------------------------------------------
// ReorderParticles — @0x316a0 (491 bytes)
// ---------------------------------------------------------------------------
// Shader 014: six SRVs, six UAVs, both groups in ascending register order:
//   t0 positions, t1 newPositions, t2 velocities, t3 phases,
//   t4 the radix sort payload (sortedToOriginal indices, a typed uint buffer),
//   t5 normals;
//   u0 sortedPositions, u1 sortedNewPositions, u2 sortedVelocities,
//   u3 sortedPhases, u4 reverseLookup, u5 sortedNormals.
// (The radix result comes from `getOutputBuffer()`; .val is the payload that
// becomes t4.)
void ReorderParticles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::RadixSortBuffer sortBuffers = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderReorderParticles;
    params.readWrite[0] = solver->mSortedPositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mSortedVelocities.mBuffer->getResourceRW();
    params.readWrite[3] = solver->mSortedPhases.mBuffer->getResourceRW();
    params.readWrite[4] = solver->mReverseLookup.mBuffer->getResourceRW();
    params.readWrite[5] = solver->mSortedNormals.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = sortBuffers.val->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mNormals.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// CollideParticles — @0x31890 (433 bytes)
// ---------------------------------------------------------------------------
// Shader 058: t0 cellBucketStarts, t1 cellBucketEnds, t2 the radix payload
// (sorted particle indices), t3 sortedNewPositions, t4 restPositions,
// t5 sortedPhases, t6 particleBounds; u0 particleNeighbors,
// u1 particleNeighborCounts.
void CollideParticles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::RadixSortBuffer sortBuffers = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderCollideParticles;
    params.readWrite[0] = solver->mParticleNeighbors.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mParticleNeighborCounts.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mLib->mCellBucketStarts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mLib->mCellBucketEnds.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = sortBuffers.val->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mRestPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mParticleBounds.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// CollideShapes — @0x31a50 (834 bytes)
// ---------------------------------------------------------------------------
// Shader 070 (cb0 = KernelParams, cb1 = the substep fraction pair), one
// dispatch:
//   t0 shape geometry, t1 shape positions, t2 shape rotations,
//   t3 shape prevPositions, t4 shape prevRotations, t5 shape flags   (when
//   mNumShapes > 0; otherwise those slots stay null),
//   t6 sortedPositions, t7 sortedNewPositions, t8 sortedPhases,
//   t9 shape BVH root, t10 shape BVH lowers, t11 shape BVH uppers   (when
//   mNumShapes != 0),
//   t12 convex records, t13 convex planes                           (when
//   mNumConvexes != 0),
//   t14 SDF records followed by t15..t30 the sixteen SDF textures   (when
//   mNumSDFs != 0);
//   u0 staticContactCounts, u1 staticContactPlanes, u2 staticContactVelocities.
// b1 receives the substep fractions (start, end) written through mSubstepParams.
void CollideShapes(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    NvFlexLibrary* lib = solver->mLib;
    NvFlex::Context* context = lib->mContext;

    const float invSubsteps = 1.0f / (float)numSubsteps;
    float* substepParams = (float*)context->map(solver->mSubstepParams.mBuffer);
    substepParams[0] = (float)substep * invSubsteps;
    substepParams[1] = (float)(substep + 1) * invSubsteps;
    context->unmap(solver->mSubstepParams.mBuffer);

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderCollideShapes;
    params.readWrite[0] = solver->mStaticContactCounts.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mStaticContactPlanes.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mStaticContactVelocities.mBuffer->getResourceRW();

    if (0 < solver->mShapes.mNumShapes)
    {
        params.readOnly[0] = solver->mShapes.mGeometry.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = solver->mShapes.mPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] = solver->mShapes.mRotations.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] =
            solver->mShapes.mPrevPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[4] =
            solver->mShapes.mPrevRotations.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[5] = solver->mShapes.mFlags.mBuffer->getResource(NvFlex::eBuffer);
    }

    params.readOnly[6] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[8] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);

    if (solver->mShapes.mNumShapes != 0)
    {
        params.readOnly[9] = solver->mShapes.mBVH.mRootNode.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[10] =
            solver->mShapes.mBVH.mNodeLowers.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[11] =
            solver->mShapes.mBVH.mNodeUppers.mBuffer->getResource(NvFlex::eBuffer);
    }

    if (lib->mConvexMeshData.mNumConvexes != 0)
    {
        params.readOnly[12] =
            lib->mConvexMeshData.mConvexes.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[13] = lib->mConvexMeshData.mPlanes.mBuffer->getResource(NvFlex::eBuffer);
    }

    if (lib->mSDFData.mNumSDFs != 0)
    {
        params.readOnly[14] = lib->mSDFData.mSDFs.mBuffer->getResource(NvFlex::eBuffer);

        // The shipped body walks the fixed sixteen-entry texture table and only
        // binds the non-null entries; the rest stay null.
        for (int i = 0; i < 16; ++i)
        {
            NvFlex::Texture3D* texture = lib->mSDFData.mTextures[i];
            if (texture != 0)
                params.readOnly[15 + i] = texture->getResource();
        }
    }

    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;
    params.secondConstantBuffer = solver->mSubstepParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// CollideTriangles — @0x31da0 (829 bytes)
// ---------------------------------------------------------------------------
// Six parameters in the shipped ABI: the third is never read, `param_4` is the
// substep and `param_5` the substep count (the fractions below are built from
// exactly those two), `param_6` is the block count.  FlexStagesCollide.h names
// them accordingly.
//
// Shader 076 (cb0 = KernelParams, cb1 = substep fractions):
//   t0 mesh vertices, t1 mesh indices, t2 BVH lowers, t3 BVH uppers,
//   t4 BVH root, t5 sortedPositions, t6 sortedNewPositions, t7 sortedPhases,
//   t8..t16 shape prevPositions/prevRotations/flags/positions/rotations/
//   geometry/BVH root/BVH lowers/BVH uppers (t8..t16 only when mNumShapes > 0),
//   t17 the flattened per-mesh table;
//   u0 staticContactCounts, u1 staticContactPlanes, u2 staticContactVelocities.
void CollideTriangles(NvFlexSolver* solver, FlexTriangleMeshData* triMesh, int unused,
                      int substep, int numSubsteps, unsigned numBlocks)
{
    (void)unused;

    NvFlexLibrary* lib = solver->mLib;
    NvFlex::Context* context = lib->mContext;

    const float invSubsteps = 1.0f / (float)numSubsteps;
    float* substepParams = (float*)context->map(solver->mSubstepParams.mBuffer);
    substepParams[0] = (float)substep * invSubsteps;
    substepParams[1] = (float)(substep + 1) * invSubsteps;
    context->unmap(solver->mSubstepParams.mBuffer);

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderCollideTriangles;
    params.readWrite[0] = solver->mStaticContactCounts.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mStaticContactPlanes.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mStaticContactVelocities.mBuffer->getResourceRW();

    if (triMesh->mVerticesArray.mBuffer != 0)
    {
        params.readOnly[0] = triMesh->mVerticesArray.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = triMesh->mIndicesArray.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] = triMesh->mBvhData.mNodeLowersArray.mBuffer->getResource(
            NvFlex::eBuffer);
        params.readOnly[3] = triMesh->mBvhData.mNodeUppersArray.mBuffer->getResource(
            NvFlex::eBuffer);
        params.readOnly[4] = triMesh->mBvhData.mRootNodeArray.mBuffer->getResource(
            NvFlex::eBuffer);
    }

    params.readOnly[5] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);

    // Shape views exist only after shapes were uploaded; the shipped code
    // guards on mNumShapes and the buffers are checked as well because this
    // solver allocates the shape arrays lazily.
    if (0 < solver->mShapes.mNumShapes)
    {
        const DxBufferExt* shapeViews[9] = {
            &solver->mShapes.mPrevPositions, &solver->mShapes.mPrevRotations,
            &solver->mShapes.mFlags,         &solver->mShapes.mPositions,
            &solver->mShapes.mRotations,     &solver->mShapes.mGeometry,
            &solver->mShapes.mBVH.mRootNode, &solver->mShapes.mBVH.mNodeLowers,
            &solver->mShapes.mBVH.mNodeUppers,
        };
        for (int i = 0; i < 9; ++i)
        {
            if (shapeViews[i]->mBuffer != 0)
                params.readOnly[8 + i] = shapeViews[i]->mBuffer->getResource(NvFlex::eBuffer);
        }
    }

    if (triMesh->mVerticesArray.mBuffer != 0)
    {
        params.readOnly[17] =
            triMesh->mTriMeshTableDevice.mBuffer->getResource(NvFlex::eBuffer);
    }

    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;
    params.secondConstantBuffer = solver->mSubstepParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// LazyClearGrid — @0x34630 (321 bytes)
// ---------------------------------------------------------------------------
// Shader 019: t0 activeIndices, t1 newPositions, t2 particleBounds;
// u0 cellBucketStarts, u1 cellBucketEnds.
void LazyClearGrid(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlexLibrary* lib = solver->mLib;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderClearCellBuckets;
    params.readWrite[0] = lib->mCellBucketStarts.mBuffer->getResourceRW();
    params.readWrite[1] = lib->mCellBucketEnds.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mActiveIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mParticleBounds.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    DxTimer timer(solver->mTimers, params.shader->desc().label);
    nvpipe::Dispatch(solver, params);
}
