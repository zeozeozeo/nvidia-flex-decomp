// flexdecomp: reconstructed D3D backend — solver stage group "density".
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  One block per
// function, each header below carrying its shipped RVA and byte size:
//
//   CalculateDensity               @0x321c0 (469 bytes)
//   SolveDensity                   @0x323a0 (523 bytes)
//   SpringsGenerateIndices         @0x325b0 (181 bytes)
//   SpringsFindParticleRange       @0x32670 (242 bytes)
//   SpringsReorderData             @0x32770 (336 bytes)
//   SolveSprings                   @0x328c0 (458 bytes)
//   CalculateAndSolveInflatables   @0x32a90 (631 bytes)
//   SolveShapes                    @0x32e10 (689 bytes)
//   Finalize                       @0x33b60 (409 bytes)
//   CalculateVorticity             @0x33340 (415 bytes)
//   SolveVelocities                @0x334e0 (497 bytes)
//   SmoothPositions                @0x342e0 (414 bytes)
//   CalculateAnisotropy            @0x34480 (425 bytes)
//
// Shader indices are the DXBC containers under
// work/dxbc/NvFlexReleaseD3D_x64/NNN.dxbc; the t<n>/u<n> slot lists are the
// `dcl_resource*`/`dcl_uav*` declarations in work/dxbc/interfaces.txt.  The
// pseudocode binds views in declared-slot order: the k-th `getResource(...)`
// call is t<k>, the k-th `getResourceRW()` call is u<k>.
#include "FlexPipeline.h"

// ---------------------------------------------------------------------------
// CalculateDensity — @0x321c0 (469 bytes)
// ---------------------------------------------------------------------------
// Only runs for a fluid solver (fluidRestDistance > 0).  Shader 034
// (CalculateDensity) or 043 (CalculateDensitySurfaceTension) when surface
// tension is enabled — one dispatch either way:
//   t0 particle neighbors, t1 neighbor counts, t2 sortedNewPositions,
//   t3 sortedPhases, t4 the radix-sort payload (declared only by 043);
//   u0 densities, u1 normals, u2 sortedNormals (u1/u2 also only in 043).
void CalculateDensity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (!(0.0f < solver->mParams.fluidRestDistance))
        return;

    NvFlexLibrary* lib = solver->mLib;
    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mParams.surfaceTension <= 0.0f
                        ? lib->mShaderCalculateDensity
                        : lib->mShaderCalculateDensitySurfaceTension;
    params.readWrite[0] = solver->mDensities.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mNormals.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mSortedNormals.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = sorted.val->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// SolveDensity — @0x323a0 (523 bytes)
// ---------------------------------------------------------------------------
// Three shader variants, one dispatch: 044 SolveDensitiesNonFluid for
// eNvFlexFeatureModeSimpleSolids, 057 SolveDensities when surface tension is
// off, 062 SolveDensitiesSurfaceTension otherwise.  All three share:
//   t0 particle neighbors, t1 neighbor counts, t2 sortedNewPositions,
//   t3 sortedPhases, t4 densities, t5 normals, t6 sortedPositions,
//   t7 sortedNormals; u0 deltas.
void SolveDensity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlexLibrary* lib = solver->mLib;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    if (solver->mDesc.featureMode == eNvFlexFeatureModeSimpleSolids)
        params.shader = lib->mShaderSolveDensitiesNonFluid;
    else if (solver->mParams.surfaceTension <= 0.0f)
        params.shader = lib->mShaderSolveDensities;
    else
        params.shader = lib->mShaderSolveDensitiesSurfaceTension;

    params.readWrite[0] = solver->mDeltas.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mDensities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mNormals.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mSortedNormals.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// SpringsGenerateIndices — @0x325b0 (181 bytes)
// ---------------------------------------------------------------------------
// Shader 053: u0 the half-spring indices; no SRV.  Grid is two threads per
// spring (one per half spring), 256 per group.  Skipped when numSprings is 0.
// No timer scope in the shipped body.
void SpringsGenerateIndices(NvFlexLibrary* lib, NvFlex::Buffer* indices,
                            NvFlex::ConstantBuffer* cb, int numSprings)
{
    if (numSprings == 0)
        return;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderSpringsGenerateIndices;
    params.readWrite[0] = indices->getResourceRW();
    params.gridDim.x = (unsigned)(numSprings * 2 + 0xff) >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = cb;

    lib->mContext->dispatch(&params);
}

// ---------------------------------------------------------------------------
// SpringsFindParticleRange — @0x32670 (242 bytes)
// ---------------------------------------------------------------------------
// Shader 040: t0 the half-spring indices, u0 the particle begin offsets, u1 the
// end offsets.  Two threads per spring, 256 per group.
void SpringsFindParticleRange(NvFlexLibrary* lib, NvFlex::Buffer* b0, NvFlex::Buffer* b1,
                              NvFlex::Buffer* b2, NvFlex::ConstantBuffer* cb,
                              int numSprings)
{
    if (numSprings == 0)
        return;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderSpringsParticleRange;
    params.readOnly[0] = b0->getResource(NvFlex::eBuffer);
    params.readWrite[0] = b1->getResourceRW();
    params.readWrite[1] = b2->getResourceRW();
    params.gridDim.x = (unsigned)(numSprings * 2 + 0xff) >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = cb;

    lib->mContext->dispatch(&params);
}

// ---------------------------------------------------------------------------
// SpringsReorderData — @0x32770 (336 bytes)
// ---------------------------------------------------------------------------
// Shader 071: t0 the half-spring indices, t1..t3 the three per-half-spring
// payload buffers; u0..u2 the reordered payload buffers.  Two threads per
// spring, 256 per group.
void SpringsReorderData(NvFlexLibrary* lib, NvFlex::Buffer* b0, NvFlex::Buffer* b1,
                        NvFlex::Buffer* b2, NvFlex::Buffer* b3, NvFlex::Buffer* b4,
                        NvFlex::Buffer* b5, NvFlex::Buffer* b6,
                        NvFlex::ConstantBuffer* cb, int numSprings)
{
    if (numSprings == 0)
        return;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderSpringsReorder;
    params.readOnly[0] = b0->getResource(NvFlex::eBuffer);
    params.readOnly[1] = b1->getResource(NvFlex::eBuffer);
    params.readOnly[2] = b2->getResource(NvFlex::eBuffer);
    params.readOnly[3] = b3->getResource(NvFlex::eBuffer);
    params.readWrite[0] = b4->getResourceRW();
    params.readWrite[1] = b5->getResourceRW();
    params.readWrite[2] = b6->getResourceRW();
    params.gridDim.x = (unsigned)(numSprings * 2 + 0xff) >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = cb;

    lib->mContext->dispatch(&params);
}

// ---------------------------------------------------------------------------
// SolveSprings — @0x328c0 (458 bytes)
// ---------------------------------------------------------------------------
// Shader 032: t0 the half-spring indices (undeclared — the shader starts at
// t1), t1 opposites, t2 lengths, t3 stiffness, t4 particle begin, t5 particle
// end, t6 reverse lookup, t7 sortedNewPositions; u0 deltas.
// Grid is eight threads per particle, 512 per group, and ignores numBlocks.
void SolveSprings(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    if (solver->mNumSprings == 0)
        return;

    const int blocks = (solver->mMaxParticles * 8 + 0x1ff) / 512;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderSolveSprings;
    params.readWrite[0] = solver->mDeltas.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mHalfSpringIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mHalfSpringOpposites.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mHalfSpringLengths.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mHalfSpringStiffness.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mHalfSpringParticleBegin.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mHalfSpringParticleEnd.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = (unsigned)blocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CalculateAndSolveInflatables — @0x32a90 (631 bytes)
// ---------------------------------------------------------------------------
// Two dispatches, one per inflatable (grid = mNumInflatables), both inside
// their own timer scope; the AMD workaround synchronises once at the end.
//   shader 035 CalculateInflatableVolume:
//     t0 inflatables, t1 sortedNewPositions, t2 reverseLookup,
//     t3 dynamicTriangles; u0 pressures.
//   shader 036 SolveInflatableVolume:
//     t0 inflatables, t1 sortedNewPositions, t2 reverseLookup,
//     t3 dynamicTriangles, t4 pressures; u0 deltas.
void CalculateAndSolveInflatables(NvFlexSolver* solver, int substep, int numSubsteps,
                                  unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    NvFlexLibrary* lib = solver->mLib;
    const int numInflatables = solver->mNumInflatables;

    {
        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = lib->mShaderCalculateInflatableVolume;
        params.readWrite[0] = solver->mPressures.mBuffer->getResourceRW();
        params.readOnly[0] = solver->mInflatables.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] = solver->mDynamicTriangles.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = (unsigned)numInflatables;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    {
        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = lib->mShaderSolveInflatableVolume;
        params.readWrite[0] = solver->mDeltas.mBuffer->getResourceRW();
        params.readOnly[0] = solver->mInflatables.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] = solver->mDynamicTriangles.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[4] = solver->mPressures.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = (unsigned)numInflatables;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    if (lib->mEnableAmdDriverBugWorkaround)
        nvpipe::Synchronize(lib);
}

// ---------------------------------------------------------------------------
// SolveShapes — @0x32e10 (689 bytes)
// ---------------------------------------------------------------------------
// One dispatch over the rigids (grid = mNumRigids), with six shader variants
// selected by whether the plastic-deformation buffers exist and by the number
// of indices per rigid (32/128 variants need SHFL and FP32 atomics):
//   t0 rigidOffsets, t1 rigidIndices, t2 rigidCoefficients,
//   t3 rigidPlasticThresholds, t4 rigidPlasticCreeps, t5 reverseLookup,
//   t6 sortedPositions, t7 rigidLocalNormals;
//   u0 deltas, u1 rigidLocalPositions, u2 sortedNewPositions, u3 rigidRotations,
//   u4 rigidTranslations, u5 normals.
// t3/t4 are bound only when both plastic buffers exist (null otherwise).
void SolveShapes(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    NvFlexLibrary* lib = solver->mLib;
    const bool plastic = solver->mRigidPlasticThresholds.mBuffer != 0 &&
                         solver->mRigidPlasticCreeps.mBuffer != 0;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    if (lib->mIsSHFLSupported && lib->mIsFP32ATOMICSupported)
    {
        const int indicesPerRigid = solver->mNumRigidIndices / solver->mNumRigids;
        if (indicesPerRigid < 0x21)
            params.shader = plastic ? lib->mShaderSolveShapesPlasticDeformation32
                                    : lib->mShaderSolveShapes32;
        else if (0x7f < indicesPerRigid)
            params.shader = plastic ? lib->mShaderSolveShapesPlasticDeformation128
                                    : lib->mShaderSolveShapes128;
        else
            params.shader = plastic ? lib->mShaderSolveShapesPlasticDeformation
                                    : lib->mShaderSolveShapes;
    }
    else
    {
        params.shader = plastic ? lib->mShaderSolveShapesPlasticDeformation
                                : lib->mShaderSolveShapes;
    }

    params.readWrite[0] = solver->mDeltas.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mRigidLocalPositions.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readWrite[3] = solver->mRigidRotations.mBuffer->getResourceRW();
    params.readWrite[4] = solver->mRigidTranslations.mBuffer->getResourceRW();
    params.readWrite[5] = solver->mNormals.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mRigidOffsets.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mRigidIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mRigidCoefficients.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = plastic
                             ? solver->mRigidPlasticThresholds.mBuffer->getResource(NvFlex::eBuffer)
                             : 0;
    params.readOnly[4] = plastic
                             ? solver->mRigidPlasticCreeps.mBuffer->getResource(NvFlex::eBuffer)
                             : 0;
    params.readOnly[5] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mRigidLocalNormals.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = (unsigned)solver->mNumRigids;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// Finalize — @0x33b60 (409 bytes)
// ---------------------------------------------------------------------------
// Scatters the sorted state back to the unsorted buffers.  Shader 030:
//   t0 sortedNewPositions, t1 sortedNewVelocities, t2 sortedDensities,
//   t3 deltas, t4 the radix-sort payload;
//   u0 positions, u1 velocities, u2 densities.
void Finalize(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderFinalize;
    params.readWrite[0] = solver->mPositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mVelocities.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mDensities.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedDensities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mDeltas.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = sorted.val->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CalculateVorticity — @0x33340 (415 bytes)
// ---------------------------------------------------------------------------
// Only when vorticity confinement is enabled.  Shader 069:
//   t0 sortedNewPositions, t1 sortedNewVelocities, t2 sortedPhases,
//   t3 particle neighbors, t4 neighbor counts, t5 staticContactPlanes,
//   t6 staticContactCounts; u0 curl.
void CalculateVorticity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (!(0.0f < solver->mParams.vorticityConfinement))
        return;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderCalculateVorticity;
    params.readWrite[0] = solver->mCurl.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mStaticContactPlanes.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mStaticContactCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// SolveVelocities — @0x334e0 (497 bytes)
// ---------------------------------------------------------------------------
// Shader 077:
//   t0 particle neighbors, t1 neighbor counts, t2 staticContactPlanes,
//   t3 staticContactCounts, t4 sortedPositions, t5 sortedVelocities,
//   t6 sortedNewVelocities, t7 sortedPhases, t8 curl;
//   u0 sortedNewPositions, u1 deltas, u2 sortedDensities, u3 potentials.
void SolveVelocities(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderSolveVelocities;
    params.readWrite[0] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mDeltas.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mSortedDensities.mBuffer->getResourceRW();
    params.readWrite[3] = solver->mPotentials.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mStaticContactPlanes.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mStaticContactCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] = solver->mSortedVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[8] = solver->mCurl.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// SmoothPositions — @0x342e0 (414 bytes)
// ---------------------------------------------------------------------------
// Only when smoothing or anisotropy is requested.  Shader 020:
//   t0 particle neighbors, t1 neighbor counts, t2 sortedPhases,
//   t3 sortedNewPositions, t4 the radix-sort payload;
//   u0 sortedPositions, u1 the pre-smoothing copy.
void SmoothPositions(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (!(0.0f < solver->mParams.anisotropyScale) && !(0.0f < solver->mParams.smoothing))
        return;

    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderSmoothPositions;
    params.readWrite[0] = solver->mSortedPositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mSmoothPositionsOriginal.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = sorted.val->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CalculateAnisotropy — @0x34480 (425 bytes)
// ---------------------------------------------------------------------------
// Only when anisotropy is requested.  Shader 067:
//   t0 particle neighbors, t1 neighbor counts, t2 the radix-sort payload,
//   t3 sortedPhases, t4 sortedPositions;
//   u0 anisotropy1, u1 anisotropy2, u2 anisotropy3.
void CalculateAnisotropy(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (!(0.0f < solver->mParams.anisotropyScale))
        return;

    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderCalculateAnisotropy;
    params.readWrite[0] = solver->mAnisotropy1.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mAnisotropy2.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mAnisotropy3.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleNeighbors.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = sorted.val->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}
