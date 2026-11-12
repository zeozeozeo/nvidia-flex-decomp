// flexdecomp: reconstructed D3D backend — flex.obj stage group "diffuse".
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   UpdateTriangles   @0x336e0 (354 bytes)
//   UpdateNormals     @0x33850 (458 bytes)
//   CreateDiffuse     @0x33d00 (406 bytes)
//   UpdateDiffuse     @0x33ea0 (450 bytes)
//   ClampDiffuse      @0x34070 (247 bytes)
//   CompactDiffuse    @0x34170 (354 bytes)
//
// Every stage builds one zeroed DispatchParams, binds the UAVs first (the
// pseudocode stores the `getResourceRW()` results at the `readWrite[]` slots)
// and then the SRVs in declared-register order, points the root constant buffer
// at the solver's KernelParams mirror, and dispatches once inside a timer
// scope.  The shipped code adds no null-shader guard; neither do these.
//
// The individual resource calls in the pseudocode's stack layout:
//   readWrite[k]  <- local_(50-8k)   virtual +0x18, no argument (getResourceRW)
//   readOnly[k]   <- local_(150-8k)  virtual +0x10, argument 1   (getResource)
#include <cmath>

#include "FlexPipeline.h"

// ---------------------------------------------------------------------------
// UpdateTriangles — @0x336e0 (354 bytes), one dispatch
// ---------------------------------------------------------------------------
// Shader 025 (UpdateTriangles): t0 sortedNewPositions, t1 sortedNewVelocities,
// t2 reverseLookup, t3 dynamicTriangles; u0 deltas (raw), u1
// dynamicTriangleNormals (stride 12).  Grid is the dynamic-triangle count
// rounded up to 256-wide groups.
void UpdateTriangles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    const unsigned blocks =
        (static_cast<unsigned>(solver->mDynamicNumTris) + 0xffu) >> 8;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderUpdateTriangles;
    params.readWrite[0] = solver->mDeltas.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mDynamicTriangleNormals.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mDynamicTriangles.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = blocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// UpdateNormals — @0x33850 (458 bytes), two dispatches
// ---------------------------------------------------------------------------
// Bindings (identical for both dispatches): u0 normalsTemp;
// t0 reverseLookup (undelivered — the shader never reads slot 0),
// t1 dynamicTriangles (stride 4), t2 dynamicTriangleNormals (stride 12).
// First dispatch: shader 041 (UpdateVertexNormals, the clear pass — its only
// declaration is u0, so the three SRVs are bound but unused), grid = particle
// capacity / 256.  Second dispatch: shader 022/068 (UpdateVertexNormals), grid
// = dynamic triangle count / 256.  The shipped code reuses one params struct
// for both, so only the shader and gridDim.x change.
void UpdateNormals(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    const unsigned particleBlocks =
        (static_cast<unsigned>(solver->mMaxParticles) + 0xffu) >> 8;
    const unsigned triBlocks =
        (static_cast<unsigned>(solver->mDynamicNumTris) + 0xffu) >> 8;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.readWrite[0] = solver->mNormalsTemp.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mReverseLookup.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mDynamicTriangles.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] =
        solver->mDynamicTriangleNormals.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    params.shader = solver->mLib->mShaderUpdateVertexNormalsInit;
    params.gridDim.x = particleBlocks;
    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    params.shader = solver->mLib->mShaderUpdateVertexNormals;
    params.gridDim.x = triBlocks;
    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CreateDiffuse — @0x33d00 (406 bytes), one dispatch, rate-gated
// ---------------------------------------------------------------------------
// Only runs when the diffuse timer has caught up with the emission rate; the
// timer is then rewound modulo the rate (fmodf of the shipped code).
// Shader 023 (CreateDiffuseParticles): t0 potentials, t1 sortedNewPositions,
// t2 sortedNewVelocities, t3 random; u0 numDiffuseParticles, u1
// diffusePositions, u2 diffuseVelocities.  Grid is the substep block count.
void CreateDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (solver->mDiffuseRate <= solver->mDiffuseTimer)
    {
        solver->mDiffuseTimer = fmodf(solver->mDiffuseTimer, solver->mDiffuseRate);

        NvFlex::DispatchParams params;
        nvpipe::ClearDispatch(params);

        params.shader = solver->mLib->mShaderCreateDiffuseParticles;
        params.readWrite[0] = solver->mNumDiffuseParticles.mBuffer->getResourceRW();
        params.readWrite[1] = solver->mDiffusePositions.mBuffer->getResourceRW();
        params.readWrite[2] = solver->mDiffuseVelocities.mBuffer->getResourceRW();
        params.readOnly[0] = solver->mPotentials.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[1] =
            solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[2] =
            solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
        params.readOnly[3] = solver->mRandom.mBuffer->getResource(NvFlex::eBuffer);
        params.gridDim.x = numBlocks;
        params.gridDim.y = 1;
        params.gridDim.z = 1;
        params.rootConstantBuffer = solver->mKernelParams.mBuffer;

        {
            DxTimer timer(solver->mTimers, params.shader->desc().label);
            nvpipe::Dispatch(solver, params);
        }
    }
}

// ---------------------------------------------------------------------------
// UpdateDiffuse — @0x33ea0 (450 bytes), one dispatch
// ---------------------------------------------------------------------------
// Shader 061 (UpdateDiffuseParticles): t0 numDiffuseParticles, t1
// cellBucketStarts, t2 cellBucketEnds, t3 particleNeighborCounts (the shader
// reads it as its contact-count buffer), t4 sortedNewPositions, t5
// sortedNewVelocities, t7 particleBounds; u0 diffusePositions, u1
// diffuseVelocities.  The pseudocode makes eight SRV calls for a seven-slot
// interface: the call for sortedPhases lands in slot 6, the register the
// shader does not declare, and slot 7 is the bounds buffer — exactly the hole
// this stage's interface documents.  Grid is the diffuse capacity / 256.
void UpdateDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    const unsigned blocks =
        (static_cast<unsigned>(solver->mMaxDiffuseParticles) + 0xffu) >> 8;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderUpdateDiffuseParticles;
    params.readWrite[0] = solver->mDiffusePositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mDiffuseVelocities.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mNumDiffuseParticles.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mLib->mCellBucketStarts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mLib->mCellBucketEnds.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] =
        solver->mParticleNeighborCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[4] =
        solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[5] =
        solver->mSortedNewVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[6] = solver->mSortedPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[7] = solver->mParticleBounds.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = blocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// ClampDiffuse — @0x34070 (247 bytes), one dispatch
// ---------------------------------------------------------------------------
// Shader 017 (ClampDiffuseParticleCount) binds no SRVs at all; u0
// numDiffuseParticles, u1 numDiffuseParticlesNew.  The shipped grid is the
// literal (1,1,1).
void ClampDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderClampDiffuseParticleCount;
    params.readWrite[0] = solver->mNumDiffuseParticles.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mNumDiffuseParticlesNew.mBuffer->getResourceRW();
    params.gridDim.x = 1;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}

// ---------------------------------------------------------------------------
// CompactDiffuse — @0x34170 (354 bytes), one dispatch
// ---------------------------------------------------------------------------
// Shader 015 (CompactDiffuseParticles): t0 numDiffuseParticles, t1
// diffusePositions, t2 diffuseVelocities; u0 numDiffuseParticlesNew, u1
// diffusePositionsNew, u2 diffuseVelocitiesNew.  Grid is the diffuse
// capacity / 256.
void CompactDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;
    (void)numBlocks;

    const unsigned blocks =
        (static_cast<unsigned>(solver->mMaxDiffuseParticles) + 0xffu) >> 8;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderCompactDiffuseParticles;
    params.readWrite[0] = solver->mNumDiffuseParticlesNew.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mDiffusePositionsNew.mBuffer->getResourceRW();
    params.readWrite[2] = solver->mDiffuseVelocitiesNew.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mNumDiffuseParticles.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mDiffusePositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mDiffuseVelocities.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = blocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }
}
