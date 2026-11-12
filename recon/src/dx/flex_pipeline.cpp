// flexdecomp: reconstructed D3D backend — flex.obj solver pipeline.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c, cross-checked
// against the already gate-verified CUDA reconstruction in
// recon/src/cuda/flex.part2.cu (same maths, same stage order).
//
// This unit holds the pipeline core and the seven stages that were needed to
// pin the pattern down; the remaining stages live in three sibling files that
// share FlexPipeline.h:
//
//   flex_stages_collide.cpp   bounds / hash / grid / reorder / collide / clear
//   flex_stages_density.cpp   density / springs / inflatables / shapes / solve
//   flex_stages_diffuse.cpp   diffuse particles / triangles / normals
//
//   InitParams    @ 0x34f40 (1610 bytes)
//   ExecuteCallback @ 0x30bb0 (239)
//   Update        @ 0x35590 (1120)
#include <cfloat>
#include <cstring>
#include <utility>

#include "FlexObjects.h"
#include "FlexPipeline.h"
#include "generated/shaders_flex.h"

namespace
{

// D3D-specific constants the shipped InitParams loads from .rdata.
const float kPi = 3.1415927410125732f;          // 0x40490fdb
const float k64Pi = 201.0619354248047f;         // 0x43490fdb
const float k60 = 60.0f;                        // 0x42700000

} // namespace

// ---------------------------------------------------------------------------
// InitParams — @0x34f40 (1610 bytes)
// ---------------------------------------------------------------------------
// Fills the 512-byte KernelParams mirror, uploads it to the solver's constant
// buffer and keeps the host copy in mKernelParamsHost (the stage dispatchers
// only touch individual fields afterwards).
//
// Offsets below are the KernelParams field offsets asserted in FlexInternal.h;
// each RHS is the pseudocode expression at the corresponding stack slot
// (local_238 is kPlanes[0], i.e. offset 0).
void InitParams(NvFlexSolver* solver, float dt, int numSubsteps)
{
    NvFlexParams& params = solver->mParams;
    KernelParams& p = solver->mKernelParamsHost;

    std::memcpy(p.kPlanes, params.planes, sizeof(p.kPlanes));

    const float radius = params.radius;
    const float invRadius = 1.0f / radius;
    const float invRestDensity = 1.0f / solver->mFluidRestDensity;

    // Kernel normalisation: poly6 = 315/(64*pi*r^9), spiky1 = 15/(pi*r^3),
    // spiky2 = 30/(pi*r^4).
    const float r3 = radius * radius * radius;
    const float spikyDenominator = kPi * r3;

    // Cohesion spline coefficients in terms of h = fluidRestDistance/radius.
    const float h = params.fluidRestDistance / radius;
    const float h2 = h * h;

    p.kNumPlanes = params.numPlanes;
    p.kRadius = radius;
    p.kRadiusSq = radius * radius;
    p.kInvRadius = invRadius;
    p.kPoly6 = 315.0f / (k64Pi * r3 * r3 * r3);
    p.kSpiky1 = 15.0f / spikyDenominator;
    p.kSpiky2 = 30.0f / (spikyDenominator * radius);

    p.kRestDensity = solver->mFluidRestDensity;
    p.kInvRestDensity = invRestDensity;
    p.kFluidRestDistance = params.fluidRestDistance;
    p.kSolidRestDistance = params.solidRestDistance;
    p.kLambdaScale = 1.0f / solver->mFluidDensityConstraintScale;
    p.kCohesion = radius * params.cohesion;
    p.kSurfaceTension =
        (invRestDensity * params.surfaceTension) / solver->mFluidSurfaceConstraintScale;
    p.kSolidPressure = params.solidPressure;
    p.kVorticityConfinement = params.vorticityConfinement;
    p.kAnisotropy = params.anisotropyScale;
    p.kAnisotropyMin = radius * params.anisotropyMin;
    p.kAnisotropyMax = radius * params.anisotropyMax;
    p.kSmoothing = params.smoothing;
    p.kViscosity = invRestDensity * params.viscosity;
    p.kFreeSurfaceDrag = params.freeSurfaceDrag;
    p.kBuoyancy = params.buoyancy;

    // The cohesion spline: -(h+1)/h^2 and (h^2+h+1)/h^2.  The shipped body writes
    // the pair as one sign-flipped float4, which is why it renders as a vector
    // xor in the pseudocode.
    p.kCohesion1 = -(h + 1.0f) / h2;
    p.kCohesion2 = (h2 + h + 1.0f) / h2;

    p.kDiffuseThreshold = solver->mMaxDiffuseParticles != 0 ? params.diffuseThreshold : 0.0f;
    p.kDiffuseBuoyancy = params.diffuseBuoyancy;
    p.kDiffuseDrag = params.diffuseDrag;
    p.kDiffuseBallistic = params.diffuseBallistic;

    p.kGravity.x = params.gravity[0];
    p.kGravity.y = params.gravity[1];
    p.kGravity.z = params.gravity[2];

    p.kDynamicFriction = params.dynamicFriction;
    p.kStaticFriction = params.staticFriction;
    p.kParticleFriction = params.particleFriction;
    p.kAdhesion = params.adhesion;
    p.kDissipation = params.dissipation;
    p.kDamping = params.damping;

    p.kCollisionDistance = params.collisionDistance;
    p.kCollisionThreshold = params.shapeCollisionMargin;
    p.kCollisionMargin = params.collisionDistance + params.shapeCollisionMargin;
    p.kCollisionMarginSq = p.kCollisionMargin * p.kCollisionMargin;

    p.kDrag = params.drag * k60;
    p.kLift = params.lift * k60;

    p.kSleepThreshold = params.sleepThreshold;
    p.kSleepThresholdSq = p.kSleepThreshold * p.kSleepThreshold;
    p.kShockPropagation = params.shockPropagation;
    p.kRestitution = params.restitution;
    p.kSOR = 1.0f / (1.0f + params.relaxationFactor);

    p.kNumParticles = solver->mNumParticles;
    p.kNumParticlesAligned = solver->mNumParticlesAligned;
    p.kDt = dt;
    p.kInvDt = 1.0f / dt;
    p.kSubStepLength = 1.0f / (float)numSubsteps;
    p.kMaxSpeed = params.maxSpeed;

    // Collision cell size: the particle radius including its collision margin.
    const float collisionRadius = radius + params.particleCollisionMargin;
    p.kInvCellEdge = 1.0f / collisionRadius;
    p.kCollideParticlesRadiusSq = collisionRadius * collisionRadius;

    p.kMaxParticles = solver->mMaxParticles;
    p.kRelaxationMode = (int)params.relaxationMode;
    p.kShapesCount = (int)solver->mSDFDeviceShapes.size();
    p.kNumShapes = solver->mShapes.mNumShapes;
    p.kNumRigids = solver->mNumRigids;
    p.kNumSprings = solver->mNumSprings;
    p.kNumTriangles = solver->mDynamicNumTris;
    p.kMaxDiffuseParticles = solver->mMaxDiffuseParticles;
    p.kDiffuseDt = (float)numSubsteps * dt;
    p.kDiffuseMaxVelocity = FLT_MAX;
    p.kRelaxationFactor = params.relaxationFactor;
    p.kDiffuseLifetime = params.diffuseLifetime;

    p.kWind.x = params.wind[0];
    p.kWind.y = params.wind[1];
    p.kWind.z = params.wind[2];

    p.kMaxNeighborsPerParticle = solver->mMaxNeighborsPerParticle;
    p.kMaxAcceleration = params.maxAcceleration;
    p.kMaxVelocityDelta = params.maxAcceleration * dt;
    p.kMaxContactsPerParticle = solver->mMaxContactsPerParticle;

    // Upload: one 512-byte write through the mapped constant buffer.
    nvpipe::UploadKernelParams(solver);
}

// ---------------------------------------------------------------------------
// Stage plumbing
// ---------------------------------------------------------------------------
// Every stage is the same five steps: build a zeroed DispatchParams, bind the
// shader and its views, set the grid, point b0 at the solver's KernelParams and
// dispatch inside a timer scope.  The view order below is the shader's register
// order (`dcl_resource` t<n> / `dcl_uav` u<n> in
// work/dxbc/interfaces.txt), which the pseudocode emits in the same order.

// ---------------------------------------------------------------------------
// Predict — @0x30e50 (338 bytes)
// ---------------------------------------------------------------------------
// Shader 073: t0 activeIndices, t1 positions, t2 phases; u0 newPositions,
// u1 velocities.
void Predict(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlexLibrary* lib = solver->mLib;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderPredict;
    params.readWrite[0] = solver->mNewPositions.mBuffer->getResourceRW();
    params.readWrite[1] = solver->mVelocities.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mActiveIndices.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mPhases.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    {
        DxTimer timer(solver->mTimers, params.shader->desc().label);
        nvpipe::Dispatch(solver, params);
    }

    if (lib->mEnableAmdDriverBugWorkaround)
        nvpipe::Synchronize(lib);
}

// ---------------------------------------------------------------------------
// CreateGrid — @0x31570 (291 bytes)
// ---------------------------------------------------------------------------
// Shader 063: t0 the radix-sorted cell ids; u0 cell bucket starts, u1 ends.
void CreateGrid(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlexLibrary* lib = solver->mLib;

    // The sort result is the key buffer of the radix sort's front side.
    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = lib->mShaderCreateGrid;
    params.readWrite[0] = lib->mCellBucketStarts.mBuffer->getResourceRW();
    params.readWrite[1] = lib->mCellBucketEnds.mBuffer->getResourceRW();
    params.readOnly[0] = sorted.key->getResource(NvFlex::eBuffer);
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
// ContinuousShockPropagation — @0x320e0 (211 bytes)
// ---------------------------------------------------------------------------
// Shader 038: t0 particleBounds, u0 sortedNewPositions.  Skipped entirely when
// shockPropagation is zero, which is the default.
void ContinuousShockPropagation(NvFlexSolver* solver, int substep, int numSubsteps,
                                unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (!(0.0f < solver->mParams.shockPropagation))
        return;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderContinuousShockPropagation;
    params.readWrite[0] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mParticleBounds.mBuffer->getResource(NvFlex::eBuffer);
    params.gridDim.x = numBlocks;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = solver->mKernelParams.mBuffer;

    nvpipe::Dispatch(solver, params);
}

// ---------------------------------------------------------------------------
// ApplyDeltas — @0x32d10 (254 bytes)
// ---------------------------------------------------------------------------
// Shader 066: t0 deltas, u0 sortedNewPositions.
void ApplyDeltas(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderApplyDeltas;
    params.readWrite[0] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mDeltas.mBuffer->getResource(NvFlex::eBuffer);
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
// SolveContacts — @0x330d0 (326 bytes)
// ---------------------------------------------------------------------------
// Shader 037: t0 contact counts, t1 planes, t2 sortedPositions, t3 contact
// velocities; u0 sortedNewPositions.
void SolveContacts(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderSolveContactsSequential;
    params.readWrite[0] = solver->mSortedNewPositions.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mStaticContactCounts.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mStaticContactPlanes.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[2] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[3] = solver->mStaticContactVelocities.mBuffer->getResource(NvFlex::eBuffer);
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
// UpdateVelocities — @0x33220 (277 bytes)
// ---------------------------------------------------------------------------
// Shader 074: t0 sortedPositions, t1 sortedNewPositions; u0 sortedNewVelocities.
void UpdateVelocities(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderUpdateVelocities;
    params.readWrite[0] = solver->mSortedNewVelocities.mBuffer->getResourceRW();
    params.readOnly[0] = solver->mSortedPositions.mBuffer->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mSortedNewPositions.mBuffer->getResource(NvFlex::eBuffer);
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
// NormalizeNormals — @0x33a20 (308 bytes)
// ---------------------------------------------------------------------------
// Last substep only.  Shader: t1 normalsTemp (t0 is the radix key, which the
// shader does not declare but the shipped code still binds), u0 normals.
void NormalizeNormals(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks)
{
    (void)substep;
    (void)numSubsteps;

    if (substep != numSubsteps - 1)
        return;

    NvFlex::RadixSortBuffer sorted = solver->mRadixSort->getOutputBuffer();

    NvFlex::DispatchParams params;
    nvpipe::ClearDispatch(params);

    params.shader = solver->mLib->mShaderNormalizeVertexNormals;
    params.readWrite[0] = solver->mNormals.mBuffer->getResourceRW();
    params.readOnly[0] = sorted.key->getResource(NvFlex::eBuffer);
    params.readOnly[1] = solver->mNormalsTemp.mBuffer->getResource(NvFlex::eBuffer);
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
// ExecuteCallback — @0x30bb0 (239 bytes)
// ---------------------------------------------------------------------------
// `mCallbacks` lives at @3656; the shipped code indexes it by stage and hands
// the registered function the sorted/front-bank buffer *objects* (not device
// pointers) together with the reverse-lookup map and the radix sort payload.
namespace
{

void ExecuteCallback(NvFlexSolver* solver, NvFlexSolverCallbackStage stage, float dt)
{
    NvFlexSolverCallback& cb = solver->mCallbacks[(int)stage];
    if (cb.function == 0)
        return;

    NvFlex::RadixSortBuffer sortBuffers = solver->mRadixSort->getOutputBuffer();

    NvFlexSolverCallbackParams params;
    params.solver = solver;
    params.userData = cb.userData;
    params.particles = reinterpret_cast<float*>(solver->mSortedNewPositions.mBuffer);
    params.velocities = reinterpret_cast<float*>(solver->mSortedNewVelocities.mBuffer);
    params.phases = reinterpret_cast<int*>(solver->mSortedPhases.mBuffer);
    params.numActive = solver->mNumParticles;
    params.dt = dt;
    params.originalToSortedMap = reinterpret_cast<const int*>(solver->mReverseLookup.mBuffer);
    params.sortedToOriginalMap = reinterpret_cast<const int*>(sortBuffers.val);

    cb.function(params);
}

} // namespace

// ---------------------------------------------------------------------------
// Update — @0x35590 (1120 bytes)
// ---------------------------------------------------------------------------
// One substep, in the shipped order.  The block counts are derived from the
// particle count: 256-particle groups for the broad/collide stages, 128-particle
// groups for the bounds/hash stages.  The integer the shipped code passes as
// the stages' `substep` argument is a constant (0 / 1 / 3) wherever the stage
// does not read it; those literals are reproduced as they are.
void Update(NvFlexSolver* solver, float dt, int substep, int numSubsteps)
{
    const unsigned blocks256 = (unsigned)((solver->mNumParticles + 0xff) >> 8);
    const unsigned blocks128 = (unsigned)((solver->mNumParticles + 0x7f) >> 7);

    Predict(solver, 0, substep, blocks256);
    ComputeBounds(solver, 0, substep, blocks128);
    CalculateParticleHash(solver, 0, substep, blocks128);
    if (solver->mNumParticles != 0)
        SortParticleHash(solver, 0, substep, blocks128);
    CreateGrid(solver, 0, substep, blocks256);
    ReorderParticles(solver, 0, substep, blocks256);
    CollideParticles(solver, 0, substep, blocks256);

    ExecuteCallback(solver, eNvFlexStageSubstepBegin, dt);

    {
        // The collision pass reads the contact counts the previous substep left
        // behind; clear them first (the shipped code clears through the same
        // two-element zero vector it keeps on the stack).
        static const unsigned kZeroValues[2] = {0, 0};
        solver->mLib->mContext->clearUnorderedAccessViewUint(
            solver->mStaticContactCounts.mBuffer, kZeroValues);
    }
    CollideTriangles(solver, &solver->mLib->mTriangleMeshData, dt, substep, numSubsteps,
                     blocks256);
    CollideShapes(solver, substep, numSubsteps, blocks128);
    ContinuousShockPropagation(solver, substep, numSubsteps, blocks256);

    int stageArg = substep;
    for (int i = 0; i < solver->mParams.numIterations; ++i)
    {
        stageArg = 0;
        ExecuteCallback(solver, eNvFlexStageIterationStart, dt);
        CalculateDensity(solver, 0, numSubsteps, blocks256);
        SolveDensity(solver, 0, numSubsteps, blocks256);
        SolveSprings(solver, 0, numSubsteps, blocks256);
        if (solver->mNumInflatables != 0)
            CalculateAndSolveInflatables(solver, 0, numSubsteps, blocks256);
        if (solver->mNumRigids != 0)
            SolveShapes(solver, 0, numSubsteps, blocks256);
        ApplyDeltas(solver, 0, numSubsteps, blocks256);
        SolveContacts(solver, 0, numSubsteps, blocks256);
        stageArg = 1;
        ExecuteCallback(solver, eNvFlexStageIterationEnd, dt);
    }

    UpdateVelocities(solver, stageArg, numSubsteps, blocks256);
    const int lastSubstep = numSubsteps - 1;

    ExecuteCallback(solver, eNvFlexStageSubstepEnd, dt);
    if (substep == lastSubstep)
        ExecuteCallback(solver, eNvFlexStageUpdateEnd, (float)numSubsteps * dt);

    CalculateVorticity(solver, 3, numSubsteps, blocks256);
    SolveVelocities(solver, 3, numSubsteps, blocks256);

    if (solver->mDynamicNumTris != 0)
    {
        UpdateTriangles(solver, 3, numSubsteps, blocks256);
        if (substep == lastSubstep)
            UpdateNormals(solver, 3, numSubsteps, blocks256);
        NormalizeNormals(solver, substep, numSubsteps, blocks256);
    }

    Finalize(solver, 3, numSubsteps, blocks256);

    solver->mDiffuseTimer += dt;

    if (substep == lastSubstep)
    {
        if (solver->mMaxDiffuseParticles != 0)
        {
            CreateDiffuse(solver, 3, numSubsteps, blocks256);
            UpdateDiffuse(solver, 3, numSubsteps, blocks256);
            ClampDiffuse(solver, 3, numSubsteps, blocks256);
            CompactDiffuse(solver, 3, numSubsteps, blocks256);

            // Ping-pong the diffuse banks: positions, velocities and counts.
            std::swap(solver->mDiffusePositions, solver->mDiffusePositionsNew);
            std::swap(solver->mDiffuseVelocities, solver->mDiffuseVelocitiesNew);
            std::swap(solver->mNumDiffuseParticles, solver->mNumDiffuseParticlesNew);
        }
        SmoothPositions(solver, 3, numSubsteps, blocks256);
        CalculateAnisotropy(solver, 3, numSubsteps, blocks256);
    }

    LazyClearGrid(solver, 3, numSubsteps, blocks256);

    if (solver->mLib->mGpuVendorId == VENDOR_ID_NVIDIA)
        solver->mLib->mContext->computeWaitForGraphics();

    solver->mStepCounter++;
}
