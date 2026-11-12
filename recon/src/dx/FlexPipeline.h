// flexdecomp: reconstructed D3D backend — solver pipeline contract.
//
// One declaration per `flex.obj` solver stage, plus the four helpers every
// stage uses.  The stage implementations are split across several translation
// units purely for size; they share this contract so the split cannot drift:
//
//   flex_pipeline.cpp        InitParams, Update, NvFlexUpdateSolver
//   flex_stages_collide.cpp  bounds / hash / grid / reorder / collide / clear
//   flex_stages_density.cpp  density / solve / springs / inflatables / shapes
//   flex_stages_diffuse.cpp  diffuse particles / triangles / normals
//
// Rules that hold for every stage (see recon/src/dx/flex_pipeline.cpp for the
// worked examples): build a zeroed `NvFlex::DispatchParams`, set `shader` from
// the library, bind `readOnly[k]` to `t<k>` and `readWrite[k]` to `u<k>` in the
// order the shader declares them (`work/dxbc/interfaces.txt`), set
// `gridDim` from the stage's group count, point `rootConstantBuffer` at
// `solver->mKernelParams.mBuffer`, and dispatch inside the timer scope the
// pseudocode shows.  The shipped code has no null-shader guard; do not add one.
#pragma once

#include "FlexObjects.h"

namespace nvpipe
{

// Zero the whole DispatchParams (the shipped code stores to every slot it uses
// and clears the rest before the call).
void ClearDispatch(NvFlex::DispatchParams& params);

// The device-layer dispatch (ContextEx::dispatch).
void Dispatch(NvFlexSolver* solver, const NvFlex::DispatchParams& params);

// @0x30de0 (98 bytes): flush + signal/wait the library sync fence.  Only
// reached when `mEnableAmdDriverBugWorkaround` is set.
void Synchronize(NvFlexLibrary* lib);

// One 512-byte upload of `solver->mKernelParamsHost` to the constant buffer.
void UploadKernelParams(NvFlexSolver* solver);

// Blocks whose thread-group count depends on the substep index clamp the value
// to the solver's particle capacity the way the shipped code does:
// `min(numBlocks, (mMaxParticles + 0xff) >> 8)`.
inline unsigned ClampBlocks(NvFlexSolver* solver, unsigned numBlocks)
{
    return numBlocks;
}

} // namespace nvpipe

// ---------------------------------------------------------------------------
// Constant-boundary stages + the substep body (flex_pipeline.cpp)
// ---------------------------------------------------------------------------
void InitParams(NvFlexSolver* solver, float dt, int numSubsteps);

// @0x35590 (1120 bytes): one substep.
void Update(NvFlexSolver* solver, float dt, int substep, int numSubsteps);

// Implemented in flex_pipeline.cpp as the first worked examples.
void Predict(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CreateGrid(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void ContinuousShockPropagation(NvFlexSolver* solver, int substep, int numSubsteps,
                                unsigned numBlocks);
void ApplyDeltas(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SolveContacts(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void UpdateVelocities(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void NormalizeNormals(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
// Implemented in flex_stages_density.cpp (declared here because Update, in this
// file, is their only caller).
void CalculateVorticity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SolveVelocities(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
// Implemented in flex_solver.cpp.
void NvFlexUpdateSolver(NvFlexSolver* solver, float dt, int numSubsteps, bool enableTimers);

// Stage groups, one header/file pair each so work stays disjoint.
#include "FlexStagesCollide.h"
#include "FlexStagesDensity.h"
#include "FlexStagesDiffuse.h"
