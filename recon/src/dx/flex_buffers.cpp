// flexdecomp: reconstructed D3D backend — flex.obj, per-attribute accessors.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  Every entry point
// here is the same three-step body the shipped code emits:
//
//     nvtxRangePushA("<Name>");
//     {Read,Write}Data(solver->mLib, <solver buffer>, <user buffer>, desc);
//     nvtxRangePop();
//
// so the validation, the range checks and the error strings all live in
// flex_copy.cpp, and the table below is the whole difference between them.
#include "FlexObjects.h"

// ---------------------------------------------------------------------------
// Particles — @0x3ea40 (97 bytes) / @0x3eb90 (97 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetParticles(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                   const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mPositions.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetParticles(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                   const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mPositions.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Rest particles — @0x3eab0 (97 bytes) / @0x3eb20 (97 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetRestParticles(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                       const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mRestPositions.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetRestParticles(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                       const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mRestPositions.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Velocities
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetVelocities(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                    const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mVelocities.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetVelocities(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                    const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mVelocities.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Phases
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetPhases(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mPhases.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetPhases(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mPhases.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Normals
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetNormals(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                 const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mNormals.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetNormals(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                 const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mNormals.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Active indices — @0x3ed60 (97 bytes) / @0x3edd0 (97 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexSetActive(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::WriteData(solver->mLib, solver->mActiveIndices.mBuffer, buffer, desc);
}

extern "C" void NvFlexGetActive(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mActiveIndices.mBuffer, desc);
}

// @0x3ee40 (42 bytes) / @0x3ee70 (13 bytes)
extern "C" int NvFlexGetActiveCount(NvFlexSolver* solver)
{
    return solver->mNumParticles;
}

extern "C" void NvFlexSetActiveCount(NvFlexSolver* solver, int count)
{
    solver->mNumParticles = count;
    solver->mNumParticlesAligned = count;
}

// ---------------------------------------------------------------------------
// Derived attributes — @0x3ec00 (115 bytes) / @0x3ee80 (102 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexGetSmoothParticles(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                         const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mSmoothPositionsOriginal.mBuffer,
                              desc);
}

extern "C" void NvFlexGetDensities(NvFlexSolver* solver, NvFlexBuffer* buffer,
                                   const NvFlexCopyDesc* desc)
{
    NvFlex::flexobj::ReadData(solver->mLib, buffer, solver->mDensities.mBuffer, desc);
}

// ---------------------------------------------------------------------------
// Simple accessors — @0x3c2c0 (8 bytes)
// ---------------------------------------------------------------------------
extern "C" NvFlexLibrary* NvFlexGetSolverLibrary(NvFlexSolver* solver)
{
    return solver->mLib;
}

// @0x3fa80 (127 bytes): the shipped body is a field-by-field copy of the whole
// params struct, i.e. a plain assignment.
extern "C" void NvFlexGetParams(NvFlexSolver* solver, NvFlexParams* params)
{
    *params = solver->mParams;
}

// ---------------------------------------------------------------------------
// Synchronisation — @0x422d0 (55 bytes)
// ---------------------------------------------------------------------------
extern "C" void NvFlexWait(NvFlexLibrary* lib)
{
    lib->mContext->fenceSet(lib->mSyncQuery);
    lib->mContext->fenceWait(lib->mSyncQuery, 0);
}

// @0x43c30 (14 bytes)
extern "C" void NvFlexGetDataAftermath(NvFlexLibrary* lib, void* data, void* status)
{
    lib->mContext->getDataAftermath(data, (unsigned)(size_t)status);
}
