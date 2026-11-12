// flexdecomp: reconstructed D3D backend — solver stage group "collide".
//
// Declarations for the stages implemented in flex_stages_collide.cpp.  This
// header is owned by that unit; every symbol and signature here comes from the
// `// ---- <Name>  @ <rva>` block in
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c (which prints the integer
// arguments first, so `substep` is param 2, `numSubsteps` param 3).
// Included by FlexPipeline.h; there is nothing to include from here.
#pragma once

#include "FlexInternal.h"

void ComputeBounds(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CalculateParticleHash(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SortParticleHash(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void ReorderParticles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CollideParticles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CollideShapes(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
// @0x31da0 (829 bytes): six arguments in the shipped body.  The body reads the
// substep / substep-count pair from its 4th and 5th arguments (the 3rd is never
// read: `Update` passes the frame dt there) and the block count from the 6th.
void CollideTriangles(NvFlexSolver* solver, FlexTriangleMeshData* triMesh, int unused,
                      int substep, int numSubsteps, unsigned numBlocks);
void LazyClearGrid(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
