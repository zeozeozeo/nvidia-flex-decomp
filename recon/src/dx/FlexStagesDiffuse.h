// flexdecomp: reconstructed D3D backend — solver stage group "diffuse".
//
// Declarations for the stages implemented in flex_stages_diffuse.cpp.  This
// header is owned by that unit; every symbol and signature here comes from the
// `// ---- <Name>  @ <rva>` block in
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c (which prints the integer
// arguments first, so `substep` is param 2, `numSubsteps` param 3).
// Included by FlexPipeline.h; there is nothing to include from here.
#pragma once

#include "FlexInternal.h"

void CreateDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void UpdateDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void ClampDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CompactDiffuse(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void UpdateTriangles(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void UpdateNormals(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
