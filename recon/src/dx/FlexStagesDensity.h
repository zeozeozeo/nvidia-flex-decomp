// flexdecomp: reconstructed D3D backend — solver stage group "density".
//
// Declarations for the stages implemented in flex_stages_density.cpp.  This
// header is owned by that unit; every symbol and signature here comes from the
// `// ---- <Name>  @ <rva>` block in
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c (which prints the integer
// arguments first, so `substep` is param 2, `numSubsteps` param 3).
// Included by FlexPipeline.h; there is nothing to include from here.
#pragma once

#include "FlexInternal.h"

void CalculateDensity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SolveDensity(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SolveSprings(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SpringsGenerateIndices(NvFlexLibrary* lib, NvFlex::Buffer* indices,
                            NvFlex::ConstantBuffer* cb, int numSprings);
// @0x32670 (242 bytes): (lib, b0, b1, b2, cb, numSprings) — five buffer/constant
// arguments before the count, exactly as the shipped body takes them.
void SpringsFindParticleRange(NvFlexLibrary* lib, NvFlex::Buffer* b0, NvFlex::Buffer* b1,
                              NvFlex::Buffer* b2, NvFlex::ConstantBuffer* cb,
                              int numSprings);
// @0x32770 (336 bytes): seven buffers plus the constant buffer, then the count.
void SpringsReorderData(NvFlexLibrary* lib, NvFlex::Buffer* b0, NvFlex::Buffer* b1,
                        NvFlex::Buffer* b2, NvFlex::Buffer* b3, NvFlex::Buffer* b4,
                        NvFlex::Buffer* b5, NvFlex::Buffer* b6,
                        NvFlex::ConstantBuffer* cb, int numSprings);
void CalculateAndSolveInflatables(NvFlexSolver* solver, int substep, int numSubsteps,
                                  unsigned numBlocks);
void SolveShapes(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void Finalize(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void CalculateAnisotropy(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
void SmoothPositions(NvFlexSolver* solver, int substep, int numSubsteps, unsigned numBlocks);
