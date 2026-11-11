// Shared declaration of the SolveShapes kernel template.
//
// The definition is reconstructed per instantiation in separate translation
// units (flex.part6.b.cu -> <true>, flex.part6.c.cu -> <false>) so that the two
// can be worked on independently; the host launch site (flex.part2.cu) only
// needs this declaration plus the explicit instantiations produced there.
//
// Signature evidence: the shipped mangled names
//   _Z11SolveShapesILb0EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if
//   _Z11SolveShapesILb1EEvPKiS1_PKfS3_S3_S1_P8XVector3IfEP8XVector4IfE8Vec3IterSA_P5XQuatIfES6_S9_SA_if
#pragma once

#include "nvflex/all.h"

// The shipped objects export both instantiations (weak, default visibility);
// without the attribute nvcc gives the host launch stub internal linkage and
// the symbol would be missing from libnvflex.so.  It must be on the first
// declaration: nvcc rejects adding it in the definition TU.
template <bool IsRigid>
__global__ void __attribute__((visibility("default")))
SolveShapes(const int* shapeOffsets, const int* shapeIndices,
                            const float* shapePositions, const float* shapeRotations,
                            const float* shapePrevPositions, const int* shapeStarts,
                            XVector3<float>* p, XVector4<float>* q,
                            Vec3Iter u, Vec3Iter u2, XQuat<float>* quats,
                            XVector3<float>* v, XVector4<float>* w,
                            Vec3Iter u3, int numShapes, float dt);
