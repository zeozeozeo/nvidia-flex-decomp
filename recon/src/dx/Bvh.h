// flexdecomp: reconstructed D3D backend — bvh.obj.
//
// bvh.obj contains the D3D compute-shader BVH builder used by the solver: the
// BVH aggregate (three device buffers plus counts), the LinearBVHBuilderGPU
// class with its embedded shader set, ResizeBVH/FreeBVH, the D3D buffer
// wrappers (declared in utils.h because the copy helpers in utils.cpp use
// them) and a large number of std:: template instantiations that the standard
// library provides.
//
// There is no MedianBVHBuilder/LinearBVHBuilderCPU in the D3D backend: the
// object inventory has no such symbols (the host-side median builder only
// exists in the CUDA library).
//
// Class layouts follow the PDB (`bin/flexkit d3dcontext --cls BVH`):
//   BVH                  three DxBufferExt @0/@32/@64, ints @96/@100/@104
//   LinearBVHBuilderGPU  buffers, shaders, radix sort, context (PDB order)
#pragma once

#include "RadixSort.h"   // NvFlex::RadixSort, RadixSortDesc/Params/Buffer, DxTimers
#include "utils.h"

// ---------------------------------------------------------------------------
// Node format
// ---------------------------------------------------------------------------
// 16 bytes: three floats and a 32-bit word with the child index (31 bits) and
// the leaf bit — the same layout as the CUDA backend's PackedNodeHalf.
struct PackedNodeHalf
{
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;
};

// ---------------------------------------------------------------------------
// BVH — three device buffers (PDB: DxBufferExt @0/@32/@64, ints @96/@100/@104)
// ---------------------------------------------------------------------------
struct BVH
{
    DxBufferExt mNodeLowers;   // @0   PackedNodeHalf[capacity]
    DxBufferExt mNodeUppers;   // @32  PackedNodeHalf[capacity]
    DxBufferExt mRootNode;     // @64  int[1]
    int mNumNodes;             // @96
    int mMaxNodes;             // @100
    int mMaxDepth;             // @104

    BVH() : mNumNodes(0), mMaxNodes(0), mMaxDepth(0) {}
};

// bvh.obj @0x2af90 (541 bytes): grow the node buffers to 1.5x the requested
// node count (keeping an existing buffer that is already large enough).
void ResizeBVH(BVH* bvh, int numNodes, NvFlex::Context* context);

// bvh.obj @0x2b1b0 (208 bytes).
void FreeBVH(BVH* bvh);

// ---------------------------------------------------------------------------
// LinearBVHBuilderGPU
// ---------------------------------------------------------------------------
// Build(BVH*, positions, indices, unused, numItems, bounds, timers):
//   - grows the BVH to 2*numItems nodes,
//   - writes the constant block {numItems},
//   - when `bounds` is null, runs ComputeTotalBounds{,.Group,,Finalize} to
//     derive a tight quantization bound, otherwise uploads 1/(extent+1e-6) and
//     the supplied lower corner,
//   - sorts the Morton keys with the backend radix sort,
//   - runs CalculateMortonCodes, CalculateKeyDeltas, BuildLeaves, clears the
//     child counters and runs BuildHierarchy.
class LinearBVHBuilderGPU
{
public:
    // bvh.obj @0x2b280 (1645 bytes).
    LinearBVHBuilderGPU(NvFlex::ContextEx* context, bool shflSupported, int smCount, bool amd);
    // bvh.obj @0x2b8f0 (992 bytes).
    ~LinearBVHBuilderGPU();

    // bvh.obj @0x2bcd0 (4592 bytes).
    void Build(BVH* bvh, DxBufferExt* positions, DxBufferExt* indices, DxBufferExt* unused,
               int numItems, const Bounds* bounds, DxTimers* timers);

private:
    DxBufferExt mDeltas;                        // @0
    DxBufferExt mRangeLefts;                    // @32
    DxBufferExt mRangeRights;                   // @64
    DxBufferExt mNumChildren;                   // @96
    DxComputeBuffer mTotalLower;                // @128
    DxBufferExt mTotalUpper;                    // @160
    DxComputeBuffer mTotalInvEdges;             // @192
    int mMaxItems;                              // @224
    DxConstantBuffer mConstantBuffer;           // @232
    NvFlex::ComputeShader* mShaderCalculateMortonCodes;       // @256
    NvFlex::ComputeShader* mShaderCalculateKeyDeltas;         // @264
    NvFlex::ComputeShader* mShaderBuildLeaves;                // @272
    NvFlex::ComputeShader* mShaderBuildHierarchy;             // @280
    NvFlex::ComputeShader* mShaderComputeTotalBounds;         // @288
    NvFlex::ComputeShader* mShaderComputeTotalBoundsGroup;    // @296
    NvFlex::ComputeShader* mShaderComputeTotalBoundsFinalize; // @304
    NvFlex::ComputeShader* mShaderComputeTotalInvEdges;       // @312
    DxBufferExt mBoundsLower;                   // @320
    DxBufferExt mBoundsUpper;                   // @352
    DxBufferExt mGroupBoundsLower;              // @384
    DxBufferExt mGroupBoundsUpper;              // @416
    NvFlex::RadixSort* mBVHRadixSort;           // @448
    int mBVHRadixSortSize;                      // @456
    bool mIsSHFLSupported;                      // @460
    int mSMCount;                               // @464
    int mMaxNumTriMeshes;                       // @468
    NvFlex::ContextEx* mContext;                // @472
};

// ---------------------------------------------------------------------------
// Debug hooks (bvh.obj emits empty bodies @0x2af50..0x2af80)
// ---------------------------------------------------------------------------
void PrintBvhRadixSortInput(NvFlex::Context* context, NvFlex::Buffer* keys,
                            NvFlex::Buffer* values, int numItems);
void PrintBvhRadixSortOutput(NvFlex::Context* context, NvFlex::Buffer* keys,
                             NvFlex::Buffer* values, int numItems);
void PrintCalculateMortonCodesInput(NvFlex::Context* context, DxBufferExt* positions,
                                    DxBufferExt* indices, DxBufferExt* lower,
                                    DxBufferExt* invEdges, int numItems);
void PrintBuildHierarchyOutput(NvFlex::Context* context, DxBufferExt* positions,
                               DxBufferExt* indices, DxBufferExt* deltas,
                               DxBufferExt* rangeLefts, DxBufferExt* rangeRights,
                               DxBufferExt* numChildren);
