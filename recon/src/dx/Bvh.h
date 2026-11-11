// flexdecomp: reconstructed D3D backend — bvh.obj.
//
// bvh.obj contains the D3D compute-shader BVH builder used by the solver: the
// BVH aggregate (three device buffers plus counts), the LinearBVHBuilderGPU
// class with its embedded shader set, ResizeBVH/FreeBVH, the D3D buffer
// wrappers (DxBuffer/DxBufferExt/DxComputeBuffer/DxConstantBuffer, declared in
// utils.h because the copy helpers in utils.cpp use them) and a large amount
// of std:: template instantiations that the standard library provides.
//
// There is no MedianBVHBuilder/LinearBVHBuilderCPU in the D3D backend: the
// object inventory has no such symbols (the host-side median builder only
// exists in the CUDA library).
//
// Class layouts follow the PDB (see `bin/flexkit d3dcontext --cls BVH`):
//   BVH                        size 112  three DxBufferExt + 3 ints
//   LinearBVHBuilderGPU        size 480  buffers, shaders, radix sort, context
#pragma once

#include "utils.h"

// ---------------------------------------------------------------------------
// Node format
// ---------------------------------------------------------------------------
// 16 bytes: two float lanes, one float lane and a 32-bit word carrying the child
// index (31 bits) and the leaf bit.  Same layout as the CUDA backend's
// PackedNodeHalf (recon/include/nvflex/recovered.h).
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

// bvh.obj @0x2af90 (541 bytes).
void ResizeBVH(BVH* bvh, int numNodes, NvFlex::Context* context);

// bvh.obj @0x2b1b0 (208 bytes).
void FreeBVH(BVH* bvh);

// ---------------------------------------------------------------------------
// Radix sort interface (radixsort.obj)
// ---------------------------------------------------------------------------
// LinearBVHBuilderGPU sorts the Morton keys with the backend radix sort.  The
// implementation lives in radixsort.obj; these declarations are the subset the
// builder binds against (PDB: NvFlex::RadixSort vtable addRef@0, release@8,
// getInputBuffer@16, getOutputBuffer@24, sort@32; RadixSortDesc is a single
// `int maxSortBlocks`, RadixSortParams is { unsigned numKeys; unsigned
// numBlocks; unsigned numBits; }).
namespace NvFlex
{

struct RadixSortDesc
{
    int maxSortBlocks;
};

struct RadixSortParams
{
    unsigned numKeys;    // @0
    unsigned numBlocks;  // @4
    unsigned numBits;    // @8
    int _pad;            // @12
};

class RadixSort
{
public:
    virtual unsigned addRef() = 0;
    virtual unsigned release() = 0;
    virtual Buffer* getInputBuffer() = 0;
    virtual Buffer* getOutputBuffer() = 0;
    virtual void sort(Context* context, const RadixSortParams* params, void* timers) = 0;

protected:
    ~RadixSort() {}
};

} // namespace NvFlex

// ---------------------------------------------------------------------------
// DxTimers — GPU timing hooks passed into Build
// ---------------------------------------------------------------------------
// The full class (name -> timer index map, timer pool) lives in flex.obj; Build
// only needs these three members (PDB: mContext@0, mTimerPool@8, mEnabled@16).
// Keep this declaration in sync with the flex.obj slice.
struct DxTimers
{
    NvFlex::Context* mContext;
    NvFlex::TimerPool* mTimerPool;
    bool mEnabled;
};

// ---------------------------------------------------------------------------
// LinearBVHBuilderGPU
// ---------------------------------------------------------------------------
// Build(BVH*, positions, indices, unused, numItems, bounds, timers):
//   - grows the BVH to 2*numItems nodes,
//   - writes the constant block {numItems},
//   - when `bounds` is null, runs ComputeTotalBounds{,.Group,,Finalize} to
//     derive a tight quantization bound, otherwise uploads 1/(extent+1e-6) and
//     the supplied lower bound,
//   - sorts Morton codes with the backend radix sort,
//   - runs CalculateMortonCodes, CalculateKeyDeltas, BuildLeaves, clears the
//     child counters and runs BuildHierarchy.
class LinearBVHBuilderGPU
{
public:
    // bvh.obj @0x2b280 (1645 bytes).
    LinearBVHBuilderGPU(NvFlex::Context* context, bool shflSupported, int smCount,
                        bool amd);
    // bvh.obj @0x2b8f0 (992 bytes).
    ~LinearBVHBuilderGPU();

    // bvh.obj @0x2bcd0 (4592 bytes).
    void Build(BVH* bvh, DxBufferExt* positions, DxBufferExt* indices,
               DxBufferExt* unused, int numItems, const Bounds* bounds, DxTimers* timers);

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
    NvFlex::ComputeShader* mShaderCalculateMortonCodes;      // @256
    NvFlex::ComputeShader* mShaderCalculateKeyDeltas;        // @264
    NvFlex::ComputeShader* mShaderBuildLeaves;               // @272
    NvFlex::ComputeShader* mShaderBuildHierarchy;            // @280
    NvFlex::ComputeShader* mShaderComputeTotalBounds;        // @288
    NvFlex::ComputeShader* mShaderComputeTotalBoundsGroup;   // @296
    NvFlex::ComputeShader* mShaderComputeTotalBoundsFinalize;// @304
    NvFlex::ComputeShader* mShaderComputeTotalInvEdges;      // @312
    DxBufferExt mBoundsLower;                   // @320
    DxBufferExt mBoundsUpper;                   // @352
    DxBufferExt mGroupBoundsLower;              // @384
    DxBufferExt mGroupBoundsUpper;              // @416
    NvFlex::RadixSort* mBVHRadixSort;           // @448
    int mBVHRadixSortSize;                      // @456
    bool mIsSHFLSupported;                      // @460
    int mSMCount;                               // @464
    int mMaxNumTriMeshes;                       // @468
    NvFlex::Context* mContext;                  // @472
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
