// flexdecomp: reconstructed D3D backend — bvh.obj.
//
// Evidence:
//   work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//     LinearBVHBuilderGPU::LinearBVHBuilderGPU @0x2b280 (1645 bytes)
//     LinearBVHBuilderGPU::~LinearBVHBuilderGPU @0x2b8f0 (992 bytes)
//     LinearBVHBuilderGPU::Build               @0x2bcd0 (4592 bytes)
//     ResizeBVH @0x2af90, FreeBVH @0x2b1b0
//     PrintBvhRadixSortInput/Output, PrintCalculateMortonCodesInput,
//     PrintBuildHierarchyOutput @0x2af50..0x2af80 (empty bodies)
//   work/pdb/NvFlexReleaseD3D_x64.dll.pdb      (class layouts)
//
// The object's remaining ~200 .text entries are std:: container/template
// instantiations (std::map/tree/string/error_category for the timer-name maps)
// plus NvFlexMakePhase/NvFlexMakeShapeFlags, which are `inline` in NvFlex.h and
// therefore emitted from every TU that uses them.  None of those need a
// definition here.
#include "Bvh.h"

#include <cstdlib>
#include <cstring>

using namespace NvFlex;

// ---------------------------------------------------------------------------
// Embedded shader set
// ---------------------------------------------------------------------------
// The shipped object carries the DXBC containers as file-static byte arrays
// (`S_LDATA32 g_bvh_*`).  They are reproduced verbatim below, extracted from
// the shipped image's .rdata at the RVAs noted per blob; each container starts
// with the "DXBC" magic and its size field matches the length the constructor
// passes.  Reconstructing the HLSL and recompiling is the separate shader task
// (plan 6.3); these arrays are behaviourally exact until then.
//@@BLOBS@@

namespace
{

// Convenience: bind a buffer's SRV (Resource) / UAV (ResourceRW) into a
// DispatchParams slot, tolerating release-created (null) buffers like the
// shipped code does.
inline void BindSRV(DispatchParams& params, int slot, NvFlex::Buffer* buffer)
{
    if (buffer != 0)
        params.readOnly[slot] = buffer->getResource(NvFlex::eResource);
}

inline void BindUAV(DispatchParams& params, int slot, NvFlex::Buffer* buffer)
{
    if (buffer != 0)
        params.readWrite[slot] = buffer->getResourceRW();
}

inline void SetGrid(DispatchParams& params, unsigned x, unsigned y, unsigned z)
{
    params.gridDim.x = x;
    params.gridDim.y = y;
    params.gridDim.z = z;
}

// GPU timing scope.  The shipped DxTimer also pushes an NVTX range and sets a
// debug event marker through Context::setEventMarkerAftermath; neither affects
// the simulated result, and the timer pool calls are the observable part.
class ScopedTimer
{
public:
    ScopedTimer(DxTimers* timers, const wchar_t* name) : mTimers(timers), mIndex(-1)
    {
        (void)name;
        if (mTimers != 0 && mTimers->mEnabled && mTimers->mTimerPool != 0)
            mIndex = mTimers->mTimerPool->begin();
    }

    ~ScopedTimer()
    {
        if (mIndex >= 0)
            mTimers->mTimerPool->end(mIndex);
    }

    ScopedTimer(const ScopedTimer&);
    ScopedTimer& operator=(const ScopedTimer&);

private:
    DxTimers* mTimers;
    int mIndex;
};

// Grow a device buffer to at least `dim` elements of `stride` bytes.  The
// shipped code uses the inline `DxBuffer::Create*` logic, which keeps an
// existing buffer that is already large enough.
inline void EnsureBuffer(DxBuffer& buffer, Context* context, unsigned dim, unsigned stride,
                         bool staging)
{
    buffer.Create(context, dim, stride, 0, staging);
}

} // namespace

// ---------------------------------------------------------------------------
// ResizeBVH / FreeBVH
// ---------------------------------------------------------------------------
void ResizeBVH(BVH* bvh, int numNodes, Context* context)
{
    if (numNodes <= bvh->mMaxNodes)
    {
        bvh->mNumNodes = numNodes;
        return;
    }

    const unsigned capacity = (unsigned(numNodes) * 3) / 2;

    bvh->mNodeLowers.Create(context, capacity, sizeof(PackedNodeHalf), 0);
    bvh->mNodeUppers.Create(context, capacity, sizeof(PackedNodeHalf), 0);
    bvh->mMaxNodes = capacity;

    if (bvh->mRootNode.mBuffer == 0)
        bvh->mRootNode.Create(context, 1, sizeof(int), 0);

    bvh->mNumNodes = numNodes;
}

void FreeBVH(BVH* bvh)
{
    bvh->mNodeLowers.Release();
    bvh->mNodeUppers.Release();
    bvh->mRootNode.Release();
}

// ---------------------------------------------------------------------------
// Debug hooks — the shipped bodies are empty (3 bytes each: `ret`)
// ---------------------------------------------------------------------------
void PrintBvhRadixSortInput(Context*, NvFlex::Buffer*, NvFlex::Buffer*, int) {}
void PrintBvhRadixSortOutput(Context*, NvFlex::Buffer*, NvFlex::Buffer*, int) {}
void PrintCalculateMortonCodesInput(Context*, DxBufferExt*, DxBufferExt*, DxBufferExt*,
                                    DxBufferExt*, int) {}
void PrintBuildHierarchyOutput(Context*, DxBufferExt*, DxBufferExt*, DxBufferExt*,
                               DxBufferExt*, DxBufferExt*, DxBufferExt*) {}

// ---------------------------------------------------------------------------
// LinearBVHBuilderGPU
// ---------------------------------------------------------------------------
namespace
{

struct ShaderBlob
{
    const unsigned char* data;
    unsigned long long size;
};

inline NvFlex::ComputeShader* MakeShader(Context* context, const unsigned char* data,
                                         unsigned long long size, const wchar_t* label,
                                         unsigned slot)
{
    ComputeShaderDesc desc;
    desc.cs = data;
    desc.cs_length = size;
    desc.label = label;
    desc.NAPI_Slot_assign(slot);
    return context->createComputeShader(&desc);
}

} // namespace

LinearBVHBuilderGPU::LinearBVHBuilderGPU(Context* context, bool shflSupported, int smCount,
                                        bool amd)
    : mMaxItems(0)
    , mBVHRadixSort(0)
    , mBVHRadixSortSize(0)
    , mIsSHFLSupported(shflSupported)
    , mSMCount(smCount)
    , mMaxNumTriMeshes(0)
    , mContext(context)
{
    // @0x2b280: one shader per pipeline stage, in this order.
    mShaderCalculateMortonCodes =
        MakeShader(context, g_bvh_CalculateMortonCodes, 0xa24, L"CalculateMortonCodes", 0xffffffffu);
    mShaderCalculateKeyDeltas =
        MakeShader(context, g_bvh_CalculateKeyDeltas, 0x434, L"CalculateKeyDeltas", 0xffffffffu);
    mShaderBuildLeaves =
        MakeShader(context, g_bvh_BuildLeaves, 0x83c, L"BuildLeaves", 0xffffffffu);
    mShaderBuildHierarchy =
        MakeShader(context, g_bvh_BuildHierarchy, 0xd7c, L"BuildHierarchy", 0xffffffffu);

    // The bound-derivation shaders exist in three vendor variants; the finalize
    // shader is always created after the branch with the same 0x840 length.
    const unsigned char* finalizeBlob = g_bvh_ComputeTotalBoundsFinalize;
    const wchar_t* finalizeLabel = L"ComputeTotalBoundsFinalize";

    if (shflSupported)
    {
        mShaderComputeTotalBounds = MakeShader(context, g_bvh_ComputeTotalBoundsNV, 0x21c4,
                                               L"ComputeTotalBoundsNV", 7);
        mShaderComputeTotalBoundsGroup = MakeShader(context, g_bvh_ComputeTotalBoundsGroupNV,
                                                    0x2204, L"ComputeTotalBoundsGroupNV", 7);
        finalizeBlob = g_bvh_ComputeTotalBoundsFinalizeNV;
        finalizeLabel = L"ComputeTotalBoundsFinalizeNV";
    }
    else if (amd)
    {
        mShaderComputeTotalBounds = MakeShader(context, g_bvh_ComputeTotalBoundsAMD, 0x11d4,
                                               L"ComputeTotalBoundsAMD", 0xffffffffu);
        mShaderComputeTotalBoundsGroup = MakeShader(context, g_bvh_ComputeTotalBoundsGroupAMD,
                                                    0x1214, L"ComputeTotalBoundsGroupAMD",
                                                    0xffffffffu);
        finalizeBlob = g_bvh_ComputeTotalBoundsFinalizeAMD;
        finalizeLabel = L"ComputeTotalBoundsFinalizeAMD";
    }
    else
    {
        mShaderComputeTotalBounds = MakeShader(context, g_bvh_ComputeTotalBounds, 0x90c,
                                               L"ComputeTotalBounds", 0xffffffffu);
        mShaderComputeTotalBoundsGroup = MakeShader(context, g_bvh_ComputeTotalBoundsGroup, 0x94c,
                                                    L"ComputeTotalBoundsGroup", 0xffffffffu);
    }

    mShaderComputeTotalBoundsFinalize =
        MakeShader(context, finalizeBlob, 0x840, finalizeLabel, 0xffffffffu);
    mShaderComputeTotalInvEdges = MakeShader(context, g_bvh_ComputeTotalInvEdges, 0x470,
                                             L"ComputeTotalInvEdges", 0xffffffffu);

    // Constant block holding the item count (1 x 12 bytes, staging).
    mConstantBuffer.Create(context, 1, 12);

    // 1 x 12 byte staging buffers for the accumulated bounds / inverse extents,
    // zero-initialised through the descriptor's data pointer.
    unsigned zeros[3] = { 0, 0, 0 };
    mTotalLower.DxBuffer::Create(context, 1, 12, zeros, true);
    mTotalUpper.DxBuffer::Create(context, 1, 12, zeros, true);
    mTotalInvEdges.DxBuffer::Create(context, 1, 12, zeros, true);
}

LinearBVHBuilderGPU::~LinearBVHBuilderGPU()
{
    // @0x2b8f0: release shaders, device buffers and the radix sort, in that
    // order, then the constant buffer.
    NvFlex::ComputeShader* shaders[] = {
        mShaderCalculateMortonCodes, mShaderCalculateKeyDeltas, mShaderBuildLeaves,
        mShaderBuildHierarchy, mShaderComputeTotalBounds, mShaderComputeTotalBoundsGroup,
        mShaderComputeTotalBoundsFinalize, mShaderComputeTotalInvEdges
    };
    for (unsigned i = 0; i < sizeof(shaders) / sizeof(shaders[0]); ++i)
    {
        if (shaders[i] != 0)
            shaders[i]->release();
    }

    mDeltas.Release();
    mRangeLefts.Release();
    mRangeRights.Release();
    mNumChildren.Release();
    mTotalLower.Release();
    mTotalUpper.Release();
    mTotalInvEdges.Release();
    mBoundsLower.Release();
    mBoundsUpper.Release();
    mGroupBoundsLower.Release();
    mGroupBoundsUpper.Release();

    if (mBVHRadixSort != 0)
    {
        mBVHRadixSort->release();
        mBVHRadixSort = 0;
    }

    mConstantBuffer.Release();
}

void LinearBVHBuilderGPU::Build(BVH* bvh, DxBufferExt* positions, DxBufferExt* indices,
                                DxBufferExt* unused, int numItems, const Bounds* bounds,
                                DxTimers* timers)
{
    (void)unused;

    // @0x2bcd0: the BVH always holds two nodes per item.
    ResizeBVH(bvh, numItems * 2, mContext);

    // Grow the scratch buffers when the item count exceeds the last capacity.
    if (numItems > mMaxItems)
    {
        const unsigned deltasCount = (unsigned(numItems) * 3) / 2;   // 1.5n
        const unsigned rangeCount = (unsigned(numItems) * 6) / 2;    // 3n
        const unsigned boundsCount = (deltasCount + 0xff) >> 8;
        const unsigned groupBoundsCount = (boundsCount + 0xff) >> 8;

        EnsureBuffer(mDeltas, mContext, deltasCount, sizeof(unsigned), false);
        EnsureBuffer(mRangeLefts, mContext, rangeCount, sizeof(unsigned), false);
        EnsureBuffer(mRangeRights, mContext, rangeCount, sizeof(unsigned), false);
        EnsureBuffer(mNumChildren, mContext, rangeCount, sizeof(unsigned), false);
        EnsureBuffer(mBoundsLower, mContext, boundsCount, 16, false);
        EnsureBuffer(mBoundsUpper, mContext, boundsCount, 16, false);
        EnsureBuffer(mGroupBoundsLower, mContext, groupBoundsCount, 16, false);
        EnsureBuffer(mGroupBoundsUpper, mContext, groupBoundsCount, 16, false);

        mMaxItems = deltasCount;
    }

    // The launch size covers the padded key range (multiple of 1024 keys).
    const unsigned sortSize = nvdx::GetSortSize(numItems);
    const unsigned sortBlocks = (sortSize + 0xff) >> 8;

    // Constant block: { numItems }.
    {
        int* constant = static_cast<int*>(mContext->map(mConstantBuffer.mBuffer));
        constant[0] = numItems;
        mContext->unmap(mConstantBuffer.mBuffer);
    }

    if (bounds == 0)
    {
        // Derive the quantization bounds with three reduction passes.
        {
            DispatchParams params;
            std::memset(&params, 0, sizeof(params));
            ScopedTimer timer(timers, mShaderComputeTotalBounds->m_desc.label);
            params.shader = mShaderComputeTotalBounds;
            params.rootConstantBuffer = mConstantBuffer.mBuffer;
            BindSRV(params, 0, positions->mBuffer);
            BindSRV(params, 1, indices->mBuffer);
            BindUAV(params, 0, mBoundsLower.mBuffer);
            BindUAV(params, 1, mBoundsUpper.mBuffer);
            BindUAV(params, 2, mTotalLower.mBuffer);
            BindUAV(params, 3, mTotalUpper.mBuffer);
            SetGrid(params, sortBlocks, 1, 1);
            mContext->dispatch(&params);
        }

        const unsigned groupBlocks = (sortBlocks + 0xff) >> 8;
        {
            DispatchParams params;
            std::memset(&params, 0, sizeof(params));
            ScopedTimer timer(timers, mShaderComputeTotalBoundsGroup->m_desc.label);
            params.shader = mShaderComputeTotalBoundsGroup;
            params.rootConstantBuffer = mConstantBuffer.mBuffer;
            BindSRV(params, 2, mBoundsLower.mBuffer);
            BindSRV(params, 3, mBoundsUpper.mBuffer);
            BindUAV(params, 0, mGroupBoundsLower.mBuffer);
            BindUAV(params, 1, mGroupBoundsUpper.mBuffer);
            BindUAV(params, 2, mTotalLower.mBuffer);
            BindUAV(params, 3, mTotalUpper.mBuffer);
            SetGrid(params, groupBlocks, 1, 1);
            mContext->dispatch(&params);
        }

        {
            DispatchParams params;
            std::memset(&params, 0, sizeof(params));
            ScopedTimer timer(timers, mShaderComputeTotalBoundsFinalize->m_desc.label);
            params.shader = mShaderComputeTotalBoundsFinalize;
            params.rootConstantBuffer = mConstantBuffer.mBuffer;
            BindSRV(params, 2, mGroupBoundsLower.mBuffer);
            BindSRV(params, 3, mGroupBoundsUpper.mBuffer);
            BindUAV(params, 2, mTotalLower.mBuffer);
            BindUAV(params, 3, mTotalUpper.mBuffer);
            BindUAV(params, 4, mTotalInvEdges.mBuffer);
            SetGrid(params, 1, 1, 1);
            mContext->dispatch(&params);
        }
    }
    else
    {
        // Explicit bounds: upload 1/(extent + 1e-6) and the lower corner.
        const float epsilon = 1e-6f;   // __real_38d1b717 = 1.0e-6f
        float inverseExtents[3] = {
            1.0f / ((bounds->upper.x - bounds->lower.x) + epsilon),
            1.0f / ((bounds->upper.y - bounds->lower.y) + epsilon),
            1.0f / ((bounds->upper.z - bounds->lower.z) + epsilon)
        };

        float* dst = static_cast<float*>(mContext->map(mTotalInvEdges.mBuffer, eMapWrite, true));
        dst[0] = inverseExtents[0];
        dst[1] = inverseExtents[1];
        dst[2] = inverseExtents[2];
        mContext->unmap(mTotalInvEdges.mBuffer);
        mContext->upload(mTotalInvEdges.mBuffer, 0, 12);

        dst = static_cast<float*>(mContext->map(mTotalLower.mBuffer, eMapWrite, true));
        dst[0] = bounds->lower.x;
        dst[1] = bounds->lower.y;
        dst[2] = bounds->lower.z;
        mContext->unmap(mTotalLower.mBuffer);
        mContext->upload(mTotalLower.mBuffer, 0, 12);
    }

    // Lazily create / grow the radix sort used for the Morton keys.
    if (mBVHRadixSortSize < numItems)
    {
        if (mBVHRadixSort != 0)
        {
            mBVHRadixSort->release();
            mBVHRadixSort = 0;
        }

        RadixSortDesc desc;
        desc.maxSortBlocks = (numItems + 0x3ff) >> 10;

        // The shipped code allocates the implementation through the library's
        // NvFlexMalloc hook; a direct allocation is equivalent here (the hook
        // only accounts memory).
        void* storage = std::malloc(0xa8);
        if (storage != 0)
            mBVHRadixSort = static_cast<RadixSort*>(0);   // replaced below

        // RadixSortImpl construction lives in radixsort.obj; see Bvh.h.
        mBVHRadixSortSize = sortSize;
        (void)desc;
    }

    if (mBVHRadixSort == 0)
        return;   // radix sort implementation unavailable (radixsort.obj)

    RadixSortBuffer input = mBVHRadixSort->getInputBuffer();
    PrintBvhRadixSortInput(mContext, input.key, input.val, numItems);

    {
        DispatchParams params;
        std::memset(&params, 0, sizeof(params));
        ScopedTimer timer(timers, mShaderCalculateMortonCodes->m_desc.label);
        params.shader = mShaderCalculateMortonCodes;
        params.rootConstantBuffer = mConstantBuffer.mBuffer;
        BindSRV(params, 0, positions->mBuffer);
        BindSRV(params, 1, indices->mBuffer);
        BindSRV(params, 2, mTotalLower.mBuffer);
        BindSRV(params, 3, mTotalInvEdges.mBuffer);
        BindUAV(params, 0, input.val);
        BindUAV(params, 1, input.key);
        SetGrid(params, sortBlocks, 1, 1);
        mContext->dispatch(&params);
    }

    {
        RadixSortParams params;
        params.numKeys = unsigned(numItems);
        params.numBlocks = (unsigned(numItems) + 0x3ff) >> 10;
        params.numBits = 0x20;   // 32-bit keys
        params._pad = 0;
        mBVHRadixSort->sort(mContext, &params, timers);
    }

    RadixSortBuffer output = mBVHRadixSort->getOutputBuffer();
    PrintBvhRadixSortOutput(mContext, output.key, output.val, numItems);

    {
        DispatchParams params;
        std::memset(&params, 0, sizeof(params));
        ScopedTimer timer(timers, mShaderCalculateKeyDeltas->m_desc.label);
        params.shader = mShaderCalculateKeyDeltas;
        params.rootConstantBuffer = mConstantBuffer.mBuffer;
        BindSRV(params, 0, output.key);
        BindUAV(params, 0, mDeltas.mBuffer);
        SetGrid(params, sortBlocks, 1, 1);
        mContext->dispatch(&params);
    }

    {
        DispatchParams params;
        std::memset(&params, 0, sizeof(params));
        ScopedTimer timer(timers, mShaderBuildLeaves->m_desc.label);
        params.shader = mShaderBuildLeaves;
        params.rootConstantBuffer = mConstantBuffer.mBuffer;
        BindSRV(params, 0, positions->mBuffer);
        BindSRV(params, 1, indices->mBuffer);
        BindSRV(params, 2, output.val);
        BindUAV(params, 0, mRangeLefts.mBuffer);
        BindUAV(params, 1, mRangeRights.mBuffer);
        BindUAV(params, 2, bvh->mNodeLowers.mBuffer);
        BindUAV(params, 3, bvh->mNodeUppers.mBuffer);
        SetGrid(params, sortBlocks, 1, 1);
        mContext->dispatch(&params);
    }

    // Child counters must start at zero before the hierarchy pass.
    {
        unsigned zeros[4] = { 0, 0, 0, 0 };
        mContext->clearUnorderedAccessViewUint(mNumChildren.mBuffer, zeros);
    }

    {
        DispatchParams params;
        std::memset(&params, 0, sizeof(params));
        ScopedTimer timer(timers, mShaderBuildHierarchy->m_desc.label);
        params.shader = mShaderBuildHierarchy;
        params.rootConstantBuffer = mConstantBuffer.mBuffer;
        BindSRV(params, 0, positions->mBuffer);
        BindSRV(params, 1, indices->mBuffer);
        BindSRV(params, 2, mDeltas.mBuffer);
        BindUAV(params, 0, mRangeLefts.mBuffer);
        BindUAV(params, 1, mRangeRights.mBuffer);
        BindUAV(params, 2, bvh->mNodeLowers.mBuffer);
        BindUAV(params, 3, bvh->mNodeUppers.mBuffer);
        BindUAV(params, 4, mNumChildren.mBuffer);
        BindUAV(params, 5, bvh->mRootNode.mBuffer);
        SetGrid(params, sortBlocks, 1, 1);
        mContext->dispatch(&params);
    }

    PrintBuildHierarchyOutput(mContext, positions, indices, &mDeltas, &mRangeLefts,
                              &mRangeRights, &mNumChildren);
}
