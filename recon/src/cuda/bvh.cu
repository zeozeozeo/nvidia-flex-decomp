// Reconstructed from NvFlexDebugCUDA_x64.a(cudabvh.o) and NvFlexReleaseCUDA_x64.a(cudabvh.o).
//
// This translation unit holds the *host* half of the original src/cuda/bvh.cu.
// The seven device kernels (CalculateMortonCodes, CalculateKeyDeltas,
// BuildLeaves, BuildHierarchy, ComputeTotalBounds, ComputeTotalInvEdges,
// cub::EmptyKernel) live in bvh.part1.cu; the host-side launch sequences are
// reproduced here exactly as nvcc lowers `kernel<<<grid,block>>>(...)`:
// `__cudaPushCallConfiguration` followed by a call to the kernel's host
// wrapper, which every CUDA object emits next to its `__device_stub__`.
//
// Reconstruction notes (see units/nvflex.bvh/*.md for the per-unit evidence):
//  * The recovered headers flatten the FleX templates: `Vec3` is
//    `XVector3_float` and `Vec4` is `XVector4_float`, so the original mangled
//    names of the member functions and of the kernel/SDK entry points cannot be
//    produced by a plain C++ declaration.  Those definitions and declarations
//    carry their exact symbol in an `asm` label instead (the demangled
//    signature the static gate compares is the mangled name's, so it matches).
//  * `BVH`, `PackedNodeHalf`, `Bounds`, `Vec3`, `LinearBVHBuilderGPU` and
//    `NvFlexLibrary` are used verbatim from nvflex/recovered.h.
//  * `__FILE__` is reproduced as the shipped object's literal so the module
//    literal set is unchanged.
//  * Asserts call `__assert_fail` directly with the shipped object's exact
//    message / file / line / function literals (the source `assert()` would
//    substitute this TU's `__FILE__`/`__PRETTY_FUNCTION__`).

#include <cuda_runtime.h>

#include <cstring>
#include <cstdio>
#include <cfloat>
#include <algorithm>
#include <vector>

#include "nvflex/all.h"
#include "NvFlex.h"

// ---------------------------------------------------------------------------
// Shipped-object literals
// ---------------------------------------------------------------------------

// `__FILE__` of the original translation unit (rodata, cudabvh.o).
static const char* const kSourceFile = "../../../src/cuda/bvh.cu";

// ---------------------------------------------------------------------------
// nvflex.util entry points (recovered; symbols are those cudabvh.o imports)
// ---------------------------------------------------------------------------

extern "C" void FlexDeviceMallocInternal(void** ptr, size_t size, const char* name,
                                         const char* file, int line)
    asm("_Z24FlexDeviceMallocInternalPPvmPKcS2_i");
extern "C" void FlexDeviceFreeInternal(void* ptr) asm("_Z22FlexDeviceFreeInternalPv");
extern "C" cudaTextureObject_t CreateTexture(const void* data, int size)
    asm("_Z13CreateTexturePK8XVector4IfEi");
extern "C" void DestroyTexture(cudaTextureObject_t texture) asm("_Z14DestroyTexturey");

// nvflex.sort
extern "C" void SortCellIndices(NvFlexLibrary* lib, int* keys, int* indices, int n, int numBits)
    asm("_Z15SortCellIndicesP13NvFlexLibraryPiS1_ii");

// nvflex.bvh device kernels (defined in bvh.part1.cu).  These are the host
// wrappers nvcc emits beside each `__device_stub__` symbol.
extern "C" void CalculateMortonCodes(const Vec4* lowers, const Vec4* uppers,
                                     const int* priorities, int numItems,
                                     const Vec3* totalLower, const Vec3* totalInvEdges,
                                     int* indices, int* keys)
    asm("_Z20CalculateMortonCodesPK8XVector4IfES2_PKiiPK8XVector3IfES8_PiS9_");
extern "C" void CalculateKeyDeltas(const int* keys, int* deltas, int n)
    asm("_Z18CalculateKeyDeltasPKiPii");
extern "C" void BuildLeaves(const Vec4* lowers, const Vec4* uppers, int numItems,
                            const int* indices, int* rangeLefts, int* rangeRights,
                            PackedNodeHalf* nodeLowers, PackedNodeHalf* nodeUppers)
    asm("_Z11BuildLeavesPK8XVector4IfES2_iPKiPiS5_P14PackedNodeHalfS7_");
extern "C" void BuildHierarchy(int numItems, int* rootNode, const int* deltas,
                               int* numChildren, volatile int* rangeLefts,
                               volatile int* rangeRights, volatile PackedNodeHalf* nodeLowers,
                               volatile PackedNodeHalf* nodeUppers)
    asm("_Z14BuildHierarchyiPiPKiS_PViS3_PV14PackedNodeHalfS6_");
extern "C" void ComputeTotalBounds(const Vec4* lowers, const Vec4* uppers, Vec3* totalLower,
                                   Vec3* totalUpper, int numItems)
    asm("_Z18ComputeTotalBoundsPK8XVector4IfES2_P8XVector3IfES5_i");
extern "C" void ComputeTotalInvEdges(const Vec3* totalLower, const Vec3* totalUpper,
                                     Vec3* totalInvEdges)
    asm("_Z20ComputeTotalInvEdgesPK8XVector3IfES2_PS0_");

// glibc's FORTIFY-checked printf (the shipped object was built with
// _FORTIFY_SOURCE and calls this symbol, not printf)
extern "C" int __printf_chk(int flag, const char* format, ...);

// glibc's failure hook, called directly so the shipped object's message,
// file, line and function literals are reproduced verbatim
extern "C" void __assert_fail(const char* assertion, const char* file, unsigned int line,
                              const char* function) __THROW __attribute__((noreturn));

// nvflex.util logging
extern "C" void FlexLogFunc(NvFlexErrorCallback cb, NvFlexErrorSeverity severity,
                            const char* msg, const char* file, int line, ...)
    asm("_Z11FlexLogFuncPFv19NvFlexErrorSeverityPKcS1_iES_S1_S1_iz");

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

namespace {

// FleX maths.h spelled these as `a > b ? a : b` / `a < b ? a : b`, which is
// what the shipped object's maxss/minss encode (NaN selects the second
// operand).  std::max/min invert the NaN result, so they are not used.
inline float Max(float a, float b) { return a > b ? a : b; }
inline float Min(float a, float b) { return a < b ? a : b; }

inline Vec3 MakeVec3(float x, float y, float z)
{
    Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

inline float Get(const Vec3& v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

inline Vec3 Vec3Min(const Vec3& a, const Vec3& b)
{
    return MakeVec3(Min(a.x, b.x), Min(a.y, b.y), Min(a.z, b.z));
}

inline Vec3 Vec3Max(const Vec3& a, const Vec3& b)
{
    return MakeVec3(Max(a.x, b.x), Max(a.y, b.y), Max(a.z, b.z));
}

// LinearBVHBuilderCPU::KeyIndexPair as declared in the recovered headers is the
// distinct type `::KeyIndexPair`; the shipped symbols use the nested
// `LinearBVHBuilderCPU::KeyIndexPair`.  Both are {uint32 key; int index}.  The
// sort only needs a local 8-byte element with the same layout.
struct BuilderKeyIndexPair
{
    uint32_t key;
    int index;

    bool operator<(const BuilderKeyIndexPair& other) const { return key < other.key; }
};

// Comparator the shipped __introselect<int*, PartitionPredicateMedian> uses:
// compare the bounds midpoints along one axis.
struct PartitionPredicateMedianCmp
{
    const Bounds* bounds;
    int axis;

    bool operator()(int a, int b) const
    {
        const Bounds& ba = bounds[a];
        const Bounds& bb = bounds[b];
        const float ma = (Get(ba.lower, axis) + Get(ba.upper, axis)) * 0.5f;
        const float mb = (Get(bb.lower, axis) + Get(bb.upper, axis)) * 0.5f;
        return ma < mb;
    }
};

// The shipped object holds the selection algorithm in its own symbol
// (__introselect<int*, PartitionPredicateMedian>) instead of folding it into
// PartitionObjectsMedian; keep it out of line for the same code shape.
__attribute__((noinline)) void Introselect(int* first, int* nth, int* last,
                                           PartitionPredicateMedianCmp comp)
{
    std::nth_element(first, nth, last, comp);
}

#define FD_CHECK_CUDA(call, text)                                                          \
    {                                                                                      \
        cudaError_t err = (call);                                                          \
        NvFlexErrorCallback cb = *reinterpret_cast<NvFlexErrorCallback*>(lib->mErrorFunc); \
        if (err != cudaSuccess && cb != NULL)                                              \
            FlexLogFunc(cb, eNvFlexLogError, "Cuda error: %d in %s", kSourceFile,           \
                        __LINE__, err, (text));                                            \
    }

}  // namespace

// ---------------------------------------------------------------------------
// BVH storage
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/bvh.cu:0 _Z9ResizeBVHR3BVHi  (line absent from DWARF)
void ResizeBVH(BVH& bvh, int numNodes)
{
    if (bvh.mMaxNodes < numNodes)
    {
        const int numToAlloc = numNodes * 3 / 2;

        FlexDeviceFreeInternal(bvh.mNodeLowers);
        FlexDeviceFreeInternal(bvh.mNodeUppers);
        DestroyTexture(bvh.mNodeLowersTex);
        DestroyTexture(bvh.mNodeUppersTex);

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&bvh.mNodeLowers),
                                 sizeof(PackedNodeHalf) * numToAlloc, "&bvh.mNodeLowers",
                                 kSourceFile, __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&bvh.mNodeUppers),
                                 sizeof(PackedNodeHalf) * numToAlloc, "&bvh.mNodeUppers",
                                 kSourceFile, __LINE__);

        bvh.mNodeLowersTex = CreateTexture(bvh.mNodeLowers, numToAlloc * sizeof(PackedNodeHalf));
        bvh.mNodeUppersTex = CreateTexture(bvh.mNodeUppers, numToAlloc * sizeof(PackedNodeHalf));

        bvh.mMaxNodes = numToAlloc;

        if (bvh.mRootNode == NULL)
        {
            FlexDeviceMallocInternal(reinterpret_cast<void**>(&bvh.mRootNode), sizeof(int),
                                     "&bvh.mRootNode", kSourceFile, __LINE__);
        }
    }
}

// ---------------------------------------------------------------------------
// Median split builder
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/bvh.cu:103 _ZN16MedianBVHBuilder5BuildER3BVHPK6Boundsi
extern "C" void MedianBVHBuilderBuild(MedianBVHBuilder* self, BVH& bvh, const Bounds* items, int n)
    asm("_ZN16MedianBVHBuilder5BuildER3BVHPK6Boundsi");
extern "C" int MedianBVHBuilderBuildRecursive(MedianBVHBuilder* self, BVH& bvh, const Bounds* bounds,
                                              int* indices, int start, int end, int depth)
    asm("_ZN16MedianBVHBuilder14BuildRecursiveER3BVHPK6BoundsPiiii");

extern "C" void MedianBVHBuilderBuild(MedianBVHBuilder* self, BVH& bvh, const Bounds* items, int n)
{
    // create a new BVH
    memset(&bvh, 0, sizeof(BVH));

    bvh.mMaxNodes = n * 2;
    bvh.mNodeLowers = new PackedNodeHalf[bvh.mMaxNodes];
    bvh.mNodeUppers = new PackedNodeHalf[bvh.mMaxNodes];
    bvh.mNumNodes = 0;
    bvh.mRootNode = NULL;

    std::vector<int> indices(n);

    for (int i = 0; i < n; ++i)
    {
        indices[i] = i;
    }

    MedianBVHBuilderBuildRecursive(self, bvh, items, &indices[0], 0, n, 0);

    __printf_chk(1, "Created BVH for %d items with %d nodes, max depth of %d\n", n,
                 bvh.mNumNodes, bvh.mMaxDepth);
}

// flexdecomp: src/cuda/bvh.cu:126 _ZN16MedianBVHBuilder10CalcBoundsEPK6BoundsPKiii
extern "C" void MedianBVHBuilderCalcBounds(Bounds* ret, MedianBVHBuilder*, const Bounds* bounds,
                                           const int* indices, int start, int end)
    asm("_ZN16MedianBVHBuilder10CalcBoundsEPK6BoundsPKiii");

extern "C" void MedianBVHBuilderCalcBounds(Bounds* ret, MedianBVHBuilder*, const Bounds* bounds,
                                           const int* indices, int start, int end)
{
    Vec3 lower = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 upper = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = start; i < end; ++i)
    {
        const Bounds& b = bounds[indices[i]];
        lower = Vec3Min(lower, b.lower);
        upper = Vec3Max(upper, b.upper);
    }

    ret->lower = lower;
    ret->upper = upper;
}

// flexdecomp: src/cuda/bvh.cu:150 _ZN16MedianBVHBuilder22PartitionObjectsMedianEPK6BoundsPiiiS0_
extern "C" int MedianBVHBuilderPartitionObjectsMedian(MedianBVHBuilder*, const Bounds* bounds,
                                                      int* indices, int start, int end,
                                                      Bounds rangeBounds)
    asm("_ZN16MedianBVHBuilder22PartitionObjectsMedianEPK6BoundsPiiiS0_");

extern "C" int MedianBVHBuilderPartitionObjectsMedian(MedianBVHBuilder*, const Bounds* bounds,
                                                      int* indices, int start, int end,
                                                      Bounds rangeBounds)
{
    if (!(end - start >= 2))
    {
        __assert_fail("end-start >= 2", kSourceFile, 0x98,
                      "int MedianBVHBuilder::PartitionObjectsMedian(const Bounds*, int*, int, "
                      "int, Bounds)");
    }

    const Vec3 edges = MakeVec3(rangeBounds.upper.x - rangeBounds.lower.x,
                                rangeBounds.upper.y - rangeBounds.lower.y,
                                rangeBounds.upper.z - rangeBounds.lower.z);

    int longestAxis;

    if ((edges.x <= edges.y) || (edges.x <= edges.z))
    {
        longestAxis = (edges.y <= edges.z) + 1;
    }
    else
    {
        longestAxis = 0;
    }

    const int k = (start + end) / 2;

    PartitionPredicateMedianCmp comp;
    comp.bounds = bounds;
    comp.axis = longestAxis;

    Introselect(&indices[start], &indices[k], &indices[end], comp);

    return k;
}

// flexdecomp: src/cuda/bvh.cu:180 _ZN16MedianBVHBuilder24PartitionObjectsMidPointEPK6BoundsPiiiS0_
extern "C" int MedianBVHBuilderPartitionObjectsMidPoint(MedianBVHBuilder*, const Bounds* bounds,
                                                        int* indices, int start, int end,
                                                        Bounds rangeBounds)
    asm("_ZN16MedianBVHBuilder24PartitionObjectsMidPointEPK6BoundsPiiiS0_");

extern "C" int MedianBVHBuilderPartitionObjectsMidPoint(MedianBVHBuilder*, const Bounds* bounds,
                                                        int* indices, int start, int end,
                                                        Bounds rangeBounds)
{
    if (!(end - start >= 2))
    {
        __assert_fail("end-start >= 2", kSourceFile, 0xb6,
                      "int MedianBVHBuilder::PartitionObjectsMidPoint(const Bounds*, int*, int, "
                      "int, Bounds)");
    }

    const Vec3 edges = MakeVec3(rangeBounds.upper.x - rangeBounds.lower.x,
                                rangeBounds.upper.y - rangeBounds.lower.y,
                                rangeBounds.upper.z - rangeBounds.lower.z);

    int longestAxis;
    float mid;

    if ((edges.x <= edges.y) || (edges.x <= edges.z))
    {
        if (edges.y <= edges.z)
        {
            longestAxis = 2;
            mid = (rangeBounds.lower.z + rangeBounds.upper.z) * 0.5f;
        }
        else
        {
            longestAxis = 1;
            mid = (rangeBounds.lower.y + rangeBounds.upper.y) * 0.5f;
        }
    }
    else
    {
        longestAxis = 0;
        mid = (rangeBounds.lower.x + rangeBounds.upper.x) * 0.5f;
    }

    int* lower = indices + start;
    int* upper = indices + end;

    while (lower != upper)
    {
        const Bounds& b = bounds[*lower];
        const float center = (Get(b.lower, longestAxis) + Get(b.upper, longestAxis)) * 0.5f;

        if (mid < center)
        {
            for (;;)
            {
                --upper;

                if (upper == lower)
                {
                    goto partition_done;
                }

                const Bounds& ub = bounds[*upper];
                const float upperCenter =
                    (Get(ub.lower, longestAxis) + Get(ub.upper, longestAxis)) * 0.5f;

                if (!(mid < upperCenter))
                {
                    break;
                }
            }

            const int tmp = *lower;
            *lower = *upper;
            *upper = tmp;
        }

        ++lower;
    }

partition_done:
    {
        const int k = static_cast<int>(lower - indices);

        if (k != end && k != start)
        {
            return k;
        }
    }

    return (start + end) / 2;
}

// flexdecomp: src/cuda/bvh.cu:204 _ZN16MedianBVHBuilder14BuildRecursiveER3BVHPK6BoundsPiiii
extern "C" int MedianBVHBuilderBuildRecursive(MedianBVHBuilder* self, BVH& bvh,
                                              const Bounds* bounds, int* indices, int start,
                                              int end, int depth)
{
    if (!(start < end))
    {
        __assert_fail("start < end", kSourceFile, 0xce,
                      "int MedianBVHBuilder::BuildRecursive(BVH&, const Bounds*, int*, int, int, "
                      "int)");
    }

    const int nodeIndex = bvh.mNumNodes++;
    const int kMaxItemsPerLeaf = 2;

    if (!(nodeIndex < bvh.mMaxNodes))
    {
        __assert_fail("nodeIndex < bvh.mMaxNodes", kSourceFile, 0xd3,
                      "int MedianBVHBuilder::BuildRecursive(BVH&, const Bounds*, int*, int, int, "
                      "int)");
    }

    if (depth > bvh.mMaxDepth)
    {
        bvh.mMaxDepth = depth;
    }

    Bounds b;
    MedianBVHBuilderCalcBounds(&b, self, bounds, indices, start, end);

    PackedNodeHalf n;

    if (end - start < kMaxItemsPerLeaf)
    {
        const int index = indices[start];

        n.x = b.lower.x;
        n.y = b.lower.y;
        n.z = b.lower.z;
        n.i = index;
        n.b = true;
        bvh.mNodeLowers[nodeIndex] = n;

        n.x = b.upper.x;
        n.y = b.upper.y;
        n.z = b.upper.z;
        n.i = index;
        n.b = false;
        bvh.mNodeUppers[nodeIndex] = n;
    }
    else
    {
        const int split = MedianBVHBuilderPartitionObjectsMedian(self, bounds, indices, start, end, b);

        const int leftChild =
            MedianBVHBuilderBuildRecursive(self, bvh, bounds, indices, start, split, depth + 1);
        const int rightChild =
            MedianBVHBuilderBuildRecursive(self, bvh, bounds, indices, split, end, depth + 1);

        n.x = b.lower.x;
        n.y = b.lower.y;
        n.z = b.lower.z;
        n.i = leftChild;
        n.b = false;
        bvh.mNodeLowers[nodeIndex] = n;

        n.x = b.upper.x;
        n.y = b.upper.y;
        n.z = b.upper.z;
        n.i = rightChild;
        n.b = false;
        bvh.mNodeUppers[nodeIndex] = n;
    }

    return nodeIndex;
}

// ---------------------------------------------------------------------------
// Linear (Morton code) builder, CPU
// ---------------------------------------------------------------------------

namespace {

// 10-bit -> 30-bit Morton spread (the shipped object's shift/xor/mask chain).
inline uint32_t ExpandBits(uint32_t x)
{
    x = (x << 16) ^ x;
    x &= 0xff0000ff;
    x = (x << 8) ^ x;
    x &= 0x0300f00f;
    x = (x << 4) ^ x;
    x &= 0x30c30c3;
    x = (x << 2) ^ x;
    x &= 0x9249249;
    return x;
}

}  // namespace

extern "C" int LinearBVHBuilderCPUBuildRecursive(LinearBVHBuilderCPU* self, BVH& bvh,
                                                 const BuilderKeyIndexPair* keys,
                                                 const Bounds* bounds, int start, int end,
                                                 int depth)
    asm("_ZN19LinearBVHBuilderCPU14BuildRecursiveER3BVHPKNS_12KeyIndexPairEPK6Boundsiii");

// flexdecomp: src/cuda/bvh.cu:245 _ZN19LinearBVHBuilderCPU5BuildER3BVHPK6Boundsi
extern "C" void LinearBVHBuilderCPUBuild(LinearBVHBuilderCPU* self, BVH& bvh, const Bounds* items,
                                         int n)
    asm("_ZN19LinearBVHBuilderCPU5BuildER3BVHPK6Boundsi");

extern "C" void LinearBVHBuilderCPUBuild(LinearBVHBuilderCPU* self, BVH& bvh, const Bounds* items,
                                         int n)
{
    memset(&bvh, 0, sizeof(BVH));

    bvh.mMaxNodes = n * 2;
    bvh.mNodeLowers = new PackedNodeHalf[bvh.mMaxNodes];
    bvh.mNodeUppers = new PackedNodeHalf[bvh.mMaxNodes];
    bvh.mNumNodes = 0;
    bvh.mRootNode = NULL;

    std::vector<BuilderKeyIndexPair> keys;
    keys.reserve(n);

    if (n > 0)
    {
        Vec3 lower = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);
        Vec3 upper = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int i = 0; i < n; ++i)
        {
            lower = Vec3Min(lower, items[i].lower);
            upper = Vec3Max(upper, items[i].upper);
        }

        // widen the range by kEpsilon on both sides so the extremes never
        // produce a coordinate outside [0,1023]
        const float kEpsilon = 0.001f;

        lower.x -= kEpsilon;
        lower.y -= kEpsilon;
        lower.z -= kEpsilon;

        const Vec3 invEdges = MakeVec3(1.0f / ((upper.x + kEpsilon) - lower.x),
                                       1.0f / ((upper.y + kEpsilon) - lower.y),
                                       1.0f / ((upper.z + kEpsilon) - lower.z));

        for (int i = 0; i < n; ++i)
        {
            const float cx = (items[i].lower.x + items[i].upper.x) * 0.5f;
            const float cy = (items[i].lower.y + items[i].upper.y) * 0.5f;
            const float cz = (items[i].lower.z + items[i].upper.z) * 0.5f;

            const int ix = static_cast<int>((cx - lower.x) * invEdges.x * 1024.0f);
            const int iy = static_cast<int>((cy - lower.y) * invEdges.y * 1024.0f);
            const int iz = static_cast<int>((cz - lower.z) * invEdges.z * 1024.0f);

            // out-of-range coordinates clamp to the maximum 10-bit value; the
            // shipped object keeps part1by2(1023) in that case
            uint32_t ux = 0x9249249u;
            uint32_t uy = 0x12492492u;
            uint32_t uz = 0x24924924u;

            if (ix < 1023)
            {
                ux = ExpandBits(static_cast<uint32_t>(ix < 0 ? 0 : ix));
            }

            if (iy < 1023)
            {
                uy = ExpandBits(static_cast<uint32_t>(iy < 0 ? 0 : iy)) << 1;
            }

            if (iz < 1023)
            {
                uz = ExpandBits(static_cast<uint32_t>(iz < 0 ? 0 : iz)) << 2;
            }

            BuilderKeyIndexPair l;
            l.key = ux | uy | uz;
            l.index = i;

            keys.push_back(l);
        }

        std::sort(keys.begin(), keys.end());
    }

    LinearBVHBuilderCPUBuildRecursive(self, bvh, &keys[0], items, 0, n, 0);

    __printf_chk(1, "Created BVH for %d items with %d nodes, max depth of %d\n", n,
                 bvh.mNumNodes, bvh.mMaxDepth);
}

// flexdecomp: src/cuda/bvh.cu:332 _ZN19LinearBVHBuilderCPU14BuildRecursiveER3BVHPKNS_12KeyIndexPairEPK6Boundsiii
extern "C" int LinearBVHBuilderCPUBuildRecursive(LinearBVHBuilderCPU*, BVH&,
                                                 const BuilderKeyIndexPair* keys,
                                                 const Bounds* bounds, int start, int end,
                                                 int depth)
    asm("_ZN19LinearBVHBuilderCPU14BuildRecursiveER3BVHPKNS_12KeyIndexPairEPK6Boundsiii");

namespace {

// highest differing bit of two Morton codes
inline uint32_t HighestBit(uint32_t x)
{
    uint32_t mask = 0x80000000u;

    if (static_cast<int32_t>(x) >= 0)
    {
        int shift = 0;
        uint32_t v = x;

        do
        {
            ++shift;
            v <<= 1;
        } while (static_cast<int32_t>(v) >= 0);

        mask = 1u << (31 - shift);
    }

    return mask;
}

}  // namespace

extern "C" int LinearBVHBuilderCPUBuildRecursive(LinearBVHBuilderCPU* self, BVH& bvh,
                                                 const BuilderKeyIndexPair* keys,
                                                 const Bounds* bounds, int start, int end,
                                                 int depth)
{
    if (!(start < end))
    {
        __assert_fail("start < end", kSourceFile, 0x14e,
                      "int LinearBVHBuilderCPU::BuildRecursive(BVH&, const "
                      "LinearBVHBuilderCPU::KeyIndexPair*, const Bounds*, int, int, int)");
    }

    const int nodeIndex = bvh.mNumNodes++;
    const int kMaxItemsPerLeaf = 2;

    if (!(nodeIndex < bvh.mMaxNodes))
    {
        __assert_fail("nodeIndex < bvh.mMaxNodes", kSourceFile, 0x153,
                      "int LinearBVHBuilderCPU::BuildRecursive(BVH&, const "
                      "LinearBVHBuilderCPU::KeyIndexPair*, const Bounds*, int, int, int)");
    }

    if (depth > bvh.mMaxDepth)
    {
        bvh.mMaxDepth = depth;
    }

    Vec3 lower = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);
    Vec3 upper = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (int i = start; i < end; ++i)
    {
        const Bounds& b = bounds[keys[i].index];
        lower = Vec3Min(lower, b.lower);
        upper = Vec3Max(upper, b.upper);
    }

    PackedNodeHalf n;

    if (end - start < kMaxItemsPerLeaf)
    {
        const int index = keys[start].index;

        n.x = lower.x;
        n.y = lower.y;
        n.z = lower.z;
        n.i = index;
        n.b = true;
        bvh.mNodeLowers[nodeIndex] = n;

        n.x = upper.x;
        n.y = upper.y;
        n.z = upper.z;
        n.i = index;
        n.b = false;
        bvh.mNodeUppers[nodeIndex] = n;
    }
    else
    {
        int split;

        if (keys[start].key == keys[end - 1].key)
        {
            split = (start + end) / 2;
        }
        else
        {
            // binary search for the first key whose highest differing bit is set
            const uint32_t mask = HighestBit(keys[start].key ^ keys[end - 1].key);

            int lo = start;
            int hi = end;

            while (hi - lo > 0)
            {
                const int mid = (hi + lo) / 2;

                if (keys[mid].key & mask)
                {
                    hi = mid;
                }
                else
                {
                    lo = mid + 1;
                }
            }

            if (lo != hi)
            {
                __assert_fail("start == end", kSourceFile, 0x147,
                              "int LinearBVHBuilderCPU::FindSplit(const "
                              "LinearBVHBuilderCPU::KeyIndexPair*, int, int)");
            }

            split = hi;
        }

        const int leftChild =
            LinearBVHBuilderCPUBuildRecursive(self, bvh, keys, bounds, start, split, depth + 1);
        const int rightChild =
            LinearBVHBuilderCPUBuildRecursive(self, bvh, keys, bounds, split, end, depth + 1);

        n.x = lower.x;
        n.y = lower.y;
        n.z = lower.z;
        n.i = leftChild;
        n.b = false;
        bvh.mNodeLowers[nodeIndex] = n;

        n.x = upper.x;
        n.y = upper.y;
        n.z = upper.z;
        n.i = rightChild;
        n.b = false;
        bvh.mNodeUppers[nodeIndex] = n;
    }

    return nodeIndex;
}

// ---------------------------------------------------------------------------
// Shared BVH lifetime helpers
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/bvh.cu:395 _Z7FreeBVHR3BVH
void FreeBVH(BVH& bvh)
{
    FlexDeviceFreeInternal(bvh.mNodeLowers);
    bvh.mNodeLowers = NULL;

    FlexDeviceFreeInternal(bvh.mNodeUppers);
    bvh.mNodeUppers = NULL;

    DestroyTexture(bvh.mNodeLowersTex);
    bvh.mNodeLowersTex = 0;

    DestroyTexture(bvh.mNodeUppersTex);
    bvh.mNodeUppersTex = 0;

    FlexDeviceFreeInternal(bvh.mRootNode);
}

// flexdecomp: src/cuda/bvh.cu:406 _Z8CloneBVHRK3BVHRS_
void CloneBVH(const BVH& hostBVH, BVH& deviceBVH)
{
    ResizeBVH(deviceBVH, hostBVH.mMaxNodes);

    cudaMemcpy(deviceBVH.mNodeLowers, hostBVH.mNodeLowers,
               hostBVH.mNumNodes * sizeof(PackedNodeHalf), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBVH.mNodeUppers, hostBVH.mNodeUppers,
               hostBVH.mNumNodes * sizeof(PackedNodeHalf), cudaMemcpyHostToDevice);

    cudaMemset(deviceBVH.mRootNode, 0, sizeof(int));
}

// ---------------------------------------------------------------------------
// GPU builder
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/bvh.cu:0 _ZN19LinearBVHBuilderGPUC1Ev  (line absent from DWARF)
// The shipped object's constructor body carries the complete-object (C1)
// symbol; the base-object form C2 sits at the same address as an alias.
extern "C" void LinearBVHBuilderGPUConstruct(LinearBVHBuilderGPU* this_)
    asm("_ZN19LinearBVHBuilderGPUC1Ev");

extern "C" void LinearBVHBuilderGPUConstruct(LinearBVHBuilderGPU* this_)
{
    this_->mIndices = NULL;
    this_->mKeys = NULL;
    this_->mDeltas = NULL;
    this_->mRangeLefts = NULL;
    this_->mRangeRights = NULL;
    this_->mNumChildren = NULL;
    this_->mTotalLower = NULL;
    this_->mTotalUpper = NULL;
    this_->mTotalInvEdges = NULL;
    this_->mMaxItems = 0;

    FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mTotalLower), sizeof(Vec3),
                             "&mTotalLower", kSourceFile, __LINE__);
    FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mTotalUpper), sizeof(Vec3),
                             "&mTotalUpper", kSourceFile, __LINE__);
    FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mTotalInvEdges), sizeof(Vec3),
                             "&mTotalInvEdges", kSourceFile, __LINE__);
}

extern "C" void LinearBVHBuilderGPUConstructC2(LinearBVHBuilderGPU* this_)
    asm("_ZN19LinearBVHBuilderGPUC2Ev") __attribute__((alias("_ZN19LinearBVHBuilderGPUC1Ev")));

// flexdecomp: src/cuda/bvh.cu:0 _ZN19LinearBVHBuilderGPUD1Ev  (line absent from DWARF)
extern "C" void LinearBVHBuilderGPUDestroy(LinearBVHBuilderGPU* this_)
    asm("_ZN19LinearBVHBuilderGPUD1Ev");

extern "C" void LinearBVHBuilderGPUDestroy(LinearBVHBuilderGPU* this_)
{
    FlexDeviceFreeInternal(this_->mIndices);
    FlexDeviceFreeInternal(this_->mKeys);
    FlexDeviceFreeInternal(this_->mDeltas);
    FlexDeviceFreeInternal(this_->mRangeLefts);
    FlexDeviceFreeInternal(this_->mRangeRights);
    FlexDeviceFreeInternal(this_->mNumChildren);
    FlexDeviceFreeInternal(this_->mTotalLower);
    FlexDeviceFreeInternal(this_->mTotalUpper);
    FlexDeviceFreeInternal(this_->mTotalInvEdges);
}

extern "C" void LinearBVHBuilderGPUDestroyC2(LinearBVHBuilderGPU* this_)
    asm("_ZN19LinearBVHBuilderGPUD2Ev") __attribute__((alias("_ZN19LinearBVHBuilderGPUD1Ev")));

// flexdecomp: src/cuda/bvh.cu:657 _ZN19LinearBVHBuilderGPU5BuildEP13NvFlexLibraryR3BVHPK8XVector4IfES7_PKiiP6Bounds
extern "C" void LinearBVHBuilderGPUBuild(LinearBVHBuilderGPU*, NvFlexLibrary* lib, BVH& bvh,
                                         const Vec4* itemLowers, const Vec4* itemUppers,
                                         const int* itemPriorities, int numItems,
                                         Bounds* totalBounds)
    asm("_ZN19LinearBVHBuilderGPU5BuildEP13NvFlexLibraryR3BVHPK8XVector4IfES7_PKiiP6Bounds");

extern "C" void LinearBVHBuilderGPUBuild(LinearBVHBuilderGPU* this_, NvFlexLibrary* lib, BVH& bvh,
                                         const Vec4* itemLowers, const Vec4* itemUppers,
                                         const int* itemPriorities, int numItems,
                                         Bounds* totalBounds)
{
    const int kNumThreadsPerBlock = 256;
    const int kNumBlocks = (numItems + kNumThreadsPerBlock - 1) / kNumThreadsPerBlock;

    ResizeBVH(bvh, numItems * 2);

    if (this_->mMaxItems < numItems)
    {
        const int itemsToAlloc = numItems * 3 / 2;
        const int nodesToAlloc = numItems * 3;

        FlexDeviceFreeInternal(this_->mIndices);
        FlexDeviceFreeInternal(this_->mKeys);
        FlexDeviceFreeInternal(this_->mDeltas);
        FlexDeviceFreeInternal(this_->mRangeLefts);
        FlexDeviceFreeInternal(this_->mRangeRights);
        FlexDeviceFreeInternal(this_->mNumChildren);

        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mIndices),
                                 itemsToAlloc * sizeof(int) * 2, "&mIndices", kSourceFile,
                                 __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mKeys),
                                 itemsToAlloc * sizeof(int) * 2, "&mKeys", kSourceFile, __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mDeltas),
                                 itemsToAlloc * sizeof(int), "&mDeltas", kSourceFile, __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mRangeLefts),
                                 nodesToAlloc * sizeof(int), "&mRangeLefts", kSourceFile,
                                 __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mRangeRights),
                                 nodesToAlloc * sizeof(int), "&mRangeRights", kSourceFile,
                                 __LINE__);
        FlexDeviceMallocInternal(reinterpret_cast<void**>(&this_->mNumChildren),
                                 nodesToAlloc * sizeof(int), "&mNumChildren", kSourceFile,
                                 __LINE__);

        this_->mMaxItems = itemsToAlloc;
    }

    const dim3 grid(kNumBlocks, 1, 1);
    const dim3 block(kNumThreadsPerBlock, 1, 1);

    if (totalBounds == NULL)
    {
        static Vec3 upper = MakeVec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        static Vec3 lower = MakeVec3(FLT_MAX, FLT_MAX, FLT_MAX);

        FD_CHECK_CUDA(cudaMemcpyAsync(this_->mTotalLower, &lower, sizeof(lower),
                                      cudaMemcpyHostToDevice, 0),
                      "cudaMemcpyAsync(mTotalLower, &lower, sizeof(lower), "
                      "cudaMemcpyHostToDevice)");
        FD_CHECK_CUDA(cudaMemcpyAsync(this_->mTotalUpper, &upper, sizeof(upper),
                                      cudaMemcpyHostToDevice, 0),
                      "cudaMemcpyAsync(mTotalUpper, &upper, sizeof(upper), "
                      "cudaMemcpyHostToDevice)");

        if (__cudaPushCallConfiguration(grid, block, 0, 0) == 0)
        {
            ComputeTotalBounds(itemLowers, itemUppers, this_->mTotalLower, this_->mTotalUpper,
                               numItems);
        }

        if (__cudaPushCallConfiguration(dim3(1, 1, 1), dim3(1, 1, 1), 0, 0) == 0)
        {
            ComputeTotalInvEdges(this_->mTotalLower, this_->mTotalUpper, this_->mTotalInvEdges);
        }
    }
    else
    {
        const float kEpsilon = 1e-4f;

        const Vec3 invEdges =
            MakeVec3(1.0f / (((totalBounds->upper).x - (totalBounds->lower).x) + kEpsilon),
                     1.0f / (((totalBounds->upper).y - (totalBounds->lower).y) + kEpsilon),
                     1.0f / (((totalBounds->upper).z - (totalBounds->lower).z) + kEpsilon));

        FD_CHECK_CUDA(cudaMemcpyAsync(this_->mTotalLower, &totalBounds->lower.x, sizeof(Vec3),
                                      cudaMemcpyHostToDevice, 0),
                      "cudaMemcpyAsync(mTotalLower, &totalBounds->lower.x, sizeof(Vec3), "
                      "cudaMemcpyHostToDevice)");
        FD_CHECK_CUDA(cudaMemcpyAsync(this_->mTotalUpper, &totalBounds->upper.x, sizeof(Vec3),
                                      cudaMemcpyHostToDevice, 0),
                      "cudaMemcpyAsync(mTotalUpper, &totalBounds->upper.x, sizeof(Vec3), "
                      "cudaMemcpyHostToDevice)");
        FD_CHECK_CUDA(cudaMemcpyAsync(this_->mTotalInvEdges, &invEdges.x, sizeof(Vec3),
                                      cudaMemcpyHostToDevice, 0),
                      "cudaMemcpyAsync(mTotalInvEdges, &invEdges.x, sizeof(Vec3), "
                      "cudaMemcpyHostToDevice)");
    }

    if (__cudaPushCallConfiguration(grid, block, 0, 0) == 0)
    {
        CalculateMortonCodes(itemLowers, itemUppers, itemPriorities, numItems,
                             this_->mTotalLower, this_->mTotalInvEdges, this_->mIndices,
                             this_->mKeys);
    }

    SortCellIndices(lib, this_->mKeys, this_->mIndices, numItems, 32);

    if (__cudaPushCallConfiguration(grid, block, 0, 0) == 0)
    {
        CalculateKeyDeltas(this_->mKeys, this_->mDeltas, numItems - 1);
    }

    if (__cudaPushCallConfiguration(grid, block, 0, 0) == 0)
    {
        BuildLeaves(itemLowers, itemUppers, numItems, this_->mIndices, this_->mRangeLefts,
                    this_->mRangeRights, bvh.mNodeLowers, bvh.mNodeUppers);
    }

    FD_CHECK_CUDA(cudaMemsetAsync(this_->mNumChildren, 0, numItems * 2 * sizeof(int), 0),
                  "cudaMemsetAsync(mNumChildren, 0, sizeof(int)*maxNodes)");

    if (__cudaPushCallConfiguration(grid, block, 0, 0) == 0)
    {
        BuildHierarchy(numItems, bvh.mRootNode, this_->mDeltas, this_->mNumChildren,
                       this_->mRangeLefts, this_->mRangeRights, bvh.mNodeLowers,
                       bvh.mNodeUppers);
    }
}
