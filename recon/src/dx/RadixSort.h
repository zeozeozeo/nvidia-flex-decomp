// flexdecomp — D3D backend: radixsort.obj.
//
// Radix sort over 32-bit keys, used by the D3D solver for the particle hash /
// BVH reorder passes.  This header is the entry point `flex.obj` (and `bvh.obj`)
// use:
//
//   NvFlex::RadixSort* sort = NvFlex::createRadixSort(context, &desc);
//   sort->sort(context, &params, timers);        // one 4-bit radix pass set
//   NvFlex::RadixSortBuffer in  = sort->getInputBuffer();   // upload here
//   NvFlex::RadixSortBuffer out = sort->getOutputBuffer();  // read here
//   sort->release();
//
// Evidence
//   FleX/bin/win64/NvFlexReleaseD3D_x64.dll.pdb, module 0003 `radixsort.obj`:
//     ...\src\compiler\vc12win64\build\win64\D3D\Core\release\radixsort.obj
//   PDB field lists (raw `llvm-pdbutil dump -types`):
//     0x2C10 RadixSortDesc   { unsigned maxSortBlocks @0 }
//     0x2CFA RadixSortParams { unsigned numKeys @0; unsigned numSortBlocks @4;
//                              unsigned bits @8 }
//     0x2C0E RadixSortBuffer { Buffer* key @0; Buffer* val @8 }
//     0x2CCD RadixSort       sizeof 8, vtable {addRef, release,
//                              getInputBuffer @16, getOutputBuffer @24,
//                              sort @32}
//     0x44A1 RadixSortImpl   bases Object@0, RadixSort@16, members
//                              m_desc@24, m_front@32, m_back@64,
//                              m_front_counters@96, m_back_counters@112,
//                              m_constantBuffer@128, m_radixSortBlockCS@136,
//                              m_radixSort1CS@144, m_radixSort2CS@152,
//                              m_radixSort3CS@160; sizeof 168.
//
// GPU passes (shader interface verified against the shipped DXBC):
//   m_radixSort1CS      080 RadixSortHistogram  grid (n,1,1)  t0 key4
//                                                            u0 counters
//   m_radixSort2CS      081 RadixSortScan       grid (2,1,1)  t0 counter4
//                                                            u0 counter4
//   m_radixSort3CS      083 RadixSortSortPairs  grid (n,1,1)  t0 key4 t1 val4
//                                                            t2 counters
//                                                            u0 key u1 val
//   m_radixSortBlockCS  082 RadixSortSort       grid (1,1,1)  t0 key4 t1 val4
//                                                            u0 key4 u1 val4
// (n = numSortBlocks; the single-block path is taken when numSortBlocks == 1.)
#pragma once

#include "Frame.h"

namespace NvFlex {

struct RadixSortDesc
{
    unsigned maxSortBlocks;   // PDB 4 bytes
};

struct RadixSortParams
{
    unsigned numKeys;         // number of keys to sort
    unsigned numSortBlocks;   // >= 1; each block covers 1024 keys
    unsigned bits;            // significand bits, consumed 4 at a time
};

struct RadixSortBuffer
{
    Buffer* key;              // uint keys
    Buffer* val;              // uint payload
};

// Abstract sort handle returned to the solver.  `getInputBuffer()` and
// `getOutputBuffer()` both report the *front* buffer: the sort is double
// buffered internally and publishes the result by swapping on completion, so
// the caller uploads into and reads from the same handle before/after `sort()`.
// Evidence: both accessors at 0x18004d3a0 / 0x18004d3d0 load identically from
// m_front.key / m_front.val.
class RadixSort : public RefCounted
{
public:
    virtual RadixSortBuffer getInputBuffer() = 0;
    virtual RadixSortBuffer getOutputBuffer() = 0;
    // Runs `bits` significand bits, 4 bits per pass.
    virtual void sort(Context* context, const RadixSortParams* params,
                      DxTimers* timers) = 0;
};

// Implementation.  Not part of the interface `flex.obj` sees; declared here
// because the DLL exports no type information and the module owns it.
class RadixSortImpl : public RadixSort
{
public:
    // PDB 0x44AA / 0x44A6: the two ping-pong halves.
    struct Side
    {
        Buffer* key;
        Buffer* val;
        Buffer* key4;   // r32g32b32a32_uint view of key
        Buffer* val4;   // r32g32b32a32_uint view of val
        void release();
    };

    struct Counter
    {
        Buffer* counter;
        Buffer* counter4;   // r32g32b32a32_uint view
        void release();
    };

    static void swap(Side& a, Side& b);
    static void swap(Counter& a, Counter& b);

    RadixSortImpl(Context* context, const RadixSortDesc* desc);

    // RadixSort
    unsigned addRef();
    unsigned release();
    RadixSortBuffer getInputBuffer();
    RadixSortBuffer getOutputBuffer();
    void sort(Context* context, const RadixSortParams* params, DxTimers* timers);

    // Bone-yard allocator: objects are created/freed through the Flex hooks so
    // an application-supplied allocator (NvFlexSetMallocFunc) round-trips.
    static void* operator new(size_t size);
    static void  operator delete(void* ptr);

private:
    ~RadixSortImpl();

    RadixSortImpl(const RadixSortImpl&);
    RadixSortImpl& operator=(const RadixSortImpl&);

    std::atomic<unsigned> m_refCount;
    RadixSortDesc         m_desc;
    Side                  m_front;
    Side                  m_back;
    Counter               m_front_counters;
    Counter               m_back_counters;
    ConstantBuffer*       m_constantBuffer;
    ComputeShader*        m_radixSortBlockCS;
    ComputeShader*        m_radixSort1CS;
    ComputeShader*        m_radixSort2CS;
    ComputeShader*        m_radixSort3CS;
};

// Factory used by the solver: allocates a RadixSortImpl of
// maxSortBlocks*1024 keys and returns its RadixSort interface.
RadixSort* createRadixSort(Context* context, const RadixSortDesc* desc);

} // namespace NvFlex

// ---------------------------------------------------------------------------
// Block sizing helpers (radixsort.obj 0x18004d290 / 0x18004d2b0 / 0x18004d2d0).
// One sort block covers 1024 keys; `GetPaddedNumBlocks` rounds the padded key
// count to 256-entry groups (the histogram's thread-block granularity).
// `inline` so every translation unit that needs them (flex.obj, bvh.obj) gets
// one definition instead of a symbol collision.
// ---------------------------------------------------------------------------

inline int GetSortSize(int numKeys)
{
    return (((numKeys & 0x3ff) != 0) + (unsigned(numKeys) >> 10)) << 10;
}

inline int GetPaddedNumBlocksRadixSort(int numKeys)
{
    return (GetSortSize(numKeys) + 0xff) >> 8;
}

inline int GetNumBlocksRadixSort(int numKeys)
{
    return (numKeys + 0x3ff) >> 10;
}
