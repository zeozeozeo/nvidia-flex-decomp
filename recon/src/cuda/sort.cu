// FleX 1.2.0 — src/cuda/sort.cu (reconstructed)
//
// Module:  nvflex.sort  (original object: NvFlex*CUDA_x64.a:cudasort.o)
//
// Evidence
//   * DWARF:      work/dwarf/nvflex.sort.json (CU entries for sort.cu, line 42/50)
//   * pseudocode: work/decomp/nvflex.sort/cudasort.c
//   * disassembly: work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/cudasort.disasm.txt
//   * device code: work/binary/gpu/dbg/nvflex.sort/sm_30.ptx (.entry names/param lists)
//   * literals:   .rodata.str1.1 / .rodata.str1.8 of cudasort.o
//
// The shipped translation unit is two host functions that drive
// cub::DeviceRadixSort::SortPairs<int,int> plus the CUB 1.3.2 radix-sort
// kernel instantiations pulled in by that call.
//
// CUB 1.3.2 cannot be used here: its device code emits shfl/vote without
// .sync, which ptxas rejects for every target >= sm_70 (and CUDA 13.4 only
// supports sm_75 and newer).  The kernels are therefore reimplemented below
// under the *original* entry-point names so that the object keeps the same
// host stub symbols and the fatbin keeps the same .entry set.  This is the
// plan's documented contingency for nvflex.sort (task 5.3); the device
// implementation is verified by G1 here and must be confirmed by G3 on a GPU
// host (see units/nvflex.sort/*.md).  The host side is bit-exact: same calls,
// same literals, same line numbers, same temp-storage protocol.
//
// nvflex/all.h pulls the recovered internal headers; under __CUDACC__ those
// include <cuda_runtime.h> themselves, so the CUDA types are the real ones.
#include "nvflex/all.h"
#include "NvFlex.h"

// The recovered NvFlexLibrary keeps its members as layout-only storage (the
// generator re-emits members whose named type cannot reproduce the DWARF
// offsets as byte arrays).  These overloads give the two members this file
// touches their DWARF types whether they arrive typed or as opaque storage.
namespace
{
template <unsigned long N>
inline void*& SortTempSlot(unsigned char (&storage)[N]) { return *reinterpret_cast<void**>(storage); }
inline void*& SortTempSlot(void*& storage) { return storage; }

template <unsigned long N>
inline int& SortTempSizeSlot(unsigned char (&storage)[N]) { return *reinterpret_cast<int*>(storage); }
inline int& SortTempSizeSlot(int& storage) { return storage; }
} // namespace

// FlexLogFunc is defined by util.cpp and declared in the original src/cuda/util.h
// (DWARF: work/dwarf/nvflex.util.json, symbol _Z11FlexLogFuncPFv19NvFlexErrorSeverityPKcS1_iES_S1_S1_iz).
// It is not part of the recovered headers yet — integrator: hoist this
// declaration into recon/include/nvflex/util.h.
extern void FlexLogFunc(NvFlexErrorCallback func, NvFlexErrorSeverity type,
                        const char* msg, const char* file, int line, ...);

// ---------------------------------------------------------------------------
// CUB compatibility surface
//
// Only the parts that appear in cudasort.o's symbols/layouts are reconstructed:
// cub::DoubleBuffer<T> (3 words, selector protocol) and
// cub::GridEvenShare<int> (9 ints, 36 bytes — matches the PTX kernel parameter
// .param .align 4 .b8 ...[36] of the upsweep/downsweep entries).
// ---------------------------------------------------------------------------
namespace cub
{

template <typename T>
struct DoubleBuffer
{
    T*  d_buffers[2];
    int selector;

    __host__ __device__ __forceinline__ DoubleBuffer()
        : selector(0)
    {
        d_buffers[0] = 0;
        d_buffers[1] = 0;
    }

    __host__ __device__ __forceinline__ DoubleBuffer(T* d_current, T* d_alternate)
        : selector(0)
    {
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    __host__ __device__ __forceinline__ T* Current() { return d_buffers[selector]; }
};

// cub/grid/grid_even_share.cuh — same field order, same 36-byte layout.
template <typename OffsetT>
struct GridEvenShare
{
    OffsetT total_grains;
    int     big_blocks;
    OffsetT big_share;
    OffsetT normal_share;
    OffsetT normal_base_offset;
    OffsetT num_items;
    int     grid_size;
    OffsetT block_offset;
    OffsetT block_end;

    __host__ __device__ __forceinline__ GridEvenShare()
        : total_grains(0), big_blocks(0), big_share(0), normal_share(0),
          normal_base_offset(0), num_items(0), grid_size(0),
          block_offset(0), block_end(0) {}

    __host__ __device__ __forceinline__ GridEvenShare(OffsetT num_items_,
                                                      int     max_grid_size,
                                                      int     schedule_granularity)
    {
        num_items          = num_items_;
        block_offset       = num_items_;
        block_end          = num_items_;
        total_grains       = (num_items_ + schedule_granularity - 1) / schedule_granularity;
        grid_size          = int(total_grains < max_grid_size ? total_grains : max_grid_size);
        OffsetT grains_per_block = total_grains / grid_size;
        big_blocks         = int(total_grains - (grains_per_block * grid_size));
        normal_share       = grains_per_block * schedule_granularity;
        normal_base_offset = big_blocks * schedule_granularity;
        big_share          = normal_share + schedule_granularity;
    }

    __device__ __forceinline__ void Init(int partition_id)
    {
        if (partition_id < big_blocks)
        {
            block_offset = partition_id * big_share;
            block_end    = block_offset + big_share;
        }
        else if (partition_id < total_grains)
        {
            block_offset = normal_base_offset + (partition_id * normal_share);
            block_end    = block_offset + normal_share;
            if (block_end > num_items) block_end = num_items;
        }
    }

    __device__ __forceinline__ void BlockInit() { Init(int(blockIdx.x)); }
};

struct DeviceRadixSort
{
    template <typename Key, typename Value>
    static cudaError_t SortPairs(void*               d_temp_storage,
                                 size_t&              temp_storage_bytes,
                                 DoubleBuffer<Key>&   d_keys,
                                 DoubleBuffer<Value>& d_values,
                                 int                  num_items,
                                 int                  begin_bit         = 0,
                                 int                  end_bit           = sizeof(Key) * 8,
                                 cudaStream_t         stream            = 0,
                                 bool                 debug_synchronous = false);
};

} // namespace cub

// ---------------------------------------------------------------------------
// Device radix sort (CUB 1.3.2 replacement)
//
// Tuning: 256 threads/block, 5-bit digits (32 bins).  The spine is striped
// bin-major: d_spine[bin * gridDim.x + block] — the layout the original PTX
// uses in the upsweep epilogue.  The dispatcher below owns the grid size, so
// the layout is internal to this translation unit.
// ---------------------------------------------------------------------------
namespace
{

enum
{
    kBlockThreads = 256,
    kItemsPerThread = 4,
    kTileItems = kBlockThreads * kItemsPerThread,
    kRadixBits = 5,
    kRadixDigits = 1 << kRadixBits,
    kMaxGridSize = 1024
};

__device__ __forceinline__ int RadixDigit(int key, int current_bit, int mask)
{
    return (key >> current_bit) & mask;
}

// Privatized per-block digit histograms -> d_spine[block][bin].
__device__ __forceinline__ void RadixUpsweep(int* d_keys, int* d_spine,
                                            int current_bit, int num_bits,
                                            cub::GridEvenShare<int> even_share)
{
    __shared__ int s_hist[kRadixDigits];

    if (threadIdx.x < kRadixDigits)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    even_share.BlockInit();
    const int mask = (1 << num_bits) - 1;

    for (int i = even_share.block_offset + int(threadIdx.x); i < even_share.block_end; i += int(blockDim.x))
        atomicAdd(&s_hist[RadixDigit(d_keys[i], current_bit, mask)], 1);

    __syncthreads();

    // striped layout: d_spine[bin * gridDim.x + block]; the original PTX
    // epilogue is `d_spine[gridDim.x * bin_idx + blockIdx.x] = bin_count`
    if (threadIdx.x < kRadixDigits)
        d_spine[int(threadIdx.x) * int(gridDim.x) + int(blockIdx.x)] = s_hist[threadIdx.x];
}

// Exclusive prefix sum over the striped spine (single block).
//
// The spine is bin-major (bin * num_blocks + block), so the global start of a
// (bin, block) region is the prefix sum of everything before it: the totals of
// all lower bins plus the preceding blocks of the same bin.  That is exactly
// what CUB's single-block spine scan produces.
__device__ __forceinline__ void RadixScan(int* d_spine, int num_counts)
{
    if (blockIdx.x > 0)
        return;

    __shared__ int s_bin_base[kRadixDigits];

    const int num_blocks = num_counts / kRadixDigits;

    if (threadIdx.x < kRadixDigits)
    {
        const int bin = int(threadIdx.x);
        int total = 0;
        for (int b = 0; b < num_blocks; ++b)
            total += d_spine[bin * num_blocks + b];
        s_bin_base[bin] = total;
    }
    __syncthreads();

    if (threadIdx.x == 0)
    {
        int running = 0;
        for (int d = 0; d < kRadixDigits; ++d)
        {
            const int total = s_bin_base[d];
            s_bin_base[d] = running;
            running += total;
        }
    }
    __syncthreads();

    if (threadIdx.x < kRadixDigits)
    {
        const int bin = int(threadIdx.x);
        int running = s_bin_base[bin];
        for (int b = 0; b < num_blocks; ++b)
        {
            const int idx = bin * num_blocks + b;
            const int count = d_spine[idx];
            d_spine[idx] = running;
            running += count;
        }
    }
}

// Stable scatter into the bin ranges produced by RadixScan.
// One lane owns one digit: it walks the block's items in input order, so the
// output is stable exactly like CUB's downsweep.
__device__ __forceinline__ void RadixDownsweep(int* d_keys_in, int* d_keys_out,
                                               int* d_values_in, int* d_values_out,
                                               int* d_spine,
                                               int current_bit, int num_bits,
                                               cub::GridEvenShare<int> even_share)
{
    even_share.BlockInit();
    const int mask = (1 << num_bits) - 1;

    if (threadIdx.x < kRadixDigits)
    {
        const int bin = int(threadIdx.x);
        int pos = d_spine[bin * int(gridDim.x) + int(blockIdx.x)];

        for (int i = even_share.block_offset; i < even_share.block_end; ++i)
        {
            const int key = d_keys_in[i];
            if (RadixDigit(key, current_bit, mask) == bin)
            {
                d_keys_out[pos] = key;
                d_values_out[pos] = d_values_in[i];
                ++pos;
            }
        }
    }
}

} // namespace

// ---- kernel entry points --------------------------------------------------
// The entry names are the original CUB 1.3.2 instantiations (nm/PTX); the
// bodies are the reimplementation above.  `visibility("default")` is required:
// nvcc 13 marks generated launch wrappers hidden while the shipped CUDA 9.2
// objects export them (GLOBAL DEFAULT), and libnvflex.so must not lose them.

// Readability aliases; the mangled names above are fixed by ABI parity.
#define FLEX_SORT_KERNEL_EMPTY   _ZN3cub11EmptyKernelIvEEvv
#define FLEX_SORT_KERNEL_SCAN    _ZN3cub19RadixSortScanKernelINS_23DeviceRadixSortDispatchILb0EiiiE13PtxScanPolicyEiEEvPT0_i
#define FLEX_SORT_KERNEL_UPSWEEP _ZN3cub22RadixSortUpsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE16PtxUpsweepPolicyELb0EiiEEvPT1_PT2_S6_iibNS_13GridEvenShareIS6_EE
#define FLEX_SORT_KERNEL_UPSWEEP_ALT _ZN3cub22RadixSortUpsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE19PtxAltUpsweepPolicyELb0EiiEEvPT1_PT2_S6_iibNS_13GridEvenShareIS6_EE
#define FLEX_SORT_KERNEL_DOWNSWEEP _ZN3cub24RadixSortDownsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE18PtxDownsweepPolicyELb0EiiiEEvPT1_S5_PT2_S7_PT3_S8_iibbNS_13GridEvenShareIS8_EE
#define FLEX_SORT_KERNEL_DOWNSWEEP_ALT _ZN3cub24RadixSortDownsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE21PtxAltDownsweepPolicyELb0EiiiEEvPT1_S5_PT2_S7_PT3_S8_iibbNS_13GridEvenShareIS8_EE

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub11EmptyKernelIvEEvv()
{
}

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub22RadixSortUpsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE16PtxUpsweepPolicyELb0EiiEEvPT1_PT2_S6_iibNS_13GridEvenShareIS6_EE(
    int* d_keys, int* d_spine, int num_items, int current_bit, int num_bits,
    bool first_pass, cub::GridEvenShare<int> even_share)
{
    (void)num_items;   // the block's item range comes from even_share
    (void)first_pass;  // CUB reads the input ping buffer on every pass
    RadixUpsweep(d_keys, d_spine, current_bit, num_bits, even_share);
}

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub22RadixSortUpsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE19PtxAltUpsweepPolicyELb0EiiEEvPT1_PT2_S6_iibNS_13GridEvenShareIS6_EE(
    int* d_keys, int* d_spine, int num_items, int current_bit, int num_bits,
    bool first_pass, cub::GridEvenShare<int> even_share)
{
    (void)num_items;   // the block's item range comes from even_share
    (void)first_pass;  // CUB reads the input ping buffer on every pass
    RadixUpsweep(d_keys, d_spine, current_bit, num_bits, even_share);
}

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub19RadixSortScanKernelINS_23DeviceRadixSortDispatchILb0EiiiE13PtxScanPolicyEiEEvPT0_i(
    int* d_spine, int num_counts)
{
    RadixScan(d_spine, num_counts);
}

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub24RadixSortDownsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE18PtxDownsweepPolicyELb0EiiiEEvPT1_S5_PT2_S7_PT3_S8_iibbNS_13GridEvenShareIS8_EE(
    int* d_keys_in, int* d_keys_out, int* d_values_in, int* d_values_out, int* d_spine,
    int num_items, int current_bit, int num_bits, bool first_pass, bool last_pass,
    cub::GridEvenShare<int> even_share)
{
    (void)num_items;    // the block's item range comes from even_share
    (void)first_pass;   // CUB reads the input ping buffer on every pass
    (void)last_pass;    // the dispatcher always writes both keys and values
    RadixDownsweep(d_keys_in, d_keys_out, d_values_in, d_values_out, d_spine,
                   current_bit, num_bits, even_share);
}

#if defined(_MSC_VER)
extern "C" __global__ void
#else
extern "C" __global__ __attribute__((visibility("default"))) void
#endif
_ZN3cub24RadixSortDownsweepKernelINS_23DeviceRadixSortDispatchILb0EiiiE21PtxAltDownsweepPolicyELb0EiiiEEvPT1_S5_PT2_S7_PT3_S8_iibbNS_13GridEvenShareIS8_EE(
    int* d_keys_in, int* d_keys_out, int* d_values_in, int* d_values_out, int* d_spine,
    int num_items, int current_bit, int num_bits, bool first_pass, bool last_pass,
    cub::GridEvenShare<int> even_share)
{
    (void)num_items;    // the block's item range comes from even_share
    (void)first_pass;   // CUB reads the input ping buffer on every pass
    (void)last_pass;    // the dispatcher always writes both keys and values
    RadixDownsweep(d_keys_in, d_keys_out, d_values_in, d_values_out, d_spine,
                   current_bit, num_bits, even_share);
}

// ---- host dispatch --------------------------------------------------------
// Same ABI as cub::DeviceRadixSort::SortPairs<int,int> (weak, header template).
// The shipped object keeps this instantiation out of line (0x41e1 bytes, weak
// symbol), and so does the reconstruction: SortCellIndices must stay a thin
// driver, not an inlined copy of the dispatch.
template <typename Key, typename Value>
#if !defined(_MSC_VER)
__attribute__((noinline, noclone))
#endif
cudaError_t cub::DeviceRadixSort::SortPairs(void*                d_temp_storage,
                                            size_t&              temp_storage_bytes,
                                            DoubleBuffer<Key>&   d_keys,
                                            DoubleBuffer<Value>& d_values,
                                            int                  num_items,
                                            int                  begin_bit,
                                            int                  end_bit,
                                            cudaStream_t         stream,
                                            bool                 debug_synchronous)
{
    // RADIX_BITS-bit digits per pass; CUB 1.3.2 uses 5 on this policy set.
    const bool sorts = (num_items > 1 && end_bit > begin_bit);

    int grid_size = (num_items + kTileItems - 1) / kTileItems;
    if (grid_size > kMaxGridSize) grid_size = kMaxGridSize;
    if (grid_size < 1) grid_size = 1;

    const size_t required = size_t(grid_size) * kRadixDigits * sizeof(int);
    if (d_temp_storage == 0)
    {
        temp_storage_bytes = required;
        return cudaSuccess;
    }

    if (!sorts)
    {
        // Nothing to sort (or no significant bits): matched CUB's dummy launch,
        // which also surfaces sticky errors on the stream.
        FLEX_SORT_KERNEL_EMPTY<<<1, 1, 0, stream>>>();
        return cudaGetLastError();
    }

    int* d_spine = static_cast<int*>(d_temp_storage);
    cub::GridEvenShare<int> even_share(num_items, grid_size, kTileItems);

    cudaError_t error = cudaSuccess;
    int current_bit = begin_bit;

    for (int pass = 0; current_bit < end_bit; ++pass)
    {
        // 5-bit digits, the last pass may carry fewer significant bits
        const int bits_remaining = end_bit - current_bit;
        const int num_bits = bits_remaining < kRadixBits ? bits_remaining : kRadixBits;
        const int selector = d_keys.selector;
        Key*   keys_in = d_keys.d_buffers[selector];
        Key*   keys_out = d_keys.d_buffers[selector ^ 1];
        Value* values_in = d_values.d_buffers[selector];
        Value* values_out = d_values.d_buffers[selector ^ 1];
        const bool first_pass = (pass == 0);
        const bool last_pass = (current_bit + num_bits >= end_bit);

        // CUB's dispatch picks between the primary and Alt policies by SM
        // version; the reconstruction runs both entries alternately so each
        // one stays instantiated in the fatbin and is exercised at runtime.

        if (pass & 1)
        {
            FLEX_SORT_KERNEL_UPSWEEP_ALT<<<grid_size, kBlockThreads, 0, stream>>>(
                    reinterpret_cast<int*>(keys_in), d_spine, num_items, current_bit, num_bits,
                    first_pass, even_share);
        }
        else
        {
            FLEX_SORT_KERNEL_UPSWEEP<<<grid_size, kBlockThreads, 0, stream>>>(
                    reinterpret_cast<int*>(keys_in), d_spine, num_items, current_bit, num_bits,
                    first_pass, even_share);
        }

        FLEX_SORT_KERNEL_SCAN<<<1, kRadixDigits, 0, stream>>>(d_spine,
                                                             grid_size * kRadixDigits);

        if (pass & 1)
        {
            FLEX_SORT_KERNEL_DOWNSWEEP_ALT<<<grid_size, kBlockThreads, 0, stream>>>(
                    reinterpret_cast<int*>(keys_in), reinterpret_cast<int*>(keys_out),
                    reinterpret_cast<int*>(values_in), reinterpret_cast<int*>(values_out),
                    d_spine, num_items, current_bit, num_bits, first_pass, last_pass,
                    even_share);
        }
        else
        {
            FLEX_SORT_KERNEL_DOWNSWEEP<<<grid_size, kBlockThreads, 0, stream>>>(
                    reinterpret_cast<int*>(keys_in), reinterpret_cast<int*>(keys_out),
                    reinterpret_cast<int*>(values_in), reinterpret_cast<int*>(values_out),
                    d_spine, num_items, current_bit, num_bits, first_pass, last_pass,
                    even_share);
        }

        error = cudaGetLastError();
        if (error != cudaSuccess)
            return error;

        d_keys.selector ^= 1;
        d_values.selector ^= 1;
        current_bit += num_bits;
    }

    if (debug_synchronous)
        error = cudaStreamSynchronize(stream);

    return error;
}

// ---------------------------------------------------------------------------
// Host entry points (original source order: sort.cu:42, sort.cu:50)
// ---------------------------------------------------------------------------

// flexdecomp: src/cuda/sort.cu:42 _Z9SortResetP13NvFlexLibrary
void SortReset(NvFlexLibrary* lib)
{
    cudaError_t err = cudaFree(SortTempSlot(lib->mSortTemp));

    NvFlexErrorCallback errorFunc = *reinterpret_cast<NvFlexErrorCallback*>(lib->mErrorFunc);

    if (err != cudaSuccess && errorFunc)
        FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
                    "../../../src/cuda/sort.cu", 44 /* .disasm: SortReset line imm */, err,
                    "cudaFree(lib->mSortTemp)");

    SortTempSizeSlot(lib->mSortMaxTempSize) = 0;
    SortTempSlot(lib->mSortTemp) = 0;
}

// flexdecomp: src/cuda/sort.cu:50 _Z15SortCellIndicesP13NvFlexLibraryPiS1_ii
void SortCellIndices(NvFlexLibrary* lib, int* keys, int* values, int n, int numBits)
{
    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    size_t sortTempSize = 0;
    cub::DeviceRadixSort::SortPairs(0, sortTempSize, d_keys, d_values, n, 0, numBits);

    NvFlexErrorCallback errorFunc = *reinterpret_cast<NvFlexErrorCallback*>(lib->mErrorFunc);

    if (size_t(SortTempSizeSlot(lib->mSortMaxTempSize)) < sortTempSize)
    {
        SortReset(lib);

        cudaError_t err = cudaMalloc(&SortTempSlot(lib->mSortTemp), sortTempSize);
        if (err != cudaSuccess && errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
                        "../../../src/cuda/sort.cu", 63 /* .disasm: cudaMalloc line imm */, err,
                        "cudaMalloc(&lib->mSortTemp, sortTempSize)");

        SortTempSizeSlot(lib->mSortMaxTempSize) = int(sortTempSize);
    }

    cub::DeviceRadixSort::SortPairs(SortTempSlot(lib->mSortTemp), sortTempSize, d_keys, d_values,
                                    n, 0, numBits);

    if (keys != d_keys.Current())
    {
        cudaError_t err = cudaMemcpyAsync(keys, d_keys.Current(), sizeof(int) * n,
                                          cudaMemcpyDeviceToDevice, 0);
        if (err != cudaSuccess && errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
                        "../../../src/cuda/sort.cu", 71 /* .disasm: keys copy line imm */, err,
                        "cudaMemcpyAsync(keys, d_keys.Current(), sizeof(int)*n, cudaMemcpyDeviceToDevice)");
    }

    if (values != d_values.Current())
    {
        cudaError_t err = cudaMemcpyAsync(values, d_values.Current(), sizeof(int) * n,
                                          cudaMemcpyDeviceToDevice, 0);
        if (err != cudaSuccess && errorFunc)
            FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
                        "../../../src/cuda/sort.cu", 74 /* .disasm: values copy line imm */, err,
                        "cudaMemcpyAsync(values, d_values.Current(), sizeof(int)*n, cudaMemcpyDeviceToDevice)");
    }
}
