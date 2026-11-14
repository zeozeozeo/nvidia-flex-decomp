// Reconstructed FleX 1.2.0 CUDA solver - host layer, part 1.
//
// This translation unit holds the earliest host units of the shipped
// cudaflex.o (source order: flex.cu:263 .. flex.cu:5324 plus the util.h
// inline timer helpers that cudaflex.o emits as weak definitions).
//
// Evidence used for every unit:
//   work/decomp/nvflex.flex/cudaflex.c            Ghidra pseudocode, by address
//   work/dwarf/nvflex.flex.json                   DIEs: decl file/line, offsets
//   work/binary/obj/dbg/NvFlexDebugCUDA_x64/
//       dump/cudaflex.disasm.txt                  objdump -drw
//   recon/include/nvflex/recovered.h              exact recovered layouts
//
// The anonymous-namespace helpers of the original carry the CUDA-generated
// wrapper namespace name in their mangling; the namespace is spelled out so
// that the reconstructed object reproduces the shipped symbols exactly.

#include <cuda_runtime.h>

#include "nvflex/all.h"
#include "NvFlex.h"

#include <cstring>
#include <map>
#include <new>
#include <vector>

// ---- helpers defined in other translation units ---------------------------
// FlexLogFunc lives in util.cpp (nvflex.util), NvFlexSetFence in this module.
extern void FlexLogFunc(NvFlexErrorCallback callback, NvFlexErrorSeverity type,
                        const char* msg, const char* file, int line, ...);
void NvFlexSetFence(NvFlexLibrary* lib);

// The shipped literal is the build-relative path of the original source;
// __FILE__ would not reproduce it and the static gate compares literal sets.
static const char* const kFlexSourceFile = "../../../src/cuda/flex.cu";

// NvFlexLibrary::mErrorFunc is recorded as an opaque 8-byte member by the
// DWARF recovery (it is an NvFlexErrorCallback); read it through memcpy.
static NvFlexErrorCallback libErrorFunc(NvFlexLibrary* lib)
{
    void* fn = 0;
    std::memcpy(&fn, lib->mErrorFunc, sizeof(fn));
    return reinterpret_cast<NvFlexErrorCallback>(fn);
}

// The namespace spelling must match the referencing TUs' declarations: on ELF
// the .set aliases bridge the debug (d59) spelling to the release (f25) one;
// on PE there are no aliases, so the definition uses the f25 spelling.
#if defined(_MSC_VER)
namespace _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207 {
#else
namespace _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207 {
#endif

// flexdecomp: flex.cu:263 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320715ExecuteCallbackEP12NvFlexSolver25NvFlexSolverCallbackStagef
void ExecuteCallback(NvFlexSolver* solver, NvFlexSolverCallbackStage stage, float dt)
{
    // NvFlexSolver::mCallbacks is recovered as an 80-byte opaque blob holding
    // five NvFlexSolverCallback records (16 B each, DWARF: NvFlexSolverCallback).
    NvFlexSolverCallback* callbacks =
        reinterpret_cast<NvFlexSolverCallback*>(solver->mCallbacks);
    NvFlexSolverCallback& callback = callbacks[int(stage)];

    if (callback.function)
    {
        NvFlexSolverCallbackParams params;
        params.solver = solver;
        params.userData = callback.userData;
        params.particles = reinterpret_cast<float*>(solver->mSortedNewPositions);
        params.velocities = reinterpret_cast<float*>(solver->mSortedNewVelocities);
        params.phases = solver->mSortedPhases;
        params.numActive = solver->numParticles;
        params.dt = dt;
        params.originalToSortedMap = solver->mReverseLookup;
        params.sortedToOriginalMap = solver->mIndices;

        callback.function(params);
    }
}

// flexdecomp: flex.cu:5200 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPvi
void* ByteOffset(void* ptr, int byteOffset)
{
    return static_cast<char*>(ptr) + byteOffset;
}

// flexdecomp: flex.cu:5201 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPKvi
void* ByteOffset(const void* ptr, int byteOffset)
{
    return static_cast<char*>(const_cast<void*>(ptr)) + byteOffset;
}

// flexdecomp: flex.cu:5204 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32078ReadDataEP12NvFlexBufferiPKvii
void ReadData(NvFlexBuffer* dstBuffer, int dstByteOffset, const void* srcPtr,
              int srcByteOffset, int byteSize)
{
    NvFlexLibrary* lib = dstBuffer->lib;

    if (dstBuffer->mapped)
    {
        if (libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                        "Reading from a mapped buffer, buffers must be unmapped before being passed to flexGet*() methods",
                        kFlexSourceFile, 0x145a);
        }
        return;
    }

    // flex.cu:5219 - host buffers are device->host, everything else device->device
    cudaMemcpyKind kind = dstBuffer->type == eNvFlexBufferHost
                              ? cudaMemcpyDeviceToHost
                              : cudaMemcpyDeviceToDevice;

    if (dstBuffer->type == 3)
    {
        size_t s = 0;

        cudaError_t err = cudaGraphicsMapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&dstBuffer->graphicsResource), 0);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x146a, err,
                        "cudaGraphicsMapResources(1, &dstBuffer->graphicsResource)");
        }

        err = cudaGraphicsResourceGetMappedPointer(
            &dstBuffer->ptr, &s, reinterpret_cast<cudaGraphicsResource_t>(dstBuffer->graphicsResource));
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x146b, err,
                        "cudaGraphicsResourceGetMappedPointer(&dstBuffer->ptr, &s, dstBuffer->graphicsResource)");
        }
    }

    void* dstPtr = dstBuffer->ptr;

    cudaError_t err = cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset),
                                      ByteOffset(srcPtr, srcByteOffset),
                                      byteSize, kind, 0);
    if (err != cudaSuccess && libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                    kFlexSourceFile, 0x1472, err,
                    "cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset), ByteOffset(srcPtr, srcByteOffset), byteSize, kind)");
    }

    if (dstBuffer->type == 3)
    {
        err = cudaGraphicsUnmapResources(1, reinterpret_cast<cudaGraphicsResource_t*>(&dstBuffer->graphicsResource), 0);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1477, err,
                        "cudaGraphicsUnmapResources(1, &dstBuffer->graphicsResource)");
        }
    }

    NvFlexSetFence(lib);
}

// flexdecomp: flex.cu:5245 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPviP12NvFlexBufferii
void WriteData(void* dstPtr, int dstByteOffset, NvFlexBuffer* srcBuffer,
               int srcByteOffset, int byteSize)
{
    NvFlexLibrary* lib = srcBuffer->lib;

    if (!srcBuffer->mapped)
    {
        // the shipped code binds the const overload here (both ByteOffset
        // overloads appear in the original call set)
        const void* srcPtr = srcBuffer->ptr;

        cudaMemcpyKind kind = srcBuffer->type == eNvFlexBufferHost
                                  ? cudaMemcpyHostToDevice
                                  : cudaMemcpyDeviceToDevice;

        cudaError_t err = cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset),
                                          ByteOffset(srcPtr, srcByteOffset),
                                          byteSize, kind, 0);
        if (err != cudaSuccess && libErrorFunc(lib))
        {
            FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                        kFlexSourceFile, 0x1496, err,
                        "cudaMemcpyAsync(ByteOffset(dstPtr, dstByteOffset), ByteOffset(srcPtr, srcByteOffset), byteSize, kind, 0)");
        }

        NvFlexSetFence(lib);
        return;
    }

    if (libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError,
                    "Writing to a mapped buffer, buffers must be unmapped before being passed to flexSet*() methods",
                    kFlexSourceFile, 0x1483);
    }
}

// flexdecomp: flex.cu:5275 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320711writeData3DEP9cudaArrayP12NvFlexBufferiii
void writeData3D(cudaArray_t dst, NvFlexBuffer* src, int pitch, int width, int height)
{
    NvFlexLibrary* lib = src->lib;

    cudaMemcpyKind kind = src->type == eNvFlexBufferHost
                              ? cudaMemcpyHostToDevice
                              : cudaMemcpyDeviceToDevice;

    cudaMemcpy3DParms copyParams = {};
    copyParams.srcPtr = make_cudaPitchedPtr(src->ptr, pitch, width, height);
    copyParams.dstArray = dst;
    // shipped code uses the slice count as extent depth (flex.cu:5289)
    copyParams.extent = make_cudaExtent(width, height, height);
    copyParams.kind = kind;

    cudaError_t err = cudaMemcpy3DAsync(&copyParams);
    if (err != cudaSuccess && libErrorFunc(lib))
    {
        FlexLogFunc(libErrorFunc(lib), eNvFlexLogError, "Cuda error: %d in %s",
                    kFlexSourceFile, 0x14ae, err, "cudaMemcpy3DAsync(&copyParams)");
    }
}

// flexdecomp: flex.cu:5297 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32078ReadDataEPFv19NvFlexErrorSeverityPKcS2_iEP12NvFlexBufferPKviPK14NvFlexCopyDesci
void ReadData(NvFlexErrorCallback errorFunc, NvFlexBuffer* dstBuffer,
              const void* srcPtr, int srcLength, const NvFlexCopyDesc* desc,
              int elementWidth)
{
    if (desc == 0)
    {
        int n = dstBuffer->elementCount;
        if (srcLength <= dstBuffer->elementCount)
            n = srcLength;

        ReadData(dstBuffer, 0, srcPtr, 0, n * elementWidth);
        return;
    }

    const int n = desc->elementCount;

    if (dstBuffer->elementCount < desc->dstOffset + n)
    {
        if (errorFunc)
        {
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexGet*() method with destination range outside of destination buffer",
                        kFlexSourceFile, 0x14b7);
        }
        return;
    }

    if (n + desc->srcOffset <= srcLength)
    {
        ReadData(dstBuffer, desc->dstOffset * elementWidth, srcPtr,
                 desc->srcOffset * elementWidth, n * elementWidth);
        return;
    }

    if (errorFunc)
    {
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexGet*() method with source range outside of source buffer",
                    kFlexSourceFile, 0x14bd);
    }
}

// flexdecomp: flex.cu:5324 _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPFv19NvFlexErrorSeverityPKcS2_iEPviP12NvFlexBufferPK14NvFlexCopyDesci
void WriteData(NvFlexErrorCallback errorFunc, void* dstPtr, int destLength,
               NvFlexBuffer* srcBuffer, const NvFlexCopyDesc* desc,
               int elementWidth)
{
    if (srcBuffer == 0)
        return;

    if (desc == 0)
    {
        int n = srcBuffer->elementCount;
        if (destLength <= srcBuffer->elementCount)
            n = destLength;

        WriteData(dstPtr, 0, srcBuffer, 0, n * elementWidth);
        return;
    }

    const int n = desc->elementCount;

    if (destLength < desc->dstOffset + n)
    {
        if (errorFunc)
        {
            FlexLogFunc(errorFunc, eNvFlexLogError,
                        "Call to an NvFlexSet*() method with destination range outside of destination buffer",
                        kFlexSourceFile, 0x14d6);
        }
        return;
    }

    if (n + desc->srcOffset <= srcBuffer->elementCount)
    {
        WriteData(dstPtr, desc->dstOffset * elementWidth, srcBuffer,
                  desc->srcOffset * elementWidth, n * elementWidth);
        return;
    }

    if (errorFunc)
    {
        FlexLogFunc(errorFunc, eNvFlexLogError,
                    "Call to an NvFlexSet*() method with source range outside of source buffer",
                    kFlexSourceFile, 0x14dc);
    }
}

}  // namespace _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207

// ---- release-build spelling of the anonymous namespace ---------------------
//
// The shipped debug and release objects embed different CUDA wrapper file names
// in the anonymous-namespace mangling:
//   debug   cudaflex.o: _GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca3207
//   release cudaflex.o: _GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca3207
// The ledger unit ids come from the debug DWARF, while the release object is the
// static-gate baseline for every symbol it defines, and the other host parts call
// these helpers under the release spelling. Both names are therefore emitted:
// the single body of each helper keeps the d59 name and the f25 name is a
// `.set` symbol alias to it, which is the one-section/two-symbol shape the
// shipped objects present.
#if !defined(_MSC_VER)
// PE build: internal symbols are free; the d59/f25 spelling pair exists only
// for the ELF static gate.
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320715ExecuteCallbackEP12NvFlexSolver25NvFlexSolverCallbackStagef\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320715ExecuteCallbackEP12NvFlexSolver25NvFlexSolverCallbackStagef, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320715ExecuteCallbackEP12NvFlexSolver25NvFlexSolverCallbackStagef\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPKvi\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPKvi, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPKvi\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPvi\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPvi, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320710ByteOffsetEPvi\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320711writeData3DEP9cudaArrayP12NvFlexBufferiii\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca320711writeData3DEP9cudaArrayP12NvFlexBufferiii, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca320711writeData3DEP9cudaArrayP12NvFlexBufferiii\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32078ReadDataEP12NvFlexBufferiPKvii\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32078ReadDataEP12NvFlexBufferiPKvii, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32078ReadDataEP12NvFlexBufferiPKvii\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32078ReadDataEPFv19NvFlexErrorSeverityPKcS2_iEP12NvFlexBufferPKviPK14NvFlexCopyDesci\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32078ReadDataEPFv19NvFlexErrorSeverityPKcS2_iEP12NvFlexBufferPKviPK14NvFlexCopyDesci, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32078ReadDataEPFv19NvFlexErrorSeverityPKcS2_iEP12NvFlexBufferPKviPK14NvFlexCopyDesci\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPviP12NvFlexBufferii\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPviP12NvFlexBufferii, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPviP12NvFlexBufferii\n");
__asm__(".globl _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPFv19NvFlexErrorSeverityPKcS2_iEPviP12NvFlexBufferPK14NvFlexCopyDesci\n"
        ".set   _ZN63_GLOBAL__N__39_tmpxft_00000f25_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPFv19NvFlexErrorSeverityPKcS2_iEPviP12NvFlexBufferPK14NvFlexCopyDesci, _ZN63_GLOBAL__N__39_tmpxft_00000d59_00000000_6_flex_cpp1_ii_dbca32079WriteDataEPFv19NvFlexErrorSeverityPKcS2_iEPviP12NvFlexBufferPK14NvFlexCopyDesci\n");
#endif  // !_MSC_VER

// ---- util.h inline helpers emitted by cudaflex.o ---------------------------
//
// CudaRangeTimerPool and CudaTimer are declared in util.h; cudaflex.o emits
// their inline bodies as weak definitions in their own sections. recovered.h
// currently carries the classes as data-only structs (no ctor/dtor
// declarations), so the shipped member symbols are emitted here through
// explicit assembler labels. The integrator still needs to declare, in the
// recovered util.h:
//     CudaRangeTimerPool::CudaRangeTimerPool();
//     CudaTimer::CudaTimer(const char*, CudaTimers&);
//     ~CudaTimer();
// and DWARF records CudaRangeTimer::mBegin/mEnd as cudaEvent_t (recovered.h
// spells them void*; the layout is identical).
//
// recovered.h stores the standard-library members as opaque byte arrays, so
// the objects behind them are reached through layout-preserving casts (offset
// 0 of the pool, reference member at offset 8 of the timer).

namespace {

typedef std::vector<CudaRangeTimer> CudaRangeTimerVector;

inline CudaRangeTimerVector& timerPoolElems(CudaRangeTimerPool* pool)
{
    return *reinterpret_cast<CudaRangeTimerVector*>(pool->mElem);
}

struct CudaTimerNameLess
{
    bool operator()(const char* a, const char* b) const
    {
        return std::strcmp(a, b) < 0;
    }
};

typedef std::map<const char*, int, CudaTimerNameLess> CudaTimerNameMap;

inline CudaTimerNameMap& timerNameMap(CudaTimer* timer)
{
    void* p = 0;
    std::memcpy(&p, timer->mNameToTimerMap, sizeof(p));
    return *reinterpret_cast<CudaTimerNameMap*>(p);
}

// CudaRangeTimerPool::mInitCapacity (util.h:196): 0xc8 in the shipped ctor and
// 0x12c0/0x18 = 200 iterations of the event-creation loop.
const int kCudaRangeTimerPoolInitCapacity = 200;

// FLEX_DATATRACK diagnostic (MSVC): buffer identity recorded by the pool ctor
#if defined(_MSC_VER)
void* g_timerDataProbe = nullptr;
size_t g_timerCapProbe = 0;
#endif

// CudaRangeTimerPool::alloc() (util.h:179), which the shipped build inlines into
// CudaTimer::CudaTimer(). GCC 15 inlines the whole vector growth path as well,
// which triples the ctor's size, so the identical work is kept out of line
// here; the ctor's spec records the two folded-in callees.
#if !defined(_MSC_VER)
__attribute__((noinline))
#endif
void cudaTimerGrowPool(CudaRangeTimerPool* pool)
{
    CudaRangeTimerVector& elems = timerPoolElems(pool);

    const size_t oldSize = elems.size();
    const size_t newSize = oldSize * 2;

    elems.resize(newSize, CudaRangeTimer());

    for (size_t i = oldSize; i < newSize; ++i)
    {
        CudaRangeTimer timer;
        cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&timer.mBegin));
        cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&timer.mEnd));
        elems[i] = timer;
    }
}

}  // namespace

// flexdecomp: util.h:219 _ZN18CudaRangeTimerPoolC1Ev
#if !defined(_MSC_VER)
void cudaRangeTimerPoolCtor(CudaRangeTimerPool* self)
    __asm__("_ZN18CudaRangeTimerPoolC1Ev");
// flexdecomp: util.h:219 _ZN18CudaRangeTimerPoolC2Ev
void cudaRangeTimerPoolCtorBase(CudaRangeTimerPool* self)
    __asm__("_ZN18CudaRangeTimerPoolC2Ev")
    __attribute__((alias("_ZN18CudaRangeTimerPoolC1Ev")));

#endif  // !_MSC_VER
void cudaRangeTimerPoolCtor(CudaRangeTimerPool* self)
{
    CudaRangeTimerVector* elems =
        ::new (static_cast<void*>(self->mElem)) CudaRangeTimerVector();

    self->mUsedElemCount = 0;

    // FLEX_CTOR diagnostic: FLEX_PROBE_NOINSERT hard-disables the pre-fill
    elems->insert(elems->begin(), kCudaRangeTimerPoolInitCapacity,
                  CudaRangeTimer());

#if defined(_MSC_VER)
    // FLEX_DATATRACK diagnostic: record buffer identity for the shutdown check
    g_timerDataProbe = elems->data();
    g_timerCapProbe = elems->capacity();
#endif

    for (int i = 0; i < kCudaRangeTimerPoolInitCapacity; ++i)
    {
        CudaRangeTimer timer;
        cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&timer.mBegin));
        cudaEventCreate(reinterpret_cast<cudaEvent_t*>(&timer.mEnd));
        (*elems)[i] = timer;
    }

#if defined(_MSC_VER)
    // FLEX_CTOR_DUMP diagnostic: post-fill buffer identity + content hash
    {
        FILE* f = nullptr;
        fopen_s(&f, "flex_ctor_dump.txt", "w");
        if (f)
        {
            fprintf(f, "data=%p size=%zu cap=%zu elem=%zu\n", (void*)elems->data(),
                    elems->size(), elems->capacity(), sizeof(CudaRangeTimer));
            unsigned char* b = reinterpret_cast<unsigned char*>(elems->data());
            size_t n = elems->capacity() * sizeof(CudaRangeTimer);
            unsigned long long h = 1469598103934665603ull;
            for (size_t i = 0; i < n; ++i)
            {
                h ^= b[i];
                h *= 1099511628211ull;
            }
            fprintf(f, "fnv=%llu\n", h);
            fclose(f);
        }
    }
#endif
}

#if defined(_MSC_VER)
extern "C" void FlexTimerDataProbe(void** outData, size_t* outCap)
{
    *outData = g_timerDataProbe;
    *outCap = g_timerCapProbe;
}
#endif

// flexdecomp: util.h:261 _ZN9CudaTimerC1EPKcR10CudaTimers
#if !defined(_MSC_VER)
void cudaTimerCtor(CudaTimer* self, const char* name, CudaTimers& timers)
    __asm__("_ZN9CudaTimerC1EPKcR10CudaTimers");
// flexdecomp: util.h:261 _ZN9CudaTimerC2EPKcR10CudaTimers
void cudaTimerCtorBase(CudaTimer* self, const char* name, CudaTimers& timers)
    __asm__("_ZN9CudaTimerC2EPKcR10CudaTimers")
    __attribute__((alias("_ZN9CudaTimerC1EPKcR10CudaTimers")));

#endif  // !_MSC_VER
void cudaTimerCtor(CudaTimer* self, const char* name, CudaTimers& timers)
{
    self->mEnabled = timers.mEnabled;

    // reference member (DWARF: const NameToTimerMap&) stored in the recovered
    // 8-byte slot
    const void* nameMap = static_cast<const void*>(timers.mNameToTimerMap);
    std::memcpy(self->mNameToTimerMap, &nameMap, sizeof(nameMap));

    self->mTimerPool = timers.mTimerPool;

    if (self->mEnabled)
    {
        CudaRangeTimerVector& elems = timerPoolElems(self->mTimerPool);

        if (self->mTimerPool->mUsedElemCount == int(elems.size()))
            cudaTimerGrowPool(self->mTimerPool);

        self->mTimerIndex = self->mTimerPool->mUsedElemCount++;

        cudaEventRecord(reinterpret_cast<cudaEvent_t>(
                            timerPoolElems(self->mTimerPool)[self->mTimerIndex].mBegin),
                        0);

        timerNameMap(self)[name] = self->mTimerIndex;
    }
}

// flexdecomp: util.h:274 _ZN9CudaTimerD1Ev
#if !defined(_MSC_VER)
void cudaTimerDtor(CudaTimer* self) __asm__("_ZN9CudaTimerD1Ev");
// flexdecomp: util.h:274 _ZN9CudaTimerD2Ev
void cudaTimerDtorBase(CudaTimer* self)
    __asm__("_ZN9CudaTimerD2Ev") __attribute__((alias("_ZN9CudaTimerD1Ev")));

#endif  // !_MSC_VER
void cudaTimerDtor(CudaTimer* self)
{
    if (self->mEnabled)
        cudaEventRecord(reinterpret_cast<cudaEvent_t>(
                            timerPoolElems(self->mTimerPool)[self->mTimerIndex].mEnd),
                        0);
}
