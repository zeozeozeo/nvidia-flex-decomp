// flexdecomp — D3D backend: frame.cpp slice.
//
// Contents
//   1. DxTimer — the per-frame profiling helper ``anonymous namespace'::DxTimer`
//      that radixsort.obj, bvh.obj and flex.obj each emit a copy of.
//   2. The complete frame.obj inventory with the provider of every symbol.
//
// Evidence
//   radixsort.obj `DxTimer::DxTimer`  @ RVA 0x4d180 (218 bytes)
//   radixsort.obj `DxTimer::~DxTimer` @ RVA 0x4d260 ( 45 bytes)
//   PDB field list 0x44C1: DxTimer {bool mEnabled@0; int mTimerIndex@4;
//      TimerPool* mTimerPool@8; Context* mContext@16}
//   PDB field list 0x44B9: DxTimers {bool mEnabled@0; map mNameToTimerMap@8;
//      map mNameToTimerMapW@24; Context* mContext@40; TimerPool* mTimerPool@48}
#include "Frame.h"

#include <windows.h>

#include <cstdlib>

// ---------------------------------------------------------------------------
// Allocation hooks (see Frame.h)
// ---------------------------------------------------------------------------
namespace NvFlex {

namespace {

void* DefaultFlexMalloc(size_t size) { return std::malloc(size); }
void  DefaultFlexFree(void* ptr)     { std::free(ptr); }

} // namespace

void* (*g_FlexMalloc)(size_t) = &DefaultFlexMalloc;
void  (*g_FlexFree)(void*)    = &DefaultFlexFree;

} // namespace NvFlex

// ---------------------------------------------------------------------------
// NVTX
//
// The shipped DLL imports nvtxRangePushW / nvtxRangePop from
// nvToolsExt64_1.dll (IAT slots 0x1800883d8 / 0x1800883d0).  This build does
// not link an import library for nvToolsExt, so the two entry points are
// resolved lazily from the module the host application has already loaded.
// When an import library becomes available, replace the resolver with the
// direct imports; behaviour is identical (both are no-ops without a profiler).
// ---------------------------------------------------------------------------
namespace {

typedef int  (*NvtxRangePushW)(const wchar_t* name);
typedef void (*NvtxRangePop)(void);

struct NvtxApi
{
    NvtxRangePushW pushW;
    NvtxRangePop   pop;

    NvtxApi() : pushW(0), pop(0)
    {
        HMODULE mod = GetModuleHandleW(L"nvToolsExt64_1.dll");
        if (!mod)
            mod = LoadLibraryW(L"nvToolsExt64_1.dll");
        if (mod)
        {
            pushW = (NvtxRangePushW)(void*)GetProcAddress(mod, "nvtxRangePushW");
            pop   = (NvtxRangePop)(void*)GetProcAddress(mod, "nvtxRangePop");
        }
    }
};

const NvtxApi& nvtx()
{
    static NvtxApi api;   // resolved once, never unloaded
    return api;
}

} // namespace

// ---------------------------------------------------------------------------
// DxTimer
// ---------------------------------------------------------------------------

DxTimer::DxTimer(DxTimers* timers, const wchar_t* name)
    : mEnabled(false)
    , mTimerIndex(0)
    , mTimerPool(0)
    , mContext(0)
{
    // The original leaves the unused fields indeterminate; zeroing them is
    // observationally equivalent (they are only read when mEnabled is set).
    if (nvtx().pushW)
        nvtx().pushW(name);

    if (!timers)
        return;

    mContext   = timers->mContext;
    mTimerPool = timers->mTimerPool;
    mEnabled   = timers->mEnabled;

    if (mEnabled)
    {
        mTimerIndex = mTimerPool->begin();
        timers->mNameToTimerMapW[name] = mTimerIndex;
    }

    // Name the GPU event; the marker length is 2*wcslen+2 bytes (the loop at
    // 0x18004d20a scans for the terminator a wchar_t at a time).
    if (mContext)
        mContext->setEventMarkerAftermath((void*)name,
                                          (unsigned)(2 * std::wcslen(name) + 2));
}

DxTimer::~DxTimer()
{
    if (nvtx().pop)
        nvtx().pop();

    if (mEnabled && mTimerPool)
        mTimerPool->end(mTimerIndex);
}

// ---------------------------------------------------------------------------
// frame.obj — complete inventory.
//
// Module 0213 of NvFlexReleaseD3D_x64.dll.pdb:
//   f:\binaries\Intermediate\vctools\crt_bld\SELF_64_amd64\crt\prebuild\
//     build\amd64\mt_obj\eh\nativecpp\frame.obj
//
// That is the MSVC C++ exception frame handler, statically linked from the
// vcruntime.  Every symbol below is provided by the toolchain (msvcrt /
// libgcc's SEH personality); none is FleX code and none is redefined by the
// reconstruction.  RVAs are image-relative (section .text at RVA 0x1000).
// ---------------------------------------------------------------------------

extern const FrameRuntimeSymbol nvflexFrameRuntimeSymbols[] =
{
    { 0x06c180, 110, "__DestructExceptionObject",              "vcruntime frame.cpp" },
    { 0x06c1f0,  33, "std::bad_exception::bad_exception",      "vcruntime frame.cpp" },
    { 0x06c214,  42, "std::bad_exception::bad_exception",      "vcruntime frame.cpp" },
    { 0x06c240,  15, "std::bad_exception::~bad_exception",     "vcruntime frame.cpp" },
    { 0x06c250,  57, "std::bad_exception::scalar deleting dtor","vcruntime frame.cpp" },
    { 0x06c28c, 193, "CatchIt",                                "vcruntime frame.cpp" },
    { 0x06c350, 137, "ExFilterRethrow",                        "vcruntime frame.cpp" },
    { 0x06c3dc,1204, "FindHandler",                            "vcruntime frame.cpp" },
    { 0x06c890, 613, "FindHandlerForForeignException",         "vcruntime frame.cpp" },
    { 0x06caf8, 215, "IsInExceptionSpec",                      "vcruntime frame.cpp" },
    { 0x06cbd0, 135, "Is_bad_exception_allowed",               "vcruntime frame.cpp" },
    { 0x06cc58,  20, "__RethrowException",                     "vcruntime frame.cpp" },
    { 0x06cc6c,  25, "__uncaught_exception",                   "vcruntime frame.cpp" },
    { 0x06cc88, 191, "_is_exception_typeof",                   "vcruntime frame.cpp" },
    { 0x06cd48,   8, "_SetWinRTOutOfMemoryExceptionCallback",  "vcruntime frame.cpp" },
    { 0x06cd50,  35, "__AdjustPointer",                        "vcruntime frame.cpp" },
    { 0x06cd74, 182, "__BuildCatchObject",                     "vcruntime frame.cpp" },
    { 0x06ce2c, 591, "__BuildCatchObjectHelper",               "vcruntime frame.cpp" },
    { 0x06d07c, 515, "__CxxCallCatchBlock",                    "vcruntime frame.cpp" },
    { 0x06d280,  71, "__CxxDetectRethrow",                     "vcruntime frame.cpp" },
    { 0x06d2c8, 502, "__CxxExceptionFilter",                   "vcruntime frame.cpp" },
    { 0x06d4c0,   6, "__CxxQueryExceptionSize",                "vcruntime frame.cpp" },
    { 0x06d4c8, 193, "__CxxRegisterExceptionObject",           "vcruntime frame.cpp" },
    { 0x06d58c, 343, "__CxxUnregisterExceptionObject",         "vcruntime frame.cpp" },
    { 0x06d6e4,  83, "__FrameUnwindFilter",                    "vcruntime frame.cpp" },
    { 0x06d738, 402, "__FrameUnwindToState",                   "vcruntime frame.cpp" },
    { 0x06d8cc, 145, "__GetPlatformExceptionInfo",             "vcruntime frame.cpp" },
    { 0x06d960, 548, "__InternalCxxFrameHandler",              "vcruntime frame.cpp" },
    { 0x06db84, 301, "__TypeMatch",                            "vcruntime frame.cpp" },
    { 0x0871c4, 244, "__CxxCallCatchBlock$filt$0",             "vcruntime funclet" },
    { 0x0872b8, 126, "__CxxCallCatchBlock$fin$1",              "vcruntime funclet" },
    { 0x087336,  26, "__CxxExceptionFilter$fin$0",             "vcruntime funclet" },
    { 0x087350,  24, "__DestructExceptionObject$filt$0",       "vcruntime funclet" },
    { 0x087368,  22, "__FrameUnwindToState$filt$0",            "vcruntime funclet" },
    { 0x08737e,  41, "__FrameUnwindToState$fin$1",             "vcruntime funclet" },
};

extern const unsigned nvflexFrameRuntimeSymbolCount =
    (unsigned)(sizeof(nvflexFrameRuntimeSymbols) / sizeof(nvflexFrameRuntimeSymbols[0]));
