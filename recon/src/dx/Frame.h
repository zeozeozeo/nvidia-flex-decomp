// flexdecomp — D3D backend: frame.cpp / "frame" slice.
//
// Evidence
//   FleX/bin/win64/NvFlexReleaseD3D_x64.dll.pdb, module 0213 `frame.obj`:
//     f:\binaries\Intermediate\vctools\crt_bld\SELF_64_amd64\crt\prebuild\
//       build\amd64\mt_obj\eh\nativecpp\frame.obj
//   The 35 .text functions of that module are the MSVC vcruntime C++ exception
//   handling frame handler (`vcruntime/frame.cpp`), statically linked into the
//   DLL by the MSVC linker.  They are *not* FleX code and must never be
//   redefined here; the toolchain supplies the equivalent machinery
//   (`__gxx_personality_seh0` / `__CxxFrameHandler3` in msvcrt).  The complete
//   inventory with its provider is in `nvflexFrameRuntimeSymbols` (Frame.cpp).
//
//   What *is* FleX-sided in the "frame" domain is the fence/timer layer:
//   `NvFlex::Fence`, `NvFlex::Timer`, `NvFlex::TimerPool` (implemented by the
//   Context slices, ContextD3D11.obj / ContextD3D12.obj) and the per-frame
//   profiling helper ``anonymous namespace'::DxTimer`` which every D3D
//   translation unit emits (radixsort.obj 0x18004d180, bvh.obj, flex.obj).
//   The latter is reconstructed and defined here.
//
//   Class layouts follow the raw PDB field list (`llvm-pdbutil dump -types`):
//     DxTimers  56 bytes  mEnabled@0, mNameToTimerMap@8, mNameToTimerMapW@24,
//                         mContext@40, mTimerPool@48
//     DxTimer   24 bytes  mEnabled@0, mTimerIndex@4, mTimerPool@8, mContext@16
//     TimerPool 32 bytes  m_timerPoolSize@8, m_usedElemCount@16,
//                         m_requestedElemCount@24; vtable
//                         {~TimerPool, begin, end, resolve, sync, clear, get,
//                          reserve}
//
// Interface contract
//   This header is the provisional home of the shared D3D core interface
//   (`Buffer`, `ConstantBuffer`, `ComputeShader`, `Context`, `TimerPool`,
//   `DispatchParams`).  It is the *only* definition of those names: other
//   slices include this header instead of redeclaring them.  Virtual methods
//   are appended at the end when a slice needs more; reordering is a bug.
//   When the Context slice publishes `Context.h`, move this block there
//   verbatim and include it from here.
#pragma once

#include <atomic>
#include <cstddef>
#include <map>
#include <cstring>
#include <cwchar>

// PDB: global-scope enum `NvFlexFormat` (not namespaced).
enum NvFlexFormat
{
    eNvFlexFormat_r32_uint          = 8,   // .rdata: BufferDesc.format of the key/counter buffers
    eNvFlexFormat_r32g32_uint       = 9,
    eNvFlexFormat_r32g32b32a32_uint = 10,  // the uint4 views (key4/val4/counter4)
    eNvFlexFormat_r8_unorm          = 11,
    eNvFlexFormat_r8g8_unorm        = 12,
    eNvFlexFormat_r8g8b8a8_unorm    = 13,
    eNvFlexFormat_r16_unorm         = 14,
    eNvFlexFormat_r16g16_unorm      = 15,
    eNvFlexFormat_r16g16b16a16_unorm= 16,
    eNvFlexFormat_d32_float         = 17,
    eNvFlexFormat_d24_unorm_s8_uint = 18,
    eNvFlexFormat_r8_snorm          = 19,
    eNvFlexFormat_r8g8_snorm        = 20,
    eNvFlexFormat_r8g8b8a8_snorm    = 21,
    eNvFlexFormat_max               = 22
};

namespace NvFlex {

// ---------------------------------------------------------------------------
// Allocation hooks
//
// Every backend object is allocated through these so that an application
// supplied allocator (NvFlexSetMallocFunc / NvFlexSetFreeFunc, utils.obj)
// round-trips.  The defaults are the CRT malloc/free, which is what the
// shipped image stores in .data at 0x180146980/0x180146988 (relocated to its
// statically linked CRT malloc/free).  Defined in Frame.cpp; other slices must
// declare, not define, them.
// ---------------------------------------------------------------------------
extern void* (*g_FlexMalloc)(size_t size);
extern void  (*g_FlexFree)(void* ptr);

// ---------------------------------------------------------------------------
// Buffer kinds (PDB `BufferTypes`, `BufferResourceTypes`, `MapType`).  Only the
// values this slice uses are spelled out; the enums are complete.
// ---------------------------------------------------------------------------

enum BufferTypes
{
    eBuffer     = 1,
    eUAV_SRV    = 2,
    eStage      = 4,
    eRaw        = 8,
    eStructured = 16,
    eIndirect   = 32,
    eShared     = 64,
    eDynamic    = 128
};

enum BufferResourceTypes
{
    eDownloadBuffer = 8,
    eNativeBuffer   = 16
};

enum MapType
{
    eMapRead         = 0,
    eMapWrite        = 1,
    eMapReadWrite    = 2,
    eMapWriteDiscard = 3
};

// ---------------------------------------------------------------------------
// Descriptors
// ---------------------------------------------------------------------------

// PDB: 24 bytes.  `dim` is the element count; the context multiplies by
// `stride`.  Evidence: RadixSortImpl ctor, 0x18004d9d4..0x18004da4e builds
// {format=8, dim=maxSortBlocks<<10, stride=4, bufferType=3, data=0}.
struct BufferDesc
{
    NvFlexFormat format;
    unsigned     dim;
    unsigned     stride;
    unsigned     bufferType;   // BufferTypes bits; 3 == eBuffer | eUAV_SRV
    void*        data;
};

// PDB: 4 bytes.
struct BufferViewDesc
{
    NvFlexFormat format;
};

// PDB: 12 bytes; stride@0, dim@4, uploadAccess@8.
struct ConstantBufferDesc
{
    unsigned stride;
    unsigned dim;
    bool     uploadAccess;
};

// PDB: 32 bytes; cs@0, cs_length@8, label@16, NvAPI_Slot@24.
struct ComputeShaderDesc
{
    const void*   cs;          // DXBC container bytes
    unsigned long long cs_length;
    const wchar_t* label;
    unsigned      NvAPI_Slot;
};

// PDB: 12 bytes (x, y, z).
struct NvFlexDim
{
    unsigned x, y, z;
};

// PDB: 12 bytes.  The constant-buffer mirror used by the radix shaders;
// `work/dxbc/NvFlexReleaseD3D_x64/08{0,1,2,3}_cs_5_0.asm` cb0:
//   uint3 gridDim; uint passStart; uint numCounters; uint numKeys;
struct RadixSortShaderParams
{
    NvFlexDim gridDim;
    unsigned  passStart;
    unsigned  numCounters;
    unsigned  numKeys;
};

// PDB: 16 bytes.
struct MappedData
{
    void*    data;
    unsigned rowPitch;
    unsigned depthPitch;
};

// ---------------------------------------------------------------------------
// GPU resource wrappers (implemented by the ContextD3D11/D3D12 slices)
// ---------------------------------------------------------------------------

class Resource;
class ResourceRW;
class Texture3D;   // 3D texture handle; declared in utils.h, used by ContextEx

// Reference counted handle.  `release()` returns the remaining count and
// destroys the object at zero (PDB `NvFlexObject::addRef/release`).
class RefCounted
{
public:
    virtual ~RefCounted() {}
    virtual unsigned addRef() = 0;
    virtual unsigned release() = 0;
};

class Buffer : public RefCounted
{
public:
    // Shader-resource view of this buffer for the given binding kind.
    // Evidence: RadixSortImpl::sort calls slot +0x10 with `eBuffer` (value 1).
    virtual Resource*   getResource(BufferTypes type) = 0;
    // Unordered-access view.  Evidence: sort calls slot +0x18.
    virtual ResourceRW* getResourceRW() = 0;
    virtual const BufferDesc& desc() const = 0;
};

class ConstantBuffer : public RefCounted
{
public:
    virtual const ConstantBufferDesc& desc() const = 0;
};

class ComputeShader : public RefCounted
{
public:
    virtual const ComputeShaderDesc& desc() const = 0;
};

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// PDB: 368 bytes.  `readOnly` has 32 slots and `readWrite` 8; the radix
// passes bind readOnly[0..2] and readWrite[0..1] (evidence: sort()).
struct DispatchParams
{
    ComputeShader*       shader;
    NvFlexDim            gridDim;
    ConstantBuffer*      rootConstantBuffer;
    ConstantBuffer*      secondConstantBuffer;
    Resource*            readOnly[32];
    ResourceRW*          readWrite[8];
    Buffer*              IndirectLaunchArgs;
};

// ---------------------------------------------------------------------------
// Context — the device-agnostic command interface.
//
// Contract: implementations (ContextD3D11, ContextD3D12) derive from this and
// override every pure virtual.  Virtuals are appended, never reordered.
// ---------------------------------------------------------------------------
class Context
{
public:
    virtual ~Context() {}

    virtual ConstantBuffer* createConstantBuffer(const ConstantBufferDesc* desc) = 0;
    virtual Buffer*         createBuffer(const BufferDesc* desc, void* initialData) = 0;
    virtual Buffer*         createBufferView(Buffer* buffer, const BufferViewDesc* desc) = 0;
    virtual ComputeShader*  createComputeShader(const ComputeShaderDesc* desc) = 0;

    virtual void* map(ConstantBuffer* buffer) = 0;
    virtual void  unmap(ConstantBuffer* buffer) = 0;

    // Propagate the resource-transition state between a buffer and its view.
    // Evidence: RadixSortImpl::sort brackets every dispatch with these calls,
    // alternating direction (base->view before, view->base after).
    virtual void copyResourceState(Buffer* dst, Buffer* src) = 0;

    virtual void dispatch(const DispatchParams* params) = 0;

    // Optional GPU crash-marker hook; returns a status code (PDB 0x2C6C).
    virtual int setEventMarkerAftermath(void* data, unsigned size) = 0;
};

// ---------------------------------------------------------------------------
// Fences and timers
// ---------------------------------------------------------------------------

// PDB: 8 bytes, refcounted handle with no additional virtuals; the waiting is
// done by `Context::fenceWait`.  Implemented by FenceD3D11/FenceD3D12.
class Fence : public RefCounted
{
public:
    virtual ~Fence() {}
};

// PDB TimerPool: 32 bytes, {~TimerPool, begin, end, resolve, sync, clear, get,
// reserve}.  `begin()` returns the timer slot index, `end(index)` closes it.
class TimerPool : public RefCounted
{
public:
    virtual int  begin() = 0;
    virtual void end(int index) = 0;
    virtual void resolve() = 0;
    virtual void sync() = 0;
    virtual void clear() = 0;

    unsigned long long m_timerPoolSize;
    unsigned long long m_usedElemCount;
    unsigned long long m_requestedElemCount;
};

// PDB `NvFlex::ltstr` / `NvFlex::ltwstr`: pointer-keyed string comparators used
// by the name -> timer index maps.
struct ltstr
{
    bool operator()(const char* a, const char* b) const { return std::strcmp(a, b) < 0; }
};

struct ltwstr
{
    bool operator()(const wchar_t* a, const wchar_t* b) const { return std::wcscmp(a, b) < 0; }
};

} // namespace NvFlex

// ---------------------------------------------------------------------------
// Per-frame profiling helper.
//
// The PDB records both types inside an anonymous namespace
// (`'anonymous-namespace'::DxTimers`, `...::DxTimer`), i.e. each translation
// unit gets its own copy.  Across TUs the copies must agree, so the
// reconstruction uses named types in the global namespace.  Every D3D
// translation unit includes this header and therefore shares one definition.
// ---------------------------------------------------------------------------

struct DxTimers
{
    bool     mEnabled;              // +0
    // PDB: std::map<char const*, int, NvFlex::ltstr>      (+8)
    std::map<const char*, int, NvFlex::ltstr>    mNameToTimerMap;
    // PDB: std::map<wchar_t const*, int, NvFlex::ltwstr>  (+24 in the MSVC build;
    // libstdc++'s node-based map is larger — see the ABI note in Frame.cpp)
    std::map<const wchar_t*, int, NvFlex::ltwstr> mNameToTimerMapW;
    NvFlex::Context*   mContext;    // +40 (MSVC)
    NvFlex::TimerPool* mTimerPool;  // +48 (MSVC)
};

// RAII scope timer.  Construction raises an NVTX range and opens a timer-pool
// slot; destruction closes it.  Evidence: `DxTimer::DxTimer` @0x18004d180 and
// `DxTimer::~DxTimer` @0x18004d260 (radixsort.obj), reconstructed below.
class DxTimer
{
public:
    DxTimer(DxTimers* timers, const wchar_t* name);
    ~DxTimer();

    bool               mEnabled;
    int                mTimerIndex;
    NvFlex::TimerPool* mTimerPool;
    NvFlex::Context*   mContext;

private:
    DxTimer(const DxTimer&);
    DxTimer& operator=(const DxTimer&);
};

// ---------------------------------------------------------------------------
// frame.obj inventory (module 0213 of NvFlexReleaseD3D_x64.dll.pdb)
//
// All 35 .text functions belong to the MSVC C++ exception frame handler and
// are supplied by the toolchain.  The table exists so the report can show the
// complete module and the provider of every symbol; nothing here is defined by
// the reconstruction.
// ---------------------------------------------------------------------------
struct FrameRuntimeSymbol
{
    unsigned    rva;       // image RVA (section 1 + 0x1000)
    unsigned    size;
    const char* name;
    const char* provider;
};

extern const FrameRuntimeSymbol nvflexFrameRuntimeSymbols[];
extern const unsigned          nvflexFrameRuntimeSymbolCount;
