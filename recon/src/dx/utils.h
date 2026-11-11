// flexdecomp: reconstructed D3D backend — shared private interfaces.
//
// Everything here is recovered from the PDB type map of
// NvFlexReleaseD3D_x64.dll.pdb (member offsets, base classes and the byte
// offset of every virtual method in the vtable).  The class layout and vtable
// order are ABI, not style: other translation units of this DLL (flex.obj,
// ContextD3D11/12, Device*) build their derived classes on these declarations,
// so include this header instead of redeclaring the types.
//
// Evidence for every interface: `llvm-pdbutil dump -types` field lists; the
// vtable order was additionally verified against the actual vtables in
// .rdata (`??_7Context@NvFlex@@6B@` etc.) and the resolved code addresses.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "NvFlex.h"

// ---------------------------------------------------------------------------
// NvFlexFormat — internal pixel-format enum (PDB: global `NvFlexFormat`)
// ---------------------------------------------------------------------------
// Values are the PDB enumeration (field list 0x1033); they index the DXGI
// conversion table in Format.cpp.
enum NvFlexFormat
{
    eNvFlexFormat_unknown = 0,
    eNvFlexFormat_r32_typeless = 1,
    eNvFlexFormat_r32_float = 2,
    eNvFlexFormat_r32g32_float = 3,
    eNvFlexFormat_r32g32b32a32_float = 4,
    eNvFlexFormat_r16_float = 5,
    eNvFlexFormat_r16g16_float = 6,
    eNvFlexFormat_r16g16b16a16_float = 7,
    eNvFlexFormat_r32_uint = 8,
    eNvFlexFormat_r32g32_uint = 9,
    eNvFlexFormat_r32g32b32a32_uint = 10,
    eNvFlexFormat_r8_unorm = 11,
    eNvFlexFormat_r8g8_unorm = 12,
    eNvFlexFormat_r8g8b8a8_unorm = 13,
    eNvFlexFormat_r16_unorm = 14,
    eNvFlexFormat_r16g16_unorm = 15,
    eNvFlexFormat_r16g16b16a16_unorm = 16,
    eNvFlexFormat_d32_float = 17,
    eNvFlexFormat_d24_unorm_s8_uint = 18,
    eNvFlexFormat_r8_snorm = 19,
    eNvFlexFormat_r8g8_snorm = 20,
    eNvFlexFormat_r8g8b8a8_snorm = 21,
    eNvFlexFormat_max = 22
};

// NvFlexContextDesc is defined by the context TU (Context.cpp); only the
// pointer is needed here.
struct NvFlexContextDesc;

// ---------------------------------------------------------------------------
// Math types used by the D3D backend
// ---------------------------------------------------------------------------
// The shipped backend uses the XVector3/XVector4 templates from the FleX maths
// header (PDB: `XVector3<float>` / `XVector4<float>`), which are not shipped as
// source.  Only the 12/16 byte layout matters for the recovered code.

template <typename T>
struct XVector3
{
    T x, y, z;

    XVector3() : x(0), y(0), z(0) {}
    XVector3(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}

    XVector3 operator-(const XVector3& v) const { return XVector3(x - v.x, y - v.y, z - v.z); }
    XVector3& operator+=(const XVector3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    XVector3& operator-=(const XVector3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
};

template <typename T>
struct XVector4
{
    T x, y, z, w;

    XVector4() : x(0), y(0), z(0), w(0) {}
    XVector4(T x_, T y_, T z_, T w_) : x(x_), y(y_), z(z_), w(w_) {}
};

// Bounds: PDB `Bounds` members lower@0, upper@12 (two XVector3<float>).
struct Bounds
{
    XVector3<float> lower;
    XVector3<float> upper;

    Bounds() {}
    Bounds(const XVector3<float>& lower_, const XVector3<float>& upper_)
        : lower(lower_), upper(upper_) {}

    XVector3<float> GetEdges() const { return upper - lower; }
};

// ---------------------------------------------------------------------------
// NvFlex::* interfaces
// ---------------------------------------------------------------------------
namespace NvFlex
{

// PDB enum NvFlex::MapType (field list 0x10CF).
enum MapType
{
    eMapRead         = 0,
    eMapWrite        = 1,
    eMapReadWrite    = 2,
    eMapWriteDiscard = 3
};

// PDB enum NvFlex::BufferResourceTypes (field list 0x10CD).
enum BufferResourceTypes
{
    eResource       = 1,
    eDefaultBuffer  = 2,
    eUploadBuffer   = 4,
    eDownloadBuffer = 8,
    eNativeBuffer   = 16
};

// PDB enum NvFlex::BufferTypes (field list 0x2C14).  BufferDesc::bufferType
// carries a combination of these flags (e.g. 0x13 = eBuffer|eUAV_SRV|eStructured).
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

// PDB NvFlex::BufferDesc: format@0 dim@4 stride@8 bufferType@12 data@16, sizeof 24.
struct BufferDesc
{
    NvFlexFormat format;
    unsigned dim;
    unsigned stride;
    unsigned bufferType;
    void* data;

    BufferDesc()
        : format(eNvFlexFormat_unknown), dim(0), stride(0), bufferType(0), data(0) {}
};

// PDB NvFlex::BufferViewDesc: format@0 (single member).
struct BufferViewDesc
{
    NvFlexFormat format;
};

// PDB NvFlex::ConstantBufferDesc: stride@0 dim@4 uploadAccess@8, sizeof 12.
struct ConstantBufferDesc
{
    unsigned stride;
    unsigned dim;
    bool uploadAccess;
};

// PDB NvFlex::MappedData: data@0 rowPitch@8 depthPitch@12, sizeof 16.
struct MappedData
{
    void* data;
    unsigned rowPitch;
    unsigned depthPitch;
};

struct ComputeShader;
struct ConstantBuffer;
struct Resource;
struct ResourceRW;
struct Buffer;

// PDB NvFlex::ComputeShaderDesc: cs@0 cs_length@8 label@16 NvAPI_Slot@24.
struct ComputeShaderDesc
{
    const void* cs;
    unsigned long long cs_length;
    const wchar_t* label;
    unsigned NvAPI_Slot;

    // 0x2a3b0: cs = 0, cs_length = 0, label = L"", NvAPI_Slot = -1
    ComputeShaderDesc() : cs(0), cs_length(0), label(L""), NvAPI_Slot(0xffffffffu) {}
};

// PDB NvFlexDim: unsigned x@0 y@4 z@8, sizeof 12.
struct NvFlexDim
{
    unsigned x, y, z;
};

// PDB NvFlex::DispatchParams: shader@0 gridDim@8 rootConstantBuffer@24
// secondConstantBuffer@32 readOnly[256]@40 readWrite[64]@296 IndirectLaunchArgs@360.
struct DispatchParams
{
    ComputeShader* shader;
    NvFlexDim gridDim;
    ConstantBuffer* rootConstantBuffer;
    ConstantBuffer* secondConstantBuffer;
    Resource* readOnly[256];
    ResourceRW* readWrite[64];
    Buffer* IndirectLaunchArgs;
};

// PDB NvFlex::Texture3DDesc: format@0 dim@4 data@24, sizeof 40.
struct Texture3DDesc
{
    NvFlexFormat format;
    NvFlexDim dim;
    unsigned mipLevels;     // @16 (unused by the recovered code paths)
    unsigned flags;         // @20
    void* data;             // @24
};

// Empty allocation policy base (PDB NvFlex::Allocable, sizeof 1).  The shipped
// object defines operator new/delete for it; see utils.cpp.
class Allocable
{
public:
    static void* operator new(std::size_t size);
    static void* operator new[](std::size_t size);
    static void operator delete(void* p);
    static void operator delete[](void* p);
};

// PDB global class `NvFlexObject`: vtable { addRef@0, release@8 }.  It has no
// virtual destructor; derived classes introduce their own.
class NvFlexObject
{
public:
    virtual unsigned addRef() = 0;
    virtual unsigned release() = 0;

protected:
    ~NvFlexObject() {}
};

// PDB NvFlex::Resource: no new virtuals, sizeof 8.
class Resource : public NvFlexObject
{
};

// PDB NvFlex::ResourceRW: virtual Resource* getResource() @16.
class ResourceRW : public NvFlexObject
{
public:
    virtual Resource* getResource() = 0;
};

// PDB NvFlex::Buffer: getResource@16 getResourceRW@24 ~Buffer@32; m_desc@8.
class Buffer : public NvFlexObject
{
public:
    virtual Resource* getResource(BufferResourceTypes type) = 0;
    virtual ResourceRW* getResourceRW() = 0;
    virtual ~Buffer() {}

    BufferDesc m_desc;
};

// PDB NvFlex::ConstantBuffer: ~ConstantBuffer@16; m_desc@8.
class ConstantBuffer : public NvFlexObject
{
public:
    virtual ~ConstantBuffer() {}

    ConstantBufferDesc m_desc;
};

// PDB NvFlex::ComputeShader: ~ComputeShader@16; m_desc@8.
class ComputeShader : public NvFlexObject
{
public:
    virtual ~ComputeShader() {}

    ComputeShaderDesc m_desc;
};

// PDB NvFlex::Stage: ~Stage@16; m_desc@8 (a staged buffer copy).
class Stage : public NvFlexObject
{
public:
    virtual ~Stage() {}

    BufferDesc m_desc;
};

// PDB NvFlex::Texture3D: getResource@16 getResourceRW@24 ~Texture3D@32; m_desc@8.
class Texture3D : public NvFlexObject
{
public:
    virtual Resource* getResource() = 0;
    virtual ResourceRW* getResourceRW() = 0;
    virtual ~Texture3D() {}

    Texture3DDesc m_desc;
};

// PDB NvFlex::Timer: ~Timer@16 (no data members).
class Timer : public NvFlexObject
{
public:
    virtual ~Timer() {}
};

// PDB NvFlex::Fence: ~Fence@16 (no data members).
class Fence : public NvFlexObject
{
public:
    virtual ~Fence() {}
};

// PDB NvFlex::TimerPool: ~TimerPool@0 begin@8 end@16 resolve@24 sync@32
// clear@40 get@48 reserve@56.
class TimerPool
{
public:
    virtual ~TimerPool() {}
    virtual int begin() = 0;
    virtual void end(int index) = 0;
    virtual void resolve() = 0;
    virtual void sync() = 0;
    virtual void clear() = 0;
    virtual void get(int index, unsigned long long* start, unsigned long long* end,
                     unsigned long long* frequency) = 0;
    virtual void reserve(unsigned long long count) = 0;
};

// PDB NvFlex::Context (sizeof 48 with the two timer maps).
//
// vtable offsets below are byte offsets into `??_7Context@NvFlex@@6B@`, read
// from .rdata and cross-checked against `ContextD3D11::*` / `ContextD3D12::*`
// vtable slot functions.
class Context
{
public:
    virtual void updateContext(const NvFlexContextDesc* desc) = 0;                       // @0
    virtual ConstantBuffer* createConstantBuffer(const ConstantBufferDesc* desc) = 0;    // @8
    virtual MappedData map(Texture3D* texture) = 0;                                      // @16
    virtual void* map(Buffer* buffer, MapType type, bool wait) = 0;                      // @24
    virtual void* map(ConstantBuffer* buffer) = 0;                                       // @32
    virtual void unmap(Texture3D* texture) = 0;                                          // @40
    virtual void unmap(Buffer* buffer) = 0;                                              // @48
    virtual void unmap(ConstantBuffer* buffer) = 0;                                      // @56
    virtual void copy(Texture3D* dst, Texture3D* src) = 0;                               // @64
    virtual void copy(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                      unsigned size) = 0;                                                // @72
    virtual void copy(ConstantBuffer* dst, Buffer* src) = 0;                             // @80
    virtual Buffer* createBuffer(const BufferDesc* desc, void* initialData) = 0;         // @88
    virtual Buffer* createBufferView(Buffer* buffer, const BufferViewDesc* desc) = 0;    // @96
    virtual void upload(Buffer* buffer, unsigned offset, unsigned size) = 0;             // @104
    virtual void upload(Buffer* buffer) = 0;                                             // @112
    virtual void download(Texture3D* texture) = 0;                                       // @120
    virtual void download(Buffer* buffer, unsigned offset, unsigned size) = 0;           // @128
    virtual void download(Buffer* buffer) = 0;                                           // @136
    virtual void copyToNative(void* dst, unsigned dstOffset, Buffer* src,
                              unsigned srcOffset, unsigned size) = 0;                    // @144
    virtual void copyToDevice(Buffer* dst, unsigned dstOffset, Buffer* src,
                              unsigned srcOffset, unsigned size,
                              BufferResourceTypes dstType) = 0;                          // @152
    virtual void copyToHost(Buffer* src, unsigned srcOffset, Buffer* dst,
                            unsigned dstOffset, unsigned size) = 0;                      // @160
    virtual void clearUnorderedAccessViewUint(Buffer* buffer, const unsigned* values) = 0; // @168
    virtual void clearUnorderedAccessViewFloat(Buffer* buffer, const float* values) = 0;   // @176
    virtual void copyResourceState(Buffer* dst, Buffer* src) = 0;                         // @184
    virtual Texture3D* createTexture3D(const Texture3DDesc* desc) = 0;                    // @192
    virtual MappedData mapDownload(Texture3D* texture) = 0;                               // @200
    virtual void unmapDownload(Texture3D* texture) = 0;                                   // @208
    virtual ComputeShader* createComputeShader(const ComputeShaderDesc* desc) = 0;        // @216
    virtual void dispatch(const DispatchParams* params) = 0;                              // @224
    virtual void dispatchIndirect(const DispatchParams* params) = 0;                      // @232
    virtual TimerPool* createTimerPool() = 0;                                             // @240
    virtual Fence* createFence() = 0;                                                     // @248
    virtual void fenceSet(Fence* fence) = 0;                                              // @256
    virtual void fenceWait(Fence* fence) = 0;                                             // @264
    virtual void startFrame() = 0;                                                        // @272
    virtual void flush() = 0;                                                             // @280
    virtual void clearState() = 0;                                                        // @288
    virtual void computeWaitForGraphics() = 0;                                            // @296
    virtual void pixBeginEvent(const wchar_t* name) = 0;                                  // @304
    virtual void pixEndEvent() = 0;                                                       // @312
    virtual void eventMarker(const wchar_t* name) = 0;                                    // @320
    virtual void contextPush() = 0;                                                       // @328
    virtual void contextPop() = 0;                                                        // @336
    virtual int setEventMarkerAftermath(void* marker, unsigned flags) = 0;                 // @344
    virtual int getDataAftermath(void* contextData, void* status) = 0;                     // @352
    virtual ~Context() {}                                                                 // @360

    // PDB: mEnableTimers@8, mNameToInternalTimerMap@16, mNameToExternalTimerMap@32.
    bool mEnableTimers;
    void* mNameToInternalTimerMap[2];
    void* mNameToExternalTimerMap[2];
};

// PDB NvFlex::ltstr / ltwstr: case-insensitive-less predicates used by the
// context timer maps.  ltwstr::operator() @0x2a3d0 compares wchar_t strings.
struct ltstr
{
    bool operator()(const char* a, const char* b) const { return std::strcmp(a, b) < 0; }
};

struct ltwstr
{
    bool operator()(const wchar_t* a, const wchar_t* b) const
    {
        // 0x2a3d0: element-wise compare, ordering by the sign of the first
        // differing UTF-16 code unit.
        while (*a == *b)
        {
            if (*a == 0)
                return false;
            ++a;
            ++b;
        }
        return *a < *b;
    }
};

} // namespace NvFlex

// ---------------------------------------------------------------------------
// NvFlexBuffer — public opaque handle (PDB sizeof 56)
// ---------------------------------------------------------------------------
// mapped@0 type@4 mBuffer@8 ptr@16 elementCount@24 elementByteStride@28
// mLastUsedFlag@32 mContext@40 mFlexLib@48.
// The constructor/destructor live in flex.obj; map()/unmap() live in
// utils.obj and are defined in utils.cpp.
struct NvFlexBuffer
{
    NvFlexBuffer(int elementCount, int elementByteStride, NvFlexBufferType type,
                 NvFlexLibrary* lib, NvFlex::Context* context, void* data);
    ~NvFlexBuffer();

    void* map(int flags);
    void unmap();

    bool mapped;                    // @0
    NvFlexBufferType type;          // @4
    NvFlex::Buffer* mBuffer;        // @8
    void* ptr;                      // @16
    int elementCount;               // @24
    int elementByteStride;          // @28
    int mLastUsedFlag;              // @32
    int _pad36;                     // @36
    NvFlex::Context* mContext;      // @40
    NvFlexLibrary* mFlexLib;        // @48
};

// ---------------------------------------------------------------------------
// NvFlexLibrary prefix accessors
// ---------------------------------------------------------------------------
// The full NvFlexLibrary struct is defined by the solver TU (flex.obj).  The
// members touched here are addressed by their PDB offsets; no layout is
// invented and a definition elsewhere does not conflict.
inline NvFlexErrorCallback NvFlexLibraryErrorFunc(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlexErrorCallback*>(reinterpret_cast<char*>(lib) + 0);
}

inline NvFlex::Context* NvFlexLibraryContext(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::Context**>(reinterpret_cast<char*>(lib) + 16);
}

inline NvFlex::ComputeShader* NvFlexLibraryShaderClearInt(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::ComputeShader**>(reinterpret_cast<char*>(lib) + 328);
}

inline NvFlex::ComputeShader* NvFlexLibraryShaderClearFloat4(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::ComputeShader**>(reinterpret_cast<char*>(lib) + 336);
}

inline NvFlex::ConstantBuffer* NvFlexLibraryClearCBInt(NvFlexLibrary* lib)
{
    // PDB: DxConstantBuffer mShaderClearConstantBufferInt @344; its mBuffer is @16.
    return *reinterpret_cast<NvFlex::ConstantBuffer**>(reinterpret_cast<char*>(lib) + 344 + 16);
}

inline NvFlex::ConstantBuffer* NvFlexLibraryClearCBFloat4(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::ConstantBuffer**>(reinterpret_cast<char*>(lib) + 368 + 16);
}

// ---------------------------------------------------------------------------
// Logging / error helper (definition in flex.obj; weak fallback in utils.cpp)
// ---------------------------------------------------------------------------
extern "C" void FlexLogFunc(NvFlexErrorCallback callback, NvFlexErrorSeverity severity,
                            const char* format, const char* file, int line, ...);

// ---------------------------------------------------------------------------
// D3D buffer wrappers (PDB: all four are plain structs without virtuals)
// ---------------------------------------------------------------------------
// DxBuffer { Buffer* mBuffer@0; BufferDesc mBufferDesc@8; } sizeof 32
// DxBufferExt   : DxBuffer          (adds no data)
// DxComputeBuffer : DxBufferExt
// DxConstantBuffer { ConstantBufferDesc mBufferDesc@0; ConstantBuffer* mBuffer@16; } sizeof 24
struct DxBuffer
{
    NvFlex::Buffer* mBuffer;      // @0
    NvFlex::BufferDesc mBufferDesc; // @8

    DxBuffer() : mBuffer(0) {}

    // DxBuffer::Create   @0x2a580 (275 bytes)
    void Create(NvFlex::Context* context, unsigned dim, unsigned stride, void* data,
                bool staging);
    // DxBuffer::CreateZero @0x2a6a0 (256 bytes)
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride, bool staging);
    // DxBuffer::Release  @0x2a7a0 (78 bytes)
    void Release();
    // DxBuffer::CopyToDevice @0x2a7f0 (116) and @0x2a870 (123)
    void CopyToDevice(NvFlex::Context* context, unsigned offset, const void* data,
                      unsigned size);
    void CopyToDevice(NvFlex::Context* context, const void* data, unsigned size);
};

struct DxBufferExt : DxBuffer
{
    DxBufferExt() {}
    void Create(NvFlex::Context* context, unsigned dim, unsigned stride, void* data);
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride);
};

struct DxComputeBuffer : DxBufferExt
{
    DxComputeBuffer() {}
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride);
};

struct DxConstantBuffer
{
    NvFlex::ConstantBufferDesc mBufferDesc; // @0
    NvFlex::ConstantBuffer* mBuffer;        // @16

    DxConstantBuffer() : mBuffer(0) {}

    void Create(NvFlex::Context* context, unsigned dim, unsigned stride);
    void Copy(NvFlex::Context* context, const void* data, unsigned size);
    void Release();
};

// ---------------------------------------------------------------------------
// Buffer clear dispatchers (utils.obj @0x4e350 / @0x4e440)
// ---------------------------------------------------------------------------
// `elementCount` is the byte length of `buffer`; the shaders work on 4-byte
// (int) and 16-byte (float4) granularity respectively.
void ClearBufferInt(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount, int value);
void ClearBufferFloat4(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount,
                       const XVector4<float>* value);

// ---------------------------------------------------------------------------
// Device-independent helpers (originally anonymous-namespace in every D3D TU)
// ---------------------------------------------------------------------------
namespace nvdx
{

// utils.obj: `GetSortSize` @0x4e300 / bvh.obj @0x2aee0 (23 bytes).
inline int GetSortSize(int count) { return ((count & 0x3ff) != 0 ? 1 : 0) + (count >> 10) << 10; }

// utils.obj: `GetPadedNumBlocksRadixSort` @0x4e320 (31 bytes).
inline int GetPadedNumBlocksRadixSort(int count)
{
    return (((count & 0x3ff) != 0 ? 1 : 0) + (count >> 10) << 10) + 0xff >> 8;
}

// utils.obj: `GetNumBlocksRadixSort` @0x4e340 (10 bytes).
inline int GetNumBlocksRadixSort(int count) { return (count + 0x3ff) >> 10; }

// bvh.obj: `CalculateSlack` @0x2a3f0 (9 bytes).
inline int CalculateSlack(int count) { return (count * 3) / 2; }

} // namespace nvdx
