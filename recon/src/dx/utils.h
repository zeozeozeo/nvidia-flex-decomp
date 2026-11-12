// flexdecomp: reconstructed D3D backend — utils.obj helpers.
//
// This header is the shared helper layer for the D3D backend.  The core device
// interface (`Buffer`, `ConstantBuffer`, `ComputeShader`, `Context`,
// descriptors, `DxTimers`) lives in Frame.h, which is the canonical definition
// for the whole backend; include that instead of redeclaring those names.
// What is added here:
//
//   * the math types the backend uses (XVector3/XVector4/Bounds),
//   * the public `NvFlexBuffer` handle and its map/unmap implementation,
//   * `ContextEx` — the buffer copy / upload / download / clear command surface
//     that utils.obj and bvh.obj dispatch through (Frame.h's Context covers the
//     radix-sort slice only; per its contract virtuals are appended, and these
//     move into Context.h once the Context slices publish it),
//   * the D3D buffer wrappers (DxBuffer / DxBufferExt / DxComputeBuffer /
//     DxConstantBuffer), used by every backend slice,
//   * offset accessors into `NvFlexLibrary` (owned by flex.obj),
//   * the small size helpers that every D3D TU carries as inlined copies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "Frame.h"
#include "NvFlex.h"

// ---------------------------------------------------------------------------
// Math types used by the D3D backend
// ---------------------------------------------------------------------------
// The shipped backend uses XVector3/XVector4 from the FleX maths header (PDB:
// `XVector3<float>` / `XVector4<float>`), which is not part of the shipped
// source.  Only the 12/16 byte layout is observable.

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

// PDB `Bounds`: lower@0, upper@12 (two XVector3<float>), 24 bytes.
struct Bounds
{
    XVector3<float> lower;
    XVector3<float> upper;

    Bounds() {}
    Bounds(const XVector3<float>& lower_, const XVector3<float>& upper_)
        : lower(lower_), upper(upper_) {}

    // bvh.obj @0x2a370 (61 bytes): upper - lower.
    XVector3<float> GetEdges() const { return upper - lower; }
};

// ---------------------------------------------------------------------------
// Additional Context command surface (append-only, see file header)
// ---------------------------------------------------------------------------
// Values passed to `Buffer::getResource`: PDB BufferResourceTypes
// { eResource = 1, eDefaultBuffer = 2, eUploadBuffer = 4, eDownloadBuffer = 8,
//   eNativeBuffer = 16 }.  Frame.h's Buffer::getResource takes the BufferTypes
// enum but the shipped call sites pass these values (1 for eResource), so the
// constants are repeated here for call sites in this slice.
enum BufferResourceValues
{
    kResource       = 1,
    kDefaultBuffer  = 2,
    kUploadBuffer   = 4,
    kDownloadBuffer = 8,
    kNativeBuffer   = 16
};

namespace NvFlex
{

// PDB NvFlex::Texture3D / Texture3DDesc — only needed by the full Context
// interface above; the texture path is implemented by the Context slices.
struct Texture3DDesc
{
    NvFlexFormat format;          // @0
    NvFlexDim dim;                // @4
    bool uploadAccess;            // @16
    bool downloadAccess;          // @17
    void* data;                   // @24
};

class Texture3D : public RefCounted
{
public:
    virtual Resource* getResource() = 0;
    virtual ResourceRW* getResourceRW() = 0;
    virtual ~Texture3D() {}
};

// Buffer copy / upload / download / clear operations.  Every method signature
// is taken from the shipped Context vtable (PDB `NvFlex::Context` field list
// 0x2C7D) and from the argument order used by utils.obj's copy helpers.
class ContextEx : public Context
{
public:
    // Keep the core Context overloads visible; the declarations below would
    // otherwise hide them (map(ConstantBuffer*), unmap(ConstantBuffer*)).
    using Context::map;
    using Context::unmap;

    virtual MappedData map(Texture3D* texture) = 0;
    virtual void* map(Buffer* buffer, MapType type, bool wait) = 0;
    virtual void unmap(Texture3D* texture) = 0;
    virtual void unmap(Buffer* buffer) = 0;
    virtual void copy(Texture3D* dst, Texture3D* src) = 0;
    virtual void copy(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                      unsigned size) = 0;
    virtual void copy(ConstantBuffer* dst, Buffer* src) = 0;
    virtual void upload(Buffer* buffer, unsigned offset, unsigned size) = 0;
    virtual void upload(Buffer* buffer) = 0;
    virtual void download(Texture3D* texture) = 0;
    virtual void download(Buffer* buffer, unsigned offset, unsigned size) = 0;
    virtual void download(Buffer* buffer) = 0;
    virtual void copyToNative(void* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                              unsigned size) = 0;
    virtual void copyToDevice(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                              unsigned size, BufferResourceTypes dstType) = 0;
    virtual void copyToHost(Buffer* src, unsigned srcOffset, Buffer* dst, unsigned dstOffset,
                            unsigned size) = 0;
    virtual void clearUnorderedAccessViewUint(Buffer* buffer, const unsigned* values) = 0;
    virtual void clearUnorderedAccessViewFloat(Buffer* buffer, const float* values) = 0;
    virtual void flush() = 0;
    virtual void computeWaitForGraphics() = 0;

    // --- solver / device-layer surface (append-only, flex.obj call sites) ---
    // Added for the solver pipeline.  The shipped `NvFlex::Context` vtable has
    // these too (PDB field list 0x2C7D); only the order inside this
    // reconstruction's vtable is ours.
    virtual Texture3D* createTexture3D(const Texture3DDesc* desc) = 0;
    virtual Fence*     createFence() = 0;
    virtual void       fenceSet(Fence* fence) = 0;
    virtual void       fenceWait(Fence* fence, unsigned timeoutUs) = 0;
    virtual void       startFrame() = 0;
    virtual void       clearState() = 0;
    virtual void       updateContext() = 0;
    virtual void       contextPop() = 0;
    virtual TimerPool* createTimerPool(unsigned capacity) = 0;
    virtual void       eventMarker(const wchar_t* label, int slot) = 0;
    virtual int        getDataAftermath(void* data, unsigned size) = 0;
};

} // namespace NvFlex

// ---------------------------------------------------------------------------
// NvFlexBuffer — public opaque handle (PDB sizeof 56)
// ---------------------------------------------------------------------------
// mapped@0 type@4 mBuffer@8 ptr@16 elementCount@24 elementByteStride@28
// mLastUsedFlag@32 mContext@40 mFlexLib@48.
// Constructor/destructor live in flex.obj; map()/unmap() live in utils.obj and
// are defined in utils.cpp.
struct NvFlexBuffer
{
    NvFlexBuffer(int elementCount, int elementByteStride, NvFlexBufferType type,
                 NvFlexLibrary* lib, NvFlex::ContextEx* context, void* data);
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
    NvFlex::ContextEx* mContext;    // @40
    NvFlexLibrary* mFlexLib;        // @48
};

// ---------------------------------------------------------------------------
// NvFlexLibrary prefix accessors
// ---------------------------------------------------------------------------
// NvFlexLibrary is defined by the solver TU (flex.obj).  The members touched
// here are addressed by their PDB offsets; nothing is invented and no layout
// conflicts with a definition elsewhere.
inline NvFlexErrorCallback NvFlexLibraryErrorFunc(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlexErrorCallback*>(reinterpret_cast<char*>(lib) + 0);
}

inline NvFlex::ContextEx* NvFlexLibraryContext(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::ContextEx**>(reinterpret_cast<char*>(lib) + 16);
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
    // PDB: DxConstantBuffer mShaderClearConstantBufferInt @344, mBuffer @16.
    return *reinterpret_cast<NvFlex::ConstantBuffer**>(reinterpret_cast<char*>(lib) + 344 + 16);
}

inline NvFlex::ConstantBuffer* NvFlexLibraryClearCBFloat4(NvFlexLibrary* lib)
{
    return *reinterpret_cast<NvFlex::ConstantBuffer**>(reinterpret_cast<char*>(lib) + 368 + 16);
}

// ---------------------------------------------------------------------------
// Logging (shipped definition in flex.obj; weak fallback in utils.cpp)
// ---------------------------------------------------------------------------
extern "C" void FlexLogFunc(NvFlexErrorCallback callback, NvFlexErrorSeverity severity,
                            const char* format, const char* file, int line, ...);

// ---------------------------------------------------------------------------
// D3D buffer wrappers (PDB: plain structs, no virtuals)
// ---------------------------------------------------------------------------
// DxBuffer { Buffer* mBuffer@0; BufferDesc mBufferDesc@8; } sizeof 32
// DxBufferExt : DxBuffer, DxComputeBuffer : DxBufferExt (no extra data)
// DxConstantBuffer { ConstantBufferDesc mBufferDesc@0; ConstantBuffer* mBuffer@16; }
struct DxBuffer
{
    NvFlex::Buffer* mBuffer;          // @0
    NvFlex::BufferDesc mBufferDesc;   // @8

    DxBuffer() : mBuffer(0) {}

    // bvh.obj @0x2a580 (275 bytes) / @0x2a6a0 (256 bytes).
    void Create(NvFlex::Context* context, unsigned dim, unsigned stride, void* data,
                bool staging = false);
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride,
                    bool staging = false);
    // bvh.obj @0x2a7a0 (78 bytes).
    void Release();
    // bvh.obj @0x2a7f0 (116) / @0x2a870 (123).
    void CopyToDevice(NvFlex::Context* context, unsigned offset, const void* data,
                      unsigned size);
    void CopyToDevice(NvFlex::Context* context, const void* data, unsigned size);
};

struct DxBufferExt : DxBuffer
{
    DxBufferExt() {}
    // bvh.obj @0x2a8f0 (29) / @0x2a910 (19).
    void Create(NvFlex::Context* context, unsigned dim, unsigned stride, void* data)
    {
        DxBuffer::Create(context, dim, stride, data, false);
    }
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride)
    {
        DxBuffer::CreateZero(context, dim, stride, false);
    }
};

struct DxComputeBuffer : DxBufferExt
{
    DxComputeBuffer() {}
    // bvh.obj @0x2a930 (25): a staging buffer with a single element, so the
    // requested element count is ignored (the shipped code passes 1).
    void CreateZero(NvFlex::Context* context, unsigned dim, unsigned stride)
    {
        (void)dim;
        DxBuffer::CreateZero(context, 1, stride, true);
    }
};

struct DxConstantBuffer
{
    NvFlex::ConstantBufferDesc mBufferDesc;   // @0
    NvFlex::ConstantBuffer* mBuffer;          // @16

    DxConstantBuffer() : mBuffer(0) {}

    // bvh.obj @0x2a960 (164) / @0x2aa10 (95) / @0x2aa70 (63).
    void Create(NvFlex::Context* context, unsigned dim, unsigned stride);
    void Copy(NvFlex::Context* context, const void* data, unsigned size);
    void Release();
};

// ---------------------------------------------------------------------------
// Buffer clear dispatchers (utils.obj @0x4e350 / @0x4e440)
// ---------------------------------------------------------------------------
// `elementCount` is the byte length of `buffer`; the shaders work on 4-byte
// (int) and 16-byte (float4) granularity.
void ClearBufferInt(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount, int value);
void ClearBufferFloat4(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount,
                       const XVector4<float>* value);

// ---------------------------------------------------------------------------
// Device-independent size helpers (inlined copies in every D3D TU)
// ---------------------------------------------------------------------------
namespace nvdx
{

// utils.obj `GetSortSize` @0x4e300 (23 bytes).
inline int GetSortSize(int count) { return (((count & 0x3ff) != 0 ? 1 : 0) + (count >> 10)) << 10; }

// utils.obj `GetPadedNumBlocksRadixSort` @0x4e320 (31 bytes).
inline int GetPadedNumBlocksRadixSort(int count)
{
    return (((((count & 0x3ff) != 0 ? 1 : 0) + (count >> 10)) << 10) + 0xff) >> 8;
}

// utils.obj `GetNumBlocksRadixSort` @0x4e340 (10 bytes).
inline int GetNumBlocksRadixSort(int count) { return (count + 0x3ff) >> 10; }

// bvh.obj `CalculateSlack` @0x2a3f0 (9 bytes).
inline int CalculateSlack(int count) { return (count * 3) / 2; }

} // namespace nvdx
