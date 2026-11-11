// flexdecomp: reconstructed D3D backend — utils.obj (17 functions).
//
// Sources of truth:
//   work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c  (pseudocode, slot usage)
//   work/pdb/NvFlexReleaseD3D_x64.dll.pdb             (class layouts, vtables)
//   work/pdb/d3d/utils.txt                            (object function inventory)
//
// The object's 17 .text functions are the anonymous-namespace copy helpers,
// the radix-sort size helpers, the two clear-buffer dispatchers and
// NvFlexBuffer::map/unmap.  Manifest-visible helpers live in utils.h so the
// other D3D translation units can share them.
#include "utils.h"

#include <cstdarg>
#include <cstdio>

using namespace NvFlex;

namespace {

// ---------------------------------------------------------------------------
// Buffer copy helpers.  Every one forwards to the Context virtual that the
// shipped code dispatches to; the "staged" variants add an explicit
// download/upload around the mapping.
// ---------------------------------------------------------------------------

// utils.obj @0x4de60 (16 bytes) — Context::copy (device->device).
void copy(Context* dstContext, Buffer* dst, unsigned dstOffset, Buffer* src,
          unsigned srcOffset, unsigned size)
{
    if (size != 0)
        dstContext->copy(dst, dstOffset, src, srcOffset, size);
}

// utils.obj @0x4de70 (19 bytes) — Context::copyToNative.
void copyToNative(Context* context, void* dst, unsigned dstOffset, Buffer* src,
                  unsigned srcOffset, unsigned size)
{
    if (size != 0)
        context->copyToNative(dst, dstOffset, src, srcOffset, size);
}

// utils.obj @0x4de90 (19 bytes) — Context::copyToHost.
void copyToHost(Context* context, Buffer* dst, unsigned dstOffset, Buffer* src,
                unsigned srcOffset, unsigned size)
{
    if (size != 0)
        context->copyToHost(src, srcOffset, dst, dstOffset, size);
}

// utils.obj @0x4deb0 (21 bytes) — Context::copyToDevice.
void copyToDevice(Context* context, Buffer* dst, unsigned dstOffset, Buffer* src,
                  unsigned srcOffset, unsigned size, BufferResourceTypes dstType)
{
    if (size != 0)
        context->copyToDevice(dst, dstOffset, src, srcOffset, size, dstType);
}

// utils.obj @0x4ded0 (132 bytes) — staged readback:
//   download(src, offset, size); map(src, read, wait); memcpy; unmap(src).
void copyToHostStaged(Context* context, void* dst, Buffer* src, unsigned srcOffset,
                      unsigned size)
{
    if (size == 0)
        return;

    context->download(src, srcOffset, size);
    void* mapped = context->map(src, eMapRead, true);
    std::memcpy(dst, mapped, size);
    context->unmap(src);
}

// utils.obj @0x4df60 (126 bytes) — staged upload:
//   map(dst, write, wait); memcpy; unmap(dst); upload(dst, offset, size).
void copyToDeviceStaged(Context* context, Buffer* dst, unsigned dstOffset, const void* src,
                        unsigned size)
{
    if (size == 0)
        return;

    void* mapped = context->map(dst, eMapWrite, true);
    std::memcpy(mapped, src, size);
    context->unmap(dst);
    context->upload(dst, dstOffset, size);
}

// utils.obj @0x4dfe0 (342 bytes) — grow `dst` to the size of `src`, then copy
// through host mappings.  The replacement buffer uses
// bufferType = 3 (eBuffer|eUAV_SRV), as the shipped constant does.
void copy(Context* dstContext, Buffer*& dst, Context* srcContext, Buffer* src)
{
    const unsigned size = src->m_desc.stride * src->m_desc.dim;

    if (dst->m_desc.stride * dst->m_desc.dim < size)
    {
        delete dst;

        void* data = std::malloc(size);
        std::memset(data, 0, size);

        BufferDesc desc;
        desc.dim = src->m_desc.dim;
        desc.stride = src->m_desc.stride;
        desc.bufferType = eBuffer | eUAV_SRV;   // 3, from the shipped descriptor
        desc.data = data;
        dst = dstContext->createBuffer(&desc, 0);
        std::free(data);
    }

    void* srcPtr = srcContext->map(src, eMapReadWrite, true);
    void* dstPtr = dstContext->map(dst, eMapReadWrite, true);
    std::memcpy(dstPtr, srcPtr, size);
    dstContext->unmap(dst);
    srcContext->unmap(src);
}

// utils.obj @0x4e140 (200 bytes) — DxBuffer variant.
void copy(Context* dstContext, DxBuffer* dst, Context* srcContext, DxBuffer* src)
{
    if (dst->mBufferDesc.dim * dst->mBufferDesc.stride <
        src->mBufferDesc.stride * src->mBufferDesc.dim)
    {
        dst->CreateZero(dstContext, src->mBufferDesc.dim, dst->mBufferDesc.stride, true);
    }

    void* srcPtr = srcContext->map(src->mBuffer, eMapReadWrite, true);
    void* dstPtr = dstContext->map(dst->mBuffer, eMapReadWrite, true);
    std::memcpy(dstPtr, srcPtr, src->mBufferDesc.stride * src->mBufferDesc.dim);
    dstContext->unmap(dst->mBuffer);
    srcContext->unmap(src->mBuffer);
}

// utils.obj @0x4e210 (177 bytes) — constant-buffer variant (map/unmap take the
// ConstantBuffer* overloads).
void copy(Context* dstContext, DxConstantBuffer* dst, Context* srcContext,
          DxConstantBuffer* src)
{
    if (dst->mBufferDesc.dim * dst->mBufferDesc.stride <
        src->mBufferDesc.dim * src->mBufferDesc.stride)
    {
        dst->Create(dstContext, src->mBufferDesc.dim, dst->mBufferDesc.stride);
    }

    void* srcPtr = srcContext->map(src->mBuffer);
    void* dstPtr = dstContext->map(dst->mBuffer);
    std::memcpy(dstPtr, srcPtr, src->mBufferDesc.dim * src->mBufferDesc.stride);
    dstContext->unmap(dst->mBuffer);
    srcContext->unmap(src->mBuffer);
}

// utils.obj @0x4e2d0 (46 bytes) — aggregate the shader description.
ComputeShader* createComputeShader(Context* context, const void* bytecode,
                                   unsigned long long bytecodeLength, const wchar_t* label,
                                   unsigned slot)
{
    ComputeShaderDesc desc;
    desc.cs = bytecode;
    desc.cs_length = bytecodeLength;
    desc.label = label;
    desc.NvAPI_Slot = slot;
    return context->createComputeShader(&desc);
}

// Clear shader constant buffers.  Layouts come from the PDB type records:
//   ClearParamsInt      lengthInWords@0 value@4 _pad1@8  _pad2@12   (sizeof 16)
//   ClearParamsFloat4   lengthInWords@0 _pad3@4  value@16           (sizeof 32)
struct ClearParamsInt
{
    int lengthInWords;
    int value;
    int _pad1;
    int _pad2;
};

struct ClearParamsFloat4
{
    int lengthInWords;
    unsigned char _pad3[12];
    XVector4<float> value;
};

} // namespace

// ---------------------------------------------------------------------------
// Radix-sort size helpers (originally per-TU anonymous namespace copies)
// ---------------------------------------------------------------------------
// The three helpers are provided inline by utils.h (nvdx::) because radixsort,
// bvh and flex all carry identical copies of them.  Nothing to define here.

// ---------------------------------------------------------------------------
// ClearBufferInt / ClearBufferFloat4 — utils.obj @0x4e350 / @0x4e440
// ---------------------------------------------------------------------------
// Dispatches the clear compute shader over `elementCount` bytes of `buffer`
// (int or float4 granularity respectively).
void ClearBufferInt(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount, int value)
{
    if (buffer == 0)
        return;

    DispatchParams params;
    std::memset(&params, 0, sizeof(params));

    params.shader = NvFlexLibraryShaderClearInt(lib);
    params.readWrite[0] = buffer->getResourceRW();

    const int numElements = elementCount >> 2;              // 4-byte elements
    params.gridDim.x = (numElements + 0xff) >> 8;           // 256 threads / group
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = NvFlexLibraryClearCBInt(lib);

    Context* context = NvFlexLibraryContext(lib);
    ClearParamsInt* cb = static_cast<ClearParamsInt*>(context->map(params.rootConstantBuffer));
    cb->lengthInWords = numElements;
    cb->value = value;
    context->unmap(params.rootConstantBuffer);
    context->dispatch(&params);
}

void ClearBufferFloat4(NvFlexLibrary* lib, NvFlex::Buffer* buffer, int elementCount,
                       const XVector4<float>* value)
{
    if (buffer == 0)
        return;

    DispatchParams params;
    std::memset(&params, 0, sizeof(params));

    params.shader = NvFlexLibraryShaderClearFloat4(lib);
    params.readWrite[0] = buffer->getResourceRW();

    const int numElements = elementCount >> 4;              // 16-byte elements
    params.gridDim.x = (numElements + 0xff) >> 8;
    params.gridDim.y = 1;
    params.gridDim.z = 1;
    params.rootConstantBuffer = NvFlexLibraryClearCBFloat4(lib);

    Context* context = NvFlexLibraryContext(lib);
    ClearParamsFloat4* cb = static_cast<ClearParamsFloat4*>(context->map(params.rootConstantBuffer));
    cb->lengthInWords = numElements;
    cb->value = *value;
    context->unmap(params.rootConstantBuffer);

    context->dispatch(&params);
}

// ---------------------------------------------------------------------------
// NvFlexBuffer::map / unmap — utils.obj @0x4e540 / @0x4e640
// ---------------------------------------------------------------------------
void* NvFlexBuffer::map(int flags)
{
    // 0x4e540: warning when the buffer is already mapped (utils.cpp:62).
    if (ptr != 0 && mFlexLib != 0)
    {
        NvFlexErrorCallback cb = NvFlexLibraryErrorFunc(mFlexLib);
        if (cb != 0)
            FlexLogFunc(cb, eNvFlexLogWarning, "Mapping a buffer that was already mapped",
                        "..\\..\\dx\\utils.cpp", 0x3e);
    }

    if (mBuffer == 0)
    {
        ptr = 0;
    }
    else if (type == eNvFlexBufferHost)
    {
        // eMapReadWrite, wait unless the caller passed eNvFlexMapDoNotWait.
        ptr = mContext->map(mBuffer, eMapReadWrite, (flags & 1) == 0);
    }
    else if (type == static_cast<NvFlexBufferType>(2))
    {
        ptr = mContext->map(mBuffer, eMapWriteDiscard, true);
    }
    else if (type == eNvFlexBufferDevice)
    {
        ptr = mBuffer->getResource(eDefaultBuffer);
    }

    return ptr;
}

void NvFlexBuffer::unmap()
{
    // 0x4e640: warning when unmapping an unmapped buffer (utils.cpp:98).
    if (ptr == 0 && mFlexLib != 0)
    {
        NvFlexErrorCallback cb = NvFlexLibraryErrorFunc(mFlexLib);
        if (cb != 0)
            FlexLogFunc(cb, eNvFlexLogWarning, "Unmapping a buffer that was not mapped",
                        "..\\..\\dx\\utils.cpp", 0x62);
    }

    // (type & ~2) == eNvFlexBufferHost covers type 0 and type 2.
    if (mBuffer != 0 && (static_cast<int>(type) & ~2) == eNvFlexBufferHost)
        mContext->unmap(mBuffer);

    ptr = 0;
}

// ---------------------------------------------------------------------------
// Public entry point kept by the first slice (flex.obj owns the shipped
// definition; this keeps the exported surface intact until it lands).
// ---------------------------------------------------------------------------
NV_FLEX_API int NvFlexGetVersion()
{
    return NV_FLEX_VERSION;   // 120, the value the shipped binary returns (@0x37c30)
}

// ---------------------------------------------------------------------------
// FlexLogFunc — shipped definition is in flex.obj (@0x34780, 77 bytes):
//   vsnprintf the message into a 2048 byte buffer, then call the user callback.
// A weak definition keeps the DLL loadable until flex.obj lands; a strong
// definition there takes precedence at link time.
// ---------------------------------------------------------------------------
extern "C" __attribute__((weak)) void FlexLogFunc(NvFlexErrorCallback callback,
                                                  NvFlexErrorSeverity severity,
                                                  const char* format, const char* file,
                                                  int line, ...)
{
    if (callback == 0)
        return;

    char message[2048];
    va_list args;
    va_start(args, line);
    std::vsnprintf(message, sizeof(message), format, args);
    va_end(args);

    callback(severity, message, file, line);
}
