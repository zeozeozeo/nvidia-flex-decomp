// flexdecomp: reconstructed D3D backend — ContextD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c, one
// `// ---- <Name> @ <rva>` block per function; the RVA is repeated below.
//
// Concurrency model (fixed decision): one D3D11 device and its immediate
// context.  Every `dispatch` binds all 32 SRV and 8 UAV slots (unused ones
// null), so no per-dispatch unbinding bookkeeping is needed; `startFrame`,
// `clearState`, `updateContext` and `contextPop` only do what the pseudocode
// shows.
#include "ContextD3D11.h"

#include <cstring>
#include <new>

#include "DeviceD3D11.h"

namespace NvFlex
{

namespace
{

// The shipped code allocates every backend object through the global
// FlexMalloc/FlexFree hooks so that NvFlexSetMallocFunc round-trips.
template <typename T, typename A1>
T* flexNew(A1 a1)
{
    void* memory = g_FlexMalloc(sizeof(T));
    return memory != 0 ? new (memory) T(a1) : 0;
}

template <typename T, typename A1, typename A2>
T* flexNew(A1 a1, A2 a2)
{
    void* memory = g_FlexMalloc(sizeof(T));
    return memory != 0 ? new (memory) T(a1, a2) : 0;
}

template <typename T, typename A1, typename A2, typename A3>
T* flexNew(A1 a1, A2 a2, A3 a3)
{
    void* memory = g_FlexMalloc(sizeof(T));
    return memory != 0 ? new (memory) T(a1, a2, a3) : 0;
}

template <typename T>
void flexDelete(T* object)
{
    if (object == 0)
        return;
    object->~T();
    g_FlexFree(object);
}

} // namespace

// ---------------------------------------------------------------------------
// Construction — DeviceD3D11::createContext @0x54fb0 (149 bytes)
// ---------------------------------------------------------------------------
// The shipped createContext fills a five-slot descriptor from the device and
// hands it to the ContextD3D11 constructor; taking the device directly is the
// same information, minus the copy.
ContextD3D11::ContextD3D11(DeviceD3D11* device)
    : m_device(0), m_renderDevice(0), m_deviceContext(0), m_renderDeviceContext(0),
      m_d3dAnnotation(0), m_useD3D11asyncCompute(false),
      m_computeFinishedWriting(0), m_graphicsFinishedReading(0),
      m_computeFinishedWriting_count(0), m_graphicsFinishedReading_count(0),
      m_renderCompletionFence(0), m_computeCompletionFence(0),
      m_sampler0(0), m_sampler1(0), m_sampler2(0), m_sampler3(0), m_sampler4(0),
      m_sampler5(0)
{
    for (unsigned i = 0; i < 16; ++i)
    {
        m_status.m_cache[i].m_query = 0;
        m_status.m_cache[i].m_queryActive = false;
    }
    m_status.m_size = 0;
    m_status.m_capacity = 16;
    m_status.m_data = m_status.m_cache;

    m_state.csConstantBuffers.m_capacity = 8;
    m_state.csConstantBuffers.m_data = m_state.csConstantBuffers.m_cache;
    m_state.csSamplers.m_capacity = 8;
    m_state.csSamplers.m_data = m_state.csSamplers.m_cache;
    m_state.csSrvs.m_capacity = 16;
    m_state.csSrvs.m_data = m_state.csSrvs.m_cache;
    m_state.csUavs.m_capacity = 8;
    m_state.csUavs.m_data = m_state.csUavs.m_cache;
    m_state.computeShader = 0;

    m_device = device->d3dDevice();
    m_renderDevice = device->renderDevice();
    m_deviceContext = device->d3dDeviceContext();
    m_renderDeviceContext = 0;
    m_useD3D11asyncCompute = device->useAsyncCompute();

    // The annotation interface only exists for a debug device; a null pointer
    // simply disables the BeginEvent/EndEvent pairs in dispatch().
    if (m_deviceContext != 0)
        m_deviceContext->QueryInterface(__uuidof(ID3DUserDefinedAnnotation),
                                        (void**)&m_d3dAnnotation);

    // One linear/clamp sampler, which is all the SDF and shape shaders sample
    // with.  Evidence: the constant pool entry the constructor loads at
    // 0x1800508xx — Filter = MIN_MAG_MIP_LINEAR, AddressU/V/W = CLAMP.
    D3D11_SAMPLER_DESC sampler;
    std::memset(&sampler, 0, sizeof(sampler));
    sampler.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampler.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sampler.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sampler.MaxAnisotropy = 1;
    sampler.MinLOD = 0.0f;
    sampler.MaxLOD = 3.402823466e+38f;   // FLT_MAX, the __real_7f7fffff constant

    if (m_device != 0)
        m_device->CreateSamplerState(&sampler, &m_sampler0);
}

// @0x50b40 (412 bytes)
ContextD3D11::~ContextD3D11()
{
    if (m_sampler0 != 0)
        m_sampler0->Release();
    m_sampler0 = 0;
    if (m_sampler1 != 0)
        m_sampler1->Release();
    m_sampler1 = 0;
    if (m_sampler2 != 0)
        m_sampler2->Release();
    m_sampler2 = 0;
    if (m_sampler3 != 0)
        m_sampler3->Release();
    m_sampler3 = 0;
    if (m_sampler4 != 0)
        m_sampler4->Release();
    m_sampler4 = 0;
    if (m_sampler5 != 0)
        m_sampler5->Release();
    m_sampler5 = 0;

    if (m_computeCompletionFence != 0)
        m_computeCompletionFence->Release();
    m_computeCompletionFence = 0;
    if (m_renderCompletionFence != 0)
        m_renderCompletionFence->Release();
    m_renderCompletionFence = 0;

    if (m_d3dAnnotation != 0)
        m_d3dAnnotation->Release();
    m_d3dAnnotation = 0;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
// @0x51530 (83 bytes)
ConstantBuffer* ContextD3D11::createConstantBuffer(const ConstantBufferDesc* desc)
{
    return flexNew<ConstantBufferD3D11>(this, desc);
}

// @0x51dd0 (190 bytes)
Buffer* ContextD3D11::createBuffer(const BufferDesc* desc, void* initialData)
{
    // eShared means "the caller already owns the buffer": `initialData` is then
    // the native ID3D11Buffer to wrap, and a failed wrap is deleted here.
    if (initialData != 0 && desc->bufferType == eShared)
    {
        BufferD3D11* buffer = flexNew<BufferD3D11>(this, static_cast<ID3D11Buffer*>(initialData));
        if (buffer->m_buffer == 0)
        {
            flexDelete(buffer);
            return 0;
        }
        return buffer;
    }
    return flexNew<BufferD3D11>(this, desc);
}

// @0x51e90 (109 bytes)
Buffer* ContextD3D11::createBufferView(Buffer* buffer, const BufferViewDesc* desc)
{
    if (buffer == 0)
        return 0;
    return flexNew<BufferD3D11>(this, static_cast<BufferD3D11*>(buffer), desc);
}

// @0x52ce0 (83 bytes)
ComputeShader* ContextD3D11::createComputeShader(const ComputeShaderDesc* desc)
{
    return flexNew<ComputeShaderD3D11>(this, desc);
}

// @0x52870 (83 bytes)
Texture3D* ContextD3D11::createTexture3D(const Texture3DDesc* desc)
{
    return flexNew<Texture3DD3D11>(this, desc);
}

// @0x538a0 (67 bytes)
Fence* ContextD3D11::createFence()
{
    return flexNew<FenceD3D11>(this);
}

// @0x536c0 (120 bytes)
TimerPool* ContextD3D11::createTimerPool(unsigned capacity)
{
    (void)capacity;
    return flexNew<TimerPoolD3D11>(this);
}

// ---------------------------------------------------------------------------
// Constant-buffer mapping — @0x51590 (76 bytes) / @0x515e0 (27 bytes)
// ---------------------------------------------------------------------------
void* ContextD3D11::map(ConstantBuffer* buffer)
{
    if (buffer == 0)
        return 0;

    ConstantBufferD3D11* cb = static_cast<ConstantBufferD3D11*>(buffer);
    D3D11_MAPPED_SUBRESOURCE mapped;
    mapped.pData = 0;
    mapped.RowPitch = 0;
    mapped.DepthPitch = 0;
    m_deviceContext->Map(cb->m_buffer, 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    return mapped.pData;
}

void ContextD3D11::unmap(ConstantBuffer* buffer)
{
    if (buffer == 0)
        return;
    ConstantBufferD3D11* cb = static_cast<ConstantBufferD3D11*>(buffer);
    m_deviceContext->Unmap(cb->m_buffer, 0);
}

// @0x51600 (51 bytes) — copy a buffer into a constant buffer.
void ContextD3D11::copy(ConstantBuffer* dst, Buffer* src)
{
    ConstantBufferD3D11* cb = dst != 0 ? static_cast<ConstantBufferD3D11*>(dst) : 0;
    BufferD3D11* buffer = src != 0 ? static_cast<BufferD3D11*>(src) : 0;
    m_deviceContext->CopyResource(cb->m_buffer, buffer->m_buffer);
}

// ---------------------------------------------------------------------------
// Dispatch — @0x52d40 (808 bytes)
// ---------------------------------------------------------------------------
// The shipped order, which matters for the D3D11 driver's hazard tracking:
//   CSSetShader -> 32 SRVs -> 8 UAVs -> b0/b1 -> sampler0 -> BeginEvent
//   -> Dispatch -> EndEvent -> unbind SRVs and UAVs.
void ContextD3D11::dispatch(const DispatchParams* params)
{
    ID3D11DeviceContext* ctx = m_deviceContext;
    if (ctx == 0)
        return;

    ComputeShaderD3D11* shader = static_cast<ComputeShaderD3D11*>(params->shader);

    ctx->CSSetShader(shader != 0 ? shader->m_cs : 0, 0, 0);

    ID3D11ShaderResourceView* srvs[32];
    for (unsigned i = 0; i < 32; ++i)
    {
        Resource* resource = params->readOnly[i];
        srvs[i] = resource != 0 ? static_cast<ResourceD3D11*>(resource)->m_srv : 0;
    }
    ctx->CSSetShaderResources(0, 32, srvs);

    ID3D11UnorderedAccessView* uavs[8];
    for (unsigned i = 0; i < 8; ++i)
    {
        ResourceRW* resource = params->readWrite[i];
        uavs[i] = resource != 0 ? static_cast<ResourceRWD3D11*>(resource)->m_uav : 0;
    }
    ctx->CSSetUnorderedAccessViews(0, 8, uavs, 0);

    if (params->rootConstantBuffer != 0)
    {
        ID3D11Buffer* cb = static_cast<ConstantBufferD3D11*>(params->rootConstantBuffer)->m_buffer;
        ctx->CSSetConstantBuffers(0, 1, &cb);
    }
    if (params->secondConstantBuffer != 0)
    {
        ID3D11Buffer* cb =
            static_cast<ConstantBufferD3D11*>(params->secondConstantBuffer)->m_buffer;
        ctx->CSSetConstantBuffers(1, 1, &cb);
    }

    ctx->CSSetSamplers(0, 1, &m_sampler0);

    const wchar_t* label = shader != 0 ? shader->m_desc.label : 0;
    if (m_d3dAnnotation != 0)
        m_d3dAnnotation->BeginEvent(label != 0 ? label : L"");

    if (params->IndirectLaunchArgs != 0)
    {
        ctx->DispatchIndirect(static_cast<BufferD3D11*>(params->IndirectLaunchArgs)->m_buffer, 0);
    }
    else
    {
        ctx->Dispatch(params->gridDim.x, params->gridDim.y, params->gridDim.z);
    }

    if (m_d3dAnnotation != 0)
        m_d3dAnnotation->EndEvent();

    ID3D11ShaderResourceView* noSrvs[32] = {0};
    ctx->CSSetShaderResources(0, 32, noSrvs);
    ID3D11UnorderedAccessView* noUavs[8] = {0};
    ctx->CSSetUnorderedAccessViews(0, 8, noUavs, 0);
}

// @0x53990 (11 bytes) — GFSDK_Aftermath_DX11_SetEventMarker.
int ContextD3D11::setEventMarkerAftermath(void* data, unsigned size)
{
    (void)data;
    (void)size;
    return 0;
}

// @0x539a0 (25 bytes) — GFSDK_Aftermath_DX11_GetData.
int ContextD3D11::getDataAftermath(void* data, unsigned size)
{
    (void)data;
    (void)size;
    return 0;
}

// ---------------------------------------------------------------------------
// Texture commands
// ---------------------------------------------------------------------------
// @0x528d0 (138 bytes)
MappedData ContextD3D11::map(Texture3D* texture)
{
    MappedData mapped;
    mapped.data = 0;
    mapped.rowPitch = 0;
    mapped.depthPitch = 0;

    if (texture == 0)
        return mapped;
    Texture3DD3D11* tex = static_cast<Texture3DD3D11*>(texture);
    if (!tex->m_desc.uploadAccess)
        return mapped;

    D3D11_MAPPED_SUBRESOURCE sub;
    sub.pData = 0;
    sub.RowPitch = 0;
    sub.DepthPitch = 0;
    m_deviceContext->Map(tex->m_uploadTexture, 0, D3D11_MAP_WRITE_DISCARD, 0, &sub);

    mapped.data = sub.pData;
    mapped.rowPitch = sub.RowPitch;
    mapped.depthPitch = sub.DepthPitch;
    return mapped;
}

// @0x52960 (81 bytes)
void ContextD3D11::unmap(Texture3D* texture)
{
    if (texture == 0)
        return;
    Texture3DD3D11* tex = static_cast<Texture3DD3D11*>(texture);
    if (!tex->m_desc.uploadAccess)
        return;

    m_deviceContext->Unmap(tex->m_uploadTexture, 0);
    m_deviceContext->CopyResource(tex->m_texture, tex->m_uploadTexture);
}

// @0x529c0 (43 bytes)
void ContextD3D11::download(Texture3D* texture)
{
    if (texture == 0)
        return;
    Texture3DD3D11* tex = static_cast<Texture3DD3D11*>(texture);
    if (!tex->m_desc.downloadAccess)
        return;
    m_deviceContext->CopyResource(tex->m_downloadTexture, tex->m_texture);
}

// @0x52ab0 (51 bytes)
void ContextD3D11::copy(Texture3D* dst, Texture3D* src)
{
    Texture3DD3D11* dstTex = dst != 0 ? static_cast<Texture3DD3D11*>(dst) : 0;
    Texture3DD3D11* srcTex = src != 0 ? static_cast<Texture3DD3D11*>(src) : 0;
    m_deviceContext->CopyResource(dstTex->m_texture, srcTex->m_texture);
}

// ---------------------------------------------------------------------------
// Buffer commands
// ---------------------------------------------------------------------------
// @0x51f00 (31 bytes) — plain resubmit of the upload copy.
void ContextD3D11::upload(Buffer* buffer)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);
    m_deviceContext->CopyResource(b->m_buffer, b->m_downloadBuffer);
}

// @0x51f20 (118 bytes) — region upload from the staging buffer.
void ContextD3D11::upload(Buffer* buffer, unsigned offset, unsigned size)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);

    D3D11_BOX box;
    box.left = offset;
    box.top = 0;
    box.front = 0;
    box.right = offset + size;
    box.bottom = 1;
    box.back = 1;
    m_deviceContext->CopySubresourceRegion(b->m_buffer, 0, offset, 0, 0,
                                           b->m_downloadBuffer, 0, &box);
}

// @0x51fa0 (31 bytes)
void ContextD3D11::download(Buffer* buffer)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);
    m_deviceContext->CopyResource(b->m_downloadBuffer, b->m_buffer);
}

// @0x51fc0 (118 bytes)
void ContextD3D11::download(Buffer* buffer, unsigned offset, unsigned size)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);

    D3D11_BOX box;
    box.left = offset;
    box.top = 0;
    box.front = 0;
    box.right = offset + size;
    box.bottom = 1;
    box.back = 1;
    m_deviceContext->CopySubresourceRegion(b->m_downloadBuffer, 0, offset, 0, 0,
                                           b->m_buffer, 0, &box);
}

// @0x51040 (145 bytes)
void* ContextD3D11::map(Buffer* buffer, MapType type, bool wait)
{
    D3D11_MAP mapType = D3D11_MAP_READ;
    switch (type)
    {
    case eMapRead:         mapType = D3D11_MAP_READ; break;
    case eMapWrite:        mapType = D3D11_MAP_WRITE; break;
    case eMapReadWrite:    mapType = D3D11_MAP_READ_WRITE; break;
    case eMapWriteDiscard: mapType = D3D11_MAP_WRITE_DISCARD; break;
    }
    const UINT flags = wait ? 0 : D3D11_MAP_FLAG_DO_NOT_WAIT;

    if (buffer == 0)
        return 0;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);

    // The CPU-visible side of a buffer is the staging copy (eStage) or, for
    // eDynamic buffers without one, the dynamic upload buffer.
    ID3D11Buffer* target = b->m_downloadBuffer != 0 ? b->m_downloadBuffer : b->m_uploadBuffer;
    if (target == 0)
        return 0;

    D3D11_MAPPED_SUBRESOURCE mapped;
    mapped.pData = 0;
    mapped.RowPitch = 0;
    mapped.DepthPitch = 0;
    if (m_deviceContext->Map(target, 0, mapType, flags, &mapped) != S_OK)
        return 0;
    return mapped.pData;
}

// @0x510e0 (27 bytes)
void ContextD3D11::unmap(Buffer* buffer)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);
    ID3D11Buffer* target = b->m_downloadBuffer != 0 ? b->m_downloadBuffer : b->m_uploadBuffer;
    if (target != 0)
        m_deviceContext->Unmap(target, 0);
}

// @0x51100 (144 bytes)
void ContextD3D11::copy(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                        unsigned size)
{
    BufferD3D11* dstBuffer = dst != 0 ? static_cast<BufferD3D11*>(dst) : 0;
    BufferD3D11* srcBuffer = src != 0 ? static_cast<BufferD3D11*>(src) : 0;

    D3D11_BOX box;
    box.left = srcOffset;
    box.top = 0;
    box.front = 0;
    box.right = srcOffset + size;
    box.bottom = 1;
    box.back = 1;
    m_deviceContext->CopySubresourceRegion(dstBuffer->m_buffer, 0, dstOffset, 0, 0,
                                           srcBuffer->m_buffer, 0, &box);
}

// @0x51190 (323 bytes)
void ContextD3D11::copyToNative(void* dst, unsigned dstOffset, Buffer* src,
                                unsigned srcOffset, unsigned size)
{
    if (src == 0)
        return;
    BufferD3D11* srcBuffer = static_cast<BufferD3D11*>(src);

    D3D11_BOX box;
    box.left = srcOffset;
    box.top = 0;
    box.front = 0;
    box.right = srcOffset + size;
    box.bottom = 1;
    box.back = 1;

    // The asynchronous-compute branch hands the copy to the render context;
    // with runOnRenderContext the compute context owns both sides.
    ID3D11Resource* nativeDst = reinterpret_cast<ID3D11Resource*>(dst);
    m_deviceContext->CopySubresourceRegion(nativeDst, 0, dstOffset, 0, 0,
                                           srcBuffer->m_buffer, 0, &box);
}

// @0x512e0 (244 bytes)
// The destination is always the buffer's device resource; `dstType` selects
// which of the *source* resources is read (disassembly of 0x1800522e0:
// dst = dst+0x14, src = src+0x14 for eDefaultBuffer, src+0x16 for
// eUploadBuffer, src+0x18 otherwise).
void ContextD3D11::copyToDevice(Buffer* dst, unsigned dstOffset, Buffer* src,
                                unsigned srcOffset, unsigned size,
                                BufferResourceTypes dstType)
{
    BufferD3D11* dstBuffer = dst != 0 ? static_cast<BufferD3D11*>(dst) : 0;
    BufferD3D11* srcBuffer = src != 0 ? static_cast<BufferD3D11*>(src) : 0;
    if (dstBuffer == 0 || dstBuffer->m_buffer == 0)
        return;

    D3D11_BOX box;
    box.left = srcOffset;
    box.top = 0;
    box.front = 0;
    box.right = srcOffset + size;
    box.bottom = 1;
    box.back = 1;

    ID3D11Resource* source = 0;
    switch (dstType)
    {
    case eDefaultBuffer:
        source = srcBuffer != 0 ? srcBuffer->m_buffer : 0;
        break;
    case eNativeBuffer:
        source = reinterpret_cast<ID3D11Resource*>(src);
        break;
    case eUploadBuffer:
        source = srcBuffer != 0 ? srcBuffer->m_uploadBuffer : 0;
        break;
    default: // eDownloadBuffer and eDeviceBuffer: read a device/staging resource
        source = srcBuffer != 0 ? srcBuffer->m_downloadBuffer : 0;
        if (source == 0 && srcBuffer != 0)
            source = srcBuffer->m_buffer;
        break;
    }

    if (source == 0)
        return;
    m_deviceContext->CopySubresourceRegion(dstBuffer->m_buffer, 0, dstOffset, 0, 0, source, 0,
                                           &box);
}

// @0x523e0 (144 bytes)
void ContextD3D11::copyToHost(Buffer* src, unsigned srcOffset, Buffer* dst,
                              unsigned dstOffset, unsigned size)
{
    BufferD3D11* srcBuffer = src != 0 ? static_cast<BufferD3D11*>(src) : 0;
    BufferD3D11* dstBuffer = dst != 0 ? static_cast<BufferD3D11*>(dst) : 0;

    D3D11_BOX box;
    box.left = srcOffset;
    box.top = 0;
    box.front = 0;
    box.right = srcOffset + size;
    box.bottom = 1;
    box.back = 1;
    m_deviceContext->CopySubresourceRegion(dstBuffer->m_downloadBuffer, 0, dstOffset, 0, 0,
                                           srcBuffer->m_buffer, 0, &box);
}

// @0x52470 (56 bytes)
void ContextD3D11::clearUnorderedAccessViewUint(Buffer* buffer, const unsigned* values)
{
    ID3D11UnorderedAccessView* uav = static_cast<ResourceRWD3D11*>(
        buffer->getResourceRW())->m_uav;
    m_deviceContext->ClearUnorderedAccessViewUint(uav, values);
}

// @0x524b0 (56 bytes)
void ContextD3D11::clearUnorderedAccessViewFloat(Buffer* buffer, const float* values)
{
    ID3D11UnorderedAccessView* uav = static_cast<ResourceRWD3D11*>(
        buffer->getResourceRW())->m_uav;
    m_deviceContext->ClearUnorderedAccessViewFloat(uav, values);
}

// @0x50570 (3 bytes) — the shipped body is empty.
void ContextD3D11::copyResourceState(Buffer* dst, Buffer* src)
{
    (void)dst;
    (void)src;
}

// ---------------------------------------------------------------------------
// Frame / synchronisation
// ---------------------------------------------------------------------------
// @0x523b0 (3 bytes)
void ContextD3D11::startFrame()
{
}

// @0x52490 (14 bytes)
void ContextD3D11::clearState()
{
    if (m_deviceContext != 0)
        m_deviceContext->ClearState();
}

// @0x53f0 (59 bytes) — D3D11 has nothing to update between frames; the shipped
// body re-binds the annotation interface and releases the previous one.
void ContextD3D11::updateContext()
{
    if (m_deviceContext == 0)
        return;
    if (m_d3dAnnotation != 0)
        m_d3dAnnotation->Release();
    m_d3dAnnotation = 0;
    m_deviceContext->QueryInterface(__uuidof(ID3DUserDefinedAnnotation),
                                    (void**)&m_d3dAnnotation);
}

// @0x534c0 (208 bytes) — flush is Flush() plus the NVTX range the shipped code
// brackets it with.
void ContextD3D11::flush()
{
    if (m_deviceContext != 0)
        m_deviceContext->Flush();
}

// @0x534a0 (93 bytes)
void ContextD3D11::computeWaitForGraphics()
{
    if (m_renderDeviceContext == 0)
        return;
    // Without the asynchronous-compute pair there is nothing to wait on: the
    // single immediate context is already ordered.
    m_deviceContext->Flush();
}

// @0x538f0 (27 bytes)
void ContextD3D11::fenceSet(Fence* fence)
{
    if (fence == 0)
        return;
    m_deviceContext->End(static_cast<FenceD3D11*>(fence)->m_query);
}

// @0x53910 (79 bytes)
void ContextD3D11::fenceWait(Fence* fence, unsigned timeoutUs)
{
    (void)timeoutUs;
    if (fence == 0)
        return;
    ID3D11Query* query = static_cast<FenceD3D11*>(fence)->m_query;
    while (m_deviceContext->GetData(query, 0, 0, 0) != S_OK)
        ;
}

// @0x52960 / @0x52970 (3 bytes each) — empty in this build.
void ContextD3D11::eventMarker(const wchar_t* label, int slot)
{
    (void)label;
    (void)slot;
}

void ContextD3D11::pixBeginEvent(const wchar_t* label)
{
    (void)label;
}

void ContextD3D11::pixEndEvent()
{
}

// @0x510d0 (320 bytes) — save the compute-stage state before a callback.
void ContextD3D11::contextPush()
{
    ID3D11ShaderResourceView* srv[16] = {0};
    ID3D11UnorderedAccessView* uav[8] = {0};
    ID3D11Buffer* cb[8] = {0};
    ID3D11SamplerState* sampler[8] = {0};

    m_deviceContext->CSGetShaderResources(0, 16, srv);
    m_deviceContext->CSGetUnorderedAccessViews(0, 8, uav);
    m_deviceContext->CSGetConstantBuffers(0, 8, cb);
    m_deviceContext->CSGetSamplers(0, 8, sampler);

    for (unsigned i = 0; i < 16; ++i)
        m_state.csSrvs.m_cache[i] = srv[i];
    for (unsigned i = 0; i < 8; ++i)
        m_state.csUavs.m_cache[i] = uav[i];
    for (unsigned i = 0; i < 8; ++i)
        m_state.csConstantBuffers.m_cache[i] = cb[i];
    for (unsigned i = 0; i < 8; ++i)
        m_state.csSamplers.m_cache[i] = sampler[i];

    m_deviceContext->ClearState();
}

// @0x51210 (402 bytes) — restore it afterwards.
void ContextD3D11::contextPop()
{
    m_deviceContext->CSSetConstantBuffers(0, 8, m_state.csConstantBuffers.m_cache);
    m_deviceContext->CSSetSamplers(0, 8, m_state.csSamplers.m_cache);
    m_deviceContext->CSSetShaderResources(0, 16, m_state.csSrvs.m_cache);
    m_deviceContext->CSSetUnorderedAccessViews(0, 8, m_state.csUavs.m_cache, 0);
    m_deviceContext->CSSetShader(m_state.computeShader, 0, 0);

    for (unsigned i = 0; i < 16; ++i)
    {
        if (m_state.csSrvs.m_cache[i] != 0)
            m_state.csSrvs.m_cache[i]->Release();
        m_state.csSrvs.m_cache[i] = 0;
    }
    for (unsigned i = 0; i < 8; ++i)
    {
        if (m_state.csUavs.m_cache[i] != 0)
            m_state.csUavs.m_cache[i]->Release();
        m_state.csUavs.m_cache[i] = 0;
    }
    for (unsigned i = 0; i < 8; ++i)
    {
        if (m_state.csConstantBuffers.m_cache[i] != 0)
            m_state.csConstantBuffers.m_cache[i]->Release();
        m_state.csConstantBuffers.m_cache[i] = 0;
    }
    for (unsigned i = 0; i < 8; ++i)
    {
        if (m_state.csSamplers.m_cache[i] != 0)
            m_state.csSamplers.m_cache[i]->Release();
        m_state.csSamplers.m_cache[i] = 0;
    }
}

// @0x524f0 (113 bytes)
void ContextD3D11::updateSubresource(Buffer* buffer, const unsigned* box, const void* data)
{
    if (buffer == 0)
        return;
    BufferD3D11* b = static_cast<BufferD3D11*>(buffer);
    D3D11_BOX d3dBox;
    std::memcpy(&d3dBox, box, sizeof(d3dBox));
    m_deviceContext->UpdateSubresource(b->m_buffer, 0, &d3dBox, data, 0, 0);
}

// ---------------------------------------------------------------------------
// Resource extent extraction
// ---------------------------------------------------------------------------
NvFlexDim ContextD3D11::extractDimBuffer(ID3D11Resource* resource)
{
    NvFlexDim dim = {0, 0, 0};
    ID3D11Buffer* buffer = 0;
    if (resource->QueryInterface(__uuidof(ID3D11Buffer), (void**)&buffer) == S_OK)
    {
        D3D11_BUFFER_DESC desc;
        buffer->GetDesc(&desc);
        dim.x = desc.ByteWidth;
        buffer->Release();
    }
    return dim;
}

NvFlexDim ContextD3D11::extractDimTexture3D(ID3D11Resource* resource)
{
    NvFlexDim dim = {0, 0, 0};
    ID3D11Texture3D* texture = 0;
    if (resource->QueryInterface(__uuidof(ID3D11Texture3D), (void**)&texture) == S_OK)
    {
        D3D11_TEXTURE3D_DESC desc;
        texture->GetDesc(&desc);
        dim.x = desc.Width;
        dim.y = desc.Height;
        dim.z = desc.Depth;
        texture->Release();
    }
    return dim;
}

NvFlexDim ContextD3D11::extractDim(ID3D11Resource* resource)
{
    D3D11_RESOURCE_DIMENSION dimension = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    resource->GetType(&dimension);
    if (dimension == D3D11_RESOURCE_DIMENSION_BUFFER)
        return extractDimBuffer(resource);
    if (dimension == D3D11_RESOURCE_DIMENSION_TEXTURE3D)
        return extractDimTexture3D(resource);
    return NvFlexDim{0, 0, 0};
}

NvFlexDim ContextD3D11::extractDim(ID3D11ShaderResourceView* view)
{
    ID3D11Resource* resource = 0;
    view->GetResource(&resource);
    NvFlexDim dim = {0, 0, 0};
    if (resource != 0)
    {
        dim = extractDim(resource);
        resource->Release();
    }
    return dim;
}

} // namespace NvFlex
