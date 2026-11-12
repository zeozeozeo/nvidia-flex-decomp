// flexdecomp: reconstructed D3D backend — BufferD3D11, ResourceD3D11 and
// ResourceRWD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   BufferD3D11(context, BufferDesc*)          @ 180051640  (787 bytes)
//   BufferD3D11(context, ID3D11Buffer*)        @ 180051a10  (241 bytes)
//   BufferD3D11(context, Buffer*, BufferViewDesc*) @ 180051b10  (390 bytes)
//   ~BufferD3D11                               @ 180051ca0  (290 bytes)
//
// Buffer types (Frame.h `BufferTypes`) drive the D3D11 mapping the shipped code
// applies, in this order:
//
//   eBuffer(1)      the default GPU buffer; BindFlags 0x88 (SHADER_RESOURCE |
//                   UNORDERED_ACCESS), MiscFlags picked by the subtype:
//                     eRaw(8)       -> 0x20  D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS
//                     eIndirect(32) -> 0x10  D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS
//                     eStructured(16)-> 0x40 D3D11_RESOURCE_MISC_BUFFER_STRUCTURED
//                                       with StructureByteStride = stride
//   eUAV_SRV(2)     create the SRV (BUFFER, or BUFFEREX when raw) and the UAV
//   eStage(4)       a STAGING buffer (CPU read|write) used by map/download
//   eDynamic(128)   a DYNAMIC write-combining upload buffer
//   eShared(64)     wrap an application ID3D11Buffer instead of creating one
#include "ContextD3D11.h"

#include <cstring>

#include "NvApiShim.h"

namespace NvFlex
{

namespace
{

// The two view descriptions the shipped constructor builds.  `desc` is the
// buffer's own BufferDesc; `raw` selects the raw/typed variants.
D3D11_SHADER_RESOURCE_VIEW_DESC makeSRVDesc(const BufferDesc& desc)
{
    D3D11_SHADER_RESOURCE_VIEW_DESC srv;
    std::memset(&srv, 0, sizeof(srv));
    // A structured view takes the struct stride from the resource, so its view
    // format must be UNKNOWN; typed views use the buffer format.
    srv.Format = (desc.bufferType & eStructured) != 0
                     ? DXGI_FORMAT_UNKNOWN
                     : g_DXGI_table.table[desc.format].format;
    srv.Buffer.FirstElement = 0;

    if (desc.bufferType & eRaw)
    {
        // Raw views address 4-byte units through the BUFFEREX dimension.
        srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFEREX;
        srv.BufferEx.FirstElement = 0;
        srv.BufferEx.NumElements = (desc.dim * desc.stride) >> 2;
        srv.BufferEx.Flags = D3D11_BUFFEREX_SRV_FLAG_RAW;
    }
    else
    {
        srv.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
        srv.Buffer.NumElements = desc.dim;
    }
    return srv;
}

D3D11_UNORDERED_ACCESS_VIEW_DESC makeUAVDesc(const BufferDesc& desc)
{
    D3D11_UNORDERED_ACCESS_VIEW_DESC uav;
    std::memset(&uav, 0, sizeof(uav));
    uav.Format = (desc.bufferType & eStructured) != 0
                     ? DXGI_FORMAT_UNKNOWN
                     : g_DXGI_table.table[desc.format].format;
    uav.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;

    if (desc.bufferType & eRaw)
    {
        uav.Buffer.NumElements = (desc.dim * desc.stride) >> 2;
        uav.Buffer.Flags = D3D11_BUFFER_UAV_FLAG_RAW;
    }
    else
    {
        uav.Buffer.NumElements = desc.dim;
    }
    return uav;
}

void fillBufferDesc(D3D11_BUFFER_DESC& d3d, const BufferDesc& desc)
{
    std::memset(&d3d, 0, sizeof(d3d));
    d3d.ByteWidth = desc.dim * desc.stride;
    d3d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;

    // MiscFlags: the subtype bits are mutually exclusive in the shipped code.
    if (desc.bufferType & eRaw)
    {
        d3d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
    }
    else if (desc.bufferType & eIndirect)
    {
        d3d.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
    }
    else if (desc.bufferType & eStructured)
    {
        d3d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
        d3d.StructureByteStride = desc.stride;
    }
}

} // namespace

// ---------------------------------------------------------------------------
// ResourceD3D11 accessors — @0x4f320 (13 bytes) / @0x4f3b0 (15 bytes)
// ---------------------------------------------------------------------------
// The three BufferD3D11 constructors store themselves as their own views, so
// getResource()/getResourceRW() are identity.
Resource* BufferD3D11::getResource(BufferTypes)
{
    return &m_resource;
}

ResourceRW* BufferD3D11::getResourceRW()
{
    return &m_resourceRW;
}

// ---------------------------------------------------------------------------
// BufferD3D11(ContextD3D11*, const BufferDesc*) — @0x51640 (787 bytes)
// ---------------------------------------------------------------------------
BufferD3D11::BufferD3D11(ContextD3D11* context, const BufferDesc* desc)
    : m_refCount(1), m_buffer(0), m_uploadBuffer(0), m_downloadBuffer(0),
      m_bufferOnGraphicsDevice(0), m_context(0)
{
    m_desc = *desc;

    ID3D11Device* device = context->device();

    D3D11_BUFFER_DESC bufferDesc;
    fillBufferDesc(bufferDesc, m_desc);

    // SRV/UAV creation needs the views to be valid, so the initial-data
    // structure is only filled when the caller supplied bytes.
    D3D11_SUBRESOURCE_DATA initial;
    D3D11_SUBRESOURCE_DATA* initialPtr = 0;
    if (desc->data != 0)
    {
        initial.pSysMem = desc->data;
        initial.SysMemPitch = desc->dim * desc->stride;
        initial.SysMemSlicePitch = 0;
        initialPtr = &initial;
    }

    if (m_desc.bufferType & eBuffer)
    {
        if (!context->m_useD3D11asyncCompute)
        {
            device->CreateBuffer(&bufferDesc, initialPtr, &m_buffer);
        }
        else
        {
            // Asynchronous compute path: the buffer lives on the graphics
            // device and is opened on the compute device through NVAPI.
            bufferDesc.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;
            context->m_renderDevice->CreateBuffer(&bufferDesc, initialPtr, &m_bufferOnGraphicsDevice);
            ID3D11Resource* opened = 0;
            if (NvApiOpenResourceOnComputeDevice(device, m_bufferOnGraphicsDevice, &opened) == 0)
            {
                m_buffer = static_cast<ID3D11Buffer*>(opened);
                m_context = context;
            }
        }
    }

    if (m_desc.bufferType & eUAV_SRV)
    {
        D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = makeSRVDesc(m_desc);
        device->CreateShaderResourceView(m_buffer, &srvDesc, &m_resource.m_srv);

        D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = makeUAVDesc(m_desc);
        device->CreateUnorderedAccessView(m_buffer, &uavDesc, &m_resourceRW.m_uav);
    }

    if (m_desc.bufferType & eStage)
    {
        // Staging copy of the buffer: map()/download() read it, upload() writes
        // it back through CopyResource.
        D3D11_BUFFER_DESC staging;
        std::memset(&staging, 0, sizeof(staging));
        staging.ByteWidth = m_desc.dim * m_desc.stride;
        staging.Usage = D3D11_USAGE_STAGING;
        staging.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
        if (m_desc.bufferType & eRaw)
            staging.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        else if (m_desc.bufferType & eIndirect)
            staging.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        else if (m_desc.bufferType & eStructured)
        {
            staging.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            staging.StructureByteStride = m_desc.stride;
        }

        device->CreateBuffer(&staging, initialPtr, &m_downloadBuffer);
    }

    if (m_desc.bufferType & eDynamic)
    {
        // CPU-writable upload buffer.
        D3D11_BUFFER_DESC dynamic;
        std::memset(&dynamic, 0, sizeof(dynamic));
        dynamic.ByteWidth = m_desc.dim * m_desc.stride;
        dynamic.Usage = D3D11_USAGE_DYNAMIC;
        dynamic.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        dynamic.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (m_desc.bufferType & eRaw)
            dynamic.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS;
        else if (m_desc.bufferType & eIndirect)
            dynamic.MiscFlags = D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS;
        else if (m_desc.bufferType & eStructured)
        {
            dynamic.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            dynamic.StructureByteStride = m_desc.stride;
        }

        device->CreateBuffer(&dynamic, initialPtr, &m_uploadBuffer);
    }
}

// ---------------------------------------------------------------------------
// BufferD3D11(ContextD3D11*, ID3D11Buffer*) — @0x51a10 (241 bytes)
// ---------------------------------------------------------------------------
// Wrap an application buffer: read its description and open the shared handle
// on this device (NvFlexRegisterD3DBuffer).
BufferD3D11::BufferD3D11(ContextD3D11* context, ID3D11Buffer* nativeBuffer)
    : m_refCount(1), m_buffer(0), m_uploadBuffer(0), m_downloadBuffer(0),
      m_bufferOnGraphicsDevice(0), m_context(0)
{
    D3D11_BUFFER_DESC bufferDesc;
    nativeBuffer->GetDesc(&bufferDesc);

    // The wrapper shares the resource with the application: keep the native
    // buffer alive and describe it with the same BufferDesc the create path
    // would have used.  (context is only needed for the async-compute path,
    // which NvFlexRegisterD3DBuffer never takes.)
    (void)context;
    m_buffer = nativeBuffer;
    m_buffer->AddRef();

    std::memset(&m_desc, 0, sizeof(m_desc));
    m_desc.format = NvFlexFormat(0);   // eNvFlexFormat_unknown
    m_desc.dim = bufferDesc.ByteWidth / (bufferDesc.StructureByteStride != 0
                                             ? bufferDesc.StructureByteStride
                                             : 1);
    m_desc.stride = bufferDesc.StructureByteStride;
    m_desc.bufferType = eBuffer | eUAV_SRV;
    if (bufferDesc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS)
        m_desc.bufferType |= eRaw;
    else if (bufferDesc.MiscFlags & D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS)
        m_desc.bufferType |= eIndirect;
    else if (bufferDesc.MiscFlags & D3D11_RESOURCE_MISC_BUFFER_STRUCTURED)
        m_desc.bufferType |= eStructured;
    m_desc.data = 0;
}

// ---------------------------------------------------------------------------
// BufferD3D11(ContextD3D11*, BufferD3D11*, const BufferViewDesc*) — @0x51b10
// ---------------------------------------------------------------------------
// Typed view over an existing buffer: same resource, new format, element count
// recomputed from the two formats' byte sizes.
BufferD3D11::BufferD3D11(ContextD3D11* context, BufferD3D11* buffer,
                         const BufferViewDesc* view)
    : m_refCount(1), m_buffer(0), m_uploadBuffer(0), m_downloadBuffer(0),
      m_bufferOnGraphicsDevice(0), m_context(0)
{
    m_desc = buffer->m_desc;

    const unsigned bytesPerElement = g_DXGI_table.table[buffer->m_desc.format].sizeInBytes;
    unsigned elements = (buffer->m_desc.dim * buffer->m_desc.stride) / bytesPerElement;

    m_desc.format = view->format;
    const unsigned viewBytes = g_DXGI_table.table[view->format].sizeInBytes;
    elements = (elements * bytesPerElement) / viewBytes;
    m_desc.dim = elements;

    m_buffer = buffer->m_buffer;
    m_buffer->AddRef();

    D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc = makeSRVDesc(m_desc);
    context->device()->CreateShaderResourceView(m_buffer, &srvDesc, &m_resource.m_srv);

    D3D11_UNORDERED_ACCESS_VIEW_DESC uavDesc = makeUAVDesc(m_desc);
    context->device()->CreateUnorderedAccessView(m_buffer, &uavDesc, &m_resourceRW.m_uav);
}

// ---------------------------------------------------------------------------
// ~BufferD3D11 — @0x51ca0 (290 bytes)
// ---------------------------------------------------------------------------
BufferD3D11::~BufferD3D11()
{
    if (m_resource.m_srv != 0)
        m_resource.m_srv->Release();
    m_resource.m_srv = 0;

    if (m_resourceRW.m_uav != 0)
        m_resourceRW.m_uav->Release();
    m_resourceRW.m_uav = 0;

    if (m_context != 0)
    {
        // Wait for the asynchronous-compute hand-off before releasing.
        ID3D11DeviceContext* ctx = m_context->deviceContext();
        ctx->End(m_context->m_computeCompletionFence);
        while (ctx->GetData(m_context->m_computeCompletionFence, 0, 0, 0) != 0)
            ;
    }

    if (m_buffer != 0)
        m_buffer->Release();
    m_buffer = 0;
    if (m_uploadBuffer != 0)
        m_uploadBuffer->Release();
    m_uploadBuffer = 0;
    if (m_downloadBuffer != 0)
        m_downloadBuffer->Release();
    m_downloadBuffer = 0;
    if (m_bufferOnGraphicsDevice != 0)
        m_bufferOnGraphicsDevice->Release();
    m_bufferOnGraphicsDevice = 0;
}

} // namespace NvFlex
