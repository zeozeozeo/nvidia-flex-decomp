// flexdecomp: reconstructed D3D backend — Texture3DD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   Texture3DD3D11(context, Texture3DDesc*) @ 180052570 (485 bytes)
//   ~Texture3DD3D11                         @ 1800527a0 (199 bytes)
//   ContextD3D11::map(Texture3D*)           @ 1800528d0 (138 bytes)
//   ContextD3D11::unmap(Texture3D*)         @ 180052960  (81 bytes)
//   ContextD3D11::download(Texture3D*)      @ 1800529c0  (43 bytes)
//   ContextD3D11::mapDownload/unmapDownload @ 1800529f0 / 180052a80
//   ContextD3D11::copy(Texture3D*,Texture3D*) @ 180052ab0 (51 bytes)
//
// Three textures per 3D texture: the GPU one (SHADER_RESOURCE |
// UNORDERED_ACCESS), an optional DYNAMIC upload copy (`uploadAccess`, mapped with
// WRITE_DISCARD and then CopyResource'd into the GPU one) and an optional
// STAGING download copy (`downloadAccess`).  MipLevels is always 1: the signed
// distance fields this backend feeds the solver are single-level.
#include "ContextD3D11.h"

#include <cstring>

namespace NvFlex
{

namespace
{

void fillTextureDesc(D3D11_TEXTURE3D_DESC& d3d, const Texture3DDesc& desc)
{
    std::memset(&d3d, 0, sizeof(d3d));
    d3d.Width = desc.dim.x;
    d3d.Height = desc.dim.y;
    d3d.Depth = desc.dim.z;
    d3d.MipLevels = 1;
    d3d.Format = g_DXGI_table.table[desc.format].format;
    d3d.Usage = D3D11_USAGE_DEFAULT;
    d3d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
    d3d.CPUAccessFlags = 0;
    d3d.MiscFlags = 0;
}

} // namespace

Resource* Texture3DD3D11::getResource()
{
    return &m_resource;
}

ResourceRW* Texture3DD3D11::getResourceRW()
{
    return &m_resourceRW;
}

// @0x52570
Texture3DD3D11::Texture3DD3D11(ContextD3D11* context, const Texture3DDesc* desc)
    : m_refCount(1), m_texture(0), m_uploadTexture(0), m_downloadTexture(0)
{
    m_desc = *desc;

    ID3D11Device* device = context->device();

    D3D11_TEXTURE3D_DESC textureDesc;
    fillTextureDesc(textureDesc, m_desc);

    D3D11_SUBRESOURCE_DATA initial;
    D3D11_SUBRESOURCE_DATA* initialPtr = 0;
    if (desc->data != 0)
    {
        const unsigned bytesPerElement = g_DXGI_table.table[desc->format].sizeInBytes;
        initial.pSysMem = desc->data;
        initial.SysMemPitch = desc->dim.y * bytesPerElement;
        initial.SysMemSlicePitch = desc->dim.y * desc->dim.z * bytesPerElement;
        initialPtr = &initial;
    }

    device->CreateTexture3D(&textureDesc, initialPtr, &m_texture);

    // Full-resource views: no view description is passed, so the views take the
    // texture's own format and dimensions.
    device->CreateShaderResourceView(m_texture, 0, &m_resource.m_srv);
    device->CreateUnorderedAccessView(m_texture, 0, &m_resourceRW.m_uav);

    if (m_desc.uploadAccess)
    {
        D3D11_TEXTURE3D_DESC upload = textureDesc;
        upload.Usage = D3D11_USAGE_DYNAMIC;
        upload.BindFlags = D3D11_BIND_SHADER_RESOURCE;
        upload.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        device->CreateTexture3D(&upload, initialPtr, &m_uploadTexture);
    }

    if (m_desc.downloadAccess)
    {
        D3D11_TEXTURE3D_DESC download = textureDesc;
        download.Usage = D3D11_USAGE_STAGING;
        download.BindFlags = 0;
        download.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        device->CreateTexture3D(&download, initialPtr, &m_downloadTexture);
    }
}

// @0x527a0
Texture3DD3D11::~Texture3DD3D11()
{
    if (m_resource.m_srv != 0)
        m_resource.m_srv->Release();
    m_resource.m_srv = 0;
    if (m_resourceRW.m_uav != 0)
        m_resourceRW.m_uav->Release();
    m_resourceRW.m_uav = 0;
    if (m_texture != 0)
        m_texture->Release();
    m_texture = 0;
    if (m_uploadTexture != 0)
        m_uploadTexture->Release();
    m_uploadTexture = 0;
    if (m_downloadTexture != 0)
        m_downloadTexture->Release();
    m_downloadTexture = 0;
}

} // namespace NvFlex
