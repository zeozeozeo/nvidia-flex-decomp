// flexdecomp: reconstructed D3D backend — ConstantBufferD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   ConstantBufferD3D11(context, ConstantBufferDesc*) @ 1800513b0 (179 bytes)
//   ~ConstantBufferD3D11                              @ 1800514b0 (116 bytes)
//   ContextD3D11::map/unmap(ConstantBuffer*)          @ 180051590 / 1800515e0
//
// ByteWidth = dim * ((stride + 15) & ~15): the shipped code rounds every element
// up to a 16-byte constant-buffer slot.  uploadAccess selects DYNAMIC +
// CPU_ACCESS_WRITE, which is what the whole solver relies on (one upload of
// KernelParams per stage through map/unmap).
#include "ContextD3D11.h"

#include <cstring>

namespace NvFlex
{

// @0x513b0
ConstantBufferD3D11::ConstantBufferD3D11(ContextD3D11* context, const ConstantBufferDesc* desc)
    : m_refCount(1), m_buffer(0), m_downloadBuffer(0)
{
    m_desc = *desc;

    const unsigned stride = (desc->stride + 0xf) & ~0xfu;

    D3D11_BUFFER_DESC bufferDesc;
    std::memset(&bufferDesc, 0, sizeof(bufferDesc));
    bufferDesc.ByteWidth = desc->dim * stride;
    if (desc->uploadAccess)
    {
        bufferDesc.Usage = D3D11_USAGE_DYNAMIC;
        bufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    }
    bufferDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    bufferDesc.MiscFlags = 0;

    context->device()->CreateBuffer(&bufferDesc, 0, &m_buffer);
}

// @0x514b0
ConstantBufferD3D11::~ConstantBufferD3D11()
{
    if (m_buffer != 0)
        m_buffer->Release();
    m_buffer = 0;
    if (m_downloadBuffer != 0)
        m_downloadBuffer->Release();
    m_downloadBuffer = 0;
}

} // namespace NvFlex
