// flexdecomp: reconstructed D3D backend — ComputeShaderD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   ComputeShaderD3D11(context, ComputeShaderDesc*) @ 180052af0 (221 bytes)
//   ~ComputeShaderD3D11                             @ 180052c80 (93 bytes)
//   ContextD3D11::createComputeShader               @ 180052ce0 (83 bytes)
//
// The bytecode is the shipped DXBC container (`flexkit d3dblobs`), never a
// recompiled shader.  `NvAPI_Slot` selects an NVAPI shader-extension slot; the
// NV-specific variants use one, the generic variants pass 0xffffffff and skip
// the NVAPI call entirely.
#include "ContextD3D11.h"

#include "NvApiShim.h"

namespace NvFlex
{

ComputeShaderD3D11::ComputeShaderD3D11(ContextD3D11* context, const ComputeShaderDesc* desc)
    : m_refCount(1), m_cs(0)
{
    m_desc = *desc;

    if (m_desc.NvAPI_Slot != 0xffffffffu)
        NvApiSetNvShaderExtnSlot(context->device(), m_desc.NvAPI_Slot);

    context->device()->CreateComputeShader(m_desc.cs, (SIZE_T)m_desc.cs_length, 0, &m_cs);

    if (m_desc.NvAPI_Slot != 0xffffffffu)
        NvApiSetNvShaderExtnSlot(context->device(), 0xffffffffu);
}

ComputeShaderD3D11::~ComputeShaderD3D11()
{
    if (m_cs != 0)
        m_cs->Release();
    m_cs = 0;
}

} // namespace NvFlex
