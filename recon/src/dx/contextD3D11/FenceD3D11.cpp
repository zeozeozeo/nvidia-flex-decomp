// flexdecomp: reconstructed D3D backend — FenceD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   FenceD3D11(context)          @ 180053740 (109 bytes)
//   ~FenceD3D11                  @ 180053840  (93 bytes)
//   ContextD3D11::createFence    @ 1800538a0  (67 bytes)
//   ContextD3D11::fenceSet       @ 1800538f0  (27 bytes)
//   ContextD3D11::fenceWait      @ 180053910  (79 bytes)
//
// A fence is one D3D11_QUERY_EVENT.  Signalling is End(query); waiting is a spin
// on GetData(query, nullptr, 0, 0) until it stops returning S_FALSE.
#include "ContextD3D11.h"

#include <cstring>

namespace NvFlex
{

// @0x53740
FenceD3D11::FenceD3D11(ContextD3D11* context) : m_refCount(1), m_query(0)
{
    D3D11_QUERY_DESC desc;
    desc.Query = D3D11_QUERY_EVENT;
    desc.MiscFlags = 0;
    context->device()->CreateQuery(&desc, &m_query);
}

// @0x53840
FenceD3D11::~FenceD3D11()
{
    if (m_query != 0)
        m_query->Release();
    m_query = 0;
}

} // namespace NvFlex
