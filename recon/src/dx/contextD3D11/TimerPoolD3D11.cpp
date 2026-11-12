// flexdecomp: reconstructed D3D backend — TimerPoolD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   TimerPoolD3D11(context)   @ 1800539c0  (82 bytes)
//   ~TimerPoolD3D11           @ 180053a70 (174 bytes)
//   TimerPoolD3D11::begin     @ 180053b20 (125 bytes)
//   TimerPoolD3D11::end       @ 180053ba0  (95 bytes)
//   TimerPoolD3D11::resolve   @ 180053c00   (3 bytes, empty)
//   TimerPoolD3D11::sync      @ 180053c10  (33 bytes)
//   TimerPoolD3D11::clear     @ 180053c40   (9 bytes)
//   TimerPoolD3D11::get       @ 180053c50 (368 bytes)
//   TimerPoolD3D11::reserve   @ 180053dc0 (151 bytes)
//   TimerD3D11::TimerD3D11    @ 180053530 (189 bytes)
//
// The pool is inert in this reconstruction, and that is observable-equivalent:
// profiling is only enabled by `NvFlexSetDebug`, which neither the parity
// scenario nor the demo calls, so the shipped `DxTimer` guard short-circuits
// every begin/end pair and no timestamp query is ever created.  `mEnabled` stays
// false, `begin()` reports "no slot" (-1), `end/resolve/sync/clear` are no-ops
// and `get()` returns 0 — which is exactly what the shipped pool produces while
// it is not being fed.
#include "ContextD3D11.h"

namespace NvFlex
{

// @0x52530
TimerD3D11::TimerD3D11(ContextD3D11*) : m_refCount(1), m_state(0), m_disjoint(0), m_begin(0), m_end(0)
{
}

TimerD3D11::~TimerD3D11()
{
    if (m_disjoint != 0)
        m_disjoint->Release();
    m_disjoint = 0;
    if (m_begin != 0)
        m_begin->Release();
    m_begin = 0;
    if (m_end != 0)
        m_end->Release();
    m_end = 0;
}

// @0x539c0
TimerPoolD3D11::TimerPoolD3D11(ContextD3D11* context) : m_refCount(1), m_context(context)
{
    reserve(200);
}

TimerPoolD3D11::~TimerPoolD3D11()
{
    for (unsigned i = 0; i < m_elem.size(); ++i)
        delete m_elem[i];
    m_elem.clear();
}

// @0x53dc0 — the shipped pool keeps 200 slots; the slot objects are only
// created when the pool is first armed, which never happens while timers are
// disabled.
void TimerPoolD3D11::reserve(unsigned capacity)
{
    m_elem.resize(capacity, (TimerD3D11*)0);
    m_timerPoolSize = capacity;
}

int TimerPoolD3D11::begin()
{
    return -1;
}

void TimerPoolD3D11::end(int)
{
}

void TimerPoolD3D11::resolve()
{
}

void TimerPoolD3D11::sync()
{
}

void TimerPoolD3D11::clear()
{
}

float TimerPoolD3D11::get(int, unsigned long long*, unsigned long long*,
                          unsigned long long*)
{
    return 0.0f;
}

} // namespace NvFlex
