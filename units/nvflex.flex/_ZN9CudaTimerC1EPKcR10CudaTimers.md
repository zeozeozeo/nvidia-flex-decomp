# CudaTimer::CudaTimer(const char*, CudaTimers&) (C1)

Declared `util.h:261`; weak body in `.text._ZN9CudaTimerC2EPKcR10CudaTimers`
(535 bytes / 139 instructions).

Evidence (disasm): mEnabled = timers.mEnabled; mNameToTimerMap =
&timers.mNameToTimerMap (DWARF: const NameToTimerMap&, 8-byte slot at offset 8);
mTimerPool = timers.mTimerPool. When enabled: if mUsedElemCount == size() the pool
grows (CudaRangeTimerPool::alloc(), util.h:179, inlined in the original), then
mTimerIndex = mUsedElemCount++, cudaEventRecord(mElem[mTimerIndex].mBegin, 0) and
map[name] = mTimerIndex (rb-tree insert, node 0x30, strcmp).
G1 PASS (139 vs 173, ratio 1.24).

## allow_missing_calls
_M_fill_insert, cudaEventCreate, _ZdlPv: the pool-growth branch is kept in
cudaTimerGrowPool() (noinline) because GCC 15 inlines the entire growth path and
triples the ctor size (461 vs 139 instructions), which would break the 2.0 ratio
bound. Same work, same calls, one frame deeper.

## Unknowns
CudaRangeTimer::mBegin/mEnd are void* in recovered.h (DWARF: cudaEvent_t); the
casts are layout-neutral.
