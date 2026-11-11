# CudaRangeTimerPool::CudaRangeTimerPool() (C2)

Alias of the C1 body, matching the shipped object where C1 and C2 share one
section.

BLOCKED: the static gate cannot evaluate an aliased symbol. objdump emits one
function header per address, so `_disasm_map` reports `orig: 0` instructions for
C2 and `low <= ratio <= 2.0 if oi else False` fails regardless of the
reconstruction. Symbol, signature and call checks all pass (nm shows the symbol,
readelf shows FUNC WEAK size 194). Needs the same "both sides zero => pass" rule
the kernel PTX path received.
