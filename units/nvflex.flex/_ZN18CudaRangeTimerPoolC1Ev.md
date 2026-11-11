# CudaRangeTimerPool::CudaRangeTimerPool() (C1)

Declared `util.h:219`; the shipped object emits one weak body in
`.text._ZN18CudaRangeTimerPoolC2Ev` (194 bytes / 48 instructions) that both C1 and
C2 name.

recovered.h carries the class data-only (no ctor declaration), so the shipped
symbol is emitted from `recon/src/cuda/flex.part1.cu` through an explicit
assembler label; C2 is an `alias` of the same body, exactly like the original.

Evidence (disasm): zero-initialises the std::vector<CudaRangeTimer> at offset 0,
sets mUsedElemCount(offset 24) = 0, fills 200 elements (_M_fill_insert), then 200
iterations (0x12c0/0x18) of cudaEventCreate(&mBegin), cudaEventCreate(&mEnd).
mInitCapacity = 200 (mov $0xc8).
G1 PASS (48 vs 44, ratio 0.92). Parity needs-gpu (cudaEventCreate).

## allow_missing_calls
_M_fill_insert and _ZdlPv: GCC 15 inlines vector<CudaRangeTimer>'s fill for this
trivial 24-byte element and needs no exception cleanup, so neither is a direct
callee of the reconstructed symbol. Behaviour identical; the instantiation is
still emitted in the module.
