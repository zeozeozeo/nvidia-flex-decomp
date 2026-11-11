# CudaTimer::~CudaTimer() (D1)

Declared `util.h:274`; 46 bytes / 12 instructions in `.text._ZN9CudaTimerD2Ev`.
`if (mEnabled) cudaEventRecord(mTimerPool->mElem[mTimerIndex].mEnd, 0);`
(+0x8 inside the 24-byte CudaRangeTimer = mEnd).
G1 PASS (12 vs 12). Parity needs-gpu.
