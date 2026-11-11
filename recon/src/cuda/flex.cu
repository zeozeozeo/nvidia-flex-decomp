// flexdecomp: src/cuda/flex.cu
//
// Shared device constant storage for the solver translation units.
//
// Evidence: the shipped PTX declares
//   .extern .const .align 4 .b8 gParams[568]
//   .extern .const .align 4 .b8 gBounds[24]
// and the recovered DWARF types are `KernelParams` (568 B, 56 members) and
// `KernelBounds` (24 B).  With CUDA separable compilation the storage must be
// defined exactly once; every other part file declares it `extern __constant__`.
//
// flexdecomp: src/cuda/flex.cu:1 gParams
// flexdecomp: src/cuda/flex.cu:1 gBounds

#include "nvflex/all.h"

#include <cuda_runtime.h>

__constant__ KernelParams gParams;
__constant__ KernelBounds gBounds;
