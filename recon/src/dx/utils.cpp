// flexdecomp: dx/utils.cpp — first slice (placeholder until the TU lands).
// The public API is C-linkage via the shipped header's `extern "C"` block, so
// including NvFlex.h gives the exported symbols their undecorated names.
#include "NvFlex.h"

NV_FLEX_API int NvFlexGetVersion()
{
    return NV_FLEX_VERSION;   // 120, the value the shipped binary returns
}
