// Build-level compatibility shim (recon/core/* stay byte-identical to FleX).
//
// maths.cpp declares the static member specialization
//     template <> XMatrix44<float> XMatrix44<float>::kIdentity(...)
// *after* including maths.h, whose inline helpers already reference
// Mat44::kIdentity.  GCC 4.8.1 accepted that ordering; modern GCC rejects it
// as "explicit specialization after instantiation".  Declaring the
// specialization before the definition restores the original semantics
// without touching the shipped source.
#pragma once
#include <sstream>   // platform.h relies on a transitive include (libstdc++ 4.8 behaviour)
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#include "maths.h"

template <>
XMatrix44<float> XMatrix44<float>::kIdentity;
