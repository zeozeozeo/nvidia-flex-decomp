// Build-level compatibility prelude for CUDA translation units only.
// platform.h in the shipped FleX sources relies on a transitive <sstream>
// include that libstdc++ 4.8 provided; modern libstdc++ does not.
#pragma once
#include <sstream>
