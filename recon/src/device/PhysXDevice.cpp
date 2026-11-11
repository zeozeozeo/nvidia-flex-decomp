// flexdecomp: reconstructed src/device/PhysXDevice.cpp (FleX 1.2.0).
//
// The shipped object work/binary/obj/dbg/NvFlexDeviceDebug_x64/PhysXDevice.cpp.o
// was built with
//   GNU C++ 4.8.1 -m64 -mtune=generic -march=x86-64 -g -O0 -std=c++11 -fPIC
//                  -fpermissive -fno-strict-aliasing -fstack-protector
// (CU DW_AT_producer).  This is the Linux PhysX device stub layer: there is no
// PhysX runtime on Linux, so every wrapper degrades to a default answer.  The
// only real input is the PHYSX_GPU_DEVICE environment variable read by
// physxDevGetCudaOrdinalWrapper, which is what
// NvFlexDeviceGetSuggestedOrdinal() ends up forwarding (see flexDevice.cpp).

#include <stddef.h>

#include "PhysXDevice.h"

// Host libc entry points used below.  <stdlib.h> is not included because
// glibc's __USE_EXTERN_INLINES replaces atoi() with an inline strtol() (and,
// since glibc 2.39, that strtol with __isoc23_strtol) under any -O build; the
// shipped objects call the plain libc symbols -- debug: atoi@plt, release:
// strtol@plt with base 10 and a NULL endptr (rel PhysXDevice.cpp.o @0x1f) --
// so the symbols are declared directly to preserve the shipped call set.
extern "C" long int strtol(const char* nptr, char** endptr, int base);
extern "C" char* getenv(const char* name);

// The shipped object was compiled with -O0 (DW_AT_producer).  Reproduce that
// optimization level here so the reconstruction has the same un-inlined call
// graph; every function in this file is a thin loader shim.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("O0")
#endif

namespace physx
{

// flexdecomp: PhysXDevice.cpp:37 _ZN5physx29physxDevGetCudaOrdinalWrapperEPi
PHYSX_DEV_STATUS physxDevGetCudaOrdinalWrapper(int* cudaDevOrdinal)
{
	// flexdecomp: PhysXDevice.cpp:39 - "PHYSX_GPU_DEVICE" literal at .rodata+0
	const char* deviceOrdinalString = getenv("PHYSX_GPU_DEVICE");
	// flexdecomp: PhysXDevice.cpp:40/41
	if (deviceOrdinalString == NULL)
		*cudaDevOrdinal = 0;
	// flexdecomp: PhysXDevice.cpp:43.  The shipped source calls atoi(); the
	// release object has it folded to strtol@plt (base 10, NULL endptr) by
	// glibc's inline atoi, and (int)strtol(nptr, NULL, 10) is exactly atoi().
	else
		*cudaDevOrdinal = (int)strtol(deviceOrdinalString, NULL, 10);

	// flexdecomp: PhysXDevice.cpp:44
	return PHYSX_DEV_OK;
}

// flexdecomp: PhysXDevice.cpp:47 _ZN5physx32physxDevUsingDedicatedGPUWrapperEv
PHYSX_DEV_STATUS physxDevUsingDedicatedGPUWrapper()
{
	// flexdecomp: PhysXDevice.cpp:50 - mov $0x0,%eax at 0x4c
	return PHYSX_DEV_OK;
}

// This wrapper is 2 instructions in the shipped release object
// (`xor %eax,%eax; ret`), which the module's -O0 pragma would not produce.
__attribute__((optimize("O3")))
// flexdecomp: PhysXDevice.cpp:54 _ZN5physx25physxDevSLIEnabledWrapperEPv
bool physxDevSLIEnabledWrapper(void*)
{
	// flexdecomp: PhysXDevice.cpp:56 - mov $0x0,%eax at 0x5b
	return false;
}

} // namespace physx
