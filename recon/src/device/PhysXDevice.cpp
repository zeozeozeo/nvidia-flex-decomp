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

#include <stdlib.h>

#include "PhysXDevice.h"

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
	// flexdecomp: PhysXDevice.cpp:43 (PLT32 atoi@0x36)
	else
		*cudaDevOrdinal = atoi(deviceOrdinalString);

	// flexdecomp: PhysXDevice.cpp:44
	return PHYSX_DEV_OK;
}

// flexdecomp: PhysXDevice.cpp:47 _ZN5physx32physxDevUsingDedicatedGPUWrapperEv
PHYSX_DEV_STATUS physxDevUsingDedicatedGPUWrapper()
{
	// flexdecomp: PhysXDevice.cpp:50 - mov $0x0,%eax at 0x4c
	return PHYSX_DEV_OK;
}

// flexdecomp: PhysXDevice.cpp:54 _ZN5physx25physxDevSLIEnabledWrapperEPv
bool physxDevSLIEnabledWrapper(void*)
{
	// flexdecomp: PhysXDevice.cpp:56 - mov $0x0,%eax at 0x5b
	return false;
}

} // namespace physx
