// flexdecomp: reconstructed PhysX device-layer declarations.
//
// Evidence: DWARF decl_file 2 of
//   work/binary/obj/dbg/NvFlexDeviceDebug_x64/{flexDevice,PhysXDevice}.cpp.o
// which names ./../../../src/device/PhysXDevice.h and records
//   namespace physx                 :44
//   physx::PHYSX_DEV_STATUS typedef :47   (underlying type int)
//   PHYSX_DEV_* enumerators         :52   (anonymous enum, int-sized)
//
// The shipped header declares no functions: flexDevice.cpp declares the one
// wrapper it calls itself, and PhysXDevice.cpp defines all three.

#ifndef FLEXDECOMP_PHYSX_DEVICE_H
#define FLEXDECOMP_PHYSX_DEVICE_H

// flexdecomp: PhysXDevice.h:44
namespace physx
{

// flexdecomp: PhysXDevice.h:47
typedef int PHYSX_DEV_STATUS;

// flexdecomp: PhysXDevice.h:52
enum
{
	PHYSX_DEV_OK = 0,
	PHYSX_DEV_UNKNOWN_ERROR = 1,
	PHYSX_DEV_INVALID_HANDLE = 2,
	PHYSX_DEV_UNINITIALIZED = 3,
	PHYSX_DEV_NV_API_UNAVAILABLE = 4,
	PHYSX_DEV_CUDA_UNAVAILABLE = 5,
	PHYSX_DEV_CUDA_MEMORY_ALLOC_FAILURE = 6,
	PHYSX_DEV_LEGACY_MODE_GPU_HANDLE = 7,
	PHYSX_DEV_PHYSX_DEV_UNAVAILABLE = 8
};

} // namespace physx

#endif // FLEXDECOMP_PHYSX_DEVICE_H
