// flexdecomp: reconstructed src/device/flexDevice.cpp (FleX 1.2.0).
//
// The shipped object work/binary/obj/dbg/NvFlexDeviceDebug_x64/flexDevice.cpp.o
// was built with
//   GNU C++ 4.8.1 -m64 -mtune=generic -march=x86-64 -g -O0 -std=c++11 -fPIC
//                  -fpermissive -fno-strict-aliasing -fstack-protector
// against CUDA 9.2's cuda.h (DWARF decl_file 3 =
// external/CUDA/9.2.148-linux/include/cuda.h) and PhysXDevice.h (decl_file 4).
//
// This translation unit is the whole of the optional NvFlexDevice helper
// library: it lazily dlopen()s the CUDA driver, fetches the PhysX "low latency"
// context-creation entry point out of the driver export table, and creates the
// context used by NvFlexInit().  All driver entry points stay in file-scope
// pointers so the library loads on hosts without libcuda.so.
//
// The CUDA driver declarations below are the subset of cuda.h (9.2.148) that
// this file actually uses; the shipped header is not vendored in the tree.
// Each one records the cuda.h line the DWARF points at.

#include "NvFlexDevice.h"

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <stddef.h>
#include <cstdio>

#include "PhysXDevice.h"

// Codegen note.  The two shipped objects disagree on this file:
//   * the debug object (DWARF source of truth) is -O0 and keeps
//     GetProcAddress / GetSuggestedDeviceOrdinal / CudaSafeDelayLoad out of
//     line;
//   * the release object (the shipped ABI, and the static gate's baseline for
//     every symbol it defines) is optimised and folds all three into their
//     single callers.
// Both shapes are therefore reproduced: the -O0 pragma below gives each
// helper's own symbol its un-inlined body, and the `*Impl` bodies are forced
// inline so NvFlexDeviceGetSuggestedOrdinal()/NvFlexDeviceCreateCudaContext()
// contain the folded code the release object has.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC optimize ("O0")
#endif

#if defined(_MSC_VER)
#define FLEX_FORCEINLINE __forceinline
#else
#define FLEX_FORCEINLINE inline __attribute__((always_inline))
#endif

// ---- cuda.h subset (external/CUDA/9.2.148-linux/include/cuda.h) ------------

// flexdecomp: cuda.h:249
typedef int CUdevice;

// flexdecomp: cuda.h:251
struct CUctx_st;

// flexdecomp: cuda.h:250
typedef struct CUctx_st* CUcontext;

// flexdecomp: cuda.h:263/264
struct CUuuid_st
{
	char bytes[16];
};

// flexdecomp: cuda.h:265
typedef struct CUuuid_st CUuuid;

// flexdecomp: cuda.h:310 (values verbatim; the full enum is not reproduced)
enum CUctx_flags_enum
{
	CU_CTX_SCHED_AUTO = 0x00,
	CU_CTX_SCHED_SPIN = 0x01,
	CU_CTX_SCHED_YIELD = 0x02,
	CU_CTX_SCHED_BLOCKING_SYNC = 0x04,
	CU_CTX_BLOCKING_SYNC = 0x04,
	CU_CTX_SCHED_MASK = 0x07,
	CU_CTX_MAP_HOST = 0x08,
	CU_CTX_LMEM_RESIZE_TO_MAX = 0x10,
	CU_CTX_FLAGS_MASK = 0x1f
};

// flexdecomp: cuda.h:1058/1467 - only the success code is tested here
enum cudaError_enum
{
	CUDA_SUCCESS = 0
};

typedef enum cudaError_enum CUresult;

// ---- driver export table (CUetblPhysXInterface) ----------------------------

// Identifier of the driver's PhysX interface export table.  The 16 bytes are
// the .rodata bytes at offset 0 of the shipped object; the symbol has external
// linkage there (GLOBAL, 16 bytes, .rodata+0).
//
// flexdecomp: flexDevice.cpp:51
// The shipped source relied on -fpermissive to narrow the ints into
// CUuuid::bytes; the casts reproduce the same 16 bytes without that flag.
extern const CUuuid CU_ETID_PhysXInterface = { {
	(char)0x0c, (char)0xa5, (char)0x0b, (char)0x8c, (char)0x10, (char)0x04, (char)0x92, (char)0x9a,
	(char)0x89, (char)0xa7, (char)0xd0, (char)0xdf, (char)0x10, (char)0xe7, (char)0x72, (char)0x86
} };

// flexdecomp: flexDevice.cpp:53
struct CUetblPhysXInterface_st
{
	// size of this structure
	// flexdecomp: flexDevice.cpp:56 (size_t, offset 0)
	size_t size;

	// flexdecomp: flexDevice.cpp:64 (offset 8)
	CUresult (*cuCtxCreateOnNode3DLowLatency)(CUcontext* pctx, unsigned int flags, CUdevice dev);
};

// flexdecomp: flexDevice.cpp:66
typedef CUetblPhysXInterface_st CUetblPhysXInterface;

// flexdecomp: flexDevice.cpp:68
typedef CUresult (*cuInit_t)(unsigned int Flags);
typedef CUresult (*cuGetExportTable_t)(const void** ppExportTable, const CUuuid* pExportTableId);
// flexdecomp: flexDevice.cpp:69
typedef CUresult (*cuDeviceGet_t)(CUdevice* device, int ordinal);
// flexdecomp: flexDevice.cpp:70
typedef CUresult (*cuCtxCreate_t)(CUcontext* pctx, unsigned int flags, CUdevice dev);
// flexdecomp: flexDevice.cpp:71
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext ctx);
// flexdecomp: flexDevice.cpp:72
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext* pctx);
// flexdecomp: flexDevice.cpp:73
typedef CUresult (*cuCtxDestroy_t)(CUcontext ctx);

// flexdecomp: flexDevice.cpp:75-80 (order as in the shipped .bss)
static cuInit_t cuInit_f;
static cuGetExportTable_t cuGetExportTable_f;
static cuDeviceGet_t cuDeviceGet_f;
static cuCtxCreate_t cuCtxCreate_f;
static cuCtxDestroy_t cuCtxDestroy_f;
static cuCtxSetCurrent_t cuCtxSetCurrent_f;
static cuCtxGetCurrent_t cuCtxGetCurrent_f;

// flexdecomp: PhysXDevice.cpp:37 - declared in flexDevice.cpp, not in the header
namespace physx
{
	PHYSX_DEV_STATUS physxDevGetCudaOrdinalWrapper(int* cudaDevOrdinal);
}

// ---- entry points ----------------------------------------------------------

// flexdecomp: flexDevice.cpp:84 - folded form (release object)
static FLEX_FORCEINLINE void* GetProcAddressImpl(void* handle, const char* name)
{
#if defined(_MSC_VER)
	return (void*)::GetProcAddress((HMODULE)handle, name);
#else
	return dlsym(handle, name);
#endif
}

// flexdecomp: flexDevice.cpp:84 _ZL14GetProcAddressPvPKc
static void* GetProcAddress(void* handle, const char* name)
{
	return GetProcAddressImpl(handle, name);
}

// flexdecomp: flexDevice.cpp:92 - folded form (release object)
static FLEX_FORCEINLINE int GetSuggestedDeviceOrdinalImpl()
{
	// flexdecomp: flexDevice.cpp:94 (ordinal is written by the PhysX wrapper)
	int ordinal;
	if (physx::physxDevGetCudaOrdinalWrapper(&ordinal) != 0)
		return -1;
	return ordinal;
}

// flexdecomp: flexDevice.cpp:92 _ZL25GetSuggestedDeviceOrdinalv
// noipa keeps this symbol's own body: without it GCC folds the function into
// NvFlexDeviceGetSuggestedOrdinal(), which has the identical inlined body,
// and the debug object's stand-alone 18-instruction body would vanish.
#if !defined(_MSC_VER)
__attribute__((noipa))
#endif
static int GetSuggestedDeviceOrdinal()
{
	return GetSuggestedDeviceOrdinalImpl();
}

// flexdecomp: flexDevice.cpp:108 - folded form (release object)
static FLEX_FORCEINLINE bool CudaSafeDelayLoadImpl(int deviceOrdinal, CUcontext* outContext)
{
	// flexdecomp: flexDevice.cpp:114 - "libcuda.so" at .rodata+0x11, RTLD_NOW
	// (the shipped Windows DLL loads nvcuda.dll; same resolution path)
#if defined(_MSC_VER)
	void* hCudaDriver = (void*)LoadLibraryA("nvcuda.dll");
#else
	void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
#endif

	// flexdecomp: flexDevice.cpp:117/119
	if (hCudaDriver == NULL)
		return false;

	// flexdecomp: flexDevice.cpp:122-127 (literal order in .rodata)
	cuInit_f = (cuInit_t)GetProcAddressImpl(hCudaDriver, "cuInit");
	cuGetExportTable_f = (cuGetExportTable_t)GetProcAddressImpl(hCudaDriver, "cuGetExportTable");
	cuCtxCreate_f = (cuCtxCreate_t)GetProcAddressImpl(hCudaDriver, "cuCtxCreate");
	cuDeviceGet_f = (cuDeviceGet_t)GetProcAddressImpl(hCudaDriver, "cuDeviceGet");
	cuCtxSetCurrent_f = (cuCtxSetCurrent_t)GetProcAddressImpl(hCudaDriver, "cuCtxSetCurrent");
	cuCtxGetCurrent_f = (cuCtxGetCurrent_t)GetProcAddressImpl(hCudaDriver, "cuCtxGetCurrent");
	cuCtxDestroy_f = (cuCtxDestroy_t)GetProcAddressImpl(hCudaDriver, "cuCtxDestroy");

	// flexdecomp: flexDevice.cpp:130
	CUetblPhysXInterface* lowlatency = NULL;

	// flexdecomp: flexDevice.cpp:134 - immediate 0x1c at 0x131
	int flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST | CU_CTX_LMEM_RESIZE_TO_MAX;
	// flexdecomp: flexDevice.cpp:135
	// Divergence (Windows only): the shipped CUDA 9.2-era device DLL calls no
	// cuInit anywhere (verified against its disassembly - no import, no
	// "cuInit" string) because 9.2-era driver entry points initialized the
	// driver lazily.  The CUDA 13 driver requires explicit initialization:
	// without cuInit(0), cuDeviceGet and
	// cuCtxCreateOnNode3DLowLatency return CUDA_ERROR_NOT_INITIALIZED (3)
	// and NvFlexDeviceCreateCudaContext would always fail.  Recorded as a
	// behaviour-preserving adaptation in units/nvflexdevice.flexdevice/.
#if defined(_MSC_VER)
	cuInit_f(0);
#endif
	cuGetExportTable_f((const void**)&lowlatency, &CU_ETID_PhysXInterface);

	// flexdecomp: flexDevice.cpp:139/140 - -1 selects device 0
	if (deviceOrdinal == -1)
		deviceOrdinal = 0;

	// flexdecomp: flexDevice.cpp:146 - the shipped object computes the
	// suggested ordinal at 0x15f and never consumes it (dead store to
	// -0x18(%rbp)); kept so the call is present in the object.
	int suggestedOrdinal = GetSuggestedDeviceOrdinalImpl();
	(void)suggestedOrdinal;

#if defined(_MSC_VER)
	// FLEX_DEVICE_TRACE diagnostic
	if (GetEnvironmentVariableA("FLEX_DEVICE_TRACE", nullptr, 0))
		fprintf(stderr, "[devtrace] hCuda=%p getTbl=%p lowlatency=%p\n", hCudaDriver,
		        (void*)cuGetExportTable_f, (const void*)lowlatency);
#endif
	// flexdecomp: flexDevice.cpp:149
	if (lowlatency)
	{
		// flexdecomp: flexDevice.cpp:151/152
		CUdevice device;
		cuDeviceGet_f(&device, deviceOrdinal);

		// flexdecomp: flexDevice.cpp:154
		CUresult result = lowlatency->cuCtxCreateOnNode3DLowLatency(outContext, flags, device);
#if defined(_MSC_VER)
		if (GetEnvironmentVariableA("FLEX_DEVICE_TRACE", nullptr, 0))
			fprintf(stderr, "[devtrace] device=%d rc=%d ctx=%p\n", deviceOrdinal,
			        (int)result, outContext ? *outContext : nullptr);
#endif

		// flexdecomp: flexDevice.cpp:155-158
		if (result != CUDA_SUCCESS)
		{
			*outContext = NULL;
			return false;
		}

		// flexdecomp: flexDevice.cpp:163/164
		cuCtxSetCurrent_f(*outContext);
		return true;
	}

	// flexdecomp: flexDevice.cpp:168
	return false;
}

// flexdecomp: flexDevice.cpp:108 _ZL17CudaSafeDelayLoadiPP8CUctx_st
static bool CudaSafeDelayLoad(int deviceOrdinal, CUcontext* outContext)
{
	return CudaSafeDelayLoadImpl(deviceOrdinal, outContext);
}

// flexdecomp: flexDevice.cpp:174 _Z31NvFlexDeviceGetSuggestedOrdinalv
int NvFlexDeviceGetSuggestedOrdinal()
{
	// flexdecomp: flexDevice.cpp:176 (inlined in the release object)
	return GetSuggestedDeviceOrdinalImpl();
}

// flexdecomp: flexDevice.cpp:180 _Z29NvFlexDeviceCreateCudaContexti
bool NvFlexDeviceCreateCudaContext(int ordinal)
{
	// flexdecomp: flexDevice.cpp:182/183 (inlined in the release object)
	CUcontext context;
	return CudaSafeDelayLoadImpl(ordinal, &context);
}

// flexdecomp: flexDevice.cpp:186 _Z30NvFlexDeviceDestroyCudaContextv
void NvFlexDeviceDestroyCudaContext()
{
	// flexdecomp: flexDevice.cpp:188/189
	CUcontext context;
	cuCtxGetCurrent_f(&context);

	// flexdecomp: flexDevice.cpp:191
	cuCtxDestroy_f(context);
}
