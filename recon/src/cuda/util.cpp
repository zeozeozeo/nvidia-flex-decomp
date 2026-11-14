// Reconstructed FleX 1.2.0 CUDA utility layer - src/cuda/util.cpp (519 lines).
//
// Evidence:
//   work/decomp/nvflex.util/util.cpp.c                                  pseudocode
//   work/binary/obj/dbg/NvFlexDebugCUDA_x64/dump/util.cpp.disasm.txt    disasm
//   work/dwarf/nvflex.util.json                    signatures, lines, layouts
//
// The shipped release object is compiled with _FORTIFY_SOURCE (FlexLogFunc calls
// __vsnprintf_chk, not vsnprintf), so the same level is requested here before any
// header is read; the debug object, built -O0 where fortification is inert, calls
// the plain libc symbol.
#if defined(__OPTIMIZE__) && !defined(_FORTIFY_SOURCE)
#define _FORTIFY_SOURCE 2
#endif

// util.cpp is host C++ and the reconstruction is compiled without the CUDA
// toolkit headers on the compiler include path (-I recon/include only), so the
// CUDA driver/runtime interfaces the shipped object referenced are declared
// below with the exact ABI of the CUDA 9.2 headers it was compiled against
// (cudaResourceDesc 64 B, cudaTextureDesc 64 B, cudaChannelFormatDesc 20 B --
// all confirmed by work/dwarf/nvflex.util.json).

// CUcontext is CUctx_st* in DWARF (util.cpp:90 signature); recovered.h would
// otherwise flatten it to void*, which changes the mangling of every unit that
// takes a context (_Z14CudaSetContextP8CUctx_st).
#define FLEXDECOMP_DEF_CUcontext
struct CUctx_st;
typedef CUctx_st* CUcontext;

#include "NvFlex.h"
#include "nvflex/all.h"

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <cstring>

#if defined(_MSC_VER)
// Windows/PE build: the CUDA driver is nvcuda.dll, loaded through the loader
// API (the shipped Windows DLL imports GetProcAddress from KERNEL32; the
// Linux build dlopen()s libcuda.so instead).
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// ---------------------------------------------------------------------------
// CUDA runtime interface (cuda_runtime_api.h, CUDA 9.2 ABI)
// ---------------------------------------------------------------------------
extern "C" {

enum cudaError { cudaSuccess = 0 };
typedef enum cudaError cudaError_t;  // DWARF: cudaError_t (4 bytes)

enum cudaChannelFormatKind {
	cudaChannelFormatKindSigned = 0,
	cudaChannelFormatKindUnsigned = 1,
	cudaChannelFormatKindFloat = 2,
};

enum cudaResourceType {
	cudaResourceTypeArray = 0,
	cudaResourceTypeMipmappedArray = 1,
	cudaResourceTypeLinear = 2,  // disasm: resType = 2 in all three CreateTexture
	cudaResourceTypePitch2D = 3,
};

enum cudaTextureAddressMode {
	cudaAddressModeWrap = 0,
	cudaAddressModeClamp = 1,
	cudaAddressModeMirror = 2,
	cudaAddressModeBorder = 3,
};

enum cudaTextureFilterMode {
	cudaFilterModePoint = 0,
	cudaFilterModeLinear = 1,
};

enum cudaTextureReadMode {
	cudaReadModeElementType = 0,  // disasm: texDesc.readMode = 0
	cudaReadModeNormalizedFloat = 1,
};

// DWARF: cudaChannelFormatDesc, 20 bytes, f at 0x10
struct cudaChannelFormatDesc {
	int x;
	int y;
	int z;
	int w;
	enum cudaChannelFormatKind f;
};

// DWARF: cudaResourceDesc, 64 bytes, resType at 0x0, union at 0x8
// Windows/CUDA 13 ABI: the runtime adds a trailing `unsigned int flags`
// (driver_types.h:1664, after the union) and reads it at every
// cudaCreateTextureObject call - without it the runtime reads past this
// struct and rejects the call with cudaErrorInvalidValue.  sizeof grows
// 64 -> 72.  (The Linux/ELF path keeps the shipped CUDA 9.2 layout the
// static gate pins.)
struct cudaResourceDesc {
	enum cudaResourceType resType;
	union {
		struct { cudaArray_t array; } array;
		struct { cudaArray_t array; size_t offset; size_t sizeInBytes; } mipmap;
		struct { void* devPtr; struct cudaChannelFormatDesc desc; size_t sizeInBytes; } linear;
		struct {
			void* devPtr;
			struct cudaChannelFormatDesc desc;
			size_t width;
			size_t height;
			size_t pitchInBytes;
		} pitch2D;
	} res;
#if defined(_MSC_VER)
	unsigned int flags;
#endif
};

// DWARF: cudaTextureDesc, 64 bytes, readMode at 0x10, borderColor at 0x18
// Windows/CUDA 13 ABI: two trailing ints (disableTrilinearOptimization,
// seamlessCubemap; cuda_runtime_api.h:9230) - sizeof 64 -> 72.
struct cudaTextureDesc {
	enum cudaTextureAddressMode addressMode[3];
	enum cudaTextureFilterMode filterMode;
	enum cudaTextureReadMode readMode;
	int sRGB;
	float borderColor[4];
	int normalizedCoords;
	unsigned int maxAnisotropy;
	enum cudaTextureFilterMode mipmapFilterMode;
	float mipmapLevelBias;
	float minMipmapLevelClamp;
	float maxMipmapLevelClamp;
#if defined(_MSC_VER)
	int disableTrilinearOptimization;
	int seamlessCubemap;
#endif
};

struct cudaResourceViewDesc;  // opaque: cudaCreateTextureObject's 4th argument (always null here)

cudaError_t cudaSetDevice(int device);
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                                    const struct cudaResourceDesc* pResDesc,
                                    const struct cudaTextureDesc* pTexDesc,
                                    const struct cudaResourceViewDesc* pResViewDesc);
cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject);

}  // extern "C"

static_assert(sizeof(cudaChannelFormatDesc) == 20, "DWARF: cudaChannelFormatDesc 20 B");
#if defined(_MSC_VER)
static_assert(sizeof(cudaResourceDesc) == 72, "CUDA 13 ABI: cudaResourceDesc 72 B");
static_assert(sizeof(cudaTextureDesc) == 72, "CUDA 13 ABI: cudaTextureDesc 72 B");
#else
static_assert(sizeof(cudaResourceDesc) == 64, "DWARF: cudaResourceDesc 64 B");
static_assert(sizeof(cudaTextureDesc) == 64, "DWARF: cudaTextureDesc 64 B");
#endif

// ---------------------------------------------------------------------------
// CUDA driver interface (cuda.h), resolved dynamically by CudaSafeDelayLoad.
// The typedefs are the DWARF types util.h declares (cuInit_t, cuCtxGetCurrent_t,
// cuCtxSetCurrent_t, cuCtxCreate_t, cuCtxDestroy_t, cuDeviceGet_t).
// ---------------------------------------------------------------------------
// DWARF: cudaError_enum / CUresult (4-byte enumeration, CUDA_SUCCESS == 0)
enum cudaError_enum { CUDA_SUCCESS = 0 };
typedef enum cudaError_enum CUresult;
typedef int CUdevice;

typedef CUresult (*cuInit_t)(unsigned int Flags);
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext* pctx);
typedef CUresult (*cuCtxSetCurrent_t)(CUcontext ctx);
typedef CUresult (*cuCtxCreate_t)(CUcontext* pctx, unsigned int flags, CUdevice dev);
typedef CUresult (*cuCtxDestroy_t)(CUcontext ctx);
typedef CUresult (*cuDeviceGet_t)(CUdevice* device, int ordinal);
typedef CUresult (*cuDriverGetVersion_t)(int* driverVersion);

// recovered.h keeps pfnCuDriverGetVersion_t opaque (typedef void*), the real
// driver prototype is cuDriverGetVersion_t above.

// Driver entry points, resolved on first use (nm: .bss 0x8..0x30)
static cuInit_t cuInit_f;
static cuCtxGetCurrent_t cuCtxGetCurrent_f;
static cuCtxSetCurrent_t cuCtxSetCurrent_f;
static cuCtxCreate_t cuCtxCreate_f;
static cuCtxDestroy_t cuCtxDestroy_f;
static cuDeviceGet_t cuDeviceGet_f;

// The error callback member of NvFlexLibrary is recovered as opaque bytes
// (recovered.h: "DWARF: NvFlexErrorCallback (opaque)"); DWARF places it at
// offset 0 with type NvFlexErrorCallback.
static inline NvFlexErrorCallback FlexGetErrorFunc(const NvFlexLibrary* lib)
{
	return *reinterpret_cast<const NvFlexErrorCallback*>(lib->mErrorFunc);
}

// flexdecomp: src/cuda/util.cpp:49 _Z11FlexLogFuncPFv19NvFlexErrorSeverityPKcS1_iES_S1_S1_iz
// DWARF: FlexLogFunc(NvFlexErrorCallback func, NvFlexErrorSeverity type,
//                    const char* msg, const char* file, int line, ...)
// disasm: 0x0..0x108, char buf[2048] at -0x8c0, vsnprintf(buf, 0x800, msg, args)
void FlexLogFunc(NvFlexErrorCallback func, NvFlexErrorSeverity type, const char* msg,
                 const char* file, int line, ...)
{
	if (func) {
		char buf[2048];

		va_list args;
		va_start(args, line);
		vsnprintf(buf, sizeof(buf), msg, args);
		va_end(args);

		func(type, buf, file, line);
	}
}

// Log calls carry the shipped translation-unit path and a trailing empty
// variadic argument, exactly as the object does (disasm 0x19b/0x1a8: r9 =
// ".rodata+0xe" (""), rcx = ".rodata+0xf" ("./../../../src/cuda/util.cpp")).
#define FLEX_LOG_ERROR(func, msg, line)                                                 \
	FlexLogFunc((func), eNvFlexLogError, (msg), "./../../../src/cuda/util.cpp", (line), "")

// flexdecomp: src/cuda/util.cpp:81 _ZL14GetProcAddressPvPKc
// The shipped release object folds this helper into all six CudaSafeDelayLoad
// call sites and keeps no out-of-line copy; at -O3 the reconstruction does the
// same, so CudaSafeDelayLoad calls dlsym directly (object parity with
// libnvflex_rel.so).  The debug object's out-of-line copy is the ledger's
// provenance for the unit, which is therefore debug-only.
#if defined(_MSC_VER)
// Windows: the helper is the KERNEL32 GetProcAddress import (the shipped
// Windows DLL resolves driver entry points through it).  The static overload
// keeps the void* call sites unchanged; the ::-qualified call inside reaches
// the Windows API.
static void* GetProcAddress(void* handle, const char* name)
{
	return (void*)::GetProcAddress((HMODULE)handle, name);
}
#else
static void* GetProcAddress(void* handle, const char* name)
{
	return dlsym(handle, name);
}
#endif

// flexdecomp: src/cuda/util.cpp:90 _Z17CreateCudaContextiPP8CUctx_st
// disasm 0x12e: cudaSetDevice(ordinal); cuCtxGetCurrent_f(outContext); return true;
bool CreateCudaContext(int deviceOrdinal, CUcontext* outContext)
{
	cudaSetDevice(deviceOrdinal);
	cuCtxGetCurrent_f(outContext);

	return true;
}

// flexdecomp: src/cuda/util.cpp:114 _Z17CudaSafeDelayLoadPFv19NvFlexErrorSeverityPKcS1_iEiPP8CUctx_st
// disasm 0x15e..0x4bb. Returns true only when this call created a context;
// an already-current context is handed back through outContext with false.
	bool CudaSafeDelayLoad(NvFlexErrorCallback errorFunc, int deviceOrdinal, CUcontext* outContext)
{
#if defined(_MSC_VER)
	// Windows: nvcuda.dll replaces libcuda.so (same exported entry-point names:
	// cuInit, cuCtxSetCurrent, cuCtxGetCurrent, cuCtxCreate, cuCtxDestroy,
	// cuDeviceGet, cuDriverGetVersion are all present in the driver DLL).
	void* hCudaDriver = (void*)LoadLibraryA("nvcuda.dll");
#else
	void* hCudaDriver = dlopen("libcuda.so", RTLD_NOW);
#endif

	if (hCudaDriver == 0) {
		if (errorFunc)
			FLEX_LOG_ERROR(errorFunc,
			               "Could not open CUDA driver, ensure you have the latest driver "
			               "(nvcuda.dll) installed and an NVIDIA GPU present on the system",
			               0x7c);
		return false;
	}

	// disasm 0x1d3..0x26b: resolution order cuInit, cuCtxSetCurrent,
	// cuCtxGetCurrent, cuCtxCreate, cuCtxDestroy, cuDeviceGet
	cuInit_f = (cuInit_t)GetProcAddress(hCudaDriver, "cuInit");
	cuCtxSetCurrent_f = (cuCtxSetCurrent_t)GetProcAddress(hCudaDriver, "cuCtxSetCurrent");
	cuCtxGetCurrent_f = (cuCtxGetCurrent_t)GetProcAddress(hCudaDriver, "cuCtxGetCurrent");
	cuCtxCreate_f = (cuCtxCreate_t)GetProcAddress(hCudaDriver, "cuCtxCreate");
	cuCtxDestroy_f = (cuCtxDestroy_t)GetProcAddress(hCudaDriver, "cuCtxDestroy");
	cuDeviceGet_f = (cuDeviceGet_t)GetProcAddress(hCudaDriver, "cuDeviceGet");

	if (cuInit_f == 0) {
		if (errorFunc)
			FLEX_LOG_ERROR(errorFunc, "cuInit missing in nvcuda.dll", 0x8a);
		return false;
	}

	CUresult status = cuInit_f(0);
	if (status != CUDA_SUCCESS) {
		if (errorFunc)
			FLEX_LOG_ERROR(errorFunc,
			               "Could not open CUDA driver, ensure you have the latest driver "
			               "(nvcuda.dll) installed and an NVIDIA GPU present on the system",
			               0x91);
		return false;
	}

	pfnCuDriverGetVersion_t pfnCuDriverGetVersion =
	    (pfnCuDriverGetVersion_t)GetProcAddress(hCudaDriver, "cuDriverGetVersion");
	if (pfnCuDriverGetVersion == 0) {
		if (errorFunc)
			FLEX_LOG_ERROR(errorFunc, "cuDriverGetVersion missing in nvcuda.dll.", 0x9a);
		return false;
	}

	int mDriverVersion;
	status = ((cuDriverGetVersion_t)pfnCuDriverGetVersion)(&mDriverVersion);
	if (status != CUDA_SUCCESS) {
		if (errorFunc)
			FLEX_LOG_ERROR(errorFunc, "Retrieving CUDA driver version failed.", 0xa2);
		return false;
	}

	// disasm 0x3cd: cmp $0x233b / jg -- 0x233c = 9020 = CUDA 9.2 driver version
	if (mDriverVersion < 0x233c) {
		if (errorFunc)
			FlexLogFunc(errorFunc, eNvFlexLogError,
			            "CUDA driver version is %u, expected at least %u",
			            "./../../../src/cuda/util.cpp", 0xa8, mDriverVersion, 0x233c);
		return false;
	}

	CUcontext currentCtx;
	cuCtxGetCurrent_f(&currentCtx);

	if (currentCtx == 0) {
		if (deviceOrdinal == -1)
			deviceOrdinal = 0;

		if (!CreateCudaContext(deviceOrdinal, outContext)) {
			if (errorFunc)
				FLEX_LOG_ERROR(errorFunc, "Could not create CUDA context\n", 0xb8);
			return false;
		}

		return true;
	}

	*outContext = currentCtx;

	return false;
}

// flexdecomp: src/cuda/util.cpp:198 _Z10CudaUnloadP8CUctx_st
void CudaUnload(CUcontext context)
{
	if (context) {
		cuCtxDestroy_f(context);
		cuCtxSetCurrent_f(0);
	}
}

// flexdecomp: src/cuda/util.cpp:208 _Z14CudaGetContextv
// DWARF: parameter storage locked (no parameters)
CUcontext CudaGetContext(void)
{
	CUcontext ctx;

	cuCtxGetCurrent_f(&ctx);

	return ctx;
}

// flexdecomp: src/cuda/util.cpp:214 _Z14CudaSetContextP8CUctx_st
void CudaSetContext(CUcontext ctx)
{
	cuCtxSetCurrent_f(ctx);
}

// flexdecomp: src/cuda/util.cpp:219 _Z16CudaContextCheckPFv19NvFlexErrorSeverityPKcS1_iEP8CUctx_st
void CudaContextCheck(NvFlexErrorCallback errorFunc, CUcontext expected)
{
	if (CudaGetContext() != expected && errorFunc)
		FLEX_LOG_ERROR(errorFunc, "Called Flex with unexpected CUDA context.", 0xdf);
}

// flexdecomp: src/cuda/util.cpp:335 _Z10DumpMemoryPKc
// disasm 0x58f: empty in the shipped object (parameter spill then return)
void DumpMemory(const char* point)
{
	(void)point;
}

// flexdecomp: src/cuda/util.cpp:339 _Z24FlexDeviceMallocInternalPPvmPKcS2_i
// The label/file/line arguments are accepted for the debug allocator and not
// used by this configuration (single cudaMalloc call, no logging).
void FlexDeviceMallocInternal(void** ptr, size_t size, const char* label, const char* file,
                              int line)
{
	cudaMalloc(ptr, size);
}

// flexdecomp: src/cuda/util.cpp:351 _Z22FlexDeviceFreeInternalPv
void FlexDeviceFreeInternal(void* ptr)
{
	cudaFree(ptr);
}

// flexdecomp: src/cuda/util.cpp:363 _Z13CreateTexturePKfi
cudaTextureObject_t CreateTexture(const float* buffer, int sizeInBytes)
{
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;  // disasm 0x624: f = 2
	resDesc.res.linear.desc.x = 32;                          // disasm 0x62b: 0x20
	resDesc.res.linear.sizeInBytes = (long)sizeInBytes;
	resDesc.res.linear.devPtr = (void*)buffer;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;  // disasm 0x654: 0

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, 0);

	return tex;
}

// flexdecomp: src/cuda/util.cpp:389 _Z13CreateTexturePK8XVector4IfEi
cudaTextureObject_t CreateTexture(const Vec4* buffer, int sizeInBytes)
{
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;  // disasm 0x6ca: f = 2
	resDesc.res.linear.desc.x = 32;                          // disasm 0x6d1: 0x20
	resDesc.res.linear.desc.y = 32;
	resDesc.res.linear.desc.z = 32;
	resDesc.res.linear.desc.w = 32;
	resDesc.res.linear.sizeInBytes = (long)sizeInBytes;
	resDesc.res.linear.devPtr = (void*)buffer;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;  // disasm 0x70f: 0

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, 0);

	return tex;
}

// flexdecomp: src/cuda/util.cpp:417 _Z13CreateTexturePKii
cudaTextureObject_t CreateTexture(const int* buffer, int sizeInBytes)
{
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.desc.f = cudaChannelFormatKindSigned;  // disasm 0x785: f = 0
	resDesc.res.linear.desc.x = 32;                           // disasm 0x78c: 0x20
	resDesc.res.linear.sizeInBytes = (long)sizeInBytes;
	resDesc.res.linear.devPtr = (void*)buffer;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;  // disasm 0x7b5: 0

	cudaTextureObject_t tex;
	cudaCreateTextureObject(&tex, &resDesc, &texDesc, 0);

	return tex;
}

// flexdecomp: src/cuda/util.cpp:443 _Z14DestroyTexturey
void DestroyTexture(cudaTextureObject_t tex)
{
	if (tex)
		cudaDestroyTextureObject(tex);
}

// flexdecomp: src/cuda/util.cpp:456 _Z9FlexAllocP13NvFlexLibraryi
// The shipped object returns ptr uninitialised when cudaMallocHost fails and no
// error callback is installed (disasm 0x82b..0x883: no store to ptr outside the
// call); reproduced verbatim.
void* FlexAlloc(NvFlexLibrary* lib, int size)
{
	void* ptr;

	cudaError_t err = cudaMallocHost(&ptr, (long)size);
	if (err != cudaSuccess) {
		NvFlexErrorCallback errorFunc = FlexGetErrorFunc(lib);
		if (errorFunc)
			FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
			            "./../../../src/cuda/util.cpp", 0x1cd, err, "cudaMallocHost(&ptr, size)");
	}

	return ptr;
}

// ---------------------------------------------------------------------------
// NvFlexBuffer (util.h:138-142).  recovered.h recovers only the DWARF data
// members of the class, so the out-of-line members are given the shipped
// mangled names directly with an assembler label.
//
// The shipped object aliases the C1/C2 (and D1/D2) symbol pairs at one address
// (nm: 0x894 / 0x9f8); the alias reproduces that exactly.
// ---------------------------------------------------------------------------
#if !defined(_MSC_VER)
extern "C" {

void NvFlexBuffer_construct(NvFlexBuffer* self, int elementSize, int byteStride, int type,
                            NvFlexLibrary* lib) __asm__("_ZN12NvFlexBufferC2EiiiP13NvFlexLibrary");
void NvFlexBuffer_construct_c1(NvFlexBuffer* self, int elementSize, int byteStride, int type,
                               NvFlexLibrary* lib) __asm__("_ZN12NvFlexBufferC1EiiiP13NvFlexLibrary")
    __attribute__((alias("_ZN12NvFlexBufferC2EiiiP13NvFlexLibrary")));

void NvFlexBuffer_destroy(NvFlexBuffer* self) __asm__("_ZN12NvFlexBufferD2Ev");
void NvFlexBuffer_destroy_d1(NvFlexBuffer* self) __asm__("_ZN12NvFlexBufferD1Ev")
    __attribute__((alias("_ZN12NvFlexBufferD2Ev")));

void* NvFlexBuffer_map(NvFlexBuffer* self) __asm__("_ZN12NvFlexBuffer3mapEv");
void NvFlexBuffer_unmap(NvFlexBuffer* self) __asm__("_ZN12NvFlexBuffer5unmapEv");

}  // extern "C"
#else
// PE build: plain C++ linkage; flex.part2.cu declares these without asm labels
void NvFlexBuffer_construct(NvFlexBuffer* self, int elementSize, int byteStride, int type,
                            NvFlexLibrary* lib);
void NvFlexBuffer_destroy(NvFlexBuffer* self);
void* NvFlexBuffer_map(NvFlexBuffer* self);
void NvFlexBuffer_unmap(NvFlexBuffer* self);
#endif

// flexdecomp: src/cuda/util.cpp:471 (util.h:138) _ZN12NvFlexBufferC2EiiiP13NvFlexLibrary
// DWARF prototype: NvFlexBuffer(NvFlexBuffer* this, int elementSize, int byteStride, int type,
//                                NvFlexLibrary* lib)
// disasm 0x894: field order mapped, type, graphicsResource, lib, ptr,
// elementCount (= elementSize), elementByteStride (= byteStride)
void NvFlexBuffer_construct(NvFlexBuffer* self, int elementSize, int byteStride, int type,
                            NvFlexLibrary* lib)
{
	self->mapped = false;
	self->type = type;
	self->graphicsResource = 0;
	self->lib = lib;
	self->ptr = 0;
	self->elementCount = elementSize;
	self->elementByteStride = byteStride;

	if (type == eNvFlexBufferHost) {
		cudaError_t err = cudaMallocHost(&self->ptr,
		                                 (long)(self->elementByteStride * self->elementCount));
		if (err != cudaSuccess) {
			NvFlexErrorCallback errorFunc = FlexGetErrorFunc(lib);
			if (errorFunc)
				FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
				            "./../../../src/cuda/util.cpp", 0x1dd, err,
				            "cudaMallocHost(&ptr, elementCount*elementByteStride)");
		}
	} else if (type == eNvFlexBufferDevice) {
		cudaError_t err = cudaMalloc(&self->ptr,
		                             (long)(self->elementByteStride * self->elementCount));
		if (err != cudaSuccess) {
			NvFlexErrorCallback errorFunc = FlexGetErrorFunc(lib);
			if (errorFunc)
				FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
				            "./../../../src/cuda/util.cpp", 0x1e1, err,
				            "cudaMalloc(&ptr, elementCount*elementByteStride)");
		}
	}
}

// flexdecomp: src/cuda/util.cpp:485 (util.h:139) _ZN12NvFlexBufferD2Ev
void NvFlexBuffer_destroy(NvFlexBuffer* self)
{
	if (self->type == eNvFlexBufferHost) {
		cudaError_t err = cudaFreeHost(self->ptr);
		if (err != cudaSuccess) {
			NvFlexErrorCallback errorFunc = FlexGetErrorFunc(self->lib);
			if (errorFunc)
				FlexLogFunc(errorFunc, eNvFlexLogError, "Cuda error: %d in %s",
				            "./../../../src/cuda/util.cpp", 0x1e9, err, "cudaFreeHost(ptr)");
		}
		self->ptr = 0;
	} else if (self->type == eNvFlexBufferDevice) {
		cudaFree(self->ptr);
		self->ptr = 0;
	}
}

// flexdecomp: src/cuda/util.cpp:499 _ZN12NvFlexBuffer3mapEv
void* NvFlexBuffer_map(NvFlexBuffer* self)
{
	if (self->mapped) {
		NvFlexErrorCallback errorFunc = FlexGetErrorFunc(self->lib);
		if (errorFunc)
			FlexLogFunc(errorFunc, eNvFlexLogWarning,
			            "Mapping a buffer that was already mapped",
			            "./../../../src/cuda/util.cpp", 0x1f7, "");
	}

	self->mapped = true;

	return self->ptr;
}

// flexdecomp: src/cuda/util.cpp:511 _ZN12NvFlexBuffer5unmapEv
void NvFlexBuffer_unmap(NvFlexBuffer* self)
{
	if (!self->mapped) {
		NvFlexErrorCallback errorFunc = FlexGetErrorFunc(self->lib);
		if (errorFunc)
			FlexLogFunc(errorFunc, eNvFlexLogWarning,
			            "Unmapping a buffer that was not mapped",
			            "./../../../src/cuda/util.cpp", 0x203, "");
	}

	self->mapped = false;
}
