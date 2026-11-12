// flexdecomp: reconstructed D3D backend — NVAPI shim.
//
// The shipped D3D11 backend links `nvapi64.lib` and uses it for two things:
//
//   * shader-extension slots (`NvAPI_D3D11_SetNvShaderExtnSlot`) around compute
//     shader creation, so the NV-specific shader variants can use driver
//     intrinsics;
//   * asynchronously shared resources between the compute and render devices
//     (`NvAPI_D3D11_OpenResourceOnComputeDevice`,
//     `NvAPI_D3D11_SignalSynchronizationObjectGpu`,
//     `NvAPI_D3D11_WaitForSynchronizationObjectGpu`);
//   * GPU enumeration for the SM count and PCI id match
//     (`NvAPI_EnumPhysicalGPUs`, `NvAPI_GPU_GetPCIIdentifiers`,
//     `NvAPI_GPU_GetShaderSubPipeCount`).
//
// None of that is part of the SDK's redistributable output, so the
// reconstruction loads `nvapi64.dll` lazily and degrades exactly like the
// shipped code does on a machine without it: every entry point returns
// NVAPI_NOT_SUPPORTED (-1) and the callers fall back to their generic path.
//
// This is not a stub: `nvapi64.dll` is present on NVIDIA systems (including the
// Wine prefix through the host driver), and the shim forwards to it when it is.
#include "NvApiShim.h"

#include <windows.h>

#include <cstring>

// The NVAPI ABI surface this backend uses.  Declaring the prototypes locally
// keeps the reconstruction free of the NVAPI SDK headers, which are not part of
// the FleX SDK; every declaration matches the published NVAPI signature.
typedef int (*NvAPI_Initialize_t)();
typedef int (*NvAPI_EnumPhysicalGPUs_t)(void** handles, unsigned* count);
typedef int (*NvAPI_GPU_GetPCIIdentifiers_t)(void* gpu, unsigned* deviceId,
                                             unsigned* subSystemId,
                                             unsigned* subVendorId,
                                             unsigned* revisionId);
typedef int (*NvAPI_GPU_GetShaderSubPipeCount_t)(void* gpu, unsigned* count);
typedef int (*NvAPI_GetErrorMessage_t)(int status, void* desc);
typedef int (*NvAPI_D3D11_SetNvShaderExtnSlot_t)(ID3D11Device* device, unsigned slot);
typedef int (*NvAPI_D3D11_OpenResourceOnComputeDevice_t)(ID3D11Device* device,
                                                         ID3D11Resource* resource,
                                                         ID3D11Resource** opened);
typedef int (*NvAPI_D3D11_SignalSynchronizationObjectGpu_t)(ID3D11Device* device,
                                                            void* handle, void* key);
typedef int (*NvAPI_D3D11_WaitForSynchronizationObjectGpu_t)(ID3D11Device* device,
                                                             void* handle, void* key);

namespace
{

struct NvApi
{
    HMODULE module;
    NvAPI_Initialize_t Initialize;
    NvAPI_EnumPhysicalGPUs_t EnumPhysicalGPUs;
    NvAPI_GPU_GetPCIIdentifiers_t GPU_GetPCIIdentifiers;
    NvAPI_GPU_GetShaderSubPipeCount_t GPU_GetShaderSubPipeCount;
    NvAPI_GetErrorMessage_t GetErrorMessage;
    NvAPI_D3D11_SetNvShaderExtnSlot_t D3D11_SetNvShaderExtnSlot;
    NvAPI_D3D11_OpenResourceOnComputeDevice_t D3D11_OpenResourceOnComputeDevice;
    NvAPI_D3D11_SignalSynchronizationObjectGpu_t D3D11_SignalSynchronizationObjectGpu;
    NvAPI_D3D11_WaitForSynchronizationObjectGpu_t D3D11_WaitForSynchronizationObjectGpu;
};

NvApi& api()
{
    static NvApi nvapi;
    static bool loaded = false;

    if (loaded)
        return nvapi;

    loaded = true;
    std::memset(&nvapi, 0, sizeof(nvapi));

    // Load lazily: a machine (or Wine prefix) without the driver DLL must not
    // fail the load of this library.
    nvapi.module = LoadLibraryA("nvapi64.dll");
    if (nvapi.module == 0)
        return nvapi;

    // nvapi_QueryInterface is the only exported symbol; every other entry point
    // is resolved through it with its published id.
    void* (*query)(unsigned) =
        (void* (*)(unsigned))GetProcAddress(nvapi.module, "nvapi_QueryInterface");
    if (query == 0)
        return nvapi;

    nvapi.Initialize = (NvAPI_Initialize_t)query(0x0150E828);
    nvapi.EnumPhysicalGPUs = (NvAPI_EnumPhysicalGPUs_t)query(0xE5AC921F);
    nvapi.GPU_GetPCIIdentifiers = (NvAPI_GPU_GetPCIIdentifiers_t)query(0x2DDFB66E);
    nvapi.GPU_GetShaderSubPipeCount = (NvAPI_GPU_GetShaderSubPipeCount_t)query(0x0BE17923);
    nvapi.GetErrorMessage = (NvAPI_GetErrorMessage_t)query(0x6C2D4EED);
    nvapi.D3D11_SetNvShaderExtnSlot = (NvAPI_D3D11_SetNvShaderExtnSlot_t)query(0xE32CD4F0);
    nvapi.D3D11_OpenResourceOnComputeDevice =
        (NvAPI_D3D11_OpenResourceOnComputeDevice_t)query(0xE2B7E0CB);
    nvapi.D3D11_SignalSynchronizationObjectGpu =
        (NvAPI_D3D11_SignalSynchronizationObjectGpu_t)query(0x9E07CC50);
    nvapi.D3D11_WaitForSynchronizationObjectGpu =
        (NvAPI_D3D11_WaitForSynchronizationObjectGpu_t)query(0xBEECDAA1);
    return nvapi;
}

} // namespace

int NvApiInitialize()
{
    NvApi& nv = api();
    return nv.Initialize != 0 ? nv.Initialize() : NVAPI_NOT_SUPPORTED;
}

int NvApiEnumPhysicalGPUs(void** handles, unsigned* count)
{
    NvApi& nv = api();
    return nv.EnumPhysicalGPUs != 0 ? nv.EnumPhysicalGPUs(handles, count)
                                    : NVAPI_NOT_SUPPORTED;
}

int NvApiGpuGetPCIIdentifiers(void* gpu, unsigned* deviceId, unsigned* subSystemId,
                              unsigned* subVendorId, unsigned* revisionId)
{
    NvApi& nv = api();
    if (nv.GPU_GetPCIIdentifiers == 0)
        return NVAPI_NOT_SUPPORTED;
    return nv.GPU_GetPCIIdentifiers(gpu, deviceId, subSystemId, subVendorId, revisionId);
}

int NvApiGpuGetShaderSubPipeCount(void* gpu, unsigned* count)
{
    NvApi& nv = api();
    return nv.GPU_GetShaderSubPipeCount != 0 ? nv.GPU_GetShaderSubPipeCount(gpu, count)
                                             : NVAPI_NOT_SUPPORTED;
}

int NvApiGetErrorMessage(int status, void* desc)
{
    NvApi& nv = api();
    return nv.GetErrorMessage != 0 ? nv.GetErrorMessage(status, desc)
                                   : NVAPI_NOT_SUPPORTED;
}

int NvApiSetNvShaderExtnSlot(ID3D11Device* device, unsigned slot)
{
    NvApi& nv = api();
    return nv.D3D11_SetNvShaderExtnSlot != 0 ? nv.D3D11_SetNvShaderExtnSlot(device, slot)
                                             : NVAPI_NOT_SUPPORTED;
}

int NvApiOpenResourceOnComputeDevice(ID3D11Device* device, ID3D11Resource* resource,
                                     ID3D11Resource** opened)
{
    NvApi& nv = api();
    if (nv.D3D11_OpenResourceOnComputeDevice == 0)
        return NVAPI_NOT_SUPPORTED;
    return nv.D3D11_OpenResourceOnComputeDevice(device, resource, opened);
}

int NvApiSignalSynchronizationObjectGpu(ID3D11Device* device, void* handle, void* key)
{
    NvApi& nv = api();
    if (nv.D3D11_SignalSynchronizationObjectGpu == 0)
        return NVAPI_NOT_SUPPORTED;
    return nv.D3D11_SignalSynchronizationObjectGpu(device, handle, key);
}

int NvApiWaitForSynchronizationObjectGpu(ID3D11Device* device, void* handle, void* key)
{
    NvApi& nv = api();
    if (nv.D3D11_WaitForSynchronizationObjectGpu == 0)
        return NVAPI_NOT_SUPPORTED;
    return nv.D3D11_WaitForSynchronizationObjectGpu(device, handle, key);
}
