// flexdecomp: reconstructed D3D backend — NVAPI shim declarations.
//
// See NvApiShim.cpp for what is forwarded and why.  Every function returns
// NVAPI_NOT_SUPPORTED (-1) when `nvapi64.dll` (or the specific entry point) is
// unavailable, which is exactly the failure mode the shipped code handles.
#pragma once

#include <d3d11.h>

#define NVAPI_OK 0
#define NVAPI_NOT_SUPPORTED (-1)

int NvApiInitialize();

int NvApiEnumPhysicalGPUs(void** handles, unsigned* count);
int NvApiGpuGetPCIIdentifiers(void* gpu, unsigned* deviceId, unsigned* subSystemId,
                              unsigned* subVendorId, unsigned* revisionId);
int NvApiGpuGetShaderSubPipeCount(void* gpu, unsigned* count);
int NvApiGetErrorMessage(int status, void* desc);

int NvApiSetNvShaderExtnSlot(ID3D11Device* device, unsigned slot);
int NvApiOpenResourceOnComputeDevice(ID3D11Device* device, ID3D11Resource* resource,
                                     ID3D11Resource** opened);
int NvApiSignalSynchronizationObjectGpu(ID3D11Device* device, void* handle, void* key);
int NvApiWaitForSynchronizationObjectGpu(ID3D11Device* device, void* handle, void* key);
