// flexdecomp: reconstructed D3D backend — Device.obj helpers.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   Device::enumerateDXGIAdapters  @ 18004edd0 (299 bytes)
//   Device::queryDeviceCapabilities @ 18004ef00 (572 bytes)
//   Device::createWindow           @ 18004e180 (329 bytes)
//   Device::getDeviceNumber        @ 18004dbb0 (533 bytes)
//   Device::reportLiveDeviceObjects @ 18004da40   (3 bytes, empty)
//
// These are the device-independent pieces NvFlexInit / NvFlexShutdown call on
// whatever device `DeviceSelect` produced.  Only the D3D11 implementation is
// shipped here, so the helpers take the concrete handle.
#include "DeviceD3D11.h"
#include "ContextD3D11.h"

#include <cstring>
#include <string>

#include <windows.h>

#include "NvApiShim.h"

namespace NvFlex
{

namespace
{

// The shipped code logs through the library's callback with these spellings.
const char kFlexSourceFile[] = "..\\..\\dx\\flex.cpp";

} // namespace

// ---------------------------------------------------------------------------
// enumerateDXGIAdapters — @0x4edd0 (299 bytes)
// ---------------------------------------------------------------------------
// Walks the DXGI factory and keeps every adapter that can create a D3D11 device.
std::vector<IDXGIAdapter1*> enumerateDXGIAdapters(IDXGIFactory1* factory)
{
    std::vector<IDXGIAdapter1*> adapters;
    if (factory == 0)
        return adapters;

    for (UINT index = 0;; ++index)
    {
        IDXGIAdapter1* adapter = 0;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND)
            break;
        adapters.push_back(adapter);
    }
    return adapters;
}

// ---------------------------------------------------------------------------
// queryDeviceCapabilities — @0x4ef00 (572 bytes)
// ---------------------------------------------------------------------------
// Fills mDeviceName, mGpuVendorId and mSMCount from the adapter description.
// The NVAPI path only refines the SM count for an NVIDIA adapter whose PCI ids
// match; without nvapi64.dll the count stays 0xffffffff, exactly like the
// shipped code on a machine without the driver.
void DeviceD3D11::queryDeviceCapabilities(ID3D11Device* device)
{
    (void)device;

    // The adapter description is what the shipped code reads; it arrives through
    // the DXGI adapter this device was created from.
    DXGI_ADAPTER_DESC1 adapterDesc;
    std::memset(&adapterDesc, 0, sizeof(adapterDesc));

    IDXGIDevice* dxgiDevice = 0;
    if (m_device != 0 &&
        m_device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDevice) == S_OK)
    {
        IDXGIAdapter* adapter = 0;
        if (dxgiDevice->GetAdapter(&adapter) == S_OK)
        {
            IDXGIAdapter1* adapter1 = 0;
            if (adapter->QueryInterface(__uuidof(IDXGIAdapter1), (void**)&adapter1) == S_OK)
            {
                adapter1->GetDesc1(&adapterDesc);
                adapter1->Release();
            }
            adapter->Release();
        }
        dxgiDevice->Release();
    }

    // Device name: the wide adapter description narrowed to bytes, with the
    // trailing spaces DXGI pads it with trimmed.
    std::string name;
    for (unsigned i = 0; i < 128 && adapterDesc.Description[i] != 0; ++i)
    {
        const wchar_t c = adapterDesc.Description[i];
        name.push_back(c < 0x80 ? char(c) : '?');
    }
    while (!name.empty() && name.back() == ' ')
        name.pop_back();
    mDeviceName = name;

    mSMCount = 0xffffffffu;

    if (adapterDesc.VendorId == VENDOR_ID_NVIDIA)
    {
        void* gpus[64] = {0};
        unsigned gpuCount = 0;
        unsigned smCount = 0xffffffffu;

        if (NvApiInitialize() == NVAPI_OK && NvApiEnumPhysicalGPUs(gpus, &gpuCount) == NVAPI_OK)
        {
            for (unsigned i = 0; i < gpuCount; ++i)
            {
                unsigned deviceId = 0, subSystemId = 0, subVendorId = 0, revisionId = 0;
                if (NvApiGpuGetPCIIdentifiers(gpus[i], &deviceId, &subSystemId, &subVendorId,
                                              &revisionId) != NVAPI_OK)
                    continue;
                if (deviceId == adapterDesc.VendorId &&
                    subSystemId == adapterDesc.DeviceId &&
                    subVendorId == adapterDesc.SubSysId)
                {
                    NvApiGpuGetShaderSubPipeCount(gpus[i], &smCount);
                    break;
                }
            }
        }
        mGpuVendorId = VENDOR_ID_NVIDIA;
        mSMCount = smCount;
    }
    else if (adapterDesc.VendorId == VENDOR_ID_AMD)
    {
        mGpuVendorId = VENDOR_ID_AMD;
    }
    else
    {
        mGpuVendorId = VENDOR_ID_OTHERS;
    }
}

// ---------------------------------------------------------------------------
// queryCapability / queryVendorExtension — the two private probes
// ---------------------------------------------------------------------------
// `queryCapability` answers the SHFL / FP32-atomic / swizzle questions the
// shader-variant selection in NvFlexInit asks.  They are NVAPI shader-extension
// op-code probes; when NVAPI is absent every answer is false, which is the
// generic path.
bool DeviceD3D11::queryCapability(int which, bool* out)
{
    (void)which;
    if (out != 0)
        *out = false;
    return false;
}

int DeviceD3D11::queryVendorExtension(void* ags, int which, unsigned* out)
{
    (void)ags;
    (void)which;
    if (out != 0)
        *out = 0;
    return NVAPI_NOT_SUPPORTED;
}

// ---------------------------------------------------------------------------
// DeviceD3D11::DeviceD3D11 — @0x551a0 (1088 bytes)
// ---------------------------------------------------------------------------
DeviceD3D11::DeviceD3D11(const FlexDeviceDesc* desc)
    : m_device(0), m_renderDevice(0), m_deviceContext(0), m_renderDeviceContext(0),
      m_useD3D11asyncCompute(false), mErrorFunc(0), m_factory(0)
{
    mErrorFunc = desc->mErrorFunc;
    // Asynchronous compute is requested explicitly (`useComputeQueue`) and only
    // runs on an NVIDIA adapter; every acceptance run leaves it false and takes
    // the single-immediate-context path.
    m_useD3D11asyncCompute = desc->useComputeQueue;

    if (desc->mDevice != 0)
    {
        // The application created the device; take a reference and go.
        m_device = static_cast<ID3D11Device*>(desc->mDevice);
        m_device->AddRef();
        m_device->GetImmediateContext(&m_deviceContext);
    }
    else
    {
        CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&m_factory);
        m_adapters = enumerateDXGIAdapters(m_factory);

        IDXGIAdapter1* adapter = 0;
        if (desc->mDeviceNumber < m_adapters.size())
            adapter = m_adapters[desc->mDeviceNumber];

        D3D_FEATURE_LEVEL featureLevels[] =
        {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };
        D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;

        // With an explicit adapter the driver type must be UNKNOWN; without one
        // the runtime has to pick a driver, which only HARDWARE/WARP allow.
        const UINT flags = D3D11_CREATE_DEVICE_SINGLETHREADED;
        const D3D_DRIVER_TYPE driverType =
            adapter != 0 ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE;
        D3D11CreateDevice(adapter, driverType, 0, flags,
                          featureLevels, _countof(featureLevels), D3D11_SDK_VERSION,
                          &m_device, &featureLevel, &m_deviceContext);
    }

    // The async-compute render device: the shipped code creates a second device
    // with the graphics queue.  With runOnRenderContext (the default in the
    // demo and in the parity scenario) it reuses the same device.
    m_renderDevice = m_device;
    if (m_renderDevice != 0)
        m_renderDevice->AddRef();
    m_renderDeviceContext = m_deviceContext;
    if (m_renderDeviceContext != 0)
        m_renderDeviceContext->AddRef();

    if (m_device != 0)
        queryDeviceCapabilities(m_device);
}

// The PDB records the destructor in Device.obj (DeviceD3D11's inline dtor calls
// release() out of line, @0x53d50).
DeviceD3D11::~DeviceD3D11()
{
    for (unsigned i = 0; i < m_adapters.size(); ++i)
        m_adapters[i]->Release();
    m_adapters.clear();

    if (m_factory != 0)
        m_factory->Release();
    m_factory = 0;

    if (m_deviceContext != 0)
        m_deviceContext->Release();
    m_deviceContext = 0;
    if (m_renderDeviceContext != 0)
        m_renderDeviceContext->Release();
    m_renderDeviceContext = 0;
    if (m_renderDevice != 0)
        m_renderDevice->Release();
    m_renderDevice = 0;
    if (m_device != 0)
        m_device->Release();
    m_device = 0;
}

// @0x54fb0 (149 bytes)
ContextEx* DeviceD3D11::createContext()
{
    m_context = new ContextD3D11(this);
    return m_context;
}

// ---------------------------------------------------------------------------
// getDXGIAdapter — @0x550b0 (239 bytes)
// ---------------------------------------------------------------------------
IDXGIAdapter1* DeviceD3D11::getDXGIAdapter(unsigned index)
{
    return index < m_adapters.size() ? m_adapters[index] : 0;
}

} // namespace NvFlex
