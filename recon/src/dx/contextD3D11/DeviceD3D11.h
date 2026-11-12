// flexdecomp: reconstructed D3D backend — DeviceD3D11.
//
// Evidence: work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c
//   DeviceD3D11::DeviceD3D11   @ 1800551a0 (1088 bytes)
//   DeviceD3D11::createContext @ 180054fb0  (149 bytes)
//   DeviceD3D11::getDXGIAdapter @ 1800550b0 (239 bytes)
//   Device::enumerateDXGIAdapters @ 18004edd0 (299 bytes)
//   Device::queryDeviceCapabilities @ 18004ef00 (572 bytes)
//
// `NvFlexInit` either hands the device an existing ID3D11Device or asks for a
// new one; the second path creates a DXGI factory, enumerates adapters and calls
// D3D11CreateDevice on the requested one.
#pragma once

#include <d3d11.h>
#include <dxgi1_2.h>

#include <vector>

#include "Device.h"

namespace NvFlex
{

class DeviceD3D11 : public Device
{
public:
    explicit DeviceD3D11(const FlexDeviceDesc* desc);
    ~DeviceD3D11() override;

    ContextEx* createContext() override;
    void queryDeviceCapabilities(ID3D11Device* device) override;
    bool queryCapability(int which, bool* out) override;
    int  queryVendorExtension(void* ags, int which, unsigned* out) override;

    // Native handles (context creation and the asynchronous-compute path).
    ID3D11Device*        d3dDevice() const { return m_device; }
    ID3D11Device*        renderDevice() const { return m_renderDevice; }
    ID3D11DeviceContext* d3dDeviceContext() const { return m_deviceContext; }
    bool                 useAsyncCompute() const { return m_useD3D11asyncCompute; }

    // @0x550b0 (239 bytes): index into the DXGI adapter list.
    IDXGIAdapter1* getDXGIAdapter(unsigned index);

    // --- PDB `NvFlex::DeviceD3D11` (sizeof 248) ---
    ID3D11Device*        m_device;               // @200
    ID3D11Device*        m_renderDevice;         // @208
    ID3D11DeviceContext* m_deviceContext;        // @216
    ID3D11DeviceContext* m_renderDeviceContext;  // @224
    bool                 m_useD3D11asyncCompute; // @232
    NvFlexErrorCallback  mErrorFunc;             // @240

    IDXGIFactory1* m_factory;                     // PDB `Device::m_pFactory`
    std::vector<IDXGIAdapter1*> m_adapters;       // enumerateDXGIAdapters

private:
    DeviceD3D11(const DeviceD3D11&);
    DeviceD3D11& operator=(const DeviceD3D11&);
};

} // namespace NvFlex
