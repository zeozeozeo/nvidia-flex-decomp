// flexdecomp: reconstructed D3D backend — device abstraction.
//
// `NvFlexLibrary` owns one `Device`; the device owns the native D3D objects and
// creates the `Context` the library and every solver dispatch through.  The
// shipped DLL carries two device classes (`DeviceD3D11`, `DeviceD3D12`) and a
// small factory (`DeviceSelect` in flex.obj) that picks one from
// `NvFlexInitDesc::computeType`.
//
// This reconstruction ships only the D3D11 device.  `NvFlexInit` with
// `eNvFlexD3D12` logs an error and returns null (see flex_api.cpp); the abstract
// class below is what `NvFlexLibrary::mDevice` points at.
//
// Only the D3D11 device is observable in the acceptance runs (the demo and the
// parity scenario both create a D3D11 device), so the interface is limited to
// what flex.obj actually calls:
//
//   * `createContext()`               — NvFlexLibrary ctor
//   * `queryDeviceCapabilities(...)`  — NvFlexInitDesc path
//   * `deviceName()` / `gpuVendorId()` / `smCount()` — NvFlexGetDeviceName /
//     the shader-variant selection in NvFlexInit
//
// Evidence for the call sites: `// ---- NvFlexInit @ 180035ba0` and
// `// ---- NvFlexLibrary::NvFlexLibrary @ 180037040` in
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.
#pragma once

#include <d3d11.h>

#include <string>

#include "FlexInternal.h"

namespace NvFlex
{

class Device
{
public:
    virtual ~Device() {}

    // Create the command interface the library and every solver use.
    virtual ContextEx* createContext() = 0;

    // Populate the vendor capability flags the solver pipeline branches on
    // (SM count, SHFL/FP32-atomic/swizzle support, vendor id).
    virtual void queryDeviceCapabilities(ID3D11Device* device) = 0;

    // `queryCapability` / `queryVendorExtension` mirror the shipped private
    // helpers the D3D11 device uses to detect the above; they are declared here
    // because NvFlexInit's vendor branch calls them through the device.
    virtual bool queryCapability(int which, bool* out) = 0;
    virtual int  queryVendorExtension(void* ags, int which, unsigned* out) = 0;

    const char* deviceName() const  { return mDeviceName.c_str(); }
    int         gpuVendorId() const { return mGpuVendorId; }
    unsigned    smCount() const     { return mSMCount; }
    ContextEx*  context() const     { return m_context; }

protected:
    Device() : mSMCount(0xffffffffu), mGpuVendorId(VENDOR_ID_OTHERS),
               mD3D12Device(false), m_context(0) {}

    // PDB `NvFlex::Device` (sizeof 184): NvFlexObject base (vtable + refcount),
    // mWindowDesc@8, mWinRect@88, m_hWnd@104, m_swapChain@112, m_pFactory@120,
    // mSMCount@128, mDeviceName@136, mGpuVendorId@168, mD3D12Device@172,
    // m_context@176.  Only the members flex.obj reads are carried here.
    unsigned    mSMCount;      // @128, 0xffffffff when unknown
    std::string mDeviceName;   // @136
    GpuVendorId mGpuVendorId;  // @168
    bool        mD3D12Device;  // @172
    ContextEx*  m_context;     // @176
};

} // namespace NvFlex
