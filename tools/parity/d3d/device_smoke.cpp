// flexdecomp: device-layer smoke test for the reconstructed D3D11 backend.
//
// Exercises the path no public entry point covers on its own, end to end:
//
//   DeviceD3D11 -> createContext -> createBuffer (structured r32_uint)
//   -> createConstantBuffer -> createComputeShader(g_Flex_ClearInt)
//   -> upload the clear parameters -> dispatch -> download -> map
//
// It links against the same object library as the DLL, so it also proves that
// the device layer can be used without a NvFlexLibrary.  Output is one stable
// line:
//
//   device=ok clear=ok first=7
//
// Any failure prints `device=fail|clear=fail|first=<value>` and exits 1, which
// is what `ninja -C recon/build-dx d3d_smoke` + wine is gated on.
#include <cstdio>
#include <cstring>

#include "ContextD3D11.h"
#include "DeviceD3D11.h"
#include "Format.h"
#include "generated/shaders_flex.h"

int main()
{
    FlexDeviceDesc desc;
    desc.mDeviceNumber = 0;
    desc.mDevice = 0;
    desc.mComputeCommandQueue = 0;
    desc.mRenderCommandQueue = 0;
    desc.useComputeQueue = false;
    desc.enablePresent = false;
    desc.mErrorFunc = 0;

    NvFlex::DeviceD3D11 device(&desc);
    if (device.d3dDevice() == 0)
    {
        std::printf("device=fail clear=fail first=0\n");
        return 1;
    }

    NvFlex::ContextEx* context = device.createContext();
    if (context == 0)
    {
        std::printf("device=fail clear=fail first=0\n");
        return 1;
    }

    // 64 r32_uint elements, structured, UAV + SRV.
    NvFlex::BufferDesc bufferDesc;
    bufferDesc.format = eNvFlexFormat_r32_uint;
    bufferDesc.dim = 64;
    bufferDesc.stride = 4;
    // eStage adds the CPU-visible staging copy that download()/map() read.
    bufferDesc.bufferType = NvFlex::eBuffer | NvFlex::eUAV_SRV | NvFlex::eStructured | NvFlex::eStage;
    bufferDesc.data = 0;
    NvFlex::Buffer* buffer = context->createBuffer(&bufferDesc, 0);

    // 2 words: lengthInWords @0, value @4.
    NvFlex::ConstantBufferDesc cbDesc;
    cbDesc.stride = 8;
    cbDesc.dim = 1;
    cbDesc.uploadAccess = true;
    NvFlex::ConstantBuffer* cb = context->createConstantBuffer(&cbDesc);

    const FlexShaderBlob* blob = findFlexShader("g_Flex_ClearInt");
    if (buffer == 0 || cb == 0 || blob == 0)
    {
        std::printf("device=ok clear=fail first=0\n");
        return 1;
    }

    NvFlex::ComputeShaderDesc shaderDesc;
    shaderDesc.cs = blob->data;
    shaderDesc.cs_length = blob->size;
    shaderDesc.label = L"ClearInt";
    shaderDesc.NvAPI_Slot = 0xffffffffu;
    NvFlex::ComputeShader* shader = context->createComputeShader(&shaderDesc);
    if (shader == 0)
    {
        std::printf("device=ok clear=fail first=0\n");
        return 1;
    }

    // Write {lengthInWords = 16, value = 7} — ClearInt works on 16-byte groups,
    // so 64 uint elements means 16 words of 16 bytes.
    int* params = static_cast<int*>(context->map(cb));
    params[0] = 16;
    params[1] = 7;
    context->unmap(cb);

    NvFlex::DispatchParams dispatch;
    std::memset(&dispatch, 0, sizeof(dispatch));
    dispatch.shader = shader;
    dispatch.gridDim.x = 1;
    dispatch.gridDim.y = 1;
    dispatch.gridDim.z = 1;
    dispatch.rootConstantBuffer = cb;
    dispatch.readWrite[0] = buffer->getResourceRW();
    context->dispatch(&dispatch);

    context->flush();

    context->download(buffer);
    unsigned* mapped = static_cast<unsigned*>(context->map(buffer, NvFlex::eMapRead, true));
    unsigned first = mapped != 0 ? mapped[0] : 0;
    context->unmap(buffer);

    std::printf("device=ok clear=%s first=%u\n", first == 7 ? "ok" : "fail", first);
    return first == 7 ? 0 : 1;
}
