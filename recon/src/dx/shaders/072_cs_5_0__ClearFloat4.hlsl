// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 072 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 072_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/072.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/072_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    int gLength;
    float3 _pad3;
    float4 gValue;
};

RWStructuredBuffer<float4> buf : register(u0);

[numthreads(256, 1, 1)]
void ClearFloat4(uint3 tid : SV_DispatchThreadID)
{
	// ClearFloat4
	if ((int)tid.x < gLength)
	{
		buf[tid.x] = gValue;
	}
}
