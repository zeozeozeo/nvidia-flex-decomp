// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 059 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 059_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/059.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/059_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    int gLength;
    int gValue;
    int _pad1;
    int _pad2;
};

RWStructuredBuffer<int> buf : register(u0);

[numthreads(256, 1, 1)]
void ClearInt(uint3 tid : SV_DispatchThreadID)
{
	// ClearInt
	if ((int)tid.x < gLength)
	{
		buf[tid.x] = gValue;
	}
}
