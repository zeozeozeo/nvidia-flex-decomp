// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 053 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 053_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/053.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/053_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    uint gNumSprings;
    float3 _fdpad0_0;
};

RWBuffer<uint> indices : register(u0);

[numthreads(256, 1, 1)]
void SpringsGenerateIndices(uint3 tid : SV_DispatchThreadID)
{
	// SpringsGenerateIndices: identity permutation over the two half edges
	if ((int)tid.x < 2 * gNumSprings)
	{
		indices[tid.x] = tid.x;
	}
}
