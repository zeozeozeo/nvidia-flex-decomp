// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 012 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 012_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/012.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/012_cs_5_0.asm, work/dxbc.json

StructuredBuffer<float3> totalLower : register(t0);

StructuredBuffer<float3> totalUpper : register(t1);

RWStructuredBuffer<float3> totalInvEdges : register(u0);

[numthreads(1, 1, 1)]
void ComputeTotalInvEdges(uint3 tid : SV_DispatchThreadID)
{
	const float3 eps = float3(9.99999975e-05f, 9.99999975e-05f, 9.99999975e-05f);
	totalInvEdges[0] = float3(1.0f, 1.0f, 1.0f) / ((totalUpper[0] - totalLower[0]) + eps);
}
