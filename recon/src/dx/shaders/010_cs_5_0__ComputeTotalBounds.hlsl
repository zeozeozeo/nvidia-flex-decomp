// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 010 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 010_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/010.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/010_cs_5_0.asm, work/dxbc.json

cbuffer constBVH : register(b0)
{
    int gNumItems;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> readBoundsLower : register(t2);

StructuredBuffer<float4> readBoundsUpper : register(t3);

RWStructuredBuffer<float3> totalLower : register(u2);

RWStructuredBuffer<float3> totalUpper : register(u3);

RWStructuredBuffer<float3> totalInvEdges : register(u4);

[numthreads(1, 1, 1)]
void ComputeTotalBounds(uint3 tid : SV_DispatchThreadID)
{
	// ComputeTotalBounds (final, single thread): fold the per-group results and
	// derive the reciprocal edge length.
	uint n = ((uint)gNumItems + 255u) >> 8;
	n = (n + 255u) >> 8;

	float3 lower = readBoundsLower[0].xyz;
	float3 upper = readBoundsUpper[0].xyz;

	for (uint i = 1; i < n; i++)
	{
		lower = min(lower, readBoundsLower[i].xyz);
		upper = max(upper, readBoundsUpper[i].xyz);
	}

	totalLower[0] = lower;
	totalUpper[0] = upper;
	totalInvEdges[0] = float3(1.0f, 1.0f, 1.0f) / ((upper - lower) + float3(9.99999975e-05f, 9.99999975e-05f, 9.99999975e-05f));
}
