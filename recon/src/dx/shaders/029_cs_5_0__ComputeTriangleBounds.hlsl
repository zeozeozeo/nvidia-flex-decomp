// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 029 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 029_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/029.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/029_cs_5_0.asm, work/dxbc.json

cbuffer mConstTriMeshInfo : register(b0)
{
    int numTris;
    float3 _fdpad0_0;
};

StructuredBuffer<float4> verticesArray : register(t0);

StructuredBuffer<int> indicesArray : register(t1);

RWStructuredBuffer<float4> lowers : register(u0);

RWStructuredBuffer<float4> uppers : register(u1);

[numthreads(256, 1, 1)]
void ComputeTriangleBounds(uint3 tid : SV_DispatchThreadID)
{
	// ComputeTriangleBounds: per-triangle min/max of its three vertex positions
	if ((int)tid.x < numTris)
	{
		const uint i0 = indicesArray[tid.x * 3u + 0u];
		const uint i1 = indicesArray[tid.x * 3u + 1u];
		const uint i2 = indicesArray[tid.x * 3u + 2u];

		const float3 p0 = verticesArray[i0].xyz;
		const float3 p1 = verticesArray[i1].xyz;
		const float3 p2 = verticesArray[i2].xyz;

		const float3 lower = min(p0, min(p1, p2));
		const float3 upper = max(p0, max(p1, p2));

		lowers[tid.x] = float4(lower, 0.0f);
		uppers[tid.x] = float4(upper, 0.0f);
	}
}
