// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 015 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 015_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/015.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/015_cs_5_0.asm, work/dxbc.json

StructuredBuffer<int> numDiffuseParticles : register(t0);

StructuredBuffer<float4> diffusePositions : register(t1);

StructuredBuffer<float4> diffuseVelocities : register(t2);

RWStructuredBuffer<int> numDiffuseParticlesNew : register(u0);

RWStructuredBuffer<float4> diffusePositionsNew : register(u1);

RWStructuredBuffer<float4> diffuseVelocitiesNew : register(u2);

[numthreads(256, 1, 1)]
void CompactDiffuseParticles(uint3 tid : SV_DispatchThreadID)
{
	// CompactDiffuseParticles: append live diffuse particles to the new buffers
	const uint n = numDiffuseParticles[0];
	if (tid.x < n)
	{
		const float4 pos = diffusePositions[tid.x];
		if (0.0f < pos.w)
		{
			uint slot;
			InterlockedAdd(numDiffuseParticlesNew[0], 1, slot);
			diffusePositionsNew[slot] = pos;
			diffuseVelocitiesNew[slot] = diffuseVelocities[tid.x];
		}
	}
}
