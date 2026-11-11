// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 040 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 040_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/040.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/040_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    uint gNumSprings;
    float3 _fdpad0_0;
};

Buffer<uint> ids : register(t0);

RWStructuredBuffer<int> starts : register(u0);

RWStructuredBuffer<int> ends : register(u1);

[numthreads(256, 1, 1)]
void SpringsParticleRange(uint3 tid : SV_DispatchThreadID)
{
	// SpringsParticleRange: start/end offsets of every run of equal particle ids
	const int numIds = 2 * gNumSprings;
	if ((int)tid.x < numIds)
	{
		const uint id = ids[tid.x];

		if (tid.x == 0)
		{
			starts[id] = 0;
		}
		else
		{
			const uint prevId = ids[tid.x - 1];
			if (prevId != id)
			{
				starts[id] = tid.x;
				ends[prevId] = tid.x;
			}
		}

		if ((int)tid.x == numIds - 1)
		{
			ends[id] = tid.x + 1;
		}
	}
}
