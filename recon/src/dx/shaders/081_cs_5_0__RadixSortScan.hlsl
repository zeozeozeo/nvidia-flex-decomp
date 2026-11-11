// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 081 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 081_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/081.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/081_cs_5_0.asm, work/dxbc.json


Buffer<uint4> countersSrc : register(t0);

RWBuffer<uint4> countersDst : register(u0);

cbuffer params : register(b0)
{
    uint3 gridDim;
    uint  passStart;
    uint  numCounters;
};

groupshared uint sScan[256];

[numthreads(256, 1, 1)]
void RadixSortScan(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	// numCounters counts scalars; the shader works on uint4 items.
	const uint numItems = numCounters >> 2;
	const uint numBlocks = (numItems + 255) >> 8;

	// Running total carried across blocks of 256 items.
	uint blockBase = 0;

	for (uint b = 0; b < numBlocks; ++b)
	{
		const uint i = b * 256 + gi;
		const bool valid = i < numItems;
		const uint4 v = valid ? countersSrc[i] : uint4(0, 0, 0, 0);

		// Per-lane sum, then an inclusive scan over the group.
		const uint lane = v.x + v.y + v.z + v.w;
		sScan[gi] = lane;
		GroupMemoryBarrierWithGroupSync();

		for (uint offset = 1; offset < 256; offset <<= 1)
		{
			const uint prev = gi >= offset ? sScan[gi - offset] : 0;
			GroupMemoryBarrierWithGroupSync();
			sScan[gi] += prev;
			GroupMemoryBarrierWithGroupSync();
		}

		const uint inclusive = sScan[gi];
		const uint blockTotal = sScan[255];
		GroupMemoryBarrierWithGroupSync();

		if (valid)
		{
			// Exclusive prefix of this item's own components, offset by the
			// running base plus the exclusive scan of the group.
			const uint exclusive = inclusive - lane + blockBase;
			countersDst[i] = uint4(exclusive, exclusive + v.x, exclusive + v.x + v.y, exclusive + v.x + v.y + v.z);
		}

		blockBase += blockTotal;
		GroupMemoryBarrierWithGroupSync();
	}
}
