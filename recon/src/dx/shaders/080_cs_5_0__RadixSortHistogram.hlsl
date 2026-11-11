// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 080 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 080_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/080.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/080_cs_5_0.asm, work/dxbc.json


Buffer<uint4> srcKey : register(t0);

RWBuffer<uint> counters : register(u0);

cbuffer params : register(b0)
{
    uint3 gridDim;
    uint  passStart;
    uint  numCounters;
    uint  numKeys;
};

// 16 digit buckets for the current 4-bit radix pass.
groupshared uint sHist[16];

[numthreads(256, 1, 1)]
void RadixSortHistogram(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	// Each group consumes 1024 keys: 4 per thread, strided by 256.
	const uint base = gid.x * 1024 + gi * 4;
	uint4 key = srcKey[gid.x * 256 + gi];

	// Keys past the end become 0xffffffff, which lands in the last bucket.
	if (base + 0 >= numKeys) key.x = 0xffffffff;
	if (base + 1 >= numKeys) key.y = 0xffffffff;
	if (base + 2 >= numKeys) key.z = 0xffffffff;
	if (base + 3 >= numKeys) key.w = 0xffffffff;

	const uint4 digit = (key >> passStart) & 15;

	if (gi < 16)
	{
		sHist[gi] = 0;
	}
	GroupMemoryBarrierWithGroupSync();

	InterlockedAdd(sHist[digit.x], 1);
	InterlockedAdd(sHist[digit.y], 1);
	InterlockedAdd(sHist[digit.z], 1);
	InterlockedAdd(sHist[digit.w], 1);
	GroupMemoryBarrierWithGroupSync();

	// counters[digit * gridDim.x + group] holds this group's bucket counts.
	if (gi < 16)
	{
		counters[gi * gridDim.x + gid.x] = sHist[gi];
	}
}
