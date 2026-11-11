// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 083 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 083_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/083.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/083_cs_5_0.asm, work/dxbc.json


Buffer<uint4> keySrc4 : register(t0);

Buffer<uint4> valSrc4 : register(t1);

// 16 digit starts per group, followed by 16 digit ends per group.
Buffer<uint> counters : register(t2);

RWBuffer<uint4> keyDst : register(u0);

RWBuffer<uint4> valDst : register(u1);

cbuffer params : register(b0)
{
    uint3 gridDim;
    uint  passStart;
    uint  numCounters;
    uint  numKeys;
};

// 1024 keys per group (4 per thread).
groupshared uint sKey[1024];
groupshared uint sVal[1024];
groupshared uint sScan[256];
groupshared uint sScan2[256];
groupshared uint sDigitStart[16];
groupshared uint sDigitEnd[16];

[numthreads(256, 1, 1)]
void RadixSortSortPairs(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	// Preload this group's digit boundaries: counters[digit * gridDim.x + group]
	// are the destination starts, the block after gridDim.x * 16 the ends.
	if (gi < 16)
	{
		sDigitStart[gi] = counters[gi * gridDim.x + gid.x];
		sDigitEnd[gi] = counters[(gid.x + gridDim.x) * 16 + gi];
	}

	uint4 key = keySrc4[tid.x];
	uint4 val = valSrc4[tid.x];

	const uint base = tid.x * 4;
	if (base + 0 >= numKeys) key.x = 0xffffffff;
	if (base + 1 >= numKeys) key.y = 0xffffffff;
	if (base + 2 >= numKeys) key.z = 0xffffffff;
	if (base + 3 >= numKeys) key.w = 0xffffffff;

	// Four stable one-bit partitions make one 4-bit pass.
	for (uint bit = passStart; bit < passStart + 4; ++bit)
	{
		const uint4 b = (key >> bit) & 1;
		const uint zeros = ((b.x ^ 1) + (b.y ^ 1)) + ((b.z ^ 1) + (b.w ^ 1));

		sScan[gi] = zeros;
		GroupMemoryBarrierWithGroupSync();

		for (uint offset = 1; offset < 256; offset <<= 1)
		{
			const uint prev = gi >= offset ? sScan[gi - offset] : 0;
			GroupMemoryBarrierWithGroupSync();
			sScan2[gi] = sScan[gi] + prev;
			GroupMemoryBarrierWithGroupSync();
			sScan[gi] = sScan2[gi];
			GroupMemoryBarrierWithGroupSync();
		}

		const uint inclusive = sScan[gi];
		const uint zerosBefore = inclusive - zeros;
		const uint totalZeros = sScan[255];
		GroupMemoryBarrierWithGroupSync();

		const uint zx = zerosBefore + 0;
		const uint zy = zerosBefore + (b.x ^ 1);
		const uint zz = zy + (b.y ^ 1);
		const uint zw = zz + (b.z ^ 1);

		const uint px = b.x ? (totalZeros + (base + 0 - zx)) : zx;
		const uint py = b.y ? (totalZeros + (base + 1 - zy)) : zy;
		const uint pz = b.z ? (totalZeros + (base + 2 - zz)) : zz;
		const uint pw = b.w ? (totalZeros + (base + 3 - zw)) : zw;

		sKey[px] = key.x;
		sKey[py] = key.y;
		sKey[pz] = key.z;
		sKey[pw] = key.w;
		sVal[px] = val.x;
		sVal[py] = val.y;
		sVal[pz] = val.z;
		sVal[pw] = val.w;
		GroupMemoryBarrierWithGroupSync();

		key = uint4(sKey[base + 0], sKey[base + 1], sKey[base + 2], sKey[base + 3]);
		val = uint4(sVal[base + 0], sVal[base + 1], sVal[base + 2], sVal[base + 3]);
		GroupMemoryBarrierWithGroupSync();
	}

	// Scatter the digit-ordered block to its global destination.
	for (uint i = gi; i < 1024; i += 256)
	{
		const uint k = sKey[i];
		const uint digit = (k >> passStart) & 15;
		const uint dest = sDigitStart[digit] + (i - sDigitEnd[digit]);
		keyDst[dest] = k;
		valDst[dest] = sVal[i];
	}
}
