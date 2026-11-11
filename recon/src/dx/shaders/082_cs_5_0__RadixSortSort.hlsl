// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 082 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 082_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/082.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/082_cs_5_0.asm, work/dxbc.json


Buffer<uint4> srcKey : register(t0);

Buffer<uint4> srcVal : register(t1);

RWBuffer<uint4> dstKey : register(u0);

RWBuffer<uint4> dstVal : register(u1);

cbuffer params : register(b0)
{
    uint3 gridDim;
    uint  passStart;
    uint  numCounters;
    uint  numKeys;
};

// 1024 keys per group (4 per thread) plus a 256-entry scan array.
groupshared uint sKey[1024];
groupshared uint sVal[1024];
groupshared uint sScan[256];
groupshared uint sScan2[256];

[numthreads(256, 1, 1)]
void RadixSortSort(uint3 tid : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
	uint4 key = srcKey[tid.x];
	uint4 val = srcVal[tid.x];

	// Keys past the end are pushed to the end of the order.
	const uint base = tid.x * 4;
	if (base + 0 >= numKeys) key.x = 0xffffffff;
	if (base + 1 >= numKeys) key.y = 0xffffffff;
	if (base + 2 >= numKeys) key.z = 0xffffffff;
	if (base + 3 >= numKeys) key.w = 0xffffffff;

	// One stable partition per bit, least significant first (32 passes).
	for (uint bit = 0; bit < 32; ++bit)
	{
		const uint4 b = (key >> bit) & 1;
		// Number of zero bits in this thread's four keys.
		const uint zeros = ((b.x ^ 1) + (b.y ^ 1)) + ((b.z ^ 1) + (b.w ^ 1));

		// Exclusive scan of the per-thread zero counts over the group, and the
		// group's total zero count.
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

		// Position of each of the four keys: zeros go to the front of the block,
		// ones to the back, both preserving the input order.
		const uint zx = zerosBefore + 0;
		const uint zy = zerosBefore + (b.x ^ 1);
		const uint zz = zy + (b.y ^ 1);
		const uint zw = zz + (b.z ^ 1);

		const uint ox = totalZeros + (base + 0 - zx);
		const uint oy = totalZeros + (base + 1 - zy);
		const uint oz = totalZeros + (base + 2 - zz);
		const uint ow = totalZeros + (base + 3 - zw);

		const uint px = b.x ? ox : zx;
		const uint py = b.y ? oy : zy;
		const uint pz = b.z ? oz : zz;
		const uint pw = b.w ? ow : zw;

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

	dstKey[tid.x] = key;
	dstVal[tid.x] = val;
}
