// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 033 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 033_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/033.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/033_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: inflatables (t0),
// positions (t1), reverseLookup (t2), indices (t3), lambdas (u0) and the AMD
// shader-intrinsics UAV (u7).  This is the AMD cross-lane variant of shader
// 035: the per-slice reduction is five rounds of AMD atomic-memory reads over
// the 32-lane slices (u7) instead of the shared-memory power-of-two walk, and
// the sixteen slices are folded by thread 0.
//
// The shipped u7 accesses are imm_atomic_cmp_exch with the running value as the
// compare operand and zero as the stored value, i.e. reads that can never
// change memory; the HLSL below spells exactly that out.

struct Inflatable
{
    int offset;
    int count;
    float targetVolume;
    float stiffness;
};

StructuredBuffer<Inflatable> inflatables : register(t0);

StructuredBuffer<float4> positions : register(t1);

StructuredBuffer<int> reverseLookup : register(t2);

StructuredBuffer<int> indices : register(t3);

RWStructuredBuffer<float> lambdas : register(u0);

// AMD shader-intrinsics UAV: only touched through the folded atomic-memory
// addresses below.
RWByteAddressBuffer amdIntrinsics : register(u7);

groupshared Inflatable sInflatable;

// g2: running sum of the corner average positions over all slices.
groupshared float3 sCentre;

// g1: running sum of the signed corner volumes over all slices.
groupshared float sVolume;

// g4 / g3: per-slice reductions of the two phases (16 slices of 32 lanes).
groupshared float3 sSliceCentre[16];
groupshared float sSliceVolume[16];

// AMD atomic-memory addresses folded into the shipped instruction stream.
static const uint kAmdReduce0 = 0x50401f04u;
static const uint kAmdReduce1 = 0x50201f04u;
static const uint kAmdReduce2 = 0x50101f04u;
static const uint kAmdReduce3 = 0x50081f04u;
static const uint kAmdReduce4 = 0x50041f04u;

// Atomic read that never writes: the compare operand is the value itself and
// the stored value is zero, so a match would store zero over zero.
float AmdAtomicRead(uint addr, float value)
{
	uint old;
	InterlockedCompareExchange(amdIntrinsics[addr], asuint(value), 0u, old);
	return asfloat(old);
}

float3 AmdReduceRound(uint addr, float3 value)
{
	return float3(AmdAtomicRead(addr, value.x), AmdAtomicRead(addr, value.y), AmdAtomicRead(addr, value.z));
}

float3 AmdAccumulate(float3 value)
{
	value += AmdReduceRound(kAmdReduce0, value);
	value += AmdReduceRound(kAmdReduce1, value);
	value += AmdReduceRound(kAmdReduce2, value);
	value += AmdReduceRound(kAmdReduce3, value);
	value += AmdReduceRound(kAmdReduce4, value);
	return value;
}

float AmdAccumulate(float value)
{
	value += AmdAtomicRead(kAmdReduce0, value);
	value += AmdAtomicRead(kAmdReduce1, value);
	value += AmdAtomicRead(kAmdReduce2, value);
	value += AmdAtomicRead(kAmdReduce3, value);
	value += AmdAtomicRead(kAmdReduce4, value);
	return value;
}

[numthreads(512, 1, 1)]
void SolveInflatableVolume(uint3 gid : SV_GroupThreadID, uint3 bid : SV_GroupID)
{
	if (gid.x == 0)
	{
		sInflatable = inflatables[bid.x];
		sVolume = 0.0f;
		sCentre = float3(0.0f, 0.0f, 0.0f);
	}
	GroupMemoryBarrierWithGroupSync();

	const int numParticles = sInflatable.count;
	const int numSlices = (numParticles + 511) / 512;
	const float invCornerCount = 1.0f / (3.0f * (float)numParticles);

	const int lane = (int)gid.x & 31;
	const int slice = (int)gid.x >> 5;

	for (int index = 0; index < numSlices; index++)
	{
		const int item = index * 512 + (int)gid.x;
		const int remaining = min(numParticles - index * 512, 512);

		float3 corner = float3(0.0f, 0.0f, 0.0f);
		if (item < numParticles)
		{
			const int tri = item + sInflatable.offset;
			const int v0 = reverseLookup[indices[tri * 3 + 0]];
			const int v1 = reverseLookup[indices[tri * 3 + 1]];
			const int v2 = reverseLookup[indices[tri * 3 + 2]];
			corner = (positions[v0].xyz + positions[v1].xyz + positions[v2].xyz) * invCornerCount;
		}

		corner = AmdAccumulate(corner);

		if (lane == 0)
		{
			sSliceCentre[slice] = corner;
		}
		GroupMemoryBarrierWithGroupSync();

		if (gid.x == 0)
		{
			// Only the slices that carry live lanes contribute; every other
			// lane wrote zero, so folding the live prefix is exact.
			const int liveSlices = min(remaining, 512) >> 5;
			float3 sliceSum = sSliceCentre[0];
			for (int k = 1; k < 16; k++)
			{
				if (k < liveSlices)
				{
					sliceSum += sSliceCentre[k];
				}
			}
			sSliceCentre[0] = sliceSum;
		}
		GroupMemoryBarrierWithGroupSync();

		const float3 sliceTotal = sSliceCentre[0];
		if (gid.x == 0)
		{
			sCentre += sliceTotal;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	GroupMemoryBarrierWithGroupSync();

	for (int index2 = 0; index2 < numSlices; index2++)
	{
		const int item2 = index2 * 512 + (int)gid.x;
		const int remaining2 = min(numParticles - index2 * 512, 512);

		float volume = 0.0f;
		if (item2 < numParticles)
		{
			const int tri2 = item2 + sInflatable.offset;
			const int v0 = reverseLookup[indices[tri2 * 3 + 0]];
			const int v1 = reverseLookup[indices[tri2 * 3 + 1]];
			const int v2 = reverseLookup[indices[tri2 * 3 + 2]];

			const float3 p0 = positions[v0].xyz;
			const float3 p1 = positions[v1].xyz;
			const float3 p2 = positions[v2].xyz;
			const float3 centre = sCentre;

			const float3 a = p0 - centre;
			const float3 b = (p1 - centre) - a;
			const float3 d = (p2 - centre) - a;

			volume = dot(a, cross(b, d));
		}

		volume = AmdAccumulate(volume);

		if (lane == 0)
		{
			sSliceVolume[slice] = volume;
		}
		GroupMemoryBarrierWithGroupSync();

		if (gid.x == 0)
		{
			const int liveSlices = min(remaining2, 512) >> 5;
			float sliceSum = sSliceVolume[0];
			for (int k = 1; k < 16; k++)
			{
				if (k < liveSlices)
				{
					sliceSum += sSliceVolume[k];
				}
			}
			sSliceVolume[0] = sliceSum;
		}
		GroupMemoryBarrierWithGroupSync();

		const float sliceTotal = sSliceVolume[0];
		if (gid.x == 0)
		{
			sVolume += sliceTotal;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (gid.x == 0)
	{
		const float target = sInflatable.targetVolume;
		const float stiffness = sInflatable.stiffness;
		const float sum = sVolume;

		const float denominator = max(abs(target * 0.01f), abs(sum));
		const float ratio = abs(target) / denominator;

		lambdas[bid.x] = ratio * ratio * ratio * (stiffness * (sum - target));
	}
}
