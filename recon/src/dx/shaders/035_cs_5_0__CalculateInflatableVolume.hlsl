// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 035 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 035_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/035.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/035_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: inflatables (t0),
// positions (t1), reverseLookup (t2), indices (t3) and lambdas (u0).  One
// thread group handles one inflatable; the group walks its triangles in
// 512-element slices, block-reducing the corner average position and then the
// signed corner volume, and thread 0 finally writes the lambda.

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

groupshared Inflatable sInflatable;

// g2: running sum of the corner average positions over all slices.
groupshared float3 sCentre;

// g1: running sum of the signed corner volumes over all slices.
groupshared float sVolume;

// g4 / g3: per-thread reduction scratch for the two slices.
groupshared float3 sReduceCentre[512];
groupshared float sReduceVolume[512];

[numthreads(512, 1, 1)]
void CalculateInflatableVolume(uint3 gid : SV_GroupThreadID, uint3 bid : SV_GroupID)
{
	if (gid.x == 0)
	{
		sInflatable = inflatables[bid.x];
		sCentre = float3(0.0f, 0.0f, 0.0f);
		sVolume = 0.0f;
	}
	GroupMemoryBarrierWithGroupSync();

	const int numParticles = sInflatable.count;
	const int numSlices = (numParticles + 511) / 512;
	const float invCornerCount = 1.0f / (3.0f * (float)numParticles);

	for (int slice = 0; slice < numSlices; slice++)
	{
		const int index = slice * 512 + (int)gid.x;
		const int remaining = min(numParticles - slice * 512, 512);

		float3 corner = float3(0.0f, 0.0f, 0.0f);
		if (index < numParticles)
		{
			const int tri = index + sInflatable.offset;
			const int v0 = reverseLookup[indices[tri * 3 + 0]];
			const int v1 = reverseLookup[indices[tri * 3 + 1]];
			const int v2 = reverseLookup[indices[tri * 3 + 2]];
			corner = (positions[v0].xyz + positions[v1].xyz + positions[v2].xyz) * invCornerCount;
		}
		sReduceCentre[gid.x] = corner;

		for (int stride = remaining >> 1; stride > 0; stride >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)gid.x < stride)
			{
				sReduceCentre[gid.x] += sReduceCentre[gid.x + stride];
			}
		}
		GroupMemoryBarrierWithGroupSync();

		if (gid.x == 0)
		{
			sCentre += sReduceCentre[0];
		}
		GroupMemoryBarrierWithGroupSync();
	}

	for (int slice2 = 0; slice2 < numSlices; slice2++)
	{
		const int index2 = slice2 * 512 + (int)gid.x;
		const int remaining2 = min(numParticles - slice2 * 512, 512);

		float volume = 0.0f;
		if (index2 < numParticles)
		{
			const int tri2 = index2 + sInflatable.offset;
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
		sReduceVolume[gid.x] = volume;

		for (int stride2 = remaining2 >> 1; stride2 > 0; stride2 >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)gid.x < stride2)
			{
				sReduceVolume[gid.x] += sReduceVolume[gid.x + stride2];
			}
		}
		GroupMemoryBarrierWithGroupSync();

		if (gid.x == 0)
		{
			sVolume += sReduceVolume[0];
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
