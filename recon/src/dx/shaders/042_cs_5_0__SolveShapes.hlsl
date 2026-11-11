// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 042 (cs_5_0).  Interface (resource bindings, thread group) comes from
// the RDEF chunk; the body follows the SHEX assembly 042_cs_5_0.asm.
//
// Entry point: SolveShapes -- one work group per shape; the shape's particles
// are reduced (average position, covariance) in groupshared memory, the rigid
// rotation is extracted from the covariance, and the per-particle corrections
// are accumulated atomically.
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/042.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/042_cs_5_0.asm, work/dxbc.json,
//           work/dxbc/interfaces.txt

StructuredBuffer<int> rigidOffsets : register(t0);
StructuredBuffer<int> rigidIndices : register(t1);
StructuredBuffer<int> rigidCoefficients : register(t2);
StructuredBuffer<int> rigidPlasticThresholds : register(t3);
StructuredBuffer<int> rigidPlasticCreeps : register(t4);
StructuredBuffer<int> reverseLookup : register(t5);
StructuredBuffer<float4> oldPositions : register(t6);
StructuredBuffer<float4> localNormals : register(t7);

RWByteAddressBuffer accum : register(u0);
RWStructuredBuffer<float3> localPositions : register(u1);
RWStructuredBuffer<float4> newPositions : register(u2);
RWStructuredBuffer<float4> rotations : register(u3);
RWStructuredBuffer<float3> translations : register(u4);
RWStructuredBuffer<float4> normals : register(u5);

groupshared float3 gSum;        // g0: average position accumulator
groupshared float3 gTranslation; // g1: shape translation
groupshared float3 gColumns[3]; // g2: 3 x float3 covariance accumulators
groupshared uint gPlastic;      // g3: "exceeded plastic threshold" flag
groupshared float gMass;        // g4: effective shape mass (0 or 1)
groupshared float3 gReduce[64]; // g5
groupshared float4 gQuat;       // g6: extracted rotation

// Quaternion rotation of v (see the bytecode: 2*w*cross(v,q) + 2*q*dot(q,v) +
// v*(w*w-1)); the same expansion is used for the local positions and normals.
static float3 Rotate(float4 q, float3 v)
{
	const float3 t = cross(v, q.xyz);
	return v * (q.w * q.w - 1.0f) + 2.0f * q.xyz * dot(q.xyz, v) + 2.0f * q.w * t;
}

static float3x3 QuatMatrix(float4 q)
{
	const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
	const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
	const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
	return float3x3(
		1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),         2.0f * (xz + wy),
		2.0f * (xy + wz),         1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
		2.0f * (xz - wy),         2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy));
}

// Atomic float accumulate on the raw accumulator buffer: the same
// imm_atomic_cmp_exch retry loop the shipped shader uses.
static void AtomicAddFloat(RWByteAddressBuffer buf, uint addr, float delta)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(buf[addr], expected, asuint(asfloat(expected) + delta), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

[numthreads(64, 1, 1)]
void SolveShapes(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
	const uint shape = gid.x;
	const uint tid = gtid.x;

	const int begin = rigidOffsets[shape];
	const int count = rigidOffsets[shape + 1] - begin;
	const int iterations = (count + 63) >> 6;

	if (tid == 0u)
	{
		gSum = float3(0.0f, 0.0f, 0.0f);
		gTranslation = translations[shape];
		gColumns[0] = float3(0.0f, 0.0f, 0.0f);
		gColumns[1] = float3(0.0f, 0.0f, 0.0f);
		gColumns[2] = float3(0.0f, 0.0f, 0.0f);
		gPlastic = 0u;
	}
	GroupMemoryBarrierWithGroupSync();

	// ---- pass 1: sum of the shape's current world positions -----------------
	for (int it = 0; it < iterations; it++)
	{
		const int idx = (it << 6) | (int)tid;
		const bool inRange = idx < count;
		const int pos = begin + idx;

		float3 v = float3(0.0f, 0.0f, 0.0f);
		if (inRange)
		{
			v = newPositions[reverseLookup[rigidIndices[pos]]].xyz;
		}
		gReduce[tid] = v;

		for (int off = 32; off > 0; off >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)tid < off)
			{
				gReduce[tid] += gReduce[tid + off];
			}
		}
		GroupMemoryBarrierWithGroupSync();

		const float3 total = gReduce[0];
		if (tid == 0u)
		{
			gSum += total;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	if (tid == 0u)
	{
		const float3 avg = gSum / (float)count;
		// keep the stored translation when the particles did not move
		gSum = (abs(avg.x - gTranslation.x) < 9.99999975e-06f &&
		        abs(avg.y - gTranslation.y) < 9.99999975e-06f &&
		        abs(avg.z - gTranslation.z) < 9.99999975e-06f) ? gTranslation : avg;
	}
	GroupMemoryBarrierWithGroupSync();

	// ---- pass 2: covariance columns sum(v * localPosition) -----------------
	const bool usePlastic = (rigidPlasticCreeps[shape] != 0);
	const float plasticThreshold = (float)rigidPlasticThresholds[shape];

	for (int it = 0; it < iterations; it++)
	{
		const int idx = (it << 6) | (int)tid;
		if (idx >= count)
		{
			continue;
		}

		const int pos = begin + idx;
		const uint slot = reverseLookup[rigidIndices[pos]];
		const float3 cur = newPositions[slot].xyz;
		const float3 d = cur - gSum;
		const float3 loc = localPositions[pos];

		if (usePlastic)
		{
			const float3 moved = cur - oldPositions[slot].xyz;
			if (plasticThreshold < dot(moved, moved))
			{
				gPlastic = 0xffffffffu;
			}
		}

		gReduce[tid] = d * loc.x;
		for (int off = 32; off > 0; off >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)tid < off)
			{
				gReduce[tid] += gReduce[tid + off];
			}
		}
		GroupMemoryBarrierWithGroupSync();
		const float3 sumX = gReduce[0];
		if (tid == 0u)
		{
			gColumns[0] += sumX;
		}
		GroupMemoryBarrierWithGroupSync();

		gReduce[tid] = d * loc.y;
		for (int off = 32; off > 0; off >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)tid < off)
			{
				gReduce[tid] += gReduce[tid + off];
			}
		}
		GroupMemoryBarrierWithGroupSync();
		const float3 sumY = gReduce[0];
		if (tid == 0u)
		{
			gColumns[1] += sumY;
		}
		GroupMemoryBarrierWithGroupSync();

		gReduce[tid] = d * loc.z;
		for (int off = 32; off > 0; off >>= 1)
		{
			GroupMemoryBarrierWithGroupSync();
			if ((int)tid < off)
			{
				gReduce[tid] += gReduce[tid + off];
			}
		}
		GroupMemoryBarrierWithGroupSync();
		const float3 sumZ = gReduce[0];
		if (tid == 0u)
		{
			gColumns[2] += sumZ;
		}
		GroupMemoryBarrierWithGroupSync();
	}

	// ---- single-thread rotation extraction ---------------------------------
	if (tid == 0u)
	{
		gMass = (gPlastic == 0u) ? 1.0f : 0.0f;

		float4 q = rotations[shape];
		const float3x3 m = float3x3(gColumns[0], gColumns[1], gColumns[2]);

		// four refinement steps of the rotation update: the axis is the summed
		// cross product of the current basis rows with the covariance rows, the
		// angle comes from the (absolute) trace; the delta is composed into q.
		for (int k = 0; k < 4; k++)
		{
			const float3x3 r = QuatMatrix(q);
			const float3 om = cross(float3(r[0][0], r[0][1], r[0][2]), m[0]) +
			                  cross(float3(r[1][0], r[1][1], r[1][2]), m[1]) +
			                  cross(float3(r[2][0], r[2][1], r[2][2]), m[2]);
			const float tr = dot(float3(r[0][0], r[0][1], r[0][2]), m[0]) +
			                 dot(float3(r[1][0], r[1][1], r[1][2]), m[1]) +
			                 dot(float3(r[2][0], r[2][1], r[2][2]), m[2]);

			const float3 axis = om * (1.0f / abs(tr)) + 9.99999972e-10f;
			const float len = length(axis);
			if (len < 9.99999972e-10f)
			{
				break;
			}

			const float3 n = axis / len;
			const float half = len * 0.5f;
			float s, c;
			sincos(half, s, c);

			const float4 dq = float4(n * s, c);
			q = float4(q.w * dq.xyz + dq.w * q.xyz + cross(q.xyz, dq.xyz),
			           q.w * dq.w - dot(q.xyz, dq.xyz));
		}

		rotations[shape] = q;
		translations[shape] = gSum;
		gQuat = q;
	}
	GroupMemoryBarrierWithGroupSync();

	// ---- pass 3: apply the rotation, plastic deformation, atomics ----------
	const float4 q = gQuat;
	const float mass = gMass;
	const float blend = (float)rigidCoefficients[shape] * (1.0f - mass);
	const bool hasMass = (0.0f < mass);

	for (int it = 0; it < iterations; it++)
	{
		const int idx = (it << 6) | (int)tid;
		if (idx >= count)
		{
			continue;
		}

		const int pos = begin + idx;
		const uint particle = (uint)rigidIndices[pos];
		const uint slot = (uint)reverseLookup[particle];

		const float3 cur = newPositions[slot].xyz;
		const float3 target = gSum + Rotate(q, localPositions[pos]);
		const float3 delta = cur - target;

		if (hasMass)
		{
			const float3 blended = (cur + blend * (target - cur)) - gSum;
			localPositions[pos] = Rotate(float4(-q.xyz, q.w), blended);
		}

		const float4 ln = localNormals[pos];
		normals[particle] = float4(Rotate(q, ln.xyz), ln.w);

		const uint addr = slot * 16u;
		AtomicAddFloat(accum, addr + 0u, -mass * delta.x);
		AtomicAddFloat(accum, addr + 4u, -mass * delta.y);
		AtomicAddFloat(accum, addr + 8u, -mass * delta.z);
		AtomicAddFloat(accum, addr + 12u, 1.0f);
	}
}
