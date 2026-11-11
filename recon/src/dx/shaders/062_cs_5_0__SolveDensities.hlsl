// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 062 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 062_cs_5_0.asm.
//
// Entry point: SolveDensities (RDEF name table: "SolveDensities::deltas").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/062.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/062_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    float4 kPlanes[8];
    int kNumPlanes;
    float kRadius;
    float kRadiusSq;
    float kInvRadius;
    float kPoly6;
    float kSpiky1;
    float kSpiky2;
    float kAkinci1;
    float kAkinci2;
    float kCohesion1;
    float kCohesion2;
    float _pad0;
    float kRestDensity;
    float kInvRestDensity;
    float kFluidRestDistance;
    float kSolidRestDistance;
    float kLambdaScale;
    float kCohesion;
    float kSurfaceTension;
    float kSolidPressure;
    float kVorticityConfinement;
    float kAnisotropy;
    float kAnisotropyMin;
    float kAnisotropyMax;
    float kSmoothing;
    float kViscosity;
    float kFreeSurfaceDrag;
    float kBuoyancy;
    float kDiffuseThreshold;
    float kDiffuseBuoyancy;
    float kDiffuseDrag;
    int kDiffuseBallistic;
    float3 kGravity;
    float kDynamicFriction;
    float kStaticFriction;
    float kParticleFriction;
    float kAdhesion;
    float kDissipation;
    float kDamping;
    float kInertiaBias;
    float kCollisionDistance;
    float kCollisionThreshold;
    float kCollisionMargin;
    float kCollisionMarginSq;
    float _pad1;
    float _pad2;
    float kDrag;
    float kLift;
    float kPlasticThreshold_deprecated;
    float kPlasticCreep_deprecated;
    float kSleepThreshold;
    float kSleepThresholdSq;
    float kShockPropagation;
    float kRestitution;
    float kSOR;
    float3 _pad3;
    int kNumParticles;
    int kNumParticlesAligned;
    float kDt;
    float kInvDt;
};

StructuredBuffer<int> contacts : register(t0);
StructuredBuffer<int> contactCounts : register(t1);
StructuredBuffer<float4> sortedNewPositions : register(t2);
StructuredBuffer<int> phases : register(t3);
StructuredBuffer<float> densities : register(t4);
StructuredBuffer<float4> sortedPositions : register(t6);
StructuredBuffer<float4> sortedNormals : register(t7);

RWByteAddressBuffer deltas : register(u0);

static const uint kPhaseSolid = 1u << 22; // 4194304

[numthreads(256, 1, 1)]
void SolveDensities(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const float4 np = sortedNewPositions[tid.x];
	if (np.w == 0.0f)
	{
		deltas.Store(tid.x * 16, uint4(0u, 0u, 0u, 0u));
		return;
	}

	const float3 pos = sortedPositions[tid.x].xyz;
	const float4 selfNormal = sortedNormals[tid.x];
	const float density = densities[tid.x];
	const uint phase = (uint)phases[tid.x];
	const int count = contactCounts[tid.x];

	const bool selfSolid = (phase & kPhaseSolid) != 0u;
	const bool useFriction = (0.0f < kDynamicFriction);

	float3 accum = float3(0.0f, 0.0f, 0.0f);
	float numContacts = 0.0f;
	float3 bestN = -selfNormal.xyz;
	float bestW = min(selfNormal.w, 0.0f);

	uint ni = tid.x;
	for (int k = 0; k < count; k++)
	{
		const uint c = (uint)contacts[ni];
		ni += (uint)kNumParticlesAligned;

		const float4 npn = sortedNewPositions[c];
		const float3 d = np.xyz - npn.xyz;
		const float d2 = dot(d, d);

		if (!((d2 <= kRadiusSq) && (0.0f < d2)))
		{
			continue;
		}

		const float dist = sqrt(d2);
		const float3 dir = d / dist;
		const float4 nb = sortedNormals[c];
		const uint nphase = (uint)phases[c];

		if (selfSolid && ((nphase & kPhaseSolid) != 0u))
		{
			// solid/solid pair: cohesion weight plus the plastic term, then the
			// normal difference against the neighbour is removed as well
			const float w = (density + densities[c]) * ((1.0f - dist * kInvRadius) * -kSpiky2);
			const float3 cohesionPush = accum - (np.w * w) * dir;
			const float q = dist * kInvRadius;
			const float poly = kCohesion1 * (q * q * q) + kCohesion2 * (q * q) - 1.0f;
			const float wp = kCohesion * poly + w;
			const float3 plasticPush = (accum - (np.w * wp) * dir) - (selfNormal.xyz - nb.xyz);

			accum = (phase != nphase) ? cohesionPush : plasticPush;
		}
		else
		{
			if (kSolidRestDistance < dist)
			{
				continue;
			}

			const float wratio = np.w / (np.w + npn.w);
			const float4 sel = (nb.w < bestW) ? nb : float4(bestN, bestW);
			const float gap = dist - kSolidRestDistance;

			const float align = dot(sel.xyz, dir);
			const float3 mirrored = (align < 0.0f) ? (dir - 2.0f * align * sel.xyz) : dir;
			const float3 usedDir = (sel.w >= -kSolidRestDistance) ? mirrored : sel.xyz;
			const float3 nrm = (sel.w == 0.0f) ? dir : usedDir;

			float3 cand = accum - (wratio * gap) * nrm;

			if (useFriction && (dot(nrm, nrm) > 0.0f))
			{
				const float3 tang = (d - (pos - sortedPositions[c].xyz)) - dot(d - (pos - sortedPositions[c].xyz), nrm) * nrm;
				const float t2 = dot(tang, tang);
				const float limit = gap * kStaticFriction;

				if (t2 < limit * limit)
				{
					cand -= tang * wratio;
				}
				else
				{
					const float f = min(rsqrt(t2) * (-gap * kParticleFriction), 1.0f);
					cand -= (tang * wratio) * f;
				}
			}

			accum = cand;
		}

		numContacts += 1.0f;
	}

	deltas.Store(tid.x * 16,
	             uint4(asuint(accum.x), asuint(accum.y), asuint(accum.z), asuint(numContacts)));
}
