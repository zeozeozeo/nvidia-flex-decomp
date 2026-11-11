// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 044 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 044_cs_5_0.asm.
//
// Entry point: SolveDensities (RDEF name table: "SolveDensities::deltas").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/044.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/044_cs_5_0.asm, work/dxbc.json

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
StructuredBuffer<float4> sortedPositions : register(t6);
StructuredBuffer<float4> sortedNormals : register(t7);

RWByteAddressBuffer deltas : register(u0);

[numthreads(256, 1, 1)]
void SolveDensities(uint3 tid : SV_DispatchThreadID)
{
	// Delta accumulation for the density-constraint solver.  Every neighbour
	// inside the kernel radius pushes the particle along the normal returned by
	// the density pass (or its reflection when the two normals oppose); the
	// tangential part is friction-limited.
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
	const int count = contactCounts[tid.x];

	float3 accum = float3(0.0f, 0.0f, 0.0f);
	float numContacts = 0.0f;
	float3 bestN = -selfNormal.xyz;
	float bestW = min(selfNormal.w, 0.0f);

	uint ni = tid.x;
	for (int k = 0; k < count; k++)
	{
		const uint n = (uint)contacts[ni];
		ni += (uint)kNumParticlesAligned;

		const float4 npn = sortedNewPositions[n];
		const float3 d = np.xyz - npn.xyz;
		const float d2 = dot(d, d);

		const bool inside = (d2 <= kRadiusSq) && (0.0f < d2);
		if (!inside)
		{
			continue;
		}

		const float dist = sqrt(d2);
		if (kSolidRestDistance < dist)
		{
			continue;
		}

		const float3 dir = d / dist;
		const float4 nb = sortedNormals[n];
		const float wratio = np.w / (np.w + npn.w);
		const float4 sel = (nb.w < bestW) ? nb : float4(bestN, bestW);
		const float gap = dist - kSolidRestDistance;

		const float align = dot(sel.xyz, dir);
		const float3 mirrored = (align < 0.0f) ? (dir - 2.0f * align * sel.xyz) : dir;
		const float3 usedDir = (sel.w >= -kSolidRestDistance) ? mirrored : sel.xyz;
		const float3 nrm = (sel.w == 0.0f) ? dir : usedDir;

		float3 cand = accum - (wratio * gap) * nrm;

		const bool applyFriction = (kDynamicFriction > 0.0f) && (dot(nrm, nrm) > 0.0f);
		if (applyFriction)
		{
			const float3 tangent = d - (pos - sortedPositions[n].xyz);
			const float proj = dot(tangent, nrm);
			const float3 t = tangent - proj * nrm;
			const float t2 = dot(t, t);

			const float invT = rsqrt(t2);
			const float staticLimit = gap * kStaticFriction;
			const bool sticking = t2 < (staticLimit * staticLimit);

			const float f = min(invT * (-gap * kParticleFriction), 1.0f);
			const float3 sliding = cand - (t * wratio) * f;
			const float3 stuck = cand - t * wratio;

			cand = sticking ? stuck : sliding;
		}

		accum = cand;
		numContacts += 1.0f;
	}

	deltas.Store(tid.x * 16,
	             uint4(asuint(accum.x), asuint(accum.y), asuint(accum.z), asuint(numContacts)));
}
