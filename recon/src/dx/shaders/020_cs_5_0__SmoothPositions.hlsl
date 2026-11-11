// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 020 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 020_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/020.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/020_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<int> phases : register(t2);

StructuredBuffer<float4> sortedNewPositionsTex : register(t3);

Buffer<uint> indices : register(t4);

RWStructuredBuffer<float4> smoothPositions : register(u0);

RWStructuredBuffer<float4> smoothPositionsOriginal : register(u1);

[numthreads(256, 1, 1)]
void SmoothPositions(uint3 tid : SV_DispatchThreadID)
{
	// SmoothPositions: self-collision filter relaxes a particle towards the
	// weighted average of its neighbours, then blends with kSmoothing.
	if ((int)tid.x < kNumParticles)
	{
		const float4 cur = sortedNewPositionsTex[tid.x];
		const uint index = indices[tid.x];

		uint phase = (uint)phases[tid.x];
		if ((phase & 4194304u) != 0u)	// 1 << 22
		{
			const int count = contactCounts[tid.x];
			uint ni = tid.x;

			float4 accum = float4(0.0f, 0.0f, 0.0f, 0.0f);
			for (int k = 0; k < count; k++)
			{
				const uint n = (uint)contacts[ni];
				ni += (uint)kNumParticlesAligned;

				const float3 pn = sortedNewPositionsTex[n].xyz;
				const float3 d = cur.xyz - pn;
				const float r2 = dot(d, d);

				const bool inside = r2 < kRadiusSq;
				const bool nonzero = 0.0f < r2;
				const bool valid = nonzero && inside;

				const float q = sqrt(r2) * kInvRadius;
				const float w = 1.0f - (q * q) * q;
				const float4 cand = float4(pn * w + accum.xyz, accum.w + w);
				accum = valid ? cand : accum;
			}

			const float h = kInvRadius * kRadius * 0.5f;
			const float h2 = h * h;
			const float norm = (1.0f - h2 * h) * 4.0f;

			const float weight = min(accum.w / norm, 1.0f);
			const bool any = 0.0f < accum.w;
			const float3 avg = accum.xyz / accum.w;

			float4 result = cur;
			if (any)
			{
				const float blend = weight * kSmoothing;
				result = float4(cur.xyz + (avg - cur.xyz) * blend, cur.w);
			}

			smoothPositionsOriginal[index] = result;
			smoothPositions[tid.x] = result;
		}
		else
		{
			smoothPositionsOriginal[index] = cur;
			smoothPositions[tid.x] = cur;
		}
	}
}
