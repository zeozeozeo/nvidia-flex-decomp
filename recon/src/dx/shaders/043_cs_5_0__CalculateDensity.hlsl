// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 043 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 043_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/043.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/043_cs_5_0.asm, work/dxbc.json

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

Buffer<uint> indices : register(t4);

RWStructuredBuffer<float> densities : register(u0);

RWStructuredBuffer<float4> normals : register(u1);

RWStructuredBuffer<float4> sortedNormals : register(u2);

[numthreads(256, 1, 1)]
void CalculateDensity(uint3 tid : SV_DispatchThreadID)
{
	// CalculateDensity (with surface-tension gradient): density sum plus the
	// Spiky2 gradient accumulated per neighbour.
	if ((int)tid.x < kNumParticles)
	{
		const float3 pos = sortedNewPositions[tid.x].xyz;
		const int count = contactCounts[tid.x];

		uint ni = tid.x;
		float4 accum = float4(0.0f, 0.0f, 0.0f, 0.0f);

		for (int k = 0; k < count; k++)
		{
			const uint n = (uint)contacts[ni];
			ni += (uint)kNumParticlesAligned;

			const float3 d = pos - sortedNewPositions[n].xyz;
			const float r2 = dot(d, d);

			const bool inside = r2 < kRadiusSq;
			const bool nonzero = 0.0f < r2;
			const bool valid = inside && nonzero;

			const float dist = sqrt(r2);
			const uint phase = (uint)phases[n];
			const float w = ((phase & 4194304u) != 0u) ? 1.0f : kSolidPressure;

			const float q = 1.0f - dist * kInvRadius;
			const float k6 = q * q * kSpiky1;
			const float3 grad = d * (q * -kSpiky2) / dist;

			const float4 cand = float4(w * k6 + accum.x, accum.yzw + grad);
			accum = valid ? cand : accum;
		}

		densities[tid.x] = max(accum.x - kRestDensity, 0.0f) * kLambdaScale;

		const uint selfPhase = (uint)phases[tid.x];
		if ((selfPhase & 4194304u) != 0u)
		{
			const float4 delta = float4(accum.yzw * kSurfaceTension, 0.0f);
			normals[indices[tid.x]] = delta;
			sortedNormals[tid.x] = delta;
		}
	}
}
