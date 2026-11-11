// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 038 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 038_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/038.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/038_cs_5_0.asm, work/dxbc.json

groupshared float3 gShockPoint;
groupshared float3 gShockDir;

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

StructuredBuffer<float3> bounds : register(t0);

RWStructuredBuffer<float4> positions : register(u0);

[numthreads(256, 1, 1)]
void ContinuousShockPropagation(uint3 tid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID)
{
	// ContinuousShockPropagation: one thread picks the extreme bound along
	// gravity, then every particle's w is scaled by the exponential of its
	// distance along the shock direction.
	if (gtid.x == 0)
	{
		const float3 bound0 = bounds[0];
		gShockPoint = bound0;

		const float3 bound1 = bounds[1];
		if (dot(kGravity, bound0) < dot(kGravity, bound1))
		{
			gShockPoint = bound1;
		}

		const float gl = dot(kGravity, kGravity);
		const bool positive = 0.0f < gl;
		float3 dir = kGravity * rsqrt(gl);
		if (!positive)
		{
			dir = float3(0.0f, 0.0f, 0.0f);
		}
		gShockDir = dir;
	}
	GroupMemoryBarrierWithGroupSync();

	if ((int)tid.x < kNumParticles)
	{
		float4 pos = positions[tid.x];
		const float3 diff = gShockPoint - pos.xyz;
		const float s = dot(diff, gShockDir) * kShockPropagation;
		pos.w = exp(s) * pos.w;
		positions[tid.x] = pos;
	}
}
