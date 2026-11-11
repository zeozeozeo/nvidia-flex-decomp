// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 073 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 073_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/073.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/073_cs_5_0.asm, work/dxbc.json

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
    float kSubStepLength;
    float kMaxSpeed;
    float kInvCellEdge;
    float kStaticTriGridInvCellEdge;
};

StructuredBuffer<int> indices : register(t0);

StructuredBuffer<float4> positions : register(t1);

StructuredBuffer<uint> particleFlags : register(t2);

RWStructuredBuffer<float4> predictedPositions : register(u0);

RWStructuredBuffer<float3> velocities : register(u1);

[numthreads(256, 1, 1)]
void Predict(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x < kNumParticles)
	{
		const int index = indices[tid.x];
		const float4 p = positions[index];
		const float3 vOld = velocities[index];
		const bool active = p.w > 0.0f;
		const uint flags = particleFlags[index];

		const float3 g = kGravity * kDt;
		// Diffuse particles are buoyant: only a fraction of gravity applies.
		const float3 vBuoyant = vOld + g * kBuoyancy;
		const float3 vPlain = vOld + g;
		float3 v = (flags & 0x400000) ? vBuoyant : vPlain;
		v = active ? v : vOld;

		// Clamp to the maximum allowed speed.
		const float lenSq = dot(v, v);
		if (kMaxSpeed * kMaxSpeed < lenSq)
		{
			v = v * (kMaxSpeed * rsqrt(lenSq));
		}

		velocities[index] = v;
		predictedPositions[index] = float4(p.xyz + v * kDt, p.w);
	}
}
