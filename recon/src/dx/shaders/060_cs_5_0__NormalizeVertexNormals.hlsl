// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 060 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 060_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/060.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/060_cs_5_0.asm, work/dxbc.json

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

Buffer<uint> indices : register(t0);

ByteAddressBuffer tempNormals : register(t1);

RWStructuredBuffer<float4> normals : register(u0);

[numthreads(256, 1, 1)]
void NormalizeVertexNormals(uint3 tid : SV_DispatchThreadID)
{
	// NormalizeVertexNormals: normalized = normalize(tempNormals[index]),
	// stored negated in the xyz of the float4 normal, w cleared.
	if ((int)tid.x < kNumParticles)
	{
		const uint index = indices[tid.x];
		const uint base = index * 16;
		float3 n = float3(asfloat(tempNormals.Load(base)),
		                  asfloat(tempNormals.Load(base + 4)),
		                  asfloat(tempNormals.Load(base + 8)));

		const float d = dot(n, n);
		const bool nonzero = 0.0f < d;
		n = n / sqrt(d);
		if (!nonzero)
		{
			n = float3(0.0f, 0.0f, 0.0f);
		}

		float4 outv = normals[index];
		outv.xyz = -n;
		outv.w = 0.0f;
		normals[index] = outv;
	}
}
