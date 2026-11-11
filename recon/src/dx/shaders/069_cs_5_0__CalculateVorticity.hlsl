// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 069 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 069_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/069.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/069_cs_5_0.asm, work/dxbc.json

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

StructuredBuffer<float4> fluidPositions : register(t0);

StructuredBuffer<float4> fluidVelocities : register(t1);

StructuredBuffer<uint> particleFlags : register(t2);

StructuredBuffer<uint> neighbors : register(t3);

StructuredBuffer<int> neighborCounts : register(t4);

RWStructuredBuffer<float4> vorticity : register(u0);

[numthreads(256, 1, 1)]
void CalculateVorticity(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x < kNumParticles)
	{
		// Only diffuse particles carry vorticity; the rest are cleared.
		if ((particleFlags[tid.x] & 0x400000) == 0)
		{
			vorticity[tid.x] = float4(0, 0, 0, 0);
			return;
		}

		const float3 pos = fluidPositions[tid.x].xyz;
		const float3 vel = fluidVelocities[tid.x].xyz;
		const int count = neighborCounts[tid.x];

		float3 w = float3(0, 0, 0);
		int j = tid.x;
		for (int c = 0; c < count; ++c)
		{
			const int n = neighbors[j];
			// Neighbour lists are strided by the aligned particle count.
			j += kNumParticlesAligned;

			if ((particleFlags[n] & 0x400000) == 0)
			{
				continue;
			}

			const float3 r = pos - fluidPositions[n].xyz;
			const float rSq = dot(r, r);
			const bool inRange = (kRadiusSq >= rSq) && (rSq > 0.0f);

			const float3 velDiff = fluidVelocities[n].xyz - vel;
			const float dist = sqrt(rSq);
			// -(1 - |r| * kInvRadius) * kSpiky2 / |r| * r
			const float falloff = (1.0f - dist * kInvRadius) * -kSpiky2;
			const float3 scaled = (r * falloff) / dist;
			const float3 cross = float3(
				scaled.y * velDiff.z - scaled.z * velDiff.y,
				scaled.z * velDiff.x - scaled.x * velDiff.z,
				scaled.x * velDiff.y - scaled.y * velDiff.x);
			w = inRange ? (w + cross) : w;
		}

		vorticity[tid.x] = float4(w, length(w));
	}
}
