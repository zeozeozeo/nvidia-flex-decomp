// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 066 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 066_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/066.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/066_cs_5_0.asm, work/dxbc.json

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
    float kCollideParticlesRadiusSq;
    int kMaxParticles;
    int kRelaxationMode;
    int kShapesCount;
    int kNumShapes;
    int kNumRigids;
    int kNumSprings;
    int kNumTriangles;
    int kMaxDiffuseParticles;
    int kDiffuseRadixSortSize;
    float kDiffuseDt;
    float kDiffuseMaxVelocity;
    float3 kDiffuseSortAxis;
    float kRelaxationFactor;
};

ByteAddressBuffer deltas : register(t0);

RWStructuredBuffer<float4> newPositions : register(u0);

[numthreads(256, 1, 1)]
void ApplyDeltas(uint3 tid : SV_DispatchThreadID)
{
	// ApplyDeltas: newPositions += delta * weight, SOR or plain relaxation
	if ((int)tid.x < kNumParticles)
	{
		float4 d = asfloat(deltas.Load4(tid.x * 16));
		float4 p = newPositions[tid.x];

		float3 r;
		if (kRelaxationMode != 0)
		{
			float w = d.w * kSOR;
			w = max(w, 1.0f);
			w = 1.0f / w;
			r = d.xyz * w + p.xyz;
		}
		else
		{
			r = d.xyz * kRelaxationFactor + p.xyz;
		}

		newPositions[tid.x] = float4(r, p.w);
	}
}
