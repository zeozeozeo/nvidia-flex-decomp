// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 026 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 026_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/026.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/026_cs_5_0.asm, work/dxbc.json

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
    float3 kStaticTriGridLower;
    float kDiffuseLifetime;
    float3 kWind;
    int kMaxNeighborsPerParticle;
    int kNumGeometryEntries;
    float kMaxAcceleration;
    float kMaxVelocityDelta;
    int kMaxContactsPerParticle;
};

StructuredBuffer<int> collisionCounts : register(t0);

StructuredBuffer<float4> collisionPlanes : register(t1);

StructuredBuffer<float4> prevPositions : register(t2);

StructuredBuffer<float4> collisionVelocities : register(t3);

RWStructuredBuffer<float4> positions : register(u0);

[numthreads(256, 1, 1)]
void SolveContacts(uint3 tid : SV_DispatchThreadID)
{
	// SolveContacts: positional contact resolution with adhesion, static and
	// dynamic friction; the average correction is applied to the position.
	if ((int)tid.x < kNumParticles)
	{
		int count = collisionCounts[tid.x];
		if (count != 0)
		{
			float4 pos = positions[tid.x];
			if (pos.w == 0.0f)
			{
				return;
			}

			count -= 1;

			const bool adhesion = kAdhesion != 0.0f;
			const bool dynamicFriction = 0.0f < kDynamicFriction;

			const float3 prev = prevPositions[tid.x].xyz;
			const float3 dp = pos.xyz - prev;

			float3 accum = float3(0.0f, 0.0f, 0.0f);
			float numContacts = 0.0f;

			for (int n = count; n >= 0; n--)
			{
				const uint ci = tid.x * (uint)kMaxContactsPerParticle + (uint)n;

				const float4 plane = collisionPlanes[ci];
				const float d = dot(pos.xyz, plane.xyz) + plane.w;
				const float dist = d - kCollisionDistance;

				const float3 cand = accum - plane.xyz * (d * kAdhesion);
				if (adhesion)
				{
					accum = cand;
				}

				if (dist < 0.0f)
				{
					const float3 p = accum - plane.xyz * dist;
					numContacts += 1.0f;

					const bool planeNonZero = 0.0f < dot(plane.xyz, plane.xyz);
					const bool frictionOn = dynamicFriction && planeNonZero;

					const float3 rel = dp - collisionVelocities[ci].xyz;
					const float vn = dot(rel, plane.xyz);
					const float3 vt = rel - plane.xyz * vn;
					const float vt2 = dot(vt, vt);
					const float rsq = rsqrt(vt2);

					const float thr = plane.w * kStaticFriction;
					const bool insideRadius = vt2 < thr * thr;

					const float slip = min(rsq * (-plane.w * kDynamicFriction), 1.0f);
					const float3 dyn = p - vt * slip;
					float3 r = (0.0f < vt2) ? dyn : p;
					if (insideRadius)
					{
						r = p - vt;
					}

					accum = frictionOn ? r : p;
				}
			}

			if (0.0f < numContacts)
			{
				float4 outv = pos;
				outv.xyz = accum * (1.0f / numContacts) + pos.xyz;
				positions[tid.x] = outv;
			}
		}
	}
}
