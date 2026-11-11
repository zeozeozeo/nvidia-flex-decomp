// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 025 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 025_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/025.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/025_cs_5_0.asm, work/dxbc.json
//
// The RDEF chunk records the original variable names: sortedNewPositions (t0),
// sortedVelocities (t1), reverseLookup (t2), indices (t3), the raw UAV accum
// (u0) and triNormals (u1).  One thread per triangle: write the face normal,
// then (when drag or lift is enabled) scatter the aerodynamic force onto the
// three corners, weighted by each corner's inverse mass (position .w) and
// counting one contribution per corner.

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
};
StructuredBuffer<float4> sortedNewPositions : register(t0);

StructuredBuffer<float4> sortedVelocities : register(t1);

StructuredBuffer<int> reverseLookup : register(t2);

StructuredBuffer<int> indices : register(t3);

RWByteAddressBuffer accum : register(u0);

RWStructuredBuffer<float3> triNormals : register(u1);

// Float atomic add through an integer compare-exchange loop; the shipped code
// is twelve inlined copies of exactly this loop.
void AtomicAddFloat(uint addr, float value)
{
	uint expected = 0u;
	for (;;)
	{
		uint old;
		InterlockedCompareExchange(accum[addr], expected, asuint(asfloat(expected) + value), old);
		if (old == expected)
		{
			return;
		}
		expected = old;
	}
}

[numthreads(256, 1, 1)]
void UpdateTriangles(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x < kNumTriangles)
	{
		const int i0 = reverseLookup[indices[tid.x * 3 + 0]];
		const int i1 = reverseLookup[indices[tid.x * 3 + 1]];
		const int i2 = reverseLookup[indices[tid.x * 3 + 2]];

		const float4 p0 = sortedNewPositions[i0];
		const float4 p1 = sortedNewPositions[i1];
		const float4 p2 = sortedNewPositions[i2];

		// Face normal: cross(p2 - p0, p1 - p0), normalised, degenerate -> zero.
		const float3 b = p1.xyz - p0.xyz;
		const float3 d = p2.xyz - p0.xyz;
		const float3 c = cross(d, b);
		const float faceLenSq = dot(c, c);
		const float3 faceNormal = (0.0f < faceLenSq) ? c * rsqrt(faceLenSq) : float3(0.0f, 0.0f, 0.0f);
		triNormals[tid.x] = faceNormal;

		if (kDrag != 0.0f || kLift != 0.0f)
		{
			const float3 v0 = sortedVelocities[i0].xyz;
			const float3 v1 = sortedVelocities[i1].xyz;
			const float3 v2 = sortedVelocities[i2].xyz;

			// Relative wind at the face, then the incidence against the face normal.
			const float3 rel = kWind - (v0 + v1 + v2) * 0.3333f;
			const float relSq = dot(rel, rel);
			const float3 relNorm = (0.0f < relSq) ? rel * rsqrt(relSq) : float3(0.0f, 0.0f, 0.0f);

			const float dn = dot(relNorm, faceNormal);
			const float3 n = (dn < 0.0f) ? -faceNormal : faceNormal;

			const float dragScale = dn * kDrag * kDt;
			const float liftScale = dn * kLift * kDt;

			const float cosIncidence = sqrt(max(1.0f - dn * dn, 0.0f));
			const float liftMagnitude = clamp(cosIncidence * liftScale, -1.0f, 1.0f) * sqrt(relSq);

			const float3 liftVector = cross(cross(rel, n), rel);
			const float liftVectorSq = dot(liftVector, liftVector);
			const float3 liftDir = (0.0f < liftVectorSq) ? liftVector * rsqrt(liftVectorSq) : float3(0.0f, 0.0f, 0.0f);

			const float3 force = dragScale * rel + liftMagnitude * liftDir;

			AtomicAddFloat((uint)i0 * 16u + 0u, force.x * p0.w);
			AtomicAddFloat((uint)i0 * 16u + 4u, force.y * p0.w);
			AtomicAddFloat((uint)i0 * 16u + 8u, force.z * p0.w);
			AtomicAddFloat((uint)i0 * 16u + 12u, 1.0f);

			AtomicAddFloat((uint)i1 * 16u + 0u, force.x * p1.w);
			AtomicAddFloat((uint)i1 * 16u + 4u, force.y * p1.w);
			AtomicAddFloat((uint)i1 * 16u + 8u, force.z * p1.w);
			AtomicAddFloat((uint)i1 * 16u + 12u, 1.0f);

			AtomicAddFloat((uint)i2 * 16u + 0u, force.x * p2.w);
			AtomicAddFloat((uint)i2 * 16u + 4u, force.y * p2.w);
			AtomicAddFloat((uint)i2 * 16u + 8u, force.z * p2.w);
			AtomicAddFloat((uint)i2 * 16u + 12u, 1.0f);
		}
	}
}
