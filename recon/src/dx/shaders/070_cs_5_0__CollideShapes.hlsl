// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 070 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 070_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/070.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/070_cs_5_0.asm, work/dxbc.json

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

struct ShapeGeometry
{
	float4 mBounds;      // xyz = local centre, w = radius / flags
	float4 mRot;         // quaternion
	float4 mPad;
};

struct Convex
{
	float4 mPlanes[3];   // three local planes; the count is in the geometry
};

struct SdfDesc
{
	int4   mDim;
	float4 mInvDim;
	float4 mPad;
};

struct FlexSDFDevice
{
	float mSubStep;      // cb1[0].x  interpolates between the previous and the
	float mSubStepPrev;  // cb1[0].y  current shape transform
	float2 mPad;
};

SamplerState sdfSampler : register(s0);

StructuredBuffer<ShapeGeometry> shapeGeometry : register(t0);

StructuredBuffer<float4> shapePositions : register(t1);

StructuredBuffer<float4> shapeRotations : register(t2);

StructuredBuffer<float4> shapePrevPositions : register(t3);

StructuredBuffer<float4> shapePrevRotations : register(t4);

StructuredBuffer<uint> shapeFlags : register(t5);

StructuredBuffer<float4> sortedPositions : register(t6);

StructuredBuffer<float4> sortedNewPositions : register(t7);

StructuredBuffer<uint> sortedPhases : register(t8);

StructuredBuffer<int> shapeBvhRootNode : register(t9);

StructuredBuffer<float4> shapeBvhNodeLowers : register(t10);

StructuredBuffer<float4> shapeBvhNodeUppers : register(t11);

StructuredBuffer<Convex> convexes : register(t12);

StructuredBuffer<float4> convexPlanes : register(t13);

StructuredBuffer<SdfDesc> sdfs : register(t14);

Texture3D<float4> sdfTex0 : register(t15);
Texture3D<float4> sdfTex1 : register(t16);
Texture3D<float4> sdfTex2 : register(t17);
Texture3D<float4> sdfTex3 : register(t18);
Texture3D<float4> sdfTex4 : register(t19);
Texture3D<float4> sdfTex5 : register(t20);
Texture3D<float4> sdfTex6 : register(t21);
Texture3D<float4> sdfTex7 : register(t22);
Texture3D<float4> sdfTex8 : register(t23);
Texture3D<float4> sdfTex9 : register(t24);
Texture3D<float4> sdfTex10 : register(t25);
Texture3D<float4> sdfTex11 : register(t26);
Texture3D<float4> sdfTex12 : register(t27);
Texture3D<float4> sdfTex13 : register(t28);
Texture3D<float4> sdfTex14 : register(t29);
Texture3D<float4> sdfTex15 : register(t30);

RWStructuredBuffer<int> collisionCounts : register(u0);

RWStructuredBuffer<float4> collisionPlanes : register(u1);

RWStructuredBuffer<float4> collisionVelocities : register(u2);

cbuffer constSubstep : register(b1)
{
	FlexSDFDevice gSubParams;
};

// Signed distance of the shape stored in texture slot `slot` at local `uvw`.
float SampleSdf(uint slot, float3 uvw)
{
	if (slot == 0)  return sdfTex0.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 1)  return sdfTex1.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 2)  return sdfTex2.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 3)  return sdfTex3.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 4)  return sdfTex4.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 5)  return sdfTex5.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 6)  return sdfTex6.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 7)  return sdfTex7.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 8)  return sdfTex8.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 9)  return sdfTex9.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 10) return sdfTex10.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 11) return sdfTex11.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 12) return sdfTex12.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 13) return sdfTex13.SampleLevel(sdfSampler, uvw, 0).x;
	if (slot == 14) return sdfTex14.SampleLevel(sdfSampler, uvw, 0).x;
	return sdfTex15.SampleLevel(sdfSampler, uvw, 0).x;
}

float3 RotateByQuat(float4 q, float3 v)
{
	return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

[numthreads(128, 1, 1)]
void CollideShapes(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const float3 pos = sortedNewPositions[tid.x].xyz;
	int count = collisionCounts[tid.x];

	// ---- global collision planes ----
	for (int p = 0; p < kNumPlanes; ++p)
	{
		const float4 plane = kPlanes[p];   // dynamic constant-buffer indexing
		const float d = dot(pos, plane.xyz) + plane.w - kCollisionDistance;
		if (kCollisionMargin >= d && count < kMaxContactsPerParticle)
		{
			const int slot = tid.x * kMaxContactsPerParticle + count;
			collisionPlanes[slot] = plane;
			// Planes report an infinite inverse mass (-1 in w).
			collisionVelocities[slot] = float4(0, 0, 0, -1.0f);
			++count;
		}
	}

	// ---- shapes through the shape BVH ----
	if (kNumShapes != 0)
	{
		const float3 prevPos = sortedPositions[tid.x].xyz;
		const uint phaseMask = sortedPhases[tid.x] & 4278190080u;

		// Swept bound of the particle over the frame, expanded by the margin.
		const float3 lower = min(pos, prevPos) - kCollisionMarginSq;
		const float3 upper = max(pos, prevPos) + kCollisionMarginSq;

		int stack[64];
		int stackSize = 0;
		int node = shapeBvhRootNode[0];
		stack[stackSize++] = node;

		while (stackSize != 0)
		{
			const int n = stack[--stackSize];
			const float4 nodeLower = shapeBvhNodeLowers[n];
			const float4 nodeUpper = shapeBvhNodeUppers[n];

			// Node misses the swept bound: skip its subtree.
			if (any(upper < nodeLower.xyz) || any(nodeUpper.xyz < lower))
			{
				continue;
			}

			const uint encoded = asuint(nodeLower.w);
			const bool isLeaf = (encoded & 0x80000000u) != 0;
			if (!isLeaf)
			{
				// interior: child index in the low bits, visit the right child
				stack[stackSize++] = n + 1;
				stack[stackSize++] = (int)(encoded & 0x7fffffffu);
				continue;
			}

			const int shapeIndex = (int)(encoded & 0x7fffffffu);
			const uint flags = shapeFlags[shapeIndex];
			if ((flags & phaseMask) == 0)
			{
				continue;
			}

			const int geometry = (int)(flags & 7);
			const float3 shapePos = shapePositions[shapeIndex].xyz;
			const float4 shapeRot = shapeRotations[shapeIndex];
			const float3 prevShapePos = shapePrevPositions[shapeIndex].xyz;
			const float4 prevShapeRot = shapePrevRotations[shapeIndex];

			// Interpolate the shape transform for this substep.
			const float3 sweptPos = lerp(prevShapePos, shapePos, gSubParams.mSubStep);
			const float4 sweptRot = normalize(lerp(prevShapeRot, shapeRot, gSubParams.mSubStep));

			const float3 local = RotateByQuat(float4(-sweptRot.xyz, sweptRot.w), pos - sweptPos);

			if ((flags & 0x80000000u) != 0)
			{
				// ---- convex shape: up to three local planes ----
				const int numPlanes = geometry;
				for (int c = 0; c < numPlanes; ++c)
				{
					const float4 localPlane = convexPlanes[shapeIndex * 3 + c];
					const float3 n = RotateByQuat(sweptRot, localPlane.xyz);
					const float d = dot(local, localPlane.xyz) + localPlane.w;
					if (d < 0.0f && count < kMaxContactsPerParticle)
					{
						const int slot = tid.x * kMaxContactsPerParticle + count;
						collisionPlanes[slot] = float4(n, d);
						collisionVelocities[slot] = float4(0, 0, 0, 0);
						++count;
					}
				}
			}
			else
			{
				// ---- signed distance field shape ----
				const SdfDesc sdf = sdfs[geometry];
				const float3 uvw = local * sdf.mInvDim.xyz + 0.5f;
				if (all(uvw > 0.0f) && all(uvw < 1.0f))
				{
					const float dist = SampleSdf((uint)geometry, uvw);
					if (dist < kCollisionDistance && count < kMaxContactsPerParticle)
					{
						const int slot = tid.x * kMaxContactsPerParticle + count;
						const float3 grad = float3(
							SampleSdf((uint)geometry, uvw + float3(sdf.mInvDim.x, 0, 0)) - dist,
							SampleSdf((uint)geometry, uvw + float3(0, sdf.mInvDim.y, 0)) - dist,
							SampleSdf((uint)geometry, uvw + float3(0, 0, sdf.mInvDim.z)) - dist);
						const float3 n = RotateByQuat(sweptRot, normalize(grad));
						collisionPlanes[slot] = float4(n, dist - kCollisionDistance);
						collisionVelocities[slot] = float4(0, 0, 0, 0);
						++count;
					}
				}
			}
		}
	}

	collisionCounts[tid.x] = count;
}
