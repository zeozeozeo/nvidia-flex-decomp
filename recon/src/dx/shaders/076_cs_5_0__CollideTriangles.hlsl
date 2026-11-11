// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 076 (cs_5_0).  Interface (constant-buffer layout, resource
// bindings, thread group) comes from the RDEF chunk; the body is derived from
// the SHEX assembly 076_cs_5_0.asm.
//
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/076.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/076_cs_5_0.asm, work/dxbc.json

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

struct MeshOffsets
{
	float4 mVertexStart;
	float4 mIndexStart;
	float4 mNodeStart;
};

StructuredBuffer<float4> triangleVerticesArray : register(t0);

StructuredBuffer<int> triangleIndicesArray : register(t1);

StructuredBuffer<float4> bvhNodeLowersArray : register(t2);

StructuredBuffer<float4> bvhNodeUppersArray : register(t3);

StructuredBuffer<int> bvhRootNodeArray : register(t4);

StructuredBuffer<float4> sortedPositions : register(t5);

StructuredBuffer<float4> sortedNewPositions : register(t6);

StructuredBuffer<uint> sortedPhases : register(t7);

StructuredBuffer<float4> shapePrevPositions : register(t8);

StructuredBuffer<float4> shapePrevRotations : register(t9);

StructuredBuffer<uint> shapeFlags : register(t10);

StructuredBuffer<float4> shapePositions : register(t11);

StructuredBuffer<float4> shapeRotations : register(t12);

StructuredBuffer<float4> shapeGeometry : register(t13);

StructuredBuffer<int> shapeBvhRootNode : register(t14);

StructuredBuffer<float4> shapeBvhNodeLowers : register(t15);

StructuredBuffer<float4> shapeBvhNodeUppers : register(t16);

StructuredBuffer<MeshOffsets> meshOffsets : register(t17);

RWStructuredBuffer<int> collisionCounts : register(u0);

RWStructuredBuffer<float4> collisionPlanes : register(u1);

RWStructuredBuffer<float4> collisionVelocities : register(u2);

cbuffer constSubstep : register(b1)
{
	float4 gSubParams;   // x / y are the substep interpolation fractions
};

float3 RotateByQuat(float4 q, float3 v)
{
	return v + 2.0f * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

[numthreads(256, 1, 1)]
void CollideTriangles(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= kNumParticles)
	{
		return;
	}

	const float3 pos = sortedNewPositions[tid.x].xyz;
	const float3 prevPos = sortedPositions[tid.x].xyz;
	int count = 0;

	if (kNumShapes != 0)
	{
		const uint phaseMask = sortedPhases[tid.x] & 4278190080u;

		// Swept bound over the frame, expanded by the collision margin.
		const float3 lower = min(pos, prevPos) - kCollisionMargin;
		const float3 upper = max(pos, prevPos) + kCollisionMargin;

		int stack[40];
		int stackSize = 0;
		stack[stackSize++] = shapeBvhRootNode[0];

		// ---- shape BVH ----
		while (stackSize != 0)
		{
			const int n = stack[--stackSize];
			const float4 nodeLower = shapeBvhNodeLowers[n];
			const float4 nodeUpper = shapeBvhNodeUppers[n];

			if (any(upper < nodeLower.xyz) || any(nodeUpper.xyz < lower))
			{
				continue;
			}

			const uint encoded = asuint(nodeLower.w);
			if ((encoded & 0x80000000u) == 0)
			{
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

			// Mesh shapes carry geometry type 4; other types are handled by
			// CollideShapes.
			if ((flags & 7) != 4)
			{
				continue;
			}

			const MeshOffsets offsets = meshOffsets[shapeIndex];
			const float3 shapePos = shapePositions[shapeIndex].xyz;
			const float4 shapeRot = shapeRotations[shapeIndex];
			const float3 prevShapePos = shapePrevPositions[shapeIndex].xyz;
			const float4 prevShapeRot = shapePrevRotations[shapeIndex];

			// Interpolated mesh transform for this substep.
			const float3 sweptPos = lerp(prevShapePos, shapePos, gSubParams.x);
			const float4 sweptRot = normalize(lerp(prevShapeRot, shapeRot, gSubParams.y));

			// Particle in mesh space.
			const float3 local = RotateByQuat(float4(-sweptRot.xyz, sweptRot.w), pos - sweptPos);
			const float3 localPrev = RotateByQuat(float4(-sweptRot.xyz, sweptRot.w), prevPos - sweptPos);
			const float3 meshLower = min(local, localPrev) - kCollisionMargin;
			const float3 meshUpper = max(local, localPrev) + kCollisionMargin;

			// Bounding-sphere rejection in mesh space.
			const float4 geom = shapeGeometry[shapeIndex];
			if (length(local - geom.xyz) > geom.w + kCollisionDistance)
			{
				continue;
			}

			// ---- triangle BVH of this mesh ----
			int triStack[40];
			int triSize = 0;
			triStack[triSize++] = bvhRootNodeArray[(int)offsets.mNodeStart.x];

			while (triSize != 0)
			{
				const int tn = triStack[--triSize];
				const float4 triLower = bvhNodeLowersArray[tn];
				const float4 triUpper = bvhNodeUppersArray[tn];

				if (any(meshUpper < triLower.xyz) || any(triUpper.xyz < meshLower))
				{
					continue;
				}

				const uint triEncoded = asuint(triLower.w);
				if ((triEncoded & 0x80000000u) == 0)
				{
					triStack[triSize++] = tn + 1;
					triStack[triSize++] = (int)(triEncoded & 0x7fffffffu);
					continue;
				}

				const int tri = (int)(triEncoded & 0x7fffffffu);
				const int base = (int)offsets.mVertexStart.x;
				const int indexBase = (int)offsets.mIndexStart.x;

				const int i0 = base + triangleIndicesArray[indexBase + tri * 3 + 0];
				const int i1 = base + triangleIndicesArray[indexBase + tri * 3 + 1];
				const int i2 = base + triangleIndicesArray[indexBase + tri * 3 + 2];
				const float3 v0 = triangleVerticesArray[i0].xyz;
				const float3 v1 = triangleVerticesArray[i1].xyz;
				const float3 v2 = triangleVerticesArray[i2].xyz;

				// Closest point on the triangle to the particle.
				const float3 e1 = v1 - v0;
				const float3 e2 = v2 - v0;
				const float3 rel = local - v0;
				const float d00 = dot(e1, e1);
				const float d01 = dot(e1, e2);
				const float d11 = dot(e2, e2);
				const float d20 = dot(rel, e1);
				const float d21 = dot(rel, e2);
				const float denom = d00 * d11 - d01 * d01;
				float v = (d11 * d20 - d01 * d21) / denom;
				float w = (d00 * d21 - d01 * d20) / denom;
				v = clamp(v, 0.0f, 1.0f);
				w = clamp(w, 0.0f, 1.0f - v);
				const float3 closest = v0 + e1 * v + e2 * w;

				const float3 delta = local - closest;
				const float dist = length(delta);
				if (dist < kCollisionDistance && count < kMaxContactsPerParticle)
				{
					// Normal back in world space.
					const float3 n = RotateByQuat(sweptRot, normalize(delta));
					const int slot = tid.x * kMaxContactsPerParticle + count;
					collisionPlanes[slot] = float4(n, dist - kCollisionDistance);
					collisionVelocities[slot] = float4(0, 0, 0, 0);
					++count;
				}
			}
		}
	}

	collisionCounts[tid.x] = count;
}
