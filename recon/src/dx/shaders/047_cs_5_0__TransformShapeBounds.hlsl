// Reconstructed from the shipped DXBC bytecode of NvFlexReleaseD3D_x64.dll,
// shader 047 (cs_5_0).  Interface (constant-buffer layout, resource bindings,
// thread group) comes from the RDEF chunk; the body is derived from the SHEX
// assembly 047_cs_5_0.asm.
//
// Entry point: TransformShapeBounds (RDEF name table: "TransformShapeBounds::lowers").
// Evidence: work/dxbc/NvFlexReleaseD3D_x64/047.dxbc,
//           work/dxbc/NvFlexReleaseD3D_x64/047_cs_5_0.asm, work/dxbc.json

cbuffer consts : register(b0)
{
    int gNumBounds;
};

struct FlexConvexMeshDevice
{
    float3 mLower;
    float mPlaneOffset;
    float3 mUpper;
    int mPlaneCount;
    int mPad[4];
};

struct FlexTriangleMeshDevice
{
    float3 mLower;
    int mIndexStart;
    float3 mUpper;
    int mVertexStart;
    int mNodeStart;
    int mPad[3];
};

StructuredBuffer<float4> positions : register(t0);
StructuredBuffer<float4> rotations : register(t1);
StructuredBuffer<float4> positionsPrev : register(t2);
StructuredBuffer<float4> rotationsPrev : register(t3);
StructuredBuffer<int> flags : register(t4);
StructuredBuffer<float4> geometry : register(t5);
StructuredBuffer<FlexConvexMeshDevice> convexMeshes : register(t6);
StructuredBuffer<FlexTriangleMeshDevice> triangleMeshes : register(t7);

RWStructuredBuffer<float4> lowers : register(u0);
RWStructuredBuffer<float4> uppers : register(u1);

// Quaternion rotation of v, transcribed from the bytecode sequence:
//   t = cross(v, q.xyz)
//   v' = v * (q.w*q.w - 1) + 2 * q.xyz * dot(q.xyz, v) + 2 * q.w * t
static float3 Rotate(float4 q, float3 v)
{
	const float3 t = cross(v, q.xyz);
	return v * (q.w * q.w - 1.0f) + 2.0f * q.xyz * dot(q.xyz, v) + 2.0f * q.w * t;
}

// Rotation matrix of a unit quaternion; row i scaled by e.i and summed with the
// absolute values gives the world-space half extent of the transformed box
// (|R| * e), which is what the bytecode accumulates.
static float3x3 QuatMatrix(float4 q)
{
	const float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
	const float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
	const float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;
	return float3x3(
		1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz),         2.0f * (xz + wy),
		2.0f * (xy + wz),         1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx),
		2.0f * (xz - wy),         2.0f * (yz + wx),        1.0f - 2.0f * (xx + yy));
}

static float3 AbsMul(float3x3 m, float3 e)
{
	const float3 r0 = abs(float3(m[0][0], m[0][1], m[0][2]));
	const float3 r1 = abs(float3(m[1][0], m[1][1], m[1][2]));
	const float3 r2 = abs(float3(m[2][0], m[2][1], m[2][2]));
	return r0 * e.x + r1 * e.y + r2 * e.z;
}

[numthreads(256, 1, 1)]
void TransformShapeBounds(uint3 tid : SV_DispatchThreadID)
{
	if ((int)tid.x >= gNumBounds)
	{
		return;
	}

	const uint i = tid.x;
	const int type = flags[i] & 7;

	// geometry is read with a .yzxw swizzle in the bytecode
	const float4 gv = geometry[i];
	const float a = gv.y;
	const float b = gv.z;
	const float c = gv.x;
	const float w = gv.w;

	float3 lo = float3(0.0f, 0.0f, 0.0f);
	float3 hi = float3(0.0f, 0.0f, 0.0f);

	switch (type)
	{
	case 0: // eNvFlexShapeSphere: radius = geometry.x
		lo = float3(-c, -c, -c);
		hi = float3(c, c, c);
		break;
	case 1: // eNvFlexShapeCapsule: radius = geometry.x, half height = geometry.y
		lo = float3(-(c + a), -c, -(c + a));
		hi = float3(c, c, c + a);
		break;
	case 2: // eNvFlexShapeBox: half extents
		lo = float3(-c, -a, -b);
		hi = float3(a, b, c);
		break;
	case 3: // eNvFlexShapeConvexMesh: scale the mesh's local bounds
	{
		const FlexConvexMeshDevice m = convexMeshes[(uint)(w - 1.0f)];
		lo = float3(c, a, b) * m.mLower;
		hi = float3(a, b, c) * float3(m.mUpper.y, m.mUpper.z, m.mUpper.x);
		break;
	}
	case 4: // eNvFlexShapeTriangleMesh: scale the mesh's local bounds
	{
		const FlexTriangleMeshDevice m = triangleMeshes[(uint)(w - 1.0f)];
		lo = float3(c, a, b) * m.mLower;
		hi = float3(a, b, c) * float3(m.mUpper.y, m.mUpper.z, m.mUpper.x);
		break;
	}
	case 5: // eNvFlexShapeSDF: radius = geometry.x
		lo = float3(0.0f, 0.0f, 0.0f);
		hi = float3(c, c, c);
		break;
	default:
		break;
	}

	const float3 extent = hi - lo;
	const float3 center = (lo + hi) * 0.5f;
	const float3 origin = positions[i].xyz;

	// current transform
	{
		const float4 q = rotations[i];
		const float3 halfExtent = AbsMul(QuatMatrix(q), extent) * 0.5f;
		const float3 c0 = Rotate(q, center) + origin;
		lowers[i] = float4(c0 - halfExtent, 0.0f);
		uppers[i] = float4(c0 + halfExtent, 0.0f);
	}

	// the shipped shader stores the union of the current and the previous
	// transform (swept bounds)
	{
		const float4 qp = rotationsPrev[i];
		const float3 halfExtent = AbsMul(QuatMatrix(qp), extent) * 0.5f;
		const float3 c1 = Rotate(qp, center) + positionsPrev[i].xyz;
		lowers[i] = float4(min(lowers[i].xyz, c1 - halfExtent), 0.0f);
		uppers[i] = float4(max(uppers[i].xyz, c1 + halfExtent), 0.0f);
	}
}
