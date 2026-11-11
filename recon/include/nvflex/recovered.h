// recovered from DWARF by flexkit headers (canonical definitions).
// Layouts are preserved exactly: opaque members keep their offsets.
#pragma once
#include <cstddef>
#include <cstdint>

#ifndef FLEXDECOMP_DEF_CUcontext
#define FLEXDECOMP_DEF_CUcontext
typedef void* CUcontext;
#endif
#ifndef FLEXDECOMP_DEF_cudaArray_t
#define FLEXDECOMP_DEF_cudaArray_t
typedef void* cudaArray_t;
#endif
#ifndef FLEXDECOMP_DEF_cudaEvent_t
#define FLEXDECOMP_DEF_cudaEvent_t
typedef void* cudaEvent_t;
#endif
#ifndef FLEXDECOMP_DEF_cudaGraphicsResource_t
#define FLEXDECOMP_DEF_cudaGraphicsResource_t
typedef void* cudaGraphicsResource_t;
#endif
#ifndef FLEXDECOMP_DEF_cudaTextureObject_t
#define FLEXDECOMP_DEF_cudaTextureObject_t
typedef unsigned long long cudaTextureObject_t;
#endif
#ifndef FLEXDECOMP_DEF_uint32_t
#define FLEXDECOMP_DEF_uint32_t
typedef unsigned int uint32_t;
#endif

struct BVH;
struct Bounds;
struct Colour;
struct CudaRangeTimer;
struct CudaRangeTimerPool;
struct CudaTimer;
struct CudaTimers;
struct DeviceBuffer_int;
struct FlexCollisionShapeData;
struct FlexConvexMeshData;
struct FlexConvexMeshInternal;
struct FlexInflatable;
struct FlexSDFData;
struct FlexSDFInternal;
struct FlexTextures;
struct FlexTriangleMeshData;
struct FlexTriangleMeshInternal;
struct KernelBounds;
struct KernelParams;
struct KeyIndexPair;
struct LinearBVHBuilderCPU;
struct LinearBVHBuilderGPU;
struct Matrix33;
struct MedianBVHBuilder;
struct NvFlexBuffer;
struct NvFlexLibrary;
struct NvFlexSolver;
struct NvResourceTracker;
struct PackedNodeHalf;
struct PartitionPredicateMedian;
struct PartitionPredictateMidPoint;
struct Plane;
struct Point3;
struct Vec3Iter;
struct XMatrix44_float;
struct XQuat_float;
struct XVector2_float;
struct XVector3_float;
struct XVector4_float;
struct dim3;
struct ltstr;
struct uint3;

typedef CudaTimer CudaDetailTimer;
typedef XMatrix44_float Mat44;
typedef XMatrix44_float Matrix44;
typedef XQuat_float Quat;
typedef XVector3_float Vec3;
typedef XVector4_float Vec4;
typedef XVector3_float Vector3;

// BVH
struct BVH {
    unsigned long long mNodeLowersTex;
    unsigned long long mNodeUppersTex;
    int* mRootNode;
    PackedNodeHalf* mNodeLowers;
    PackedNodeHalf* mNodeUppers;
    int mNumNodes;
    int mMaxNodes;
    int mMaxDepth;
};

// XVector3<float>
struct XVector3_float {
    float x;
    float y;
    float z;
};

// Bounds
struct Bounds {
    Vec3 lower;
    Vec3 upper;
};

// Colour
struct Colour {
    float r;
    float g;
    float b;
    float a;
};

// CudaRangeTimer
struct CudaRangeTimer {
    void* mBegin;
    void* mEnd;
    float mTime;
};

// CudaRangeTimerPool
struct CudaRangeTimerPool {
    unsigned char mElem[24];  // DWARF: vector_CudaRangeTimer_ std__allocator_CudaRangeTimer_ _ (opaque)
    int mUsedElemCount;
    // static member mInitCapacity (const int)
};

// CudaTimer
struct CudaTimer {
    bool mEnabled;
    int mTimerIndex;
    unsigned char mNameToTimerMap[8];  // DWARF: const NameToTimerMap_ (opaque)
    CudaRangeTimerPool* mTimerPool;
};

// CudaTimers
struct CudaTimers {
    bool mEnabled;
    unsigned char mNameToTimerMap[48];  // DWARF: NameToTimerMap (opaque)
    CudaRangeTimerPool* mTimerPool;
};

// DeviceBuffer<int>
struct DeviceBuffer_int {
    int* mPtr;
    int mCapacity;
    unsigned long long mTexture;
};

// FlexCollisionShapeData
struct FlexCollisionShapeData {
    float* mAabbMin;
    float* mAabbMax;
    void* mGeometry;  // DWARF: NvFlexCollisionGeometry_*
    Vec4* mPositions;
    Vec4* mRotations;
    Vec4* mPrevPositions;
    Vec4* mPrevRotations;
    int* mFlags;
    BVH mBVH;
    unsigned int mNumShapes;
    unsigned int mMaxShapes;
};

// FlexConvexMeshData
struct FlexConvexMeshData {
    Vec4* mPlanes;
    unsigned long long mPlanesTex;
    FlexConvexMeshInternal* mConvexes;
    int mNumConvexes;
    int mMaxConvexes;
    int mMaxPlanesPerConvex;
    int* mFreeList;
};

// FlexConvexMeshInternal
struct FlexConvexMeshInternal {
    int mPlaneCount;
    int mPlaneOffset;
    Vec3 mLower;
    Vec3 mUpper;
};

// FlexInflatable
struct FlexInflatable {
    int mStartTri;
    int mNumTris;
    float mRestVolume;
    float mConstraintScale;
};

// FlexSDFData
struct FlexSDFData {
    FlexSDFInternal* mSDFTableDevice;
    FlexSDFInternal* mSDFTableHost;
    int mNumSDFs;
    int mMaxSDFs;
    int* mFreeList;
};

// FlexSDFInternal
struct FlexSDFInternal {
    int mDim[3];
    float mInvDim[3];
    unsigned long long mFieldTex;
    void* mFieldArray;
};

// FlexTextures
struct FlexTextures {
    unsigned long long sortedPositionsTex;
    unsigned long long sortedNewPositionsTex;
    unsigned long long sortedNewVelocitiesTex;
    unsigned long long restPositionsTex;
    unsigned long long cellStartsTex;
    unsigned long long cellEndsTex;
    unsigned long long contactsTex;
    unsigned long long normalsTex;
    unsigned long long curlTex;
    unsigned long long shapeGeometryTex;
    unsigned long long shapePositionsTex;
    unsigned long long shapeRotationsTex;
    unsigned long long shapePrevPositionsTex;
    unsigned long long shapePrevRotationsTex;
};

// FlexTriangleMeshData
struct FlexTriangleMeshData {
    FlexTriangleMeshInternal* mMeshTableDevice;
    FlexTriangleMeshInternal* mMeshTableHost;
    int mNumMeshes;
    int mMaxMeshes;
    int* mFreeList;
};

// FlexTriangleMeshInternal
struct FlexTriangleMeshInternal {
    unsigned char mVertices[24];  // DWARF: DeviceBuffer_XVector4_float_ _ (opaque)
    DeviceBuffer_int mIndices;
    BVH mBVH;
    int mNumTris;
    int mNumVertices;
    int mMaxTris;
    int mMaxVertices;
    Vec3 mLower;
    Vec3 mUpper;
};

// KernelBounds
struct KernelBounds {
    float kLower[3];
    float kUpper[3];
};

// KernelParams
struct KernelParams {
    float kPlanes[32];
    int kNumPlanes;
    int kMaxContactsPerParticle;
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
    float kDiffuseLifetime;
    float kGravity[3];
    float kDynamicFriction;
    float kStaticFriction;
    float kParticleFriction;
    float kAdhesion;
    float kDissipation;
    float kDamping;
    float kCollisionDistance;
    float kCollisionThreshold;
    float kCollisionMargin;
    float kCollisionMarginSq;
    float kDrag;
    float kLift;
    float kSleepThreshold;
    float kSleepThresholdSq;
    float kShockPropagation;
    float kRestitution;
    float kSOR;
    FlexTextures kTextures;
    FlexConvexMeshData kConvexMeshData;
    FlexTriangleMeshData kTriangleMeshData;
    FlexSDFData kSDFData;
};

// KeyIndexPair
struct KeyIndexPair {
    unsigned int key;
    int index;
};

// LinearBVHBuilderCPU
struct LinearBVHBuilderCPU {
};

// LinearBVHBuilderGPU
struct LinearBVHBuilderGPU {
    int* mIndices;
    int* mKeys;
    int* mDeltas;
    int* mRangeLefts;
    int* mRangeRights;
    int* mNumChildren;
    Vec3* mTotalLower;
    Vec3* mTotalUpper;
    Vec3* mTotalInvEdges;
    int mMaxItems;
};

// Matrix33
struct Matrix33 {
    XVector3_float cols[3];
};

// MedianBVHBuilder
struct MedianBVHBuilder {
};

// NvFlexBuffer
struct NvFlexBuffer {
    bool mapped;
    int type;
    void* graphicsResource;
    NvFlexLibrary* lib;
    void* ptr;
    int elementCount;
    int elementByteStride;
};

// NvResourceTracker
struct NvResourceTracker {
    unsigned char mSolver[24];  // DWARF: vector_NvFlexSolver__ std__allocator_NvFlexSolver__ _ (opaque)
    unsigned char mTriangleMesh[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
    unsigned char mSDF[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
    unsigned char mConvexMesh[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
};

// NvFlexLibrary
struct NvFlexLibrary {
    unsigned char mErrorFunc[8];  // DWARF: NvFlexErrorCallback (opaque)
    int* mCellStarts;
    int* mCellEnds;
    LinearBVHBuilderGPU* mBVHBuilder;
    unsigned char mStaticTriLowers[24];  // DWARF: DeviceBuffer_XVector4_float_ _ (opaque)
    unsigned char mStaticTriUppers[24];  // DWARF: DeviceBuffer_XVector4_float_ _ (opaque)
    FlexTriangleMeshData mTriangleMeshData;
    FlexConvexMeshData mConvexMeshData;
    FlexSDFData mSDFData;
    char mDeviceName[256];
    void* mMemoryFence;
    void* mCudaContext;
    bool mCudaOwner;
    CudaTimers mTimers;
    void* mSortTemp;
    int mSortMaxTempSize;
    NvResourceTracker mResourceTracker;
};

// NvFlexSolver
struct NvFlexSolver {
    Vec4* mPositions;
    Vec4* mNewPositions;
    Vec3* mVelocities;
    Vec4* mRestPositions;
    Vec4* mDeltas;
    int* mPhases;
    int* mActiveIndices;
    float* mDensities;
    float* mPotentials;
    Vec4* mCurl;
    Vec4* mNormals;
    Vec4* mAnisotropy1;
    Vec4* mAnisotropy2;
    Vec4* mAnisotropy3;
    Vec4* mSortedPositions;
    Vec4* mSortedNewPositions;
    Vec4* mSortedNewVelocities;
    int* mSortedPhases;
    float* mSortedDensities;
    int mMaxContactsPerParticle;
    Vec4* mStaticContactPlanes;
    Vec4* mStaticContactVelocities;
    int* mStaticContactCounts;
    unsigned char mDesc[20];  // DWARF: NvFlexSolverDesc (opaque)
    int mMaxNeighborsPerParticle;
    int* mParticleNeighbors;
    int* mParticleNeighborCounts;
    Vec4* mPlanes;
    int* mSpringIndices;
    float* mSpringLengths;
    float* mSpringStiffness;
    int mNumSprings;
    int mMaxSprings;
    int* mRigidOffsets;
    int* mRigidIndices;
    Vec3* mRigidLocalPositions;
    Vec4* mRigidLocalNormals;
    float* mRigidCoefficients;
    float* mRigidPlasticThresholds;
    float* mRigidPlasticCreeps;
    Quat* mRigidRotations;
    Vec3* mRigidTranslations;
    int mNumRigids;
    int mMaxRigids;
    int mNumRigidIndices;
    int mMaxRigidIndices;
    int mMaxPlasticEntries;
    int* mDynamicTriangles;
    Vec3* mDynamicTriangleNormals;
    int mDynamicNumTris;
    int mDynamicMaxTris;
    int* mDynamicTriangleEdges;
    int mDynamicNumTriEdges;
    float* mPressures;
    FlexInflatable* mInflatables;
    int mNumInflatables;
    int mMaxInflatables;
    Vec4* mDiffusePositions;
    Vec4* mDiffuseVelocities;
    Vec4* mDiffusePositionsNew;
    Vec4* mDiffuseVelocitiesNew;
    int* mNumDiffuseParticles;
    int* mNumDiffuseParticlesNew;
    int mMaxDiffuseParticles;
    int* mNumActiveDiffuseParticlesHost;
    float mDiffuseRate;
    float mDiffuseTimer;
    int* mCellIds;
    int* mIndices;
    int* mReverseLookup;
    int numParticles;
    int mMaxParticles;
    Vec3* mParticleLower;
    Vec3* mParticleUpper;
    unsigned char mParams[312];  // DWARF: NvFlexParams (opaque)
    void* mStartDetailEvent;
    void* mStopDetailEvent;
    void* mStartTotalEvent;
    void* mStopTotalEvent;
    void* mLatencyTimerBegin;
    void* mLatencyTimerEnd;
    bool mLatencyTimerBeginRecorded;
    bool mLatencyTimerEndRecorded;
    FlexCollisionShapeData mShapes;
    bool mDebug;
    float* mRandom;
    float mFluidRestDensity;
    float mFluidDensityConstraintScale;
    float mFluidSurfaceConstraintScale;
    FlexTextures mTextures;
    int mStepCounter;
    unsigned char mCallbacks[80];  // DWARF: NvFlexSolverCallback_5_ (opaque)
    NvFlexLibrary* mLib;
};

// PackedNodeHalf
struct PackedNodeHalf {
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;
};

// PartitionPredicateMedian
struct PartitionPredicateMedian {
    Bounds* bounds;
    int axis;
};

// PartitionPredictateMidPoint
struct PartitionPredictateMidPoint {
    Bounds* bounds;
    int axis;
    float mid;
};

// Plane
struct Plane {
};

// Point3
struct Point3 {
    float x;
    float y;
    float z;
};

// Vec3Iter
struct Vec3Iter {
    float* x;
};

// XMatrix44<float>
struct XMatrix44_float {
    float columns[4][4];
    // static member kIdentity (XMatrix44<float>)
};

// XQuat<float>
struct XQuat_float {
    float x;
    float y;
    float z;
    float w;
};

// XVector2<float>
struct XVector2_float {
    float x;
    float y;
};

// XVector4<float>
struct XVector4_float {
    float x;
    float y;
    float z;
    float w;
};

// dim3
struct dim3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

// ltstr
struct ltstr {
};

// uint3
struct uint3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

enum Preset {
    kRed = 0,
    kGreen = 1,
    kBlue = 2,
    kWhite = 3,
    kBlack = 4,
};

enum ResourceType {
    eSolver = 0,
    eTriangleMesh = 1,
    eSDF = 2,
    eConvexMesh = 3,
};

typedef void* pfnCuDriverGetVersion_t;

