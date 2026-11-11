// recovered from DWARF by flexkit headers (canonical definitions).
// Layouts are preserved exactly: opaque members keep their offsets.
#pragma once
#if defined(__CUDACC__)
#include <cuda_runtime.h>  // dim3/uint3/cuda* types are CUDA-provided
#endif
#include <cstddef>
#include <cstdint>

#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_CUcontext)
#define FLEXDECOMP_DEF_CUcontext
typedef void* CUcontext;
#endif
#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_cudaArray_t)
#define FLEXDECOMP_DEF_cudaArray_t
typedef void* cudaArray_t;
#endif
#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_cudaEvent_t)
#define FLEXDECOMP_DEF_cudaEvent_t
typedef void* cudaEvent_t;
#endif
#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_cudaGraphicsResource_t)
#define FLEXDECOMP_DEF_cudaGraphicsResource_t
typedef void* cudaGraphicsResource_t;
#endif
#if !defined(__CUDACC__) && !defined(FLEXDECOMP_DEF_cudaTextureObject_t)
#define FLEXDECOMP_DEF_cudaTextureObject_t
typedef unsigned long long cudaTextureObject_t;
#endif

// XMatrix44<float> (recovered as a template)
template <typename T> struct alignas(16) XMatrix44 {
    T columns[4][4];
    static XMatrix44 kIdentity;
};

// XQuat<float> (recovered as a template)
template <typename T> struct XQuat {
    T x;
    T y;
    T z;
    T w;
};

// XVector2<float> (recovered as a template)
template <typename T> struct XVector2 {
    T x;
    T y;
};

// XVector3<float> (recovered as a template)
template <typename T> struct XVector3 {
    T x;
    T y;
    T z;
};

// XVector4<float> (recovered as a template)
template <typename T> struct XVector4 {
    T x;
    T y;
    T z;
    T w;
};

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
struct ltstr;

typedef CudaTimer CudaDetailTimer;
typedef XMatrix44<float> Mat44;
typedef XMatrix44<float> Matrix44;
typedef XQuat<float> Quat;
typedef XVector3<float> Vec3;
typedef XVector4<float> Vec4;
typedef XVector3<float> Vector3;

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
struct alignas(8) CudaRangeTimer {
    void* mBegin;
    void* mEnd;
    float mTime;
};

// CudaRangeTimerPool
struct alignas(8) CudaRangeTimerPool {
    unsigned char mElem[24];  // DWARF: vector_CudaRangeTimer_ std__allocator_CudaRangeTimer_ _ (opaque)
    int mUsedElemCount;
    // static member mInitCapacity (const int)
};

// CudaTimer
struct CudaTimer {
    bool mEnabled;
    unsigned char _pad4[3];
    int mTimerIndex;
    unsigned char mNameToTimerMap[8];  // DWARF: const NameToTimerMap_ (opaque)
    CudaRangeTimerPool* mTimerPool;
};

// CudaTimers
struct alignas(8) CudaTimers {
    bool mEnabled;
    unsigned char _pad8[7];
    unsigned char mNameToTimerMap[48];  // DWARF: NameToTimerMap (opaque)
    CudaRangeTimerPool* mTimerPool;
};

// DeviceBuffer<int>
struct alignas(8) DeviceBuffer_int {
    int* mPtr;
    int mCapacity;
    unsigned char _pad16[4];
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
    unsigned char _pad40[4];
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
struct alignas(8) FlexTextures {
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
    unsigned char _pad344[4];
    FlexTextures kTextures;
    FlexConvexMeshData kConvexMeshData;
    FlexTriangleMeshData kTriangleMeshData;
    FlexSDFData kSDFData;
};

// KeyIndexPair
struct KeyIndexPair {
    unsigned char key[4];  // DWARF: uint32_t (opaque)
    int index;
};

// LinearBVHBuilderCPU
struct LinearBVHBuilderCPU {
};

// LinearBVHBuilderGPU
struct alignas(8) LinearBVHBuilderGPU {
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
    XVector3<float> cols[3];
};

// MedianBVHBuilder
struct MedianBVHBuilder {
};

// NvFlexBuffer
struct NvFlexBuffer {
    bool mapped;
    unsigned char _pad4[3];
    int type;
    void* graphicsResource;
    NvFlexLibrary* lib;
    void* ptr;
    int elementCount;
    int elementByteStride;
};

// NvResourceTracker
struct alignas(8) NvResourceTracker {
    unsigned char mSolver[24];  // DWARF: vector_NvFlexSolver__ std__allocator_NvFlexSolver__ _ (opaque)
    unsigned char mTriangleMesh[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
    unsigned char mSDF[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
    unsigned char mConvexMesh[24];  // DWARF: vector_unsigned int_ std__allocator_unsigned int_ _ (opaque)
};

// NvFlexLibrary
struct alignas(8) NvFlexLibrary {
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
    unsigned char _pad472[7];
    CudaTimers mTimers;
    void* mSortTemp;
    int mSortMaxTempSize;
    unsigned char _pad552[4];
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
    unsigned char _pad160[4];
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
    unsigned char _pad360[4];
    int* mDynamicTriangles;
    Vec3* mDynamicTriangleNormals;
    int mDynamicNumTris;
    int mDynamicMaxTris;
    int* mDynamicTriangleEdges;
    int mDynamicNumTriEdges;
    unsigned char _pad400[4];
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
    unsigned char _pad480[4];
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
    unsigned char _pad912[6];
    FlexCollisionShapeData mShapes;
    bool mDebug;
    unsigned char _pad1048[7];
    float* mRandom;
    float mFluidRestDensity;
    float mFluidDensityConstraintScale;
    float mFluidSurfaceConstraintScale;
    unsigned char _pad1072[4];
    FlexTextures mTextures;
    int mStepCounter;
    unsigned char _pad1192[4];
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
struct alignas(8) PartitionPredicateMedian {
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
struct alignas(16) Plane {
};

// Point3
struct Point3 {
    float x;
    float y;
    float z;
};

// Vec3Iter
struct alignas(8) Vec3Iter {
    float* x;
};

// ltstr
struct ltstr {
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

