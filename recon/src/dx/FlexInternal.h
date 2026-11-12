// flexdecomp: reconstructed D3D backend — internal contract.
//
// Every type in this header is an *internal* type of the shipped D3D backend:
// it is never seen by an application, only by the backend's own translation
// units.  The layouts therefore only have to agree with the code in this
// reconstruction, not with an external ABI.  They are nevertheless written out
// exactly as the PDB reports them, for two reasons:
//
//   * `KernelParams` *is* an external contract: it is the constant-buffer
//     mirror the shipped DXBC shaders read at cb0, so it must be byte-exact.
//     Field order and offsets come from the RDEF chunk of the shipped
//     containers and from PDB field list 0x3088; the static_asserts below pin
//     the offsets the shaders address.
//   * `NvFlexLibrary` has hard-coded member offsets in utils.h (see the
//     `NvFlexLibrary*` accessors there); using the PDB layout keeps those
//     accessors valid and makes the Ghidra pseudocode transcribe 1:1.
//
// Provenance, unless stated otherwise:
//   `llvm-pdbutil dump -types FleX/bin/win64/NvFlexReleaseD3D_x64.dll.pdb`
//   (cached in work/pdb/d3d/types.txt; `flexkit d3dcontext --cls <Name>`
//   prints the same field list with its members resolved).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "Bvh.h"
#include "RadixSort.h"
#include "utils.h"

namespace NvFlex {
class Device;   // defined in Device.h (Step 2 contract)
}

// ---------------------------------------------------------------------------
// HLSL value types
// ---------------------------------------------------------------------------
// The shipped backend declares its GPU-visible structs with these names
// (PDB field lists 0x3086/0x3087/0x3085 use the HLSL spelling `float3`/
// `float4`).  HLSL packs them at 4-byte alignment, which is what
// `KernelParams` needs (kGravity is 12 bytes at offset 256, followed
// immediately by kDynamicFriction at 268), so these are plain aggregates with
// natural alignment 4 — *not* float4-aligned vectors.
struct float3
{
    float x, y, z;
};

struct float4
{
    float x, y, z, w;
};

// ---------------------------------------------------------------------------
// GpuVendorId — PDB enum 0x2CF3
// ---------------------------------------------------------------------------
// The values are PCI vendor ids; VENDOR_ID_OTHERS is the shipped sentinel for
// "not recognised" and is deliberately not -1 (PDB field list 0x2CF2).
enum GpuVendorId
{
    VENDOR_ID_NVIDIA = 0x10de,
    VENDOR_ID_AMD    = 0x1002,
    VENDOR_ID_OTHERS = 0x00ee
};

// ---------------------------------------------------------------------------
// KernelParams — the solver constant buffer (512 bytes)
// ---------------------------------------------------------------------------
// PDB field list 0x3088, struct KernelParams.  Every stage uploads this whole
// block to b0 before dispatching; the shaders index it by byte offset.
struct KernelParams
{
    float4 kPlanes[8];              // @0     planes of the current static contacts
    int    kNumPlanes;              // @128
    float  kRadius;                 // @132
    float  kRadiusSq;               // @136
    float  kInvRadius;              // @140
    float  kPoly6;                  // @144
    float  kSpiky1;                 // @148
    float  kSpiky2;                 // @152
    float  kAkinci1;                // @156
    float  kAkinci2;                // @160
    float  kCohesion1;              // @164
    float  kCohesion2;              // @168
    float  _pad0;                   // @172   (gap in the shipped layout)
    float  kRestDensity;            // @176
    float  kInvRestDensity;         // @180
    float  kFluidRestDistance;      // @184
    float  kSolidRestDistance;      // @188
    float  kLambdaScale;            // @192
    float  kCohesion;               // @196
    float  kSurfaceTension;         // @200
    float  kSolidPressure;          // @204
    float  kVorticityConfinement;   // @208
    float  kAnisotropy;             // @212
    float  kAnisotropyMin;          // @216
    float  kAnisotropyMax;          // @220
    float  kSmoothing;              // @224
    float  kViscosity;              // @228
    float  kFreeSurfaceDrag;        // @232
    float  kBuoyancy;               // @236
    float  kDiffuseThreshold;       // @240
    float  kDiffuseBuoyancy;        // @244
    float  kDiffuseDrag;            // @248
    int    kDiffuseBallistic;       // @252
    float3 kGravity;                // @256
    float  kDynamicFriction;        // @268
    float  kStaticFriction;         // @272
    float  kParticleFriction;       // @276
    float  kAdhesion;               // @280
    float  kDissipation;            // @284
    float  kDamping;                // @288
    float  kInertiaBias;            // @292
    float  kCollisionDistance;      // @296
    float  kCollisionThreshold;     // @300
    float  kCollisionMargin;        // @304
    float  kCollisionMarginSq;      // @308
    float  _pad1;                   // @312
    float  _pad2;                   // @316
    float  kDrag;                   // @320
    float  kLift;                   // @324
    float  kPlasticThreshold_deprecated;  // @328
    float  kPlasticCreep_deprecated;      // @332
    float  kSleepThreshold;         // @336
    float  kSleepThresholdSq;       // @340
    float  kShockPropagation;       // @344
    float  kRestitution;            // @348
    float  kSOR;                    // @352
    float3 _pad3;                   // @356   (gap in the shipped layout)
    int    kNumParticles;           // @368
    int    kNumParticlesAligned;    // @372
    float  kDt;                     // @376
    float  kInvDt;                  // @380
    float  kSubStepLength;          // @384
    float  kMaxSpeed;               // @388
    float  kInvCellEdge;            // @392
    float  kStaticTriGridInvCellEdge;  // @396
    float  kCollideParticlesRadiusSq;  // @400
    int    kMaxParticles;           // @404
    int    kRelaxationMode;         // @408
    int    kShapesCount;            // @412
    int    kNumShapes;              // @416
    int    kNumRigids;              // @420
    int    kNumSprings;             // @424
    int    kNumTriangles;           // @428
    int    kMaxDiffuseParticles;    // @432
    int    kDiffuseRadixSortSize;   // @436
    float  kDiffuseDt;              // @440
    float  kDiffuseMaxVelocity;     // @444
    float3 kDiffuseSortAxis;        // @448
    float  kRelaxationFactor;       // @460
    float3 kStaticTriGridLower;     // @464
    float  kDiffuseLifetime;        // @476
    float3 kWind;                   // @480
    int    kMaxNeighborsPerParticle;  // @492
    int    kNumGeometryEntries;     // @496
    float  kMaxAcceleration;        // @500
    float  kMaxVelocityDelta;       // @504
    int    kMaxContactsPerParticle; // @508
};

static_assert(sizeof(float3) == 12, "HLSL float3 is 12 bytes");
static_assert(sizeof(float4) == 16, "HLSL float4 is 16 bytes");
static_assert(sizeof(KernelParams) == 512, "KernelParams must match the shader cb0");
static_assert(offsetof(KernelParams, kNumPlanes) == 128, "kNumPlanes");
static_assert(offsetof(KernelParams, kRadius) == 132, "kRadius");
static_assert(offsetof(KernelParams, kGravity) == 256, "kGravity");
static_assert(offsetof(KernelParams, kNumParticles) == 368, "kNumParticles");
static_assert(offsetof(KernelParams, kDt) == 376, "kDt");
static_assert(offsetof(KernelParams, kInvCellEdge) == 392, "kInvCellEdge");
static_assert(offsetof(KernelParams, kMaxParticles) == 404, "kMaxParticles");
static_assert(offsetof(KernelParams, kDiffuseSortAxis) == 448, "kDiffuseSortAxis");
static_assert(offsetof(KernelParams, kStaticTriGridLower) == 464, "kStaticTriGridLower");
static_assert(offsetof(KernelParams, kWind) == 480, "kWind");
static_assert(offsetof(KernelParams, kMaxContactsPerParticle) == 508,
              "kMaxContactsPerParticle");

// ---------------------------------------------------------------------------
// Device description (NvFlexInitDesc / FlexDeviceDesc)
// ---------------------------------------------------------------------------
// PDB `FlexDeviceDesc`, sizeof 56, polymorphic: the vtable pointer sits at 0 and
// the class is only ever destroyed through it.
struct FlexDeviceDesc
{
    virtual ~FlexDeviceDesc() {}

    unsigned          mDeviceNumber;         // @8   DXGI adapter index
    void*             mDevice;               // @16  caller-supplied ID3D11Device
    void*             mComputeCommandQueue;  // @24  D3D12 only
    void*             mRenderCommandQueue;   // @32  D3D12 only
    bool              useComputeQueue;       // @40
    bool              enablePresent;         // @41
    NvFlexErrorCallback mErrorFunc;          // @48
};

static_assert(sizeof(FlexDeviceDesc) == 56, "FlexDeviceDesc");
static_assert(offsetof(FlexDeviceDesc, mDeviceNumber) == 8, "mDeviceNumber");
static_assert(offsetof(FlexDeviceDesc, mErrorFunc) == 48, "mErrorFunc");

// ---------------------------------------------------------------------------
// Triangle-mesh, convex-mesh, SDF and shape storage (flex.obj)
// ---------------------------------------------------------------------------
// These bundles are what NvFlexCreateTriangleMesh / …DistanceField /
// …ConvexMesh hand back through the opaque handles, and what
// FlattenTriangleMeshes() and the shape rebuild consume each step.

// PDB sizeof 224.
struct FlexTriangleMeshHost
{
    DxComputeBuffer      mVertices;      // @0
    DxComputeBuffer      mIndices;       // @32
    BVH                  mBVH;           // @64
    int                  mNumTris;       // @176
    int                  mNumVertices;   // @180
    int                  mNumIndices;    // @184
    int                  mMaxTris;       // @188
    int                  mMaxVertices;   // @192
    XVector3<float>      mLower;         // @196
    XVector3<float>      mUpper;         // @208
};

// PDB `BVHArray`, sizeof 96.
struct BVHArray
{
    DxBufferExt mNodeLowersArray;   // @0
    DxBufferExt mNodeUppersArray;   // @32
    DxBufferExt mRootNodeArray;     // @64
};

// PDB sizeof 232.
struct FlexTriangleMeshData
{
    std::vector<FlexTriangleMeshHost*> mTriMeshTableHost;   // @0
    DxComputeBuffer                    mTriMeshTableDevice; // @24
    DxBufferExt                        mVerticesArray;      // @56
    DxBufferExt                        mIndicesArray;       // @88
    BVHArray                           mBvhData;            // @120
    int                                mMaxNumVertices;     // @216
    int                                mMaxNumTriangles;    // @220
    int                                mMaxNumMeshes;       // @224
};

// PDB sizeof 48: one entry of the flattened per-mesh table.
struct FlexTriangleMeshDevice
{
    float4 mLower;        // @0    xyz used, w unused
    float4 mUpper;        // @16
    int    mIndexStart;   // @32
    int    mVertexStart;  // @36
    int    mNodeStart;    // @40
    int    mPad;          // @44
};

// PDB `FlexConvexMeshDevice`, sizeof 48: one entry of the per-convex record
// table the shape shaders read.
struct FlexConvexMeshDevice
{
    float4 mLower;       // @0
    float4 mUpper;       // @16
    int    mPlaneOffset; // @32
    int    mPlaneCount;  // @36
    int    mPad[2];      // @40
};

static_assert(sizeof(FlexConvexMeshDevice) == 48, "FlexConvexMeshDevice");

// PDB sizeof 88.
struct FlexConvexMeshData
{
    DxComputeBuffer mPlanes;               // @0    float4 per plane
    DxComputeBuffer mConvexes;             // @32   FlexConvexMeshDevice entries
    int             mNumConvexes;          // @64
    int             mMaxConvexes;          // @68
    int             mMaxPlanesPerConvex;   // @72
    int*            mFreeList;             // @80
};

// PDB sizeof 72.
struct FlexSDFData
{
    std::vector<NvFlex::Texture3D*> mTextures;   // @0
    DxComputeBuffer                 mSDFs;       // @24  FlexSDFDevice entries
    int                             mNumSDFs;    // @56
    int                             mMaxSDFs;    // @60
    int*                            mFreeList;   // @64
};

// PDB sizeof 408.
struct FlexShapeData
{
    DxBufferExt  mAabbMin;            // @0
    DxBufferExt  mAabbMax;            // @32
    DxBufferExt  mGeometry;           // @64
    DxBufferExt  mPositions;          // @96
    DxBufferExt  mRotations;          // @128
    DxBufferExt  mPrevPositions;      // @160
    DxBufferExt  mPrevRotations;      // @192
    DxBufferExt  mFlags;              // @224
    BVH          mBVH;                // @256
    int          mNumShapes;          // @368
    int          mMaxShapes;          // @372
    XVector3<float> mLower;           // @376
    XVector3<float> mUpper;           // @388
    bool         mRebuildShapeData;   // @400
};

// PDB `NvResourceTracker`, sizeof 96: the library's record of every solver and
// mesh object it handed out, so NvFlexShutdown can release them.
struct NvResourceTracker
{
    std::vector<NvFlexSolver*> mSolver;        // @0
    std::vector<unsigned>      mTriangleMesh;  // @24
    std::vector<unsigned>      mSDF;           // @48
    std::vector<unsigned>      mConvexMesh;    // @72
};

// ---------------------------------------------------------------------------
// NvFlexLibrary — PDB field list 0x2F4B, sizeof 1440 (0x5a0)
// ---------------------------------------------------------------------------
// The offset accessors in utils.h depend on this layout: mErrorFunc@0,
// mContext@16, the shader pointers from 280, mShaderClearConstantBufferInt@344
// and …Float4@368 (DxConstantBuffer, whose ConstantBuffer* sits at +16).
struct NvFlexLibrary
{
    // flex.obj @0x37040 (263 bytes): default-initialise the owned bundles.
    NvFlexLibrary();
    // flex.obj @0x37bb0 (114 bytes): release the std::vector storages.
    ~NvFlexLibrary();

    NvFlexErrorCallback  mErrorFunc;                       // @0
    NvFlex::Device*      mDevice;                          // @8
    NvFlex::ContextEx*   mContext;                          // @16
    char                 mDeviceName[256];                 // @24
    NvFlex::ComputeShader* mShaderPredict;                 // @280
    NvFlex::ComputeShader* mShaderCalculateBounds;         // @288
    NvFlex::ComputeShader* mShaderCalculateBoundsGroup;    // @296
    NvFlex::ComputeShader* mShaderCalculateBoundsFinalize; // @304
    NvFlex::ComputeShader* mShaderCalculateParticleHash;   // @312
    NvFlex::ComputeShader* mShaderCreateGrid;              // @320
    NvFlex::ComputeShader* mShaderClearInt;                // @328
    NvFlex::ComputeShader* mShaderClearFloat4;             // @336
    DxConstantBuffer       mShaderClearConstantBufferInt;  // @344
    DxConstantBuffer       mShaderClearConstantBufferFloat4; // @368
    NvFlex::ComputeShader* mShaderClearCellBuckets;        // @392
    NvFlex::ComputeShader* mShaderReorderParticles;        // @400
    NvFlex::ComputeShader* mShaderCollideParticles;        // @408
    NvFlex::ComputeShader* mShaderCollideShapes;           // @416
    NvFlex::ComputeShader* mShaderContinuousShockPropagation; // @424
    NvFlex::ComputeShader* mShaderCalculateDensity;        // @432
    NvFlex::ComputeShader* mShaderCalculateDensitySurfaceTension; // @440
    NvFlex::ComputeShader* mShaderSolveDensities;          // @448
    NvFlex::ComputeShader* mShaderSolveDensitiesNonFluid;  // @456
    NvFlex::ComputeShader* mShaderSolveDensitiesSurfaceTension; // @464
    NvFlex::ComputeShader* mShaderCalculateInflatableVolume; // @472
    NvFlex::ComputeShader* mShaderSolveInflatableVolume;   // @480
    NvFlex::ComputeShader* mShaderSolveSprings;            // @488
    NvFlex::ComputeShader* mShaderSolveShapes;             // @496
    NvFlex::ComputeShader* mShaderSolveShapes32;           // @504
    NvFlex::ComputeShader* mShaderSolveShapes128;          // @512
    NvFlex::ComputeShader* mShaderSolveShapesPlasticDeformation; // @520
    NvFlex::ComputeShader* mShaderSolveShapesPlasticDeformation32; // @528
    NvFlex::ComputeShader* mShaderSolveShapesPlasticDeformation128; // @536
    NvFlex::ComputeShader* mShaderApplyDeltas;             // @544
    NvFlex::ComputeShader* mShaderSolveContactsSequential; // @552
    NvFlex::ComputeShader* mShaderSolveContactsAveraged;   // @560
    NvFlex::ComputeShader* mShaderSolveContactsAccumulate; // @568
    NvFlex::ComputeShader* mShaderCollideTriangles;        // @576
    NvFlex::ComputeShader* mShaderUpdateVelocities;        // @584
    NvFlex::ComputeShader* mShaderCalculateVorticity;      // @592
    NvFlex::ComputeShader* mShaderSolveVelocities;         // @600
    NvFlex::ComputeShader* mShaderUpdateTriangles;         // @608
    NvFlex::ComputeShader* mShaderUpdateTrianglesInit;     // @616
    NvFlex::ComputeShader* mShaderUpdateVertexNormalsInit; // @624
    NvFlex::ComputeShader* mShaderUpdateVertexNormals;     // @632
    NvFlex::ComputeShader* mShaderNormalizeVertexNormals;  // @640
    NvFlex::ComputeShader* mShaderFinalize;                // @648
    NvFlex::ComputeShader* mShaderCreateDiffuseParticles;  // @656
    NvFlex::ComputeShader* mShaderUpdateDiffuseParticles;  // @664
    NvFlex::ComputeShader* mShaderClampDiffuseParticleCount; // @672
    NvFlex::ComputeShader* mShaderCompactDiffuseParticles; // @680
    NvFlex::ComputeShader* mShaderSmoothPositions;         // @688
    NvFlex::ComputeShader* mShaderCalculateAnisotropy;     // @696
    NvFlex::ComputeShader* mShaderComputeTriangleBounds;   // @704
    NvFlex::ComputeShader* mShaderTransformShapeBounds;    // @712
    NvFlex::ComputeShader* mShaderSpringsGenerateIndices;  // @720
    NvFlex::ComputeShader* mShaderSpringsParticleRange;    // @728
    NvFlex::ComputeShader* mShaderSpringsReorder;          // @736
    DxBufferExt            mCellBucketStarts;              // @744
    DxBufferExt            mCellBucketEnds;                // @776
    LinearBVHBuilderGPU*   mBVHBuilder;                     // @808
    DxConstantBuffer       mConstTriMeshInfo;              // @816
    DxBufferExt            mStaticTriLowers;               // @840
    DxBufferExt            mStaticTriUppers;               // @872
    int                    mMaxStaticTris;                 // @904
    bool                   mIsSHFLSupported;               // @908
    bool                   mIsFP32ATOMICSupported;         // @909
    void*                  mAGSContext;                     // @912
    bool                   mIsSwizzleSupported;            // @920
    bool                   mEnableAmdDriverBugWorkaround;  // @921
    GpuVendorId            mGpuVendorId;                   // @924
    int                    mSMCount;                       // @928
    NvFlex::Fence*         mSyncQuery;                     // @936
    FlexTriangleMeshData   mTriangleMeshData;              // @944
    FlexConvexMeshData     mConvexMeshData;                // @1176
    FlexSDFData            mSDFData;                       // @1264
    bool                   mRebuildTriangleData;           // @1336
    NvResourceTracker      mResourceTracker;               // @1344
};

static_assert(sizeof(NvFlexLibrary) == 1440, "NvFlexLibrary is 0x5a0 bytes");
static_assert(offsetof(NvFlexLibrary, mContext) == 16, "mContext");
static_assert(offsetof(NvFlexLibrary, mShaderPredict) == 280, "mShaderPredict");
static_assert(offsetof(NvFlexLibrary, mShaderClearInt) == 328, "mShaderClearInt");
static_assert(offsetof(NvFlexLibrary, mShaderClearConstantBufferInt) == 344,
              "mShaderClearConstantBufferInt");
static_assert(offsetof(NvFlexLibrary, mShaderClearConstantBufferFloat4) == 368,
              "mShaderClearConstantBufferFloat4");
static_assert(offsetof(NvFlexLibrary, mCellBucketStarts) == 744, "mCellBucketStarts");
static_assert(offsetof(NvFlexLibrary, mBVHBuilder) == 808, "mBVHBuilder");
static_assert(offsetof(NvFlexLibrary, mMaxStaticTris) == 904, "mMaxStaticTris");
static_assert(offsetof(NvFlexLibrary, mGpuVendorId) == 924, "mGpuVendorId");
static_assert(offsetof(NvFlexLibrary, mSMCount) == 928, "mSMCount");
static_assert(offsetof(NvFlexLibrary, mTriangleMeshData) == 944, "mTriangleMeshData");
static_assert(offsetof(NvFlexLibrary, mConvexMeshData) == 1176, "mConvexMeshData");
static_assert(offsetof(NvFlexLibrary, mSDFData) == 1264, "mSDFData");
static_assert(offsetof(NvFlexLibrary, mResourceTracker) == 1344, "mResourceTracker");

// DxConstantBuffer's ConstantBuffer* sits at +16 (utils.h relies on it).
static_assert(offsetof(DxConstantBuffer, mBuffer) == 16, "DxConstantBuffer::mBuffer");

// ---------------------------------------------------------------------------
// NvFlexSolver — PDB field list 0x41D0, sizeof 3872
// ---------------------------------------------------------------------------
struct NvFlexSolver
{
    // flex.obj `NvFlexSolver::NvFlexSolver` — the shipped body only builds the
    // members that have constructors.
    NvFlexSolver();
    // flex.obj @0x3c1f0 (55 bytes): releases the SDF-shape vector.
    ~NvFlexSolver();

    KernelParams              mKernelParamsHost;            // @0
    DxConstantBuffer          mKernelParams;                // @512
    DxBufferExt               mPositions;                   // @536
    DxBufferExt               mNewPositions;                // @568
    DxBufferExt               mVelocities;                  // @600
    DxBufferExt               mRestPositions;               // @632
    DxBufferExt               mDeltas;                      // @664
    DxBufferExt               mPhases;                      // @696
    DxBufferExt               mActiveIndices;               // @728
    DxBufferExt               mDensities;                   // @760
    DxBufferExt               mPotentials;                  // @792
    DxBufferExt               mCurl;                        // @824
    DxBufferExt               mNormals;                     // @856
    DxBufferExt               mSortedNormals;               // @888
    DxBufferExt               mNormalsTemp;                 // @920
    DxBufferExt               mAnisotropy1;                 // @952
    DxBufferExt               mAnisotropy2;                 // @984
    DxBufferExt               mAnisotropy3;                 // @1016
    DxBufferExt               mSortedPositions;             // @1048
    DxBufferExt               mSortedVelocities;            // @1080
    DxBufferExt               mSortedNewPositions;          // @1112
    DxBufferExt               mSortedNewVelocities;         // @1144
    DxBufferExt               mSortedPhases;                // @1176
    DxBufferExt               mSortedDensities;             // @1208
    DxBufferExt               mStaticContactPlanes;         // @1240
    DxBufferExt               mStaticContactVelocities;     // @1272
    DxBufferExt               mStaticContactCounts;         // @1304
    int                       mMaxNeighborsPerParticle;     // @1336
    DxBufferExt               mParticleNeighbors;           // @1344
    DxBufferExt               mParticleNeighborCounts;      // @1376
    int                       mMaxContactsPerParticle;      // @1408
    DxBufferExt               mPlanes;                      // @1416
    DxBufferExt               mSpringIndices;               // @1448
    DxBufferExt               mSpringLengths;               // @1480
    DxBufferExt               mSpringStiffness;             // @1512
    int                       mNumSprings;                  // @1544
    int                       mMaxSprings;                  // @1548
    DxBufferExt               mHalfSpringIndices;           // @1552
    DxBufferExt               mHalfSpringOpposites;          // @1584
    DxBufferExt               mHalfSpringLengths;           // @1616
    DxBufferExt               mHalfSpringStiffness;         // @1648
    DxBufferExt               mHalfSpringParticleBegin;     // @1680
    DxBufferExt               mHalfSpringParticleEnd;       // @1712
    DxConstantBuffer          mHalfSpringConstantBuffer;    // @1744
    int                       mMaxHalfSprings;              // @1768
    DxBufferExt               mRigidOffsets;                // @1776
    DxBufferExt               mRigidIndices;                // @1808
    DxBufferExt               mRigidLocalPositions;         // @1840
    DxBufferExt               mRigidLocalNormals;           // @1872
    DxBufferExt               mRigidCoefficients;           // @1904
    DxBufferExt               mRigidPlasticThresholds;      // @1936
    DxBufferExt               mRigidPlasticCreeps;          // @1968
    DxBufferExt               mRigidRotations;              // @2000
    DxBufferExt               mRigidTranslations;           // @2032
    int                       mNumRigids;                   // @2064
    int                       mMaxRigids;                   // @2068
    int                       mNumRigidIndices;             // @2072
    int                       mMaxRigidIndices;             // @2076
    int                       mMaxPlasticEntries;           // @2080
    DxBufferExt               mDynamicTriangles;            // @2088
    DxBufferExt               mDynamicTriangleNormals;      // @2120
    int                       mDynamicNumTris;              // @2152
    int                       mDynamicMaxTris;              // @2156
    DxBufferExt               mDynamicTriangleEdges;        // @2160
    DxBufferExt               mDynamicNumTriEdges;          // @2192
    DxBufferExt               mPressures;                   // @2224
    DxComputeBuffer           mInflatables;                 // @2256
    int                       mNumInflatables;              // @2288
    int                       mMaxInflatables;              // @2292
    DxBufferExt               mDiffusePositions;            // @2296
    DxBufferExt               mDiffuseVelocities;           // @2328
    DxBufferExt               mDiffusePositionsNew;         // @2360
    DxBufferExt               mDiffuseVelocitiesNew;        // @2392
    DxComputeBuffer           mNumDiffuseParticles;         // @2424
    DxComputeBuffer           mNumDiffuseParticlesNew;      // @2456
    int                       mMaxDiffuseParticles;         // @2488
    int                       mNumActiveDiffuseParticles;   // @2492
    DxBufferExt               mDiffuseSortSize;             // @2496
    DxBufferExt               mDiffuseLaunchArg;            // @2528
    float                     mDiffuseRate;                 // @2560
    float                     mDiffuseTimer;                // @2564
    DxBufferExt               mCellIds;                     // @2568
    DxBufferExt               mReverseLookup;               // @2600
    int                       mNumParticles;                // @2632
    int                       mNumParticlesAligned;         // @2636
    int                       mMaxParticles;                // @2640
    DxBufferExt               mParticleBounds;              // @2648
    DxBufferExt               mBoundsLower;                 // @2680
    DxBufferExt               mBoundsUpper;                 // @2712
    DxBufferExt               mGroupBoundsLower;            // @2744
    DxBufferExt               mGroupBoundsUpper;            // @2776
    NvFlexSolverDesc          mDesc;                        // @2808
    NvFlexParams              mParams;                      // @2828
    DxBufferExt               mSDFShapeDX;                  // @3136
    std::vector<FlexSDFData*> mSDFDeviceShapes;             // @3168
    FlexShapeData             mShapes;                      // @3192
    bool                      mDebug;                       // @3600
    DxBufferExt               mRandom;                      // @3608
    float                     mFluidRestDensity;            // @3640
    float                     mFluidDensityConstraintScale; // @3644
    float                     mFluidSurfaceConstraintScale; // @3648
    int                       mStepCounter;                 // @3652
    NvFlexSolverCallback      mCallbacks[5];                // @3656
    DxBufferExt               mSmoothPositionsOriginal;     // @3736
    int                       mRadixSortSize;               // @3768
    NvFlex::RadixSortImpl*    mRadixSort;                   // @3776
    DxConstantBuffer          mSubstepParams;               // @3784
    DxConstantBuffer          mNumShapes;                   // @3808
    DxTimers*                 mTimers;                      // @3832
    int                       mNumDetailTimers;             // @3840
    NvFlexDetailTimer*        mDetailTimers;                // @3848
    bool                      mLatencyTimerBeginRecorded;   // @3856
    bool                      mLatencyTimerEndRecorded;     // @3857
    bool                      mLatencyTimerEnable;          // @3858
    int                       mLatencyTimerIndex;           // @3860
    NvFlexLibrary*            mLib;                         // @3864
};

static_assert(sizeof(NvFlexSolver) == 3872, "NvFlexSolver");
static_assert(offsetof(NvFlexSolver, mKernelParams) == 512, "mKernelParams");
static_assert(offsetof(NvFlexSolver, mPositions) == 536, "mPositions");
static_assert(offsetof(NvFlexSolver, mNumParticles) == 2632, "mNumParticles");
static_assert(offsetof(NvFlexSolver, mDesc) == 2808, "mDesc");
static_assert(offsetof(NvFlexSolver, mParams) == 2828, "mParams");
static_assert(offsetof(NvFlexSolver, mShapes) == 3192, "mShapes");
static_assert(offsetof(NvFlexSolver, mRadixSort) == 3776, "mRadixSort");
static_assert(offsetof(NvFlexSolver, mTimers) == 3832, "mTimers");
static_assert(offsetof(NvFlexSolver, mLib) == 3864, "mLib");

// ---------------------------------------------------------------------------
// Logging helper used by every reconstructed entry point
// ---------------------------------------------------------------------------
// flex.obj logs through the library's error callback.  The shipped signature is
// FlexLogFunc(callback, severity, format, file, line, ...) with `file` carrying
// the "..\\..\\dx\\flex.cpp" spelling and `line` the __LINE__ of the call site.
