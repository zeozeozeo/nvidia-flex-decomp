// flexdecomp: reconstructed D3D backend — flex.obj object model.
//
// The mesh / distance-field / convex-mesh stores and the library's resource
// tracker.  Layouts come from FlexInternal.h (PDB field lists); behaviour from
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c.  They are free functions in
// `NvFlex::flexobj` because the corresponding structs are plain data in
// FlexInternal.h and the shipped class methods have no external ABI.
//
// Status of this unit: the three stores (create / allocate / free / release /
// resize), the tracker and their store updates (`FlexSDFData::Update`,
// `FlexConvexMeshData::Update`) are complete, as are `FlattenTriangleMeshes`,
// `TightPack3D` and the public mesh entry points
// (NvFlexCreateTriangleMesh … NvFlexGetConvexMeshBounds).
#pragma once

#include "FlexInternal.h"

// @0x37c40 (471 bytes): rebuild the library's flattened triangle arrays from
// the per-mesh tables.  Global scope like the shipped free function.
void FlattenTriangleMeshes(NvFlexLibrary* lib);

namespace NvFlex
{
namespace flexobj
{

// ---------------------------------------------------------------------------
// Triangle-mesh store
// ---------------------------------------------------------------------------
// @0x2fdb0 (312 bytes): reserve a per-mesh host record, returning its handle.
int TriangleMeshAllocate(FlexTriangleMeshData* data, NvFlex::ContextEx* context);
// @0x2fef0 (84 bytes): release a per-mesh host record.
void TriangleMeshFree(FlexTriangleMeshData* data, int index);
// @0x2f630 (403 bytes): drop the flattened GPU arrays.
void TriangleMeshRelease(FlexTriangleMeshData* data);
// @0x2f7d0 (1490 bytes): grow the flattened GPU arrays to hold the given totals.
void TriangleMeshResize(FlexTriangleMeshData* data, NvFlex::ContextEx* context,
                        int numVertices, int numTris, int numMeshes);

// ---------------------------------------------------------------------------
// SDF store — FlexSDFData, sizeof 72
// ---------------------------------------------------------------------------
void SDFCreate(FlexSDFData* data, NvFlex::ContextEx* context, bool shflSupported);
void SDFRelease(FlexSDFData* data);
int  SDFAllocate(FlexSDFData* data);
void SDFFree(FlexSDFData* data, int index);

// ---------------------------------------------------------------------------
// Convex-mesh store — FlexConvexMeshData, sizeof 88
// ---------------------------------------------------------------------------
void ConvexCreate(FlexConvexMeshData* data, NvFlex::ContextEx* context,
                  bool shflSupported, int smCount);
void ConvexRelease(FlexConvexMeshData* data);
int  ConvexAllocate(FlexConvexMeshData* data);
void ConvexFree(FlexConvexMeshData* data, int index);

// ---------------------------------------------------------------------------
// Resource tracker — NvResourceTracker, sizeof 96
// ---------------------------------------------------------------------------
// The tracker keeps every solver and every mesh/DF/convex handle the library
// handed out, so NvFlexShutdown can destroy whatever the application leaked.
void TrackerAdd(NvResourceTracker* tracker, NvFlexSolver* solver);
int  TrackerGet(NvResourceTracker* tracker, const std::vector<unsigned>& pool, unsigned handle);
void TrackerRemove(NvResourceTracker* tracker, std::vector<unsigned>& pool, unsigned handle);
// @0x30a50 (217 bytes)
void TrackerCleanup(NvResourceTracker* tracker, NvFlexLibrary* lib);

// ---------------------------------------------------------------------------
// Buffer copy layer
// ---------------------------------------------------------------------------
// The library-level pair validates the element stride and the copy range and
// reports the shipped error strings; the context-level pair picks the transport
// from the user buffer's type.  Every NvFlexSet*()/NvFlexGet*() entry point is
// one call into these.
void ReadData(NvFlex::ContextEx* context, NvFlexBuffer* destination, unsigned dstOffset,
              NvFlex::Buffer* source, unsigned srcOffset, unsigned size);
void WriteData(NvFlex::ContextEx* context, NvFlex::Buffer* destination, unsigned dstOffset,
               NvFlexBuffer* source, unsigned srcOffset, unsigned size);
void ReadData(NvFlexLibrary* lib, NvFlexBuffer* destination, NvFlex::Buffer* source,
              const NvFlexCopyDesc* desc);
void WriteData(NvFlexLibrary* lib, NvFlex::Buffer* destination, NvFlexBuffer* source,
               const NvFlexCopyDesc* desc);

// ---------------------------------------------------------------------------
// Host-side mesh record — FlexTriangleMeshHost, sizeof 224
// ---------------------------------------------------------------------------
// @0x2f4f0 (82 bytes)
void TriangleMeshHostInit(FlexTriangleMeshHost* host);

// ---------------------------------------------------------------------------
// Mesh-API slice (appended): store updates and the phase-helper dependency
// ---------------------------------------------------------------------------
// These are the helpers the public mesh entry points below dispatch into; they
// belong to the same flex.obj object model and are implemented in
// flex_objects.cpp.
//
// @0x30200 (414 bytes): rebuild one SDF volume as a Texture3D and publish its
// record in the mSDFs table.
void SDFUpdate(FlexSDFData* data, NvFlex::ContextEx* context, int index,
               int dimX, int dimY, int dimZ, NvFlex::Buffer* field);
// @0x2f3b0 (305 bytes): upload one convex's plane set and publish its record in
// the mConvexes table.
void ConvexUpdate(FlexConvexMeshData* data, NvFlex::ContextEx* context, int index,
                  NvFlexBuffer* planes, int numPlanes,
                  const XVector3<float>* lower, const XVector3<float>* upper);
// @0x2e520 (450 bytes): dense sphere packing at the origin.
int TightPack3D(float radius, float separation, XVector3<float>* points, int maxPoints);

} // namespace flexobj
} // namespace NvFlex
