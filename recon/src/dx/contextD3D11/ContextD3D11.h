// flexdecomp: reconstructed D3D backend — ContextD3D11.obj object model.
//
// Every class below is an *internal* type of ContextD3D11.obj: it is only ever
// reached through the abstract `NvFlex::Buffer` / `ConstantBuffer` /
// `ComputeShader` / `Texture3D` / `TimerPool` / `Fence` handles declared in
// Frame.h and utils.h, so the vtable order here is this reconstruction's own.
// The member names and order follow the PDB (`llvm-pdbutil dump -types`:
// BufferD3D11 sizeof 120, ConstantBufferD3D11 56, ComputeShaderD3D11 64,
// Texture3DD3D11 112, TimerD3D11 56, TimerPoolD3D11 64, FenceD3D11 32) so the
// Ghidra bodies transcribe 1:1.
//
// Evidence for every function: `// ---- <Name> @ <rva>` blocks of
// work/decomp/nvflexd3d/NvFlexReleaseD3D_x64.dll.c; the per-function RVA is
// repeated in the source comments.
#pragma once

#include <d3d11.h>
#include <d3d11_1.h>

#include <vector>

#include "Format.h"
#include "utils.h"

namespace NvFlex
{

class ContextD3D11;
class DeviceD3D11;

// ---------------------------------------------------------------------------
// Reference counting
// ---------------------------------------------------------------------------
// PDB: every native object starts with `Object` (vtable + atomic refcount at
// +8) and allocates through the global FlexMalloc/FlexFree hooks (Frame.h).
// `addRef` is the InterlockedIncrement the shipped code emits at every
// `NvFlex::*::addRef` entry point; `release` decrements and destroys at zero
// through the scalar deleting destructor.
template <typename T>
unsigned flexAddRef(long& refCount)
{
    return unsigned(++refCount);
}

template <typename T>
unsigned flexRelease(T* object, long& refCount)
{
    const long remaining = --refCount;
    if (remaining == 0)
    {
        object->~T();
        g_FlexFree(object);
    }
    return unsigned(remaining);
}

// ---------------------------------------------------------------------------
// CPU-visible views
// ---------------------------------------------------------------------------
// `Resource*` / `ResourceRW*` in DispatchParams are these two objects: dispatch
// only needs the native view they wrap.  They are owned by the buffer or
// texture that created them and never outlive it, which is exactly how the
// shipped code treats them (dispatch reads the pointer, never add-refs it).
class ResourceD3D11 : public Resource
{
public:
    explicit ResourceD3D11(ID3D11ShaderResourceView* srv = 0) : m_srv(srv) {}

    // PDB ResourceD3D11 sizeof 16: vtable@0, m_srv@8.
    unsigned addRef() override { return 1; }
    unsigned release() override { return 1; }

    ID3D11ShaderResourceView* m_srv;   // @8
};

class ResourceRWD3D11 : public ResourceRW
{
public:
    explicit ResourceRWD3D11(ID3D11UnorderedAccessView* uav = 0) : m_uav(uav) {}

    // PDB ResourceRWD3D11 sizeof 32: vtable@0, m_uav@24.
    unsigned addRef() override { return 1; }
    unsigned release() override { return 1; }

    ID3D11UnorderedAccessView* m_uav;  // @24
};

// ---------------------------------------------------------------------------
// BufferD3D11 — PDB sizeof 120
// ---------------------------------------------------------------------------
// Three constructors (ContextD3D11.obj @0x51640, @0x51a10, @0x51b10):
//   * from a BufferDesc            — the normal path, see createBuffer
//   * from a native ID3D11Buffer   — wraps an application buffer (eShared)
//   * from a Buffer + BufferViewDesc — creates a typed view over a buffer
class BufferD3D11 : public Buffer
{
public:
    // @0x51640 (787 bytes)
    BufferD3D11(ContextD3D11* context, const BufferDesc* desc);
    // @0x51a10 (241 bytes) — NvFlexRegisterD3DBuffer / eShared path.
    BufferD3D11(ContextD3D11* context, ID3D11Buffer* nativeBuffer);
    // @0x51b10 (390 bytes) — NvFlexRegisterD3DBuffer(view) path.
    BufferD3D11(ContextD3D11* context, BufferD3D11* buffer, const BufferViewDesc* view);
    // @0x51ca0 (290 bytes)
    ~BufferD3D11() override;

    unsigned addRef() override { return flexAddRef<BufferD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    Resource*   getResource(BufferTypes type) override;
    ResourceRW* getResourceRW() override;
    const BufferDesc& desc() const override { return m_desc; }

    // Native handles used by the Context copy/upload/download commands.
    ID3D11Buffer* m_buffer;                 // @80
    ID3D11Buffer* m_uploadBuffer;           // @88
    ID3D11Buffer* m_downloadBuffer;         // @96
    ID3D11Buffer* m_bufferOnGraphicsDevice; // @104
    ContextD3D11* m_context;                // @112

    BufferDesc          m_desc;             // PDB: BufferDesc storage, before m_buffer
    ResourceD3D11       m_resource;
    ResourceRWD3D11     m_resourceRW;

private:
    BufferD3D11(const BufferD3D11&);
    BufferD3D11& operator=(const BufferD3D11&);
};

// ---------------------------------------------------------------------------
// ConstantBufferD3D11 — PDB sizeof 56
// ---------------------------------------------------------------------------
class ConstantBufferD3D11 : public ConstantBuffer
{
public:
    // @0x513b0 (179 bytes)
    ConstantBufferD3D11(ContextD3D11* context, const ConstantBufferDesc* desc);
    // @0x514b0 (116 bytes)
    ~ConstantBufferD3D11() override;

    unsigned addRef() override { return flexAddRef<ConstantBufferD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    const ConstantBufferDesc& desc() const override { return m_desc; }

    ID3D11Buffer* m_buffer;         // @40
    ID3D11Buffer* m_downloadBuffer; // @48

    ConstantBufferDesc m_desc;

private:
    ConstantBufferD3D11(const ConstantBufferD3D11&);
    ConstantBufferD3D11& operator=(const ConstantBufferD3D11&);
};

// ---------------------------------------------------------------------------
// ComputeShaderD3D11 — PDB sizeof 64
// ---------------------------------------------------------------------------
class ComputeShaderD3D11 : public ComputeShader
{
public:
    // @0x52af0 (221 bytes)
    ComputeShaderD3D11(ContextD3D11* context, const ComputeShaderDesc* desc);
    // @0x52c80 (93 bytes)
    ~ComputeShaderD3D11() override;

    unsigned addRef() override { return flexAddRef<ComputeShaderD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    const ComputeShaderDesc& desc() const override { return m_desc; }

    ID3D11ComputeShader* m_cs;      // @56

    ComputeShaderDesc m_desc;
};

// ---------------------------------------------------------------------------
// Texture3DD3D11 — PDB sizeof 112
// ---------------------------------------------------------------------------
class Texture3DD3D11 : public Texture3D
{
public:
    // @0x52570 (485 bytes)
    Texture3DD3D11(ContextD3D11* context, const Texture3DDesc* desc);
    // @0x527a0 (199 bytes)
    ~Texture3DD3D11() override;

    unsigned addRef() override { return flexAddRef<Texture3DD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    Resource*   getResource() override;
    ResourceRW* getResourceRW() override;
    const Texture3DDesc& desc() const { return m_desc; }

    ID3D11Texture3D* m_texture;         // @88
    ID3D11Texture3D* m_uploadTexture;   // @96
    ID3D11Texture3D* m_downloadTexture; // @104

    Texture3DDesc  m_desc;
    ResourceD3D11  m_resource;
    ResourceRWD3D11 m_resourceRW;
};

// ---------------------------------------------------------------------------
// Timers — PDB TimerD3D11 sizeof 56, TimerPoolD3D11 sizeof 64
// ---------------------------------------------------------------------------
// The acceptance runs never enable timers (`DxTimers::mEnabled` is only set by
// NvFlexSetDebug), and the shipped `TimerPoolD3D11::begin` still runs, so the
// pool keeps its slot life cycle; the GPU queries are created lazily.
class TimerD3D11
{
public:
    // @0x52530 (189 bytes)
    explicit TimerD3D11(ContextD3D11* context);
    ~TimerD3D11();

    unsigned addRef() { return flexAddRef<TimerD3D11>(m_refCount); }
    unsigned release() { return flexRelease(this, m_refCount); }

    long            m_refCount;
    int             m_state;      // @24  (0 = free)
    ID3D11Query*    m_disjoint;   // @32
    ID3D11Query*    m_begin;      // @40
    ID3D11Query*    m_end;        // @48
};

class TimerPoolD3D11 : public TimerPool
{
public:
    // @0x529c0 (82 bytes)
    explicit TimerPoolD3D11(ContextD3D11* context);
    ~TimerPoolD3D11() override;

    unsigned addRef() override { return flexAddRef<TimerPoolD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    // @0x52b20 (125 bytes)
    int  begin() override;
    // @0x52c40 (368 bytes)
    void end(int index) override;
    void resolve() override;
    void sync() override;
    void clear() override;
    // @0x53c50 (368 bytes) — elapsed seconds for one slot.
    void reserveTimers(unsigned count) override { reserve(count); }
    float get(int index, unsigned long long* frequency = 0, unsigned long long* begin = 0,
              unsigned long long* end = 0) override;

    // @0x52dc0 (151 bytes) — allocates `capacity` timer objects.
    void reserve(unsigned capacity);

    std::vector<TimerD3D11*> m_elem;   // @32
    ContextD3D11*            m_context; // @56
};

// ---------------------------------------------------------------------------
// FenceD3D11 — PDB sizeof 32
// ---------------------------------------------------------------------------
class FenceD3D11 : public Fence
{
public:
    // @0x53740 (109 bytes)
    explicit FenceD3D11(ContextD3D11* context);
    // @0x53840 (93 bytes)
    ~FenceD3D11() override;

    unsigned addRef() override { return flexAddRef<FenceD3D11>(m_refCount); }
    unsigned release() override { return flexRelease(this, m_refCount); }
    long     m_refCount;

    ID3D11Query* m_query;   // @24
};

// ---------------------------------------------------------------------------
// ContextD3D11 — PDB sizeof 912
// ---------------------------------------------------------------------------
// Cached compute-stage state, so `contextPush`/`contextPop` can save and
// restore it around an application callback (PDB
// `NvFlex::VectorCached<T, N>`: m_data@0, m_capacity@8, m_cache@16, m_size@272
// for N = 16; the 8-slot instantiations are 140 bytes).
template <typename T, unsigned N>
class VectorCached
{
public:
    VectorCached() : m_data(0), m_capacity(0), m_size(0)
    {
        for (unsigned i = 0; i < N; ++i)
            m_cache[i] = T();
    }

    static const unsigned s_staticCapacity = N;

    // @0x53440 / @0x534b0 / @0x53510 (12 bytes): the cache is the static
    // storage, so allocate() only reports its address.
    T* allocate() { return m_cache; }
    // @0x53490 / @0x534c0 / @0x53520 (20 bytes): nothing is heap owned.
    void cleanup() {}

    // @0x53050 (10 bytes)
    T& operator[](unsigned i) { return m_data[i]; }

    // ContextD3D11.obj `VectorCached<T,16>::allocateBack` (used by flush()).
    unsigned allocateBack() { return m_size < N ? m_size++ : N - 1; }

    T*       m_data;
    unsigned m_capacity;
    T        m_cache[N];
    unsigned m_size;
};

class ContextD3D11 : public ContextEx
{
public:
    struct Status
    {
        ID3D11Query* m_query;        // @0
        bool         m_queryActive;  // @8
    };

    struct State
    {
        VectorCached<ID3D11Buffer*, 8>             csConstantBuffers;  // @0
        VectorCached<ID3D11SamplerState*, 8>       csSamplers;         // @88
        VectorCached<ID3D11ShaderResourceView*, 16> csSrvs;            // @176
        VectorCached<ID3D11UnorderedAccessView*, 8> csUavs;            // @328
        ID3D11ComputeShader*                       computeShader;      // @416
    };

    explicit ContextD3D11(DeviceD3D11* device);
    ~ContextD3D11() override;

    // --- Context ---
    ConstantBuffer* createConstantBuffer(const ConstantBufferDesc* desc) override;
    Buffer*         createBuffer(const BufferDesc* desc, void* initialData) override;
    Buffer*         createBufferView(Buffer* buffer, const BufferViewDesc* desc) override;
    ComputeShader*  createComputeShader(const ComputeShaderDesc* desc) override;

    void* map(ConstantBuffer* buffer) override;
    void  unmap(ConstantBuffer* buffer) override;
    void  copyResourceState(Buffer* dst, Buffer* src) override;
    void  dispatch(const DispatchParams* params) override;
    int   setEventMarkerAftermath(void* data, unsigned size) override;

    // --- ContextEx ---
    MappedData map(Texture3D* texture) override;
    void*      map(Buffer* buffer, MapType type, bool wait) override;
    void       unmap(Texture3D* texture) override;
    void       unmap(Buffer* buffer) override;
    void       copy(Texture3D* dst, Texture3D* src) override;
    void       copy(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                    unsigned size) override;
    void       copy(ConstantBuffer* dst, Buffer* src) override;
    void       upload(Buffer* buffer, unsigned offset, unsigned size) override;
    void       upload(Buffer* buffer) override;
    void       download(Texture3D* texture) override;
    void       download(Buffer* buffer, unsigned offset, unsigned size) override;
    void       download(Buffer* buffer) override;
    void       copyToNative(void* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                            unsigned size) override;
    void       copyToDevice(Buffer* dst, unsigned dstOffset, Buffer* src, unsigned srcOffset,
                            unsigned size, BufferResourceTypes dstType) override;
    void       copyToHost(Buffer* src, unsigned srcOffset, Buffer* dst, unsigned dstOffset,
                          unsigned size) override;
    void       clearUnorderedAccessViewUint(Buffer* buffer, const unsigned* values) override;
    void       clearUnorderedAccessViewFloat(Buffer* buffer, const float* values) override;
    void       flush() override;
    void       computeWaitForGraphics() override;

    Texture3D* createTexture3D(const Texture3DDesc* desc) override;
    Fence*     createFence() override;
    void       fenceSet(Fence* fence) override;
    void       fenceWait(Fence* fence, unsigned timeoutUs) override;
    void       startFrame() override;
    void       clearState() override;
    void       updateContext() override;
    void       contextPop() override;
    TimerPool* createTimerPool(unsigned capacity) override;
    void       eventMarker(const wchar_t* label, int slot) override;
    int        getDataAftermath(void* data, unsigned size) override;

    // --- ContextD3D11 private surface (PDB) ---

    // @0x52970 / @0x52980: PIX markers are empty in this build.
    void pixBeginEvent(const wchar_t* label);
    void pixEndEvent();
    // @0x510d0 (320 bytes) / @0x51210 (402 bytes)
    void contextPush();
    // @0x524f0 (113 bytes)
    void updateSubresource(Buffer* buffer, const unsigned* box, const void* data);
    // @0x52370 (3 bytes): the shipped body is empty.
    void computeWaitForGraphicsInternal() {}
    // @0x4fda0 / @0x4fec0 / @0x4ff50 / @0x50060: resource extent extraction.
    static NvFlexDim extractDimBuffer(ID3D11Resource* resource);
    static NvFlexDim extractDimTexture1D(ID3D11Resource* resource);
    static NvFlexDim extractDimTexture2D(ID3D11Resource* resource);
    static NvFlexDim extractDimTexture3D(ID3D11Resource* resource);
    static NvFlexDim extractDim(ID3D11Resource* resource);
    static NvFlexDim extractDim(ID3D11ShaderResourceView* view);

    // Native accessors (NvFlexGetDeviceAndContext).
    ID3D11Device*        device() const { return m_device; }
    ID3D11DeviceContext* deviceContext() const { return m_deviceContext; }

    // --- PDB layout ---
    ID3D11Device*        m_device;                  // @64
    ID3D11Device*        m_renderDevice;            // @72
    ID3D11DeviceContext* m_deviceContext;           // @80
    ID3D11DeviceContext* m_renderDeviceContext;     // @88
    ID3DUserDefinedAnnotation* m_d3dAnnotation;      // @96

    bool   m_useD3D11asyncCompute;                          // @104
    void*  m_computeFinishedWriting;                        // @112
    void*  m_graphicsFinishedReading;                       // @120
    unsigned long long m_computeFinishedWriting_count;      // @128
    unsigned long long m_graphicsFinishedReading_count;      // @136
    ID3D11Query* m_renderCompletionFence;                   // @144
    ID3D11Query* m_computeCompletionFence;                  // @152
    ID3D11SamplerState* m_sampler0;                         // @160
    ID3D11SamplerState* m_sampler1;                         // @168
    ID3D11SamplerState* m_sampler2;                         // @176
    ID3D11SamplerState* m_sampler3;                         // @184
    ID3D11SamplerState* m_sampler4;                         // @192
    ID3D11SamplerState* m_sampler5;                         // @200
    VectorCached<Status, 16> m_status;                      // @208
    State m_state;                                         // @488

private:
    ContextD3D11(const ContextD3D11&);
    ContextD3D11& operator=(const ContextD3D11&);
};

} // namespace NvFlex
