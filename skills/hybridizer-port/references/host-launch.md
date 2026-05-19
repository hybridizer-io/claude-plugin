# Host launch

How a managed program invokes a Hybridizer-built kernel.

## Standard launch (the basic-samples idiom)

```csharp
cuda.GetDeviceProperties(out cudaDeviceProp prop, 0);

HybRunner runner = SatelliteLoader.Load()
    .SetDistrib(prop.multiProcessorCount * 16, 128);   // (gridX, blockX) short form

dynamic wrapped = runner.Wrap(new MyKernels());
wrapped.MyEntryPoint(N, a, b);                          // dynamic call → satellite DLL

cuda.ERROR_CHECK(cuda.DeviceSynchronize());
```

## `SatelliteLoader.Load()`

Lives in `src/0.Utils/Utilities/SatelliteLoader.cs` in basic-samples — **not part of the runtime**, copy it into your project. Its body is:

```csharp
var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
var sat = Directory.GetFiles(dir, "*_CUDA.dll").First();
return HybRunner.Cuda(sat);
```

For explicit backend selection: `HybRunner.Cuda(name)`, `HybRunner.OMP(name)`, `HybRunner.AVX(name)`, `HybRunner.AVX512()`, `HybRunner.HIP(name)`.

Parameterless `new HybRunner()` works only if the assembly has `[assembly: HybRunnerDefaultSatelliteName("Foo_CUDA.dll")]`. Otherwise pass the satellite path explicitly.

## `SetDistrib` overloads

- `SetDistrib(grid, block)` — 1-D.
- `SetDistrib(gridX, gridY, blockX, blockY, blockZ, shmemBytes)` — full form. **Use this for any kernel that allocates shared memory dynamically** — the `shmemBytes` argument must be ≥ what `SharedMemoryAllocator<T>` will allocate.

## `runner.Wrap(instance)`

Returns a `dynamic` proxy. Method calls on the proxy resolve via reflection to a P/Invoke into the satellite. **Instance state is marshalled along with arguments** — fields on `instance` become kernel args. For static `[EntryPoint]` methods, the instance is just a vessel.

## Streams

```csharp
cuda.StreamCreate(out cudaStream_t s);
wrapped.SetStream(s).MyKernel(args);   // fluent; enqueues into s
cuda.StreamSynchronize(s);
cuda.StreamDestroy(s);
```

## Per-call distribution override

Chain `.SetDistrib(...)` on `wrapped` (the proxy) right before the call to use a different launch geometry without rebuilding the runner:

```csharp
wrapped.SetDistrib(grid, block, shmem).MyKernel(args);
```

## Error checking

```csharp
cuda.ERROR_CHECK(cuda.DeviceSynchronize());
```

`cuda.ERROR_CHECK(cudaError_t)` is the standard guard. **Errors from kernel launches surface on the next sync, not at the launch site** — so a `DeviceSynchronize()` after a kernel batch is the only place those errors become visible.

## Dual-path (managed fallback)

Calling `MyKernels.MyEntryPoint(...)` *directly* (no `.Wrap`) runs the **managed** version — handy for unit tests, golden-output capture, and CPU-vs-GPU diffing. The kernel body is just normal C# in that case. This is why the idiomatic body is `Parallel.For(0, N, i => ...)` — the same body works under both the GPU and managed paths.

## `FloatResidentArray` / `ResidentArrayGeneric<T>` init

These types allocate once on the device and survive across many kernel calls. The lifecycle quirk: a freshly constructed resident array has `tab = null`, `_dPtr = 0`, `Status = NoAction`. The CUDA marshaller `SerializeResidentArray` only refreshes when `Status == DeviceNeedsRefresh`, and lazily allocates `DevicePointer`.

Choose the init pattern by the array's role:

### Kernel writes first (KV caches, attention scratch, output activations)

```csharp
_resident = new FloatResidentArray(size);

// 1. Force device allocation NOW so the first kernel call doesn't race a lazy alloc.
_ = _resident.DevicePointer;

// 2. Tell the marshaller "device is the truth" so it never tries to upload
//    an empty host buffer over our kernel writes.
_resident.Status = ResidentArrayStatus.HostNeedsRefresh;
```

### Host fills first (LUTs, RoPE tables, pre-computed constants)

```csharp
_resident = new FloatResidentArray(host.Length);
unsafe {
    fixed (float* src = host) {
        Buffer.MemoryCopy(src, (void*)_resident.HostPointer, byteSize, byteSize);
    }
}
_resident.Status = ResidentArrayStatus.DeviceNeedsRefresh;
// → marshaller's first pass does cuda.Memcpy(H→D) and allocates _dPtr if needed
```

(Touching `HostPointer` allocates the host buffer; `DeviceNeedsRefresh` triggers `RefreshDevice()` in the marshaller.)

### The failure mode

Doing nothing — `Status = NoAction`, no eager allocation — produces a hard-to-debug symptom: the device buffer is allocated lazily, the kernel writes to *uninitialised* device memory, and downstream output collapses to a constant value (e.g. all tokens become the same one). No CUDA error, kernels run, tokens collapse silently.

### Bypass caveat

If the array will be passed to a DllImport-bypassed kernel (see [perf-tuning.md](perf-tuning.md)), the bypass path is responsible for replicating the marshaller's `RefreshDevice()`-when-`DeviceNeedsRefresh` step. Either call `arr.RefreshDevice()` explicitly before the first bypassed call, or let a struct-cache do it (`ResidentStructCache.Get` checks `Status == DeviceNeedsRefresh` and refreshes). Forgetting this silently produces all-zero/all-garbage output.

## Manual device memory (when you need full control)

```csharp
cuda.Malloc(out IntPtr p, bytes);
cuda.Memcpy(...);
cuda.MemcpyAsync(..., stream);
cuda.Free(p);
```

Pass the `IntPtr` directly as a kernel arg; Hybridizer treats it as a typed device pointer based on the parameter signature.

## Related

- The build that produces the satellite: [build-pipeline.md](build-pipeline.md)
- What the kernel body can contain: [device-code.md](device-code.md)
- Cutting marshaller overhead: [perf-tuning.md](perf-tuning.md)
- CUDA graph capture (different launch path entirely): [graph-capture.md](graph-capture.md)
