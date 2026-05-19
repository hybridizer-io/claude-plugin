# Performance tuning

The high-impact moves once the port is functionally correct. Apply at step 8 of the [methodology.md](methodology.md), after a GPU profile shows real signal (which means after memcpies have been driven down at step 6).

## `DllImport` bypass for hot kernels

The single biggest win once kernel bodies are tight: bypass `HybRunner.Wrap`'s per-call marshaller for hot kernels.

### Why bypass

Every wrapped call (`dynamic = runner.Wrap(...)`, then `dynamic.X(args)`) goes through `HybRunner.AddParametersToStack` → `marshalManagedToNative` → `NativeSerializer.InitialVisit` for every class/interface parameter — **including `FloatResidentArray` and `ResidentArrayGeneric<T>`**.

Per param per call: `cudaMalloc` + `cudaMemcpy(H→D)` + `cudaDeviceSynchronize` (via marshaller-internal `StreamSynchronize`) + `cudaFree`. For a forward pass with hundreds of kernel launches × several resident params each, that's tens of thousands of `cudaMemcpy` and `cudaMalloc` / `cudaFree` calls — even though the underlying device data is persistent. The **struct wrapper** is re-serialised every call.

### What the bypass does

Pre-allocate the 32-byte native struct mirror **once per resident array** at construction time, then call the generated `_ExternCWrapperStream_CUDA` symbol via `NativeLibrary.GetExport` and pass the cached device-side struct pointer.

### Native struct layout (both `FloatResidentArray` and `ResidentArrayGeneric<T>`)

From the emitted `generated-cuda/hybridizer.includes.cuda.h`:

| offset | bytes | field | notes |
|---|---|---|---|
| 0–7 | 8 | `hybridobject` base (`_vtable` / `_typeid` union) | kernel doesn't read; can be zero |
| 8–15 | 8 | `tab` (data pointer) | what the kernel actually dereferences |
| 16–23 | 8 | `_count` (int64) | not load-bearing |
| 24–27 | 4 | `_status` (int) | host-side concern |
| 28–31 | 4 | padding | |

Total: 32 bytes. **C# source field order differs** from C++ layout (Hybridizer reorders for alignment) — do not infer layout from `typeof(FloatResidentArray).GetFields()`. Use the emitted `.h` as ground truth.

### Implementation skeleton

```csharp
// One-time per resident array.
public static unsafe nint MaterialiseStruct(FloatResidentArray arr) {
    if (arr.Status == ResidentArrayStatus.DeviceNeedsRefresh)
        arr.RefreshDevice();                                  // *** crucial — see below ***
    cuda.Malloc(out nint dev, (size_t)32);
    byte* buf = stackalloc byte[32];
    for (int i = 0; i < 32; i++) buf[i] = 0;
    *(long*)(buf + 8)  = (long)arr.DevicePointer;             // tab
    *(long*)(buf + 16) = arr.Count;                           // _count
    cuda.Memcpy(dev, (nint)buf, (size_t)32, cudaMemcpyKind.cudaMemcpyHostToDevice);
    return dev;
}

// Per kernel call.
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate int KernelDel(int gx, int gy, int gz, int bx, int by, int bz, int shared,
                       nint stream, nint result, nint values, /* ... */ int rows);

nint lib = NativeLibrary.Load(SatelliteLoader.CudaSatellitePath()!);
nint fp  = NativeLibrary.GetExport(lib,
    "Mynsx46Mathx46Q8Kernelsx46MatVecMulFullyResident_ExternCWrapperStream_CUDA");
var del = Marshal.GetDelegateForFunctionPointer<KernelDel>(fp);

del(gridDimX, 1, 1, blockDim, 1, 1, /*shared*/ 0, /*stream*/ nint.Zero,
    cachedResult, cachedValues, /* ... */, rows);
```

### Critical bug that masks the win

`HybRunner`'s `NativeSerializer.SerializeResidentArray` does **two** things per call:

1. Override the struct's `tab` field with the device pointer (the obvious part).
2. **If `Status == DeviceNeedsRefresh`, call `arr.RefreshDevice()`** — i.e. `cudaMemcpy(host→device)` of the actual data.

Step 2 is invisible from the call site but load-bearing. Any resident array populated with the "host-fills-first" pattern (`HostPointer` bulk-copy + `Status = DeviceNeedsRefresh`) relies on the next wrapped call to push the data to device.

A bypass that skips this step produces **silent garbage**: kernels run, no CUDA error, all results are 0 or whatever was in fresh device memory. Symptom is "argmax tokens all collapse to a constant".

Fix is one line in the struct cache `Get`:

```csharp
if (arr.Status == ResidentArrayStatus.DeviceNeedsRefresh)
    arr.RefreshDevice();
```

See [host-launch.md](host-launch.md) for the full resident-array init protocol.

### Wrapper variants and stream handling

Each kernel emits **three** host wrappers in `generated-cuda/<Class>.cu`:

- **`_ExternCWrapper_CUDA`** — non-stream. Launches into default stream + calls `cudaDeviceSynchronize` at the end. **`HybRunner.Wrap` defaults to this** (`UseStream` is false by default).
- **`_ExternCWrapperStream_CUDA`** — takes a `cudaStream_t` arg, launches into that stream, **no sync**. Returns immediately. **This is what the bypass should call.**
- **`_ExternCWrapperGridSync_CUDA`** / **`_ExternCWrapperStreamGridSync_CUDA`** — cooperative-group launch variants; only needed for cross-block sync.

The Stream variant + default stream (`nint.Zero`) serialises correctly with any `HybRunner` wrapped calls still in the mix, since they also use the default stream.

### Symbol mangling

Hybridizer mangles `.` to `x46` (ASCII 0x2E = 46 decimal). `Foo.Bar.Baz.Method_ExternCWrapperStream_CUDA` becomes `Foox46Barx46Bazx46Method_ExternCWrapperStream_CUDA`. Confirm with `nm -D <satellite.dll>` (Linux) or `dumpbin /exports` (Windows).

### Mixed signatures (resident + `float[]` params)

Some kernels have both resident arrays and plain `float*` arguments. For full bypass, the caller manages the `float*` buffers explicitly:

- **Outputs the host reads** (e.g. RmsNorm `sumOut` for the scale calc): persistent `cudaMalloc` of 4 bytes once. `cudaMemsetAsync` to zero before the reduce. `cudaMemcpy(D→H)` after.
- **Read-only inputs** (e.g. norm weights): pre-upload to a persistent `FloatResidentArray` once at ctor; pass `arr.DevicePointer` (raw, not the cached struct pointer) to the kernel as the `float*` arg.

### Build hygiene

Add `-DNO_EXCEPTION` to the `nvcc` command line in `Directory.Build.targets`. Strips `hybridizer::runtime::hostinit`'s per-call `cudaMallocManaged` of the exception-stack buffer. Safe no-op for any project whose kernels don't throw on device.

### Measured impact (TinyLlama Q8_0, 32-token forward, RTX 5070 + WSL2)

| CUDA API | Wrapped baseline | + bypass (resident) | + RmsNorm bypass | Reduction |
|---|---|---|---|---|
| `cudaMemcpy` | 92,296 | 15,632 | 2,431 | **−97 %** |
| `cudaMalloc` | 47,175 | 9,028 | 786 | **−98 %** |
| `cudaFree` | 46,805 | 8,288 | (drops out of top list) | **−99 %** |
| `cudaDeviceSynchronize` | 29,600 | 6,660 | (drops out of top list) | **−99 %** |
| `cudaLaunchKernel` | 14,800 | 14,800 | 14,800 | unchanged |

Tok/s under nsys (instrumented): 1.84 → 6.08 → **7.80** (4.2×).  
Tok/s native (no profiler): ~8.4 → ~8.1 → re-measure. **Most of nsys's reported per-call cost is instrumentation, not real time** — see the nsys overhead section below.

## `extern "C"` wrappers in `intrinsics.cuh`

Any CUDA runtime API not exposed by `Hybridizer.Runtime.CUDAImports.cuda` can be wrapped in `extern "C" DLL_PUBLIC` helpers inside `intrinsics.cuh`. They get compiled into the satellite and resolved from C# via `NativeLibrary.GetExport`, same pattern as `_ExternCWrapperStream_CUDA` kernel wrappers.

`intrinsics.cuh` is `#include`'d once into `generated-cuda/hybridizer.all.cuda.cu`, compiled by `nvcc` with the same flags as the kernels. Any host-side `extern "C"` function declared with `DLL_PUBLIC` becomes an exported symbol on the satellite DLL.

```cpp
extern "C" {
DLL_PUBLIC int llama_stream_create(cudaStream_t* streamOut) {
    return (int)cudaStreamCreate(streamOut);
}
DLL_PUBLIC int llama_graph_launch(cudaGraphExec_t exec, cudaStream_t stream) {
    return (int)cudaGraphLaunch(exec, stream);
}
}
```

```csharp
private delegate int StreamCreateDel(out nint streamOut);

private static T Resolve<T>(nint lib, string name) where T : System.Delegate {
    nint sym = NativeLibrary.GetExport(lib, name);
    return Marshal.GetDelegateForFunctionPointer<T>(sym);
}

_streamCreate = Resolve<StreamCreateDel>(CudaInvoke.LibraryHandle, "llama_stream_create");
```

### Key points

1. **`DLL_PUBLIC` is defined by Hybridizer's `hybridizer.cuda.cuh`.** Always available inside `intrinsics.cuh` because it's included from the all-cuda TU which pulls the Hybridizer headers first.
2. **Header guards are essential.** `#ifndef __MYPROJECT_INTRINSICS__` etc. Each `[IntrinsicInclude("intrinsics.cuh")]`-decorated class triggers an include of the header into its `.cu`, and all those `.cu` files are themselves `#include`'d into `hybridizer.all.cuda.cu`. The guard means the header is processed exactly once in the single TU — non-inline `extern "C"` symbols defined inside aren't multiply-defined.
3. **Hybridizer's `cudaStream_t` is a thin `IntPtr` wrapper.** `new cudaStream_t(nint)` and implicit `cudaStream_t → nint` both work. C# stores `nint`, the C++ wrapper sees `cudaStream_t`. Same for `cudaGraph_t` / `cudaGraphExec_t`.
4. **No extra CUDA headers needed.** `intrinsics.cuh` is already inside a `.cu` TU; all CUDA runtime types and functions are available. Just `#include` what you specifically need (`<cuda/functional>` for `cuda::maximum<>` etc.).
5. **Symbol mangling.** `extern "C"` is required so the name in the DLL is exactly `llama_graph_launch` (not a mangled C++ name). Verify with `strings <satellite>.dll | grep llama_`.

## Profiling — counts and ratios, not absolute times

`nsys profile --stats=true` instruments every CUDA API entry/exit. In tight-loop, small-payload workloads the instrumentation overhead is **comparable to the work itself**.

Observed asymmetry on TinyLlama Q8_0, RTX 5070, 32-token forward:

| | Under nsys | Native (Windows) |
|---|---|---|
| Wrapped HybRunner baseline | 1.84 tok/s | ~8.4 tok/s |
| + `DllImport` bypass | 7.80 tok/s | ~8.1 tok/s |
| nsys / native ratio | 4.2× speedup | ~marginal |

The bypass's 4× win under nsys is real — those CUDA API calls really did go away — but most of the time it freed was *instrumentation time*, not real CUDA driver time. On native, the marshaller's `cudaMalloc` / `cudaFree` / `cudaMemcpy` were cheap enough that eliminating 90 % of them barely registered.

### How to use nsys without being misled

- **Trust the counts** in `cuda_api_sum` and `cuda_gpu_mem_size_sum` — they reflect what the program asked CUDA to do.
- **Trust the proportions** between rows in the same report (which APIs dominate).
- **Don't trust the absolute "Time (s)" or "Avg (ns)"** as the real cost on a clean run.
- **Always re-measure native tok/s** before declaring a perf win.

### Where to run the profiler

- **On WSL2, `nsys` and `compute-sanitizer` wedge `dotnet` + CUDA** (see [gotchas.md](gotchas.md)). Profile on native Windows or use an in-process profiler.
- **Profiling and graph capture are mutually exclusive** — see [graph-capture.md](graph-capture.md).

## Related

- The resident-array init protocol that the bypass depends on: [host-launch.md](host-launch.md)
- The graph-capture path, which depends on the same `ResidentStructCache`: [graph-capture.md](graph-capture.md)
- The methodology positioning (this is step 8 work): [methodology.md](methodology.md)
- Where to find samples that show these patterns: [samples-index.md](samples-index.md)
