# CUDA graph capture

CUDA stream capture is more restrictive than its docs suggest. These are the gotchas hit while wiring it into a Hybridizer-built inference engine — re-read before attempting graph capture.

## When to use it (and when not)

Graph capture collapses many small launches into a single replay, amortising the CPU launch overhead. Useful when:

- The captured body runs many times with identical structure (e.g. per-token decode).
- CPU launch overhead is on the critical path — typically when each kernel is short and the host can't queue the next launch fast enough.

When it does **not** help (or hurts):

- At high tok/s (the CPU launch overhead is already overlapping with GPU compute) the WDDM `cudaGraphLaunch` cost can exceed what you save. A real attempt regressed 98 → 80 tok/s on RTX 5070 + WDDM at iter 7.A.7.b — the engine was already compute-bound enough that the replay overhead wasn't compensated. Graph path is best kept opt-in (e.g. behind a `LLAMA_ENABLE_GRAPH=1` env var) until measured.

## Gotchas (concrete, all hit in practice)

### 1. Legacy default stream cannot be captured

`cudaStreamBeginCapture(stream0, ...)` returns an error. You must capture into a stream created with `cudaStreamCreate`. **Side effect:** the new stream is "blocking" w.r.t. the legacy default — every legacy-default-stream operation syncs with it.

### 2. `ResidentStructCache.Materialise` triggers a sync `cudaMemcpy` on first `Get`

Each `FloatResidentArray` / `ResidentArrayGeneric<T>` has a 32-byte device struct that the `DllImport`-bypass kernel takes as a pointer parameter (see [perf-tuning.md](perf-tuning.md)). The first `ResidentStructCache.Get(arr)` for a given array does `cudaMalloc` (OK during capture) **plus a sync `cudaMemcpy`** (NOT OK during capture).

**Fix:** run a non-captured warmup of the body once before `BeginCapture`, so every `Get` hits the dictionary on the captured pass. A single warmup-call materialises all the resident structs the body will touch.

### 3. `cudaStreamCaptureModeRelaxed` is NOT a free pass for sync calls

Docs imply Relaxed allows sync operations during capture. In practice, **operations that would make the legacy stream depend on the capturing stream are still rejected** — even in Relaxed mode. That includes:

- `cuda.Memcpy(devicePtr, hostBuf, ..., H2D)` (sync, legacy-default-stream) where the deviceptr is read by the captured stream → ERROR.
- `cuda.MemsetAsync(devicePtr, 0, ..., default)` where `default` is `cudaStream_t(0)` and the deviceptr is touched by the captured stream → ERROR: *"operation would make the legacy stream depend on a capturing blocking stream"*.

When hunting this error, grep for `cuda.MemsetAsync(... default)` and `cuda.Memcpy(...)` calls in the captured body's call tree. **Route every async op to the capture stream**, not the legacy default.

### 4. Profiling and capture are mutually exclusive

Per-kernel `cudaDeviceSynchronize` (typical in profiling helpers) aborts capture. Gate the graph-capture path on a "profiling disabled" flag and fall back to direct dispatch when profiling is on — kernel ranking is still available, just without the graph collapse.

### 5. `cudaGraphInstantiateWithFlags` over `cudaGraphInstantiate`

CUDA 12+ added the `cudaGraphInstantiateFlagUpload` flag that uploads the graph to the device at instantiate time. Saves a one-time per-replay setup cost on WDDM. Pass this flag in the wrapper.

> ⚠ A subtle bug: an earlier attempt set `cudaGraphInstantiateFlagUpload` in a way that turned out to be invalid and was misdiagnosed as "GPU-bound at the existing tok/s level" — only later did the wrapper get fixed. If a graph capture appears to regress perf with no obvious cause, double-check the `instantiate` call is well-formed.

### 6. Stream drain before `BeginCapture`

After warmup, kernels are still queued async on the capture stream. Add `cudaStreamSynchronize` between warmup and `BeginCapture` so the stream is idle when capture starts. Otherwise phantom pending-work dependencies can re-trigger the legacy-stream conflict errors.

### 7. Diagnostic move: log `Materialise`

When capture aborts without an obvious reason, a one-line `Console.Error.WriteLine` in `ResidentStructCache.Materialise` surfaces which array is being first-touched inside capture. If you see no `Materialise` calls during capture, the warmup is working — point your investigation at sync calls (#3 above) instead.

## Calling CUDA graph APIs from C# — `extern "C"` wrappers

`Hybridizer.Runtime.CUDAImports.cuda` doesn't expose the graph API. Wrap each needed entry point in `intrinsics.cuh` as `extern "C" DLL_PUBLIC` and resolve from C# via `NativeLibrary.GetExport`. This is the same plumbing used for kernel bypass — see [perf-tuning.md](perf-tuning.md) for the canonical pattern.

```cpp
// intrinsics.cuh
extern "C" {
DLL_PUBLIC int llama_stream_create(cudaStream_t* streamOut) {
    return (int)cudaStreamCreate(streamOut);
}
DLL_PUBLIC int llama_stream_begin_capture(cudaStream_t stream, int mode) {
    return (int)cudaStreamBeginCapture(stream, (cudaStreamCaptureMode)mode);
}
DLL_PUBLIC int llama_stream_end_capture(cudaStream_t stream, cudaGraph_t* graphOut) {
    return (int)cudaStreamEndCapture(stream, graphOut);
}
DLL_PUBLIC int llama_graph_instantiate(cudaGraphExec_t* execOut, cudaGraph_t graph) {
    return (int)cudaGraphInstantiateWithFlags(execOut, graph, cudaGraphInstantiateFlagUpload);
}
DLL_PUBLIC int llama_graph_launch(cudaGraphExec_t exec, cudaStream_t stream) {
    return (int)cudaGraphLaunch(exec, stream);
}
}
```

`DLL_PUBLIC` is defined by `hybridizer.cuda.cuh` (expands to `__declspec(dllexport)` on Windows / `__attribute__((visibility("default")))` on Linux). It's always available because `intrinsics.cuh` is `#include`'d into `hybridizer.all.cuda.cu` which pulls the Hybridizer headers first.

## Related

- The extern-C wrapper plumbing in detail: [perf-tuning.md](perf-tuning.md)
- The `ResidentStructCache` it depends on: [perf-tuning.md](perf-tuning.md)
- Why this conflicts with `cudaDeviceSynchronize`-based profiling: [perf-tuning.md](perf-tuning.md)
