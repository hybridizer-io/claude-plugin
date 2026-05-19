# Device code — what works and what doesn't

What can legally appear inside a `[Kernel]` or `[EntryPoint]` body, and the C# features that do **not** transcode.

## Thread / block indexing

Exposed as **static property accesses**, not function calls — they look almost identical to CUDA C++:

```csharp
int tid = threadIdx.x + blockIdx.x * blockDim.x;   // also .y/.z; *64 variants exist for 64-bit grids
```

These resolve via `[IntrinsicConstant]` to `threadIdx.x` etc. in the emitted `.cu`.

## Synchronization

```csharp
CUDAIntrinsics.__syncthreads();
CUDAIntrinsics.__syncwarp();
CUDAIntrinsics.__threadfence();
```

Memory fences live on the same static class.

**Important:** `__syncthreads()` is a **no-op on the OMP flavor**, because OMP teams don't have CUDA-style block scope. Any reduction kernel that relies on `__syncthreads` between phases needs a separate OMP body — see [reductions.md](reductions.md).

## Atomics

Atomics ride on `System.Threading.Interlocked.*` — the `.builtins` XML (shipped as `hybridizer.cuda.builtins`) maps `Interlocked.Add(ref x, v)` → `atomicAdd`, `Interlocked.Increment` → `atomicInc`, etc.

You can ship a custom `.builtins` next to the `.csproj` and pass it to the CLI to add mappings. This is also the hook for `System.Numerics.Vector<T>` → SIMD intrinsics on AVX flavors.

**Do not use `AtomicExpr.apply`.** The lambda-driven `AtomicExpr.apply(ref target, val, (a, b) => op)` path is flagged buggy. Use intrinsic stubs instead:

```csharp
[IntrinsicInclude("intrinsics.cuh")]
public static class Atomics
{
    [IntrinsicFunction("atomicAdd")]
    public static float Add(ref float target, float val)
    {
        // managed CAS-loop fallback
        ...
    }

    [IntrinsicFunction("atomicMax")]
    public static float Max(ref float target, float val) { ... }
}
```

`atomicAdd(float*, float)` is built into CUDA since compute 2.0 — no header needed. `atomicMax(float*, float)` is **not** built in — ship a CAS-loop implementation in a project-root `intrinsics.cuh` using `__int_as_float` / `__float_as_int` bit reinterpretation, and add `[IntrinsicInclude("intrinsics.cuh")]` to the helper class.

For OMP: `hybridizer.omp.h` defines `atomicAdd` / `atomicMax` as plain non-atomic macros (`#define atomicAdd(p,a) {(*p) += a;}`). These **race** under `#pragma omp parallel`. Reduction kernels must split per flavor and the OMP body must use a sequential or `Parallel.For` shape that doesn't depend on the intrinsics.

## Shared memory

`SharedMemoryAllocator<T>` is the C# façade for dynamic shared memory:

```csharp
var alloc = new SharedMemoryAllocator<float>();
float[] tileA = alloc.allocate(blockDim.x * blockDim.y);
float[] tileB = alloc.allocate(blockDim.x * blockDim.y);
```

Multiple allocations concatenate. **Reserve the total byte count via `SetDistrib(grid, block, shmemBytes)` on the host** — the default 8 KB on `[EntryPoint(SharedSize=...)]` is a hard limit otherwise.

## Device-side `printf`

```csharp
Console.Out.Write($"tid={tid} val={x}");
```

Inside a kernel this transcodes to `printf`. Output is buffered and only flushes on `cuda.DeviceSynchronize()` or stream sync.

## CPU dual-path quirk

The *same* `[EntryPoint]` method is just normal C# when called directly (not via `runner.Wrap(...)`). Samples lean on this for correctness verification: run once on CPU with `Parallel.For`, run once via the GPU wrapper, diff.

The idiomatic kernel body is therefore:

```csharp
Parallel.For(0, N, i => { /* body */ });
```

rather than a manual `tid`-stride loop, because the same body works in both modes.

## Memory-resident objects

`FloatResidentArray`, `ResidentArrayGeneric<T>`, anything implementing `IResidentArray` / `IResidentData` — allocate once on the device, mutate via `RefreshHost()` / `RefreshDevice()`, track sync state via `Status`. This is the right tool for weights and static tensors that survive across many kernel calls (way cheaper than re-marshalling).

**Init gotcha:** see [host-launch.md](host-launch.md) for the eager-DevicePointer + `Status = HostNeedsRefresh` pattern. Skipping it means the first kernel reads uninitialised device memory.

## Manual device memory

When you need full control:

```csharp
cuda.Malloc(out IntPtr p, bytes);
cuda.Memcpy(...);
cuda.MemcpyAsync(..., stream);
cuda.Free(p);
```

Pass the `IntPtr` directly as a kernel arg; Hybridizer treats it as a typed device pointer based on the parameter signature.

## Language features NOT supported in device code

- **`new` for reference types** (heap alloc). Value types are fine. Use `[AllocatableType]` + `hybridizer::allocate<T>()` for the rare cases you need a device heap.
- **`string`**, **`lock`**, **`foreach`** (hidden try/catch), **exceptions** (catch blocks), **recursion**.
- **Generic methods on non-generic types.** Generic *types* / *classes* are fine (template-specialised). Generic *methods* are not — workaround: put the generic method on a generic stub type.
- **Closure-captured lambdas.** Lambdas as kernel parameters work *only* if expressed inline at the call site — no field-stored delegates, no captures of outer locals.

## `MathF.*` aborts the transcoder

The cuda/omp builtins only map `System.Math.*` (the `double` overloads), **not** `MathF.*` (`float`). Calling `MathF.Exp`, `MathF.Sqrt`, etc. inside a kernel aborts transcoding with:

```
[ERROR] : 0X60AC : Cannot get IL for method Exp -- Did you forget an intrinsic or a builtin?
```

The error is silent in MSBuild output (build still "succeeds"), but every `.cu` / `.cpp` file in `generated-{cuda,omp}/` comes out as 24-line boilerplate with no methods. The resulting satellite has zero entry points, and every test calling `wrapped.MyKernel(...)` fails with `EntryPointNotFoundException`.

**Fix:** use `(float)System.Math.X(double)`:

```csharp
// WRONG — aborts transcode:
gate[i] = val * (1.0f / (1.0f + MathF.Exp(-val))) * up[i];

// RIGHT:
gate[i] = val * (1.0f / (1.0f + (float)System.Math.Exp(-val))) * up[i];
```

The double→float cast is a single instruction; nvcc/g++ optimise it inline. No perf cost vs. `MathF` on either flavor.

**Alternative:** decorate a helper with `[IntrinsicFunction("expf")]` + `GenerateSource=true` so CUDA picks the native `expf` and the body still works as managed fallback. More work; only worth it if the double→float cast turns out to be measurably hot, which it shouldn't be.

## Polymorphism in kernels

Virtual calls work via function pointers by default (`-ufp` flag, on by default). For zero-overhead dispatch use `[HybridTemplateConcept]` on the interface + `[HybridRegisterTemplate(Specialize=typeof(Impl))]` to force template specialisation. See `src/6.Advanced/InterfacesReduction` in the basic samples.

## Related

- Attribute reference: [attributes.md](attributes.md)
- Build pipeline: [build-pipeline.md](build-pipeline.md)
- Host launch: [host-launch.md](host-launch.md)
- Reductions (where `__syncthreads` quirks bite): [reductions.md](reductions.md)
- Gotchas digest: [gotchas.md](gotchas.md)
