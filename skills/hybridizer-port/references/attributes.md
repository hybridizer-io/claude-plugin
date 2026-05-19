# Attributes

All attributes live in `Hybridizer.Runtime.CUDAImports` (`/mnt/d/hybridizer-runtime-cudaimports/src/API.cs` in a typical install).

## Kernel markers

- **`[Kernel]`** — device-callable function (not a host entry-point). Required on interface methods and on every implementation when using virtual dispatch in kernels.
- **`[EntryPoint]`** — extends `[Kernel]`; the host-launchable entry. Properties:
  - `SharedSize` — default 8192 bytes. Hard cap unless overridden via host-side `SetDistrib(grid, block, shmemBytes)`.
  - `Nvrtc` — JIT compile flag.
  - `Name` — override the mangled native name (useful when wrapping the kernel through `NativeLibrary.GetExport`).
- **`[HybridizerIgnore]`** — exclude a member from transcoding. Can target a specific flavor only: `[HybridizerIgnore("OMP")]` or `[HybridizerIgnore("CUDA")]`. Use for per-flavor `[EntryPoint]` splits when one body doesn't lower to the other backend.

## Native-bridge attributes

- **`[IntrinsicFunction("::cosf")]`** — C# stub that maps to a native function in the emitted source.
  - `IsNaked=true` — the function has no memory side effects (enables more aggressive optimisation).
  - `GenerateSource=true` — also emit the C# body, so OMP/CPU fallback works without a separate intrinsic mapping.
- **`[IntrinsicType("half2")]`** — struct/class wraps a native type.
  - `HandleAsValueType`, `NotVectorizable`, `VectorizedType=typeof(...)` — modifiers controlling marshalling and AVX SIMD form.
- **`[IntrinsicConstant("__hybridizer_threadIdxX")]`** — compile-time constant binding. This is how `threadIdx.x` works: it's a static property under the hood, bound at codegen.
- **`[IntrinsicInclude("my_header.cuh")]`** — force inclusion of a custom header in the generated source. `nvcc` finds it via `-I.` on the command line. Use this when you ship hand-written `extern "C"` wrappers or CAS-loop atomic implementations.
- **`[AllocatableType("my_allocator")]`** — the type is allocatable via `hybridizer::allocate<T>()` in device code. Rare; only needed when you need a device heap.

## Memory / perf hints on parameters or fields

- **`[In]`** — host→device only, skip readback.
- **`[Out]`** — device→host only, skip the upload.
- **(no attribute)** — both directions, default.
- **`[ReadOnly]`**, **`[WriteOnly]`** — finer access hints.
- **`[HybridConstant(Location = ConstantLocation.ConstantMemory)]`** — place static field in CUDA `__constant__` memory. The initializer is captured at codegen time, **runtime mutation does not propagate**. Use for small stable LUTs/stencils.
- **`[LaunchBounds(maxThreads, minBlocksPerMulti)]`** — emits `__launch_bounds__` on the kernel.

## Polymorphism without vtables

Virtual calls in kernels work via function pointers by default (the `-ufp` CLI flag is on by default). For zero-overhead dispatch:

- **`[HybridTemplateConcept]`** on the interface — tells Hybridizer to specialise via C++ templates rather than function pointers.
- **`[HybridRegisterTemplate(Specialize = typeof(Impl))]`** on the type using the interface — enumerates concrete instantiations to emit.

This is how to keep virtual calls in kernels without paying for indirect branches. See `src/6.Advanced/InterfacesReduction` in the basic samples.

## Misc

- **`[ICustomMarshalledSize(bytes)]`** plus implementing `ICustomMarshalled` — custom marshalling for types whose native layout differs from managed.
- **`[HybRunnerDefaultSatelliteNameAttribute("Foo_CUDA.dll")]`** (assembly-level) — lets parameterless `new HybRunner()` find the satellite without `SatelliteLoader.Load()`.

## Related

- What runs on the device: [device-code.md](device-code.md)
- How attributes flow through the build: [build-pipeline.md](build-pipeline.md)
- How `[EntryPoint]` methods get launched: [host-launch.md](host-launch.md)
