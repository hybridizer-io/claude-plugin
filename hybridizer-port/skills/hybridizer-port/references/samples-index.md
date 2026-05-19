# Samples & external references

Where to find authoritative examples and documentation. Read the matching sample first when starting on a new pattern.

## Hybridizer install (NuGet)

Hybridizer ships as the `Hybridizer` NuGet tool. Pick the install shape that matches the workflow:

| Shape | Install | Invocation |
|---|---|---|
| Global tool | `dotnet tool install --global Hybridizer` | `hybridizer` |
| Local tool | `dotnet new tool-manifest && dotnet tool install Hybridizer` | `dotnet hybridizer` |
| Template | `dotnet new install Hybridizer.App.Template && dotnet new hybridizer-app -n MyProject` | `dotnet hybridizer` (manifest is generated) |

Runtime files (includes, OMP runtime source, builtins XML) ship inside the NuGet package; the tool resolves them automatically. Run `hybridizer --display-license-details` to see which flavors are licensed before scaffolding.

Host toolchain: `nvcc` 13.x at `/usr/local/cuda/bin/nvcc` is the verified CUDA chain; `g++ -fopenmp` for OMP.

## Basic samples repo

Available at https://github.com/hybridizer-io/hybridizer-basic-samples — the source of truth for Hybridizer-idiomatic project structure and the canonical patterns.

| Sample | Teaches |
|---|---|
| `src/1.Simple/Builtin` | Minimal `[EntryPoint]` + `SatelliteLoader.Load()` + builtins file |
| `src/1.Simple/Reduction` | Hardcoded reduction, canonical shape-B skeleton |
| `src/3.Maths/NaiveMatrix` | Matmul with explicit `threadIdx.x` / `blockIdx.x`, `[In]` / `[Out]` annotations, `SetDistrib` for 2-D grid |
| `src/6.Advanced/LambdaReduction` | `Func<float,float,float>` operator parameter at the `[EntryPoint]` call site |
| `src/6.Advanced/InterfacesReduction` | `ILocalReductor` instance arg; `[Kernel]` on interface + impls; virtual calls via function pointers |
| `src/6.Advanced/GenericReduction` | `[HybridTemplateConcept]` + struct reductors + `[HybridRegisterTemplate]` — zero-overhead dispatch. Has the canonical `Atomics` helper with `[IntrinsicFunction("atomicAdd")]` / `("atomicMax")` stubs |
| `src/6.Advanced/GenericFunctions` | User-defined generic `Vector` class (not `System.Numerics.Vector<T>`) |

The `Directory.Build.targets` shipped with the samples does the `nvcc -V` → `$(CUDAVersion)` and `nvidia-smi --query-gpu=compute_cap` → `$(GenCodeArg)` auto-detection that any real project also wants. Copy it (or its substance) and adapt.

## Hybridizer GitHub wiki

The authoritative external documentation:

→ **https://github.com/hybridizer-io/hybridizer-basic-samples/wiki**

The wiki has been the canonical reference for attribute semantics, builtins file format, and runtime API. Check it for anything not covered in this skill — and if you find a gap between the wiki and what's documented here, prefer the wiki and consider opening an issue against this plugin.

## NuGet runtime package

- `Hybridizer.Runtime.CUDAImports` v3.4.0 — the runtime API: attributes, `cuda.*` bindings, `FloatResidentArray`, `cooperative_groups`, `CUDAIntrinsics`. Inspect the NuGet package contents to read the API surface.

## Builtins XML

Maps .NET methods / types to native equivalents.

- `hybridizer.cuda.builtins` — CUDA flavor (maps `Math.Exp` → `exp`, `Interlocked.Add(ref int, int)` → `atomicAdd`, etc.).
- `hybridizer.c.builtins` — OMP/AVX flavors (maps to stdlib).

You can ship a custom `.builtins` next to the `.csproj` and pass it via `--builtins`. This is the mechanism that satisfies `NumericsVectorMustBeBuiltin` — for `System.Numerics.Vector<T>` to transpile, its methods must be mapped here. CUDA prefers scalar (SIMT), so `Vector<T>` builtins mostly matter for AVX flavors.

## Verifying the install

Quick checks before invoking the toolchain:

```bash
# Hybridizer transcoder present and licensed?
hybridizer --display-license-details                # global tool
# or
dotnet hybridizer --display-license-details         # local tool

# nvcc + driver versions (CUDA flavor)
nvcc -V
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# OMP support (OMP flavor)
g++ -fopenmp -dM -E -x c++ - < /dev/null | grep _OPENMP
```

## Related

- How everything above hooks into MSBuild: [build-pipeline.md](build-pipeline.md)
- Attribute reference: [attributes.md](attributes.md)
- The host launch idiom from the samples, in detail: [host-launch.md](host-launch.md)
