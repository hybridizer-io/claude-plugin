# Build pipeline

End-to-end flow from `dotnet build` to a runnable satellite DLL.

## Install the Hybridizer CLI

Hybridizer ships as the **`Hybridizer` NuGet tool** (https://www.nuget.org/packages/Hybridizer). Three install shapes:

| Shape | Install command | Invocation | When to use |
|---|---|---|---|
| **Global tool** | `dotnet tool install --global Hybridizer` | `hybridizer` (in `PATH`) | One install, all repos. Easiest for a dev machine. |
| **Local tool** | `dotnet new tool-manifest && dotnet tool install Hybridizer` | `dotnet hybridizer` (after `dotnet tool restore`) | Tool version pinned per-repo via `.config/dotnet-tools.json`. Required for reproducible CI builds. |
| **Project template** | `dotnet new install Hybridizer.App.Template` then `dotnet new hybridizer-app -n MyProject` | `dotnet hybridizer` (template installs the local tool for you) | Starting a fresh project — wires manifest, MSBuild targets, and a sample kernel in one shot. |

The build runtime (CUDA headers, OMP runtime source, builtins XML) ships inside the NuGet package — the tool resolves them automatically. You don't normally need to point `$(HybridizerIncludes)` anywhere; only override if you have a custom build of the transcoder.

## License-gated flavors

Run:

```bash
hybridizer --display-license-details        # global tool
# or
dotnet hybridizer --display-license-details # local tool
```

The output reports which flavors are licensed (subset of CUDA, OMP, HIP, AVX, AVX2, AVX512). Pass only those to `--flavors`; anything else fails at transcode time. Do this once at scaffold and trust it for the session.

## `Directory.Build.props` — declare the invocation once

Put the CLI invocation in a `Directory.Build.props` at the repo root. One source of truth, inherited by every project:

```xml
<Project>
  <PropertyGroup>
    <!-- "hybridizer" for global tool, "dotnet hybridizer" for local tool,
         or an absolute path for a custom standalone build. -->
    <HybridizerTool>hybridizer</HybridizerTool>
  </PropertyGroup>
</Project>
```

Then in each project's `.csproj`, invoke via a plain `<Exec>` — **no custom MSBuild Task, no `UsingTask`, no abstraction layer**. Direct `Exec` keeps the command visible and debuggable; anyone can copy-paste it into a terminal.

**Quoting:** invoke `<Exec Command="$(HybridizerTool) …" />` without quotes around `$(HybridizerTool)` so that `dotnet hybridizer` (which contains a space) is split into two argv entries. If the user picks the "custom path" install (single binary path with no spaces), unquoted still works.

## Per-project pipeline order

Gated by `<CompileCUDA>enable</CompileCUDA>` and/or `<CompileOMP>enable</CompileOMP>`.

1. **`Build`** (standard `csc`) → managed `bin/$(Config)/$(TF)/$(TargetName).dll`. `$(TargetName)` follows `<AssemblyName>` override (e.g. `llama-csharp.dll`, not `LlamaCsharp.dll`). Use `$(TargetName)` in downstream paths, **not** `$(MSBuildProjectName)`.
2. **`DetectCudaVersion`** (`BeforeTargets="GenerateCUDA"`) → runs `nvcc -V`, sets `$(CUDAVersion)`.
3. **`DetectGPUArch`** (`BeforeTargets="CompileCUDA"`) → runs `nvidia-smi --query-gpu=compute_cap | sort -t. -k1,1nr -k2,2nr | head -1`, sets `$(GenCodeArg)` (e.g. `-gencode arch=compute_120,code=sm_120`).
4. **`GenerateCUDA`** and/or **`GenerateOMP`** (`AfterTargets="Build"`, per-project `Exec`) → invokes `$(HybridizerTool)` per flavor into a flavor-specific working directory (`generated-cuda/`, `generated-omp/`).
5. **`CompileCUDA`** (`AfterTargets="GenerateCUDA"`) → `nvcc` on the rollup → `$(OutDir)$(TargetName)_CUDA.dll` (an ELF `.so` on Linux, named `.dll` for `SatelliteLoader` convention).
6. **`CompileOMP`** (`AfterTargets="GenerateOMP"`) → `g++ -fopenmp` on the rollup → `$(OutDir)lib$(TargetName)_OMP.so`.
7. **Runtime:** `SatelliteLoader.LoadCuda()` / `.LoadOmp()` finds the satellite next to the host assembly via glob.

Keep `Directory.Build.targets` for shared targets (CUDA-version detection, GPU-arch detection, the `CompileCUDA`/`CompileOMP` `nvcc`/`g++` steps). The `$(HybridizerTool)` *property* belongs in `Directory.Build.props`; the targets that use it live in `Directory.Build.targets`.

## Verified CLI invocation (full HYBRIDIZE mode)

```
"$(HybridizerTool)" \
    --dll-fullpaths "$(OutDir)Hybridizer.Runtime.CUDAImports.dll;$(OutDir)$(TargetName).dll" \
    --flavors CUDA \
    --working-directory generated-cuda \
    --builtins "$(HybridizerIncludes)/hybridizer.cuda.builtins" \
    --generate-line-info --use-function-pointers
```

Full CLI signature:

```
$(HybridizerTool) \
  --dll-fullpaths <assembly.dll>[;<asm2.dll>] \
  --flavors CUDA[;OMP;HIP;AVX;AVX2;AVX512] \
  --working-directory <out-dir> \
  [--pdb-fullpaths <pdbs>] \
  [--license <file-or-string>] \
  [--generate-line-info] [-ufp] [-uha] \
  [--builtins <files>] \
  [--lines-per-file N]
```

Supported flavors: **CUDA, OMP, HIP, AVX, AVX2, AVX512** (subject to license — see "License-gated flavors" above). There's also a JAVA wrapper-generation mode (`--java`).

**Builtins file differs per flavor:**

- CUDA: `hybridizer.cuda.builtins` (maps `Math.Exp` → `exp`, etc. — CUDA built-ins).
- OMP: `hybridizer.c.builtins` (maps `Math.Exp` → `::exp` — stdlib).

Both live in `$(HybridizerIncludes)`. Pass by absolute path to be safe; the tool's own search rules can be implicit.

## Files emitted per flavor

**`generated-cuda/`** (with `--flavors CUDA`):

- `hybridizer.all.cuda.cu` — **the rollup, `#include`s every per-type `.cu` file.** ⚠ Compile **only this file** with `nvcc`. Compiling both the rollup and the per-type files double-defines every symbol (`multiple definition of HybridizerGetTypeID`, etc.).
- `<Namespace>.<Type>.cu` per type with `[EntryPoint]` methods.
- `hybridizer.dispatch.cuda.cu` — dispatch stubs.
- `hybridizer.includes.cuda.h`, `hybridizer.templates.cuda.h` — headers.
- `hybridizer.{config,base.config}` (XML), `hybridizer.dispatchants.xml`, `hybridizer.typeids.cuda.xml` — metadata.

**`generated-omp/`**: same structure, rollup is `hybridizer.all.omp.cpp`. Compile that one with `g++`.

**Side artifacts at project root** (created by the Hybridizer transcoder): `hybridizer.cuda.def`, `hybridizer.omp.def` — Windows-style symbol export lists. Add `hybridizer.*.def` to `.gitignore`.

## Working `nvcc` invocation (Linux, sm_120)

```
nvcc -O3 -gencode arch=compute_120,code=sm_120 -std=c++17 \
     --expt-relaxed-constexpr -Xcompiler -fPIC \
     -Xcudafe --diag_suppress=144 -Xcudafe --diag_suppress=177 \
     -Xcudafe --diag_suppress=555 -Xcudafe --diag_suppress=128 \
     -I. -I$(HybridizerIncludes) -Igenerated-cuda \
     generated-cuda/hybridizer.all.cuda.cu \
     -shared -o $(OutDir)$(TargetName)_CUDA.dll -lcuda
```

## Working `g++` invocation for OMP

```
g++ -O3 -std=c++17 -fopenmp -fPIC -shared \
    -I. -I$(HybridizerIncludes) -Igenerated-omp \
    generated-omp/hybridizer.all.omp.cpp \
    $(HybridizerIncludes)/../sources/hybridizer.omp.cpp \
    -o $(OutDir)lib$(TargetName)_OMP.so
```

Pulls in `hybridizer.omp.cpp` from the install's `sources/` directory. There are four benign warnings in the runtime headers (`-Wreturn-type` + `-Wformat-security`). Ignore.

## C# source-side gotchas

- **`[In]` / `[Out]` parameter attributes come from `System.Runtime.InteropServices`**, not `Hybridizer.Runtime.CUDAImports`. Add `using System.Runtime.InteropServices;` to any kernel file.
- **`HybRunner.Wrap(instance)` needs an instance** — a class containing `[EntryPoint]` methods must be a regular class, **not `static class`**. The instance is just a vessel for static-method dispatch; its fields don't have to be meaningful.
- For projects with an `<AssemblyName>` override, use **`$(TargetName)`** (the AssemblyName) in MSBuild output paths, not `$(MSBuildProjectName)`.

## Test-output satellite copy

For projects whose unit tests need to `dlopen` the satellite:

```xml
<Target Name="CopyHybridizerSatellites" AfterTargets="Build">
  <ItemGroup>
    <_Sat Include="..\bin\$(Configuration)\$(TargetFramework)\*_CUDA.dll" />
    <_Sat Include="..\bin\$(Configuration)\$(TargetFramework)\lib*_OMP.so" />
  </ItemGroup>
  <Copy SourceFiles="@(_Sat)" DestinationFolder="$(OutDir)" SkipUnchangedFiles="true" />
</Target>
```

Without this, `SatelliteLoader` (which searches `Assembly.GetExecutingAssembly().Location`'s directory) won't find the satellite when running under `dotnet test`, because the test host copies the managed DLL but not its natives.

## Related

- Host-side launch idiom: [host-launch.md](host-launch.md)
- Device-side language subset: [device-code.md](device-code.md)
- Profiling the resulting binary: [perf-tuning.md](perf-tuning.md)
