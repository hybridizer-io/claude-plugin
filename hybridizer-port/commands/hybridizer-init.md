---
name: hybridizer-init
description: Scaffold Hybridizer into an existing C# project — Directory.Build.props, Directory.Build.targets, a hello-world [EntryPoint], and the SatelliteLoader helper. Use this at step 4 of the migration methodology, after tests and managed profiling are done.
---

Scaffold Hybridizer into the current C# project so the build chain compiles a satellite end-to-end before any real kernel is written. This is step 4 of the migration methodology — the smoke-test infrastructure that proves the round-trip works.

If `$ARGUMENTS` is non-empty, treat it as the target project name (the `.csproj` filename without extension). Otherwise, locate the most likely project: a single `.csproj` in the working tree, or ask the user to pick if there are several.

## What to do

1. **Locate or install the Hybridizer CLI.** Two editions exist:
   - **Free NuGet `Hybridizer` tool** — CUDA only, emits cubin + cpp/cu wrapper. Same profiling/debugging features as paid.
   - **Paid standalone** — all flavors (CUDA / OMP / HIP / AVX / AVX2 / AVX512), emits readable `.cu` / `.cpp` source.

   Detect in this order:
   - `which hybridizer` → global free tool. Use the invocation `hybridizer`.
   - `.config/dotnet-tools.json` lists `hybridizer` → local free tool. Use `dotnet hybridizer` (run `dotnet tool restore` first if needed).
   - Neither → **ask the user** which install path to take:
     - **Quickest, CUDA-only** — `dotnet new install Hybridizer.App.Template` then `dotnet new hybridizer-app -n <Project>`. Generates a project with the free tool manifest, MSBuild targets, and a sample kernel already wired. Best when CUDA is the only target and source-inspection isn't needed.
     - **Free global tool** — `dotnet tool install --global Hybridizer`. CUDA-only, `hybridizer` in `PATH`.
     - **Free local tool** — `dotnet new tool-manifest && dotnet tool install Hybridizer`. CUDA-only, `dotnet hybridizer`. Reproducible CI builds.
     - **Paid standalone** — user has a `Hybridizer.Application` binary from a paid license install. They give us the absolute path. Required for OMP / HIP / AVX flavors or for inspecting/modifying the generated CUDA source.

   Record the resolved invocation (e.g. `hybridizer` / `dotnet hybridizer` / `/path/to/Hybridizer.Application`) — this becomes the value of `$(HybridizerTool)` in step 3.

2. **Read the license to learn supported flavors:**

   ```bash
   <invocation-from-step-1> --display-license-details
   ```

   Note the licensed flavors (CUDA, OMP, HIP, AVX, AVX2, AVX512). Only scaffold targets for flavors the user has licensed AND wants to use. If only CUDA is listed, the user is on the free tool (or a paid CUDA-only license) — that's fine for a CUDA-only scaffold. If OMP / HIP / AVX appear, the user is on the paid standalone and you can wire those flavors.

   Also check the host toolchain:
   - `nvcc -V` succeeds (skip if CUDA isn't licensed/wanted).
   - `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` reports a compute capability (skip if CUDA isn't licensed/wanted).
   - `g++ -fopenmp -dM -E -x c++ - < /dev/null | grep _OPENMP` confirms OMP support (skip if OMP isn't licensed/wanted).

   If a required host tool is missing, stop and report. Don't scaffold against a broken toolchain.

3. **Create `Directory.Build.props` at the repo root** if absent. It declares the Hybridizer invocation as MSBuild properties, inherited by every project:

   ```xml
   <Project>
     <PropertyGroup>
       <!-- One of: "hybridizer" (global tool), "dotnet hybridizer" (local tool),
            or an absolute path to a Hybridizer.Application binary. -->
       <HybridizerTool>hybridizer</HybridizerTool>
       <!-- Optional: only set if the user has a custom includes directory.
            The NuGet tool resolves includes automatically; leave HybridizerIncludes
            unset and let the tool find them, unless the user reports include errors. -->
     </PropertyGroup>
   </Project>
   ```

   If `Directory.Build.props` already exists, merge in `<HybridizerTool>` rather than overwriting.

4. **Create `Directory.Build.targets` at the repo root** with the shared `DetectCudaVersion` / `DetectGPUArch` / `CompileCUDA` / `CompileOMP` targets. See `hybridizer-port/skills/hybridizer-port/references/build-pipeline.md` for the verified target bodies. The pattern auto-detects `$(CUDAVersion)` and `$(GenCodeArg)` so each project doesn't have to.

5. **Edit the target `.csproj`:**
   - Add `<CompileCUDA>enable</CompileCUDA>` (and/or `<CompileOMP>enable</CompileOMP>`) to a `<PropertyGroup>` — only for flavors confirmed licensed in step 2.
   - Add `<PackageReference Include="Hybridizer.Runtime.CUDAImports" Version="3.4.0" />`.
   - Add a `GenerateCUDA` `<Target>` that `<Exec>`s `$(HybridizerTool)` with the correct flags (no quotes around `$(HybridizerTool)` — it might be `dotnet hybridizer` which needs to split on the space).
   - If the project has `<AssemblyName>`, use `$(TargetName)` in output paths, not `$(MSBuildProjectName)`.

6. **Add `SatelliteLoader.cs`** to the project. If the user installed via `Hybridizer.App.Template`, a working copy is already in the generated project — reuse it. Otherwise grab one from the hybridizer-basic-samples repo (`src/0.Utils/Utilities/SatelliteLoader.cs` in https://github.com/hybridizer-io/hybridizer-basic-samples) or inline the body from [host-launch.md](../skills/hybridizer-port/references/host-launch.md).

7. **Add a hello-world `[EntryPoint]`** — a vector-add or sum kernel — plus a `Main` that calls it with `runner.Wrap(...)` and asserts the output. Keep the surface minimal; the goal is to prove the round-trip, not demonstrate features. Use `Parallel.For` in the body so the same code works under managed dispatch and OMP. If the user installed via the template, an example kernel is already in the generated `Program.cs`; adapt that instead of inventing a new one.

8. **Add to `.gitignore`:**
   - `generated-cuda/`
   - `generated-omp/`
   - `hybridizer.*.def`
   - `bin/` and `obj/` if not already.

9. **Build:**
   ```
   dotnet build -c Release
   ```
   Confirm `bin/<config>/<tfm>/<projname>_CUDA.dll` (and/or `lib<projname>_OMP.so`) appears.

10. **Run the hello-world** and confirm it prints the expected output.

11. **Report:**
    - Files added / modified.
    - Hybridizer invocation chosen (`hybridizer` / `dotnet hybridizer` / custom path).
    - Licensed flavors reported by `--display-license-details`.
    - Whether CUDA, OMP, or both were enabled.
    - Whether the smoke-test kernel produced correct output.
    - Next step per the methodology: "port your first real kernel (step 5)."

## Don't

- Scaffold if there's already a working Hybridizer integration — instead inspect it, report findings, and ask what the user actually wants changed.
- Scaffold a flavor the license doesn't cover. `--display-license-details` is the source of truth.
- Add a custom MSBuild `<UsingTask>` for invoking Hybridizer. A plain `<Exec>` is the maintained pattern; custom Tasks add maintenance surface for no benefit.
- Promise the user they can read the generated `.cu` if they're on the free NuGet tool — they'll get a cubin wrapper, not source. Source emission requires the paid standalone.

## Reference

- `hybridizer-port/skills/hybridizer-port/references/build-pipeline.md` — full target bodies, CLI flags, `nvcc` and `g++` invocations.
- `hybridizer-port/skills/hybridizer-port/references/host-launch.md` — for the `SatelliteLoader` + `HybRunner` smoke-test pattern.
- `hybridizer-port/skills/hybridizer-port/references/methodology.md` — for the step-4 positioning.
