---
name: hybridizer-init
description: Scaffold Hybridizer into an existing C# project — Directory.Build.props, Directory.Build.targets, a hello-world [EntryPoint], and the SatelliteLoader helper. Use this at step 4 of the migration methodology, after tests and managed profiling are done.
---

Scaffold Hybridizer into the current C# project so the build chain compiles a satellite end-to-end before any real kernel is written. This is step 4 of the migration methodology — the smoke-test infrastructure that proves the round-trip works.

If `$ARGUMENTS` is non-empty, treat it as the target project name (the `.csproj` filename without extension). Otherwise, locate the most likely project: a single `.csproj` in the working tree, or ask the user to pick if there are several.

## What to do

1. **Verify prerequisites:**
   - `Hybridizer.Application` reachable at the path declared in `$(HybridizerTool)` (default `/mnt/d/hybridizer-software-suite/publish/MAIN/Hybridizer.Application`).
   - `nvcc -V` succeeds (skip if the user only wants OMP).
   - `nvidia-smi` reports a compute capability (skip if OMP-only).
   - If any prerequisite is missing, stop and report. Don't scaffold against a broken toolchain.

2. **Create `Directory.Build.props` at the repo root** if absent. It declares the tool paths as MSBuild properties, inherited by every project:

   ```xml
   <Project>
     <PropertyGroup>
       <HybridizerTool>/mnt/d/hybridizer-software-suite/publish/MAIN/Hybridizer.Application</HybridizerTool>
       <HybridizerIncludes>/mnt/d/hybridizer-software-suite/publish/MAIN/includes</HybridizerIncludes>
     </PropertyGroup>
   </Project>
   ```

   If it already exists, merge the two properties in rather than overwriting.

3. **Create `Directory.Build.targets` at the repo root** with the shared `DetectCudaVersion` / `DetectGPUArch` / `CompileCUDA` / `CompileOMP` targets. See `skills/hybridizer-port/references/build-pipeline.md` for the verified target bodies. The pattern auto-detects `$(CUDAVersion)` and `$(GenCodeArg)` so each project doesn't have to.

4. **Edit the target `.csproj`:**
   - Add `<CompileCUDA>enable</CompileCUDA>` (and/or `<CompileOMP>enable</CompileOMP>`) to a `<PropertyGroup>`.
   - Add `<PackageReference Include="Hybridizer.Runtime.CUDAImports" Version="3.4.0" />`.
   - Add a `GenerateCUDA` `<Target>` that `<Exec>`s `"$(HybridizerTool)"` with the correct flags. Do **not** include the BASIC-edition flags (`--jit-cuda-version`, `--jit-compil-options`, `--additional-jit-headers`, `--nvrtc`).
   - If the project has `<AssemblyName>`, use `$(TargetName)` in output paths, not `$(MSBuildProjectName)`.

5. **Add `SatelliteLoader.cs`** to the project (copy from `/mnt/d/hybridizer-basic-samples/src/0.Utils/Utilities/SatelliteLoader.cs` if available, otherwise inline the body).

6. **Add a hello-world `[EntryPoint]`** — a vector-add or sum kernel — plus a `Main` that calls it with `runner.Wrap(...)` and asserts the output. Keep the surface minimal; the goal is to prove the round-trip, not demonstrate features. Use `Parallel.For` in the body so the same code works under managed dispatch and OMP.

7. **Add to `.gitignore`:**
   - `generated-cuda/`
   - `generated-omp/`
   - `hybridizer.*.def`
   - `bin/` and `obj/` if not already.

8. **Build:**
   ```
   dotnet build -c Release
   ```
   Confirm `bin/<config>/<tfm>/<projname>_CUDA.dll` (and/or `lib<projname>_OMP.so`) appears.

9. **Run the hello-world** and confirm it prints the expected output.

10. **Report:**
    - Files added / modified.
    - Whether CUDA, OMP, or both were enabled.
    - Whether the smoke-test kernel produced correct output.
    - Next step per the methodology: "port your first real kernel (step 5)."

## Don't

- Scaffold if there's already a working Hybridizer integration — instead inspect it, report findings, and ask what the user actually wants changed.
- Use the BASIC `hybridizer` dotnet global tool.
- Add a custom MSBuild `<UsingTask>` for invoking Hybridizer. A plain `<Exec>` is the maintained pattern; custom Tasks add maintenance surface for no benefit.

## Reference

- `skills/hybridizer-port/references/build-pipeline.md` — full target bodies, CLI flags, `nvcc` and `g++` invocations.
- `skills/hybridizer-port/references/host-launch.md` — for the `SatelliteLoader` + `HybRunner` smoke-test pattern.
- `skills/hybridizer-port/references/methodology.md` — for the step-4 positioning.
