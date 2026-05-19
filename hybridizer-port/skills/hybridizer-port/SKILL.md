---
name: hybridizer-port
description: Migration assistant for porting C# code to CUDA via Hybridizer. Knows the attribute set, build pipeline, device-code restrictions, host launch idioms, kernel patterns (reductions, cooperative blocks, warp shuffle, CUB), CUDA graph capture, performance tuning, and a curated list of gotchas to avoid.
when_to_use: When the user is porting C# code to GPU with Hybridizer, debugging Hybridizer transcode/build errors, writing or reviewing kernels marked [Kernel]/[EntryPoint]/[IntrinsicFunction], integrating the `Hybridizer` NuGet tool into a build, working with HybRunner / SatelliteLoader / FloatResidentArray, or profiling CUDA performance from a Hybridizer-built binary.
allowed-tools: Read, Glob, Grep, Bash(dotnet *), Bash(nvcc *), Bash(nvidia-smi *), Bash(git status), Bash(git diff *), Bash(git log *)
---

# Hybridizer port

Helps migrate existing C# code to CUDA using Hybridizer. The skill encodes the moving parts of a real port: attributes, build pipeline, what does and doesn't compile to device code, how to launch from C#, kernel patterns that actually perform, and the bugs to dodge.

## How to use this skill

This file is the index. For any specific topic, read the matching file under `references/`. Don't try to memorise everything up front — load the page you need, when you need it.

| Need to know... | Read |
|---|---|
| The 8-step migration plan (tests → profile → refactor → infra → first kernel → cut memcpies → GPU profile → tune) | [references/methodology.md](references/methodology.md) |
| Attributes: `[Kernel]`, `[EntryPoint]`, `[IntrinsicFunction]`, `[HybridTemplateConcept]`, `[HybridConstant]`, `[In]`/`[Out]`, `[LaunchBounds]`, `[HybridizerIgnore]` | [references/attributes.md](references/attributes.md) |
| What runs on the GPU: language subset, `threadIdx`/`__syncthreads`, `SharedMemoryAllocator`, atomic intrinsics, missing `MathF`/`Half` builtins | [references/device-code.md](references/device-code.md) |
| Build pipeline: NuGet `Hybridizer` tool install shapes, `--display-license-details`, `$(HybridizerTool)` property, MSBuild target order, flavors (CUDA/OMP/HIP/AVX*) | [references/build-pipeline.md](references/build-pipeline.md) |
| Host launch: `HybRunner`, `SatelliteLoader.Load`, `SetDistrib`/`Wrap`, streams, `FloatResidentArray` init | [references/host-launch.md](references/host-launch.md) |
| Kernel patterns: cooperative blocks, warp shuffle, CUB BlockReduce, quantized split layout, per-flavor `[EntryPoint]` split | [references/kernel-patterns.md](references/kernel-patterns.md) |
| Reduction skeleton + 4 op-delivery styles + why OMP needs a separate body | [references/reductions.md](references/reductions.md) |
| CUDA graph capture: when to use it, capture mode, pre-warm, legacy↔capture dependency rules | [references/graph-capture.md](references/graph-capture.md) |
| Performance tuning: `DllImport` bypass for hot kernels, `extern "C"` wrappers in `intrinsics.cuh`, memcpy reduction strategy, profiling workflow | [references/perf-tuning.md](references/perf-tuning.md) |
| Bugs to avoid: `AtomicExpr` is buggy, `MathF.*` aborts transcode, OMP wrapper race, 4-rows-per-block trap, xUnit parallel kills NativeSerializer, WSL2 profiler wedge, stale satellite | [references/gotchas.md](references/gotchas.md) |
| Where to find samples + wiki + intrinsic catalogue | [references/samples-index.md](references/samples-index.md) |

## Slash commands

- `/hybridizer-init` — scaffold `Directory.Build.props` with `$(HybridizerTool)` and wire a hello-world kernel into an existing C# project.
- `/hybridizer-port <file-or-method>` — propose a kernel for a target function (delegates to `hybridizer-porter` subagent).
- `/hybridizer-review` — diff pending changes against the gotchas checklist (delegates to `hybridizer-reviewer` subagent).
- `/hybridizer-profile` — set up profiling: in-process on WSL, `nsys` on Windows.
- `/hybridizer-log on|off` — toggle the opt-in porting log (appends every prompt to `porting-to-hybridizer.md`).

## Subagents

- **hybridizer-porter** — analyses a C# function and proposes a kernel shape (parallelism, atomics, shared memory).
- **hybridizer-reviewer** — checks a candidate kernel for the known gotchas before you spend time compiling.
- **hybridizer-builder** — discovers the Hybridizer CLI (global tool / local tool / custom path), reads `--display-license-details` for supported flavors, invokes the transcoder correctly, and parses its output into actionable errors.

## House rules (read these even if you skip everything else)

1. **`MathF.*` does not exist in Hybridizer's builtins.** Use `(float)System.Math.X(double)` in kernels. `MathF.*` aborts the transcoder with error `0X60AC` and emits empty `.cu` files.
2. **Do not use `AtomicExpr.apply`** — flagged buggy by the maintainer. Use `[IntrinsicFunction("atomicAdd")]` / `("atomicMax")` stubs over a static `Atomics` class.
3. **Pick the Hybridizer edition that fits the task, and let `--display-license-details` decide the flavors.** The free `Hybridizer` NuGet tool is CUDA-only and emits cubin + wrapper (full profiling/debugging — *not* a stripped-down variant). The paid standalone supports all flavors (CUDA / OMP / HIP / AVX / AVX2 / AVX512) and emits readable `.cu` / `.cpp` source. Set `$(HybridizerTool)` in `Directory.Build.props` (`hybridizer` / `dotnet hybridizer` / absolute path) and invoke via a plain `<Exec>`. Only pass `--flavors` the license reports as available.
4. **Initialise `FloatResidentArray` eagerly** — touch `DevicePointer` and set `Status = HostNeedsRefresh` before the first kernel writes to it. Otherwise the kernel reads uninitialised device memory.
5. **Perf regressions between steps 3 and 6 are expected** — only investigate *functional* regressions during refactor. Perf comes back at steps 7 and 8.
6. **Garbled CUDA output is usually a stale `.dll` satellite.** Clean rebuild before reading the kernel.
7. **`nsys`/`compute-sanitizer` wedge `dotnet` on WSL2.** Profile on native Windows or use the in-process profiler.
8. **Disable xUnit parallelism for Hybridizer tests** — concurrent kernel launches corrupt `NativeSerializer`'s internal dictionary.
