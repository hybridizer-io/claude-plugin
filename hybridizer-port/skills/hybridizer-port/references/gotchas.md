# Gotchas

Hard-won failure modes — each section names the symptom, the cause, and the fix. Cross-references point to fuller treatments elsewhere in this skill.

## `MathF.*` aborts the transcoder

**Symptom:** `EntryPointNotFoundException : Unable to find an entry point named '?' in shared library`. Build "succeeds" but every `.cu` / `.cpp` in `generated-{cuda,omp}/` is 24-line boilerplate.

**Cause:** Hybridizer's `.builtins` files only map `System.Math.*` (the `double` overloads). `MathF.Exp`, `MathF.Sqrt`, etc. inside a kernel body abort transcoding with `0X60AC : Cannot get IL for method Exp -- Did you forget an intrinsic or a builtin?`.

**Fix:** `(float)System.Math.X(double)`. The cast is a single instruction; no perf cost vs. `MathF`.

**Diagnostic:** when a kernel suddenly fails with `EntryPointNotFoundException`:

1. Check `generated-cuda/<ClassName>.cu` is bigger than ~24 lines.
2. If empty, run `$(HybridizerTool)` manually and grep stderr for `0X60AC`.
3. The error includes the source line — look for `MathF.*` there.

See [device-code.md](device-code.md) for the full treatment.

## `AtomicExpr.apply` is buggy — use intrinsic stubs

**Rule:** do not use `Hybridizer.Runtime.CUDAImports.AtomicExpr.apply(ref target, val, (a, b) => op)` in any new kernel.

**Cause:** the lambda-driven path is flagged buggy by the Hybridizer maintainer.

**Fix:** local `Atomics` helper with `[IntrinsicFunction("atomicAdd")]` / `("atomicMax")` stubs and managed CAS-loop fallbacks. Canonical pattern is in `basic-samples/src/6.Advanced/GenericReduction/Program.cs`. See [device-code.md](device-code.md).

## OMP wrapper replicates the whole body across the team

**Symptom:** OMP kernel produces wrong numeric output where the CUDA equivalent is correct. Common shape: output looks like `1/size`.

**Cause:** the generated OMP `_ExternCWrapper_OMP` is `#pragma omp parallel { kernel(); }` — **every OMP thread runs the body in full**. A sequential `for` over a mutable array stomps; multiple threads read/modify/write the same `x[i]`.

**Fix:** use `Parallel.For` for any in-place mutation phase (lowered to `#pragma omp parallel for` with distributed iterations). Sequential `for` is OK for read-only-plus-write-single-slot patterns (race-but-idempotent). See [reductions.md](reductions.md).

## `FloatResidentArray` first-write reads uninitialised device memory

**Symptom:** downstream output collapses to a constant value (e.g. argmax tokens all become 0 or another constant). No CUDA error.

**Cause:** a freshly constructed `FloatResidentArray` has `Status = NoAction`. The marshaller doesn't refresh, but it does lazily allocate `DevicePointer` via `cuda.Malloc`, returning **uninitialised** device memory. For arrays the kernel writes first, this looks-like-it-works (kernel writes, the writes persist) but races the lazy alloc with the launch in subtle ways.

**Fix:** for kernel-writes-first arrays (KV caches, scratch, output activations):

```csharp
_resident = new FloatResidentArray(size);
_ = _resident.DevicePointer;                                // force allocation now
_resident.Status = ResidentArrayStatus.HostNeedsRefresh;    // device is the truth
```

For host-fills-first arrays (LUTs, RoPE tables): bulk-copy via `HostPointer` and set `Status = DeviceNeedsRefresh`.

See [host-launch.md](host-launch.md).

## Anti-pattern: K rows per block in matvec

**Symptom:** A "share the inner-loop load across K rows" matvec regresses tok/s (real measurement: 91 → 75 tok/s with K=4).

**Cause:** block-count collapse on small matrices + register pressure halving occupancy on big ones + bounds-check branches preventing MAC fusion. Adaptive 1-vs-K dispatch only partially recovers.

**Prefer instead:** warp-shuffle reduction (no register pressure change), kernel fusion across matvecs that share an input vector, or persistent-block / cooperative-groups graph-launch tricks. See [kernel-patterns.md](kernel-patterns.md).

## Disable xUnit parallelism for Hybridizer tests

**Symptom:** `System.InvalidOperationException : Operations that change non-concurrent collections must have exclusive access. A concurrent update was performed on this collection and corrupted its state.` deep in `NativeSerializer.VisitObject` → `Dictionary.FindValue`.

**Cause:** `HybRunner`'s wrapped-runner singleton holds shared `NativeSerializer` state that isn't thread-safe. Concurrent kernel-using tests corrupt the internal dictionary.

**Fix:** `xunit.runner.json` with

```json
{
  "parallelizeAssembly": false,
  "parallelizeTestCollections": false,
  "maxParallelThreads": 1
}
```

Wire it into the test project via `<None Update="xunit.runner.json"><CopyToOutputDirectory>PreserveNewest</CopyToOutputDirectory></None>`. Don't switch back to parallel without a thread-safe wrapper around the `HybRunner` singleton.

## Free NuGet tool vs paid standalone — pick the one that fits the task

**Symptom (only relevant if the user wants to inspect/modify generated CUDA):** `hybridizer.generated.cpp` starts with `char __hybridizer_cubin_module_data[]`. You can't read the kernel as `.cu` source; you got a wrapper around a cubin blob.

**Cause:** the free `Hybridizer` NuGet tool emits cubin + wrapper. That's its design; it's not a bug or a misconfiguration. The free tool still supports full profiling, debugging, and `#line` directives — the only thing it doesn't do is emit readable source.

**Fix:** if the user needs source emission (or any of OMP / HIP / AVX flavors), they need the paid standalone `Hybridizer.Application`. There's no flag on the free tool to turn this on. If the user only needs CUDA execution and doesn't care about reading the generated source, the free tool is fully sufficient.

See [build-pipeline.md](build-pipeline.md) "Install the Hybridizer CLI" for the full free-vs-paid breakdown.

## Stale satellite — clean rebuild before debugging the kernel

**Symptom:** A CUDA kernel that "should work" produces obviously wrong output after a code change.

**Cause:** the loaded `_CUDA.dll` (or `lib*_OMP.so`) satellite doesn't match the `.cs` source. Shared `bin/` directories between WSL and Windows are a common source of mismatch — a Linux ELF named `*_CUDA.dll` and a Windows DLL of the same name can stomp each other.

**Diagnostic protocol:** first time a recently-changed kernel produces garbled output:

1. **Clean rebuild.** `dotnet clean && dotnet build -c Release`, or `rm -rf bin obj && dotnet build`.
2. **Confirm the satellite mtime is fresher than the `.cs`.** `ls -la bin/Release/net8.0/*_CUDA.dll` vs the source.
3. **Only THEN read the kernel for actual bugs.** Most "garbled output" reports turn out to be stale builds.

**Bonus rule:** if `compute-sanitizer` (Windows) reports 0 errors on the broken-looking output, the kernel is provably memory- and race-correct under the test inputs. The issue is elsewhere (stale build, dispatcher misroute, etc.).

## WSL2 CUDA profiler wedge

**Symptom:** `dotnet run -- ... --backend cuda` exits 139 (SIGSEGV) with zero stdout/stderr. `nsys profile` report SQLite is "SKIPPED: does not contain CUDA trace data". The crash is in `libnvidia-ptxjitcompiler::__strlen_evex` at addr `0x1` (visible in `dmesg` and via `dotnet --DbgEnableMiniDump=1` coredumps).

**Cause:** the dotnet test/profile host loads multiple library contexts when instrumented; one of them tickles a stale-pointer bug in the WSL2 NVIDIA driver's PTX JIT compiler. The bug applies to **plain `dotnet run --backend cuda` as well as `nsys` / `compute-sanitizer`** on at least one CUDA 13 / driver 596.36 WSL2 setup.

**Workarounds:**

1. **Run on native Windows.** PowerShell or `cmd.exe` with the same `dotnet run` / `nsys profile` command works cleanly.
2. **In-process profiler** when you only need ranking (not absolute times) — a `LLAMA_KERNEL_PROFILE=1`-style env var hooked to a `CudaInvoke.Profile(name, err)` helper that wraps each kernel in `cudaDeviceSynchronize` + timing. Forces 2–3× slowdown during profiling but gives correct relative cost. Only works on a host where CUDA itself runs.

**How to spot it in a future session:** if `dotnet run --backend cuda` exits 139 with empty output on WSL2, that's the wedge — not a code regression. Ask the user to validate on Windows. Non-CUDA backends (`--backend managed`, `--backend omp`) still work on WSL and can be used for correctness smoke tests.

## nsys per-call cost is inflated 2–4×

**Rule:** in tight-loop, small-payload workloads (LLM token decode: thousands of cheap launches per second), nsys's instrumentation overhead is **comparable to the work itself**.

- Trust **counts** in `cuda_api_sum` and `cuda_gpu_mem_size_sum`.
- Trust **proportions** between rows in the same report.
- **Don't trust** absolute "Time (s)" or "Avg (ns)" as real cost on a clean run.
- **Always re-measure native tok/s** before declaring a perf win.

See [perf-tuning.md](perf-tuning.md).

## Q8_0 doesn't fit cuBLAS

**Symptom of the wrong impulse:** "let's just use cuBLAS for the matvec."

**Why it won't fit:** cuBLAS INT8 takes **per-tensor** scaling (one scale per matrix or per row). Q8_0 is **per-block** (one FP16 scale per 32-element block). No public cuBLAS / cuBLASLt API consumes this layout.

**Why dequant + cuBLAS would be slower:** DRAM bandwidth on the weight side doubles (1 byte/element → 2 bytes/element). The matvec is memory-bound, so doubling reads ≈ halves throughput. cuBLAS won't recover that.

**Tensor cores don't help:** they want per-tensor scaling. Re-quantising to per-tensor loses the accuracy that per-block scaling exists to preserve.

**CUTLASS?** Same issue. You'd write essentially the same kernel as a cooperative-block warp-shuffle matvec, just with CUTLASS templates instead of Hybridizer-emitted CUDA. Diminishing returns.

**What to do instead:**

1. Optimise the existing cooperative kernel further (warp shuffle, register tiling).
2. Kernel fusion (fused QKV / W1+W3) to amortise launch overhead.
3. CUDA graphs to replay the per-token forward with one host call.
4. Change the quant format (FP8 / per-tensor INT8) — model-quality conversation, not just perf.

## Related

- The methodology that surfaces these gotchas in order: [methodology.md](methodology.md)
- The build settings that prevent half of them: [build-pipeline.md](build-pipeline.md)
- The host-launch init protocol: [host-launch.md](host-launch.md)
