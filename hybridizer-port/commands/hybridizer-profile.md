---
name: hybridizer-profile
description: Set up profiling for a Hybridizer-built binary — in-process per-kernel timing for fast iteration, nsys for full traces. Picks the right approach based on host platform and warns about the WSL2 wedge.
---

Help the user pick and wire up a profiling approach for the current Hybridizer project.

## First: pick the approach

Profile **selection** depends on the host platform and what the user wants to know:

- **Linux/WSL2** with the CUDA backend → **in-process profiler is the only option**. The WSL2 NVIDIA PTX-JIT-compiler wedge SIGSEGVs `dotnet` under `nsys profile` and `compute-sanitizer` (and often even under plain `dotnet run --backend cuda`) on this driver/CUDA combination. Don't waste time trying nsys in WSL.
- **Native Windows** → either works. Prefer `nsys` for full system traces; in-process is fine for quick per-kernel ranking.
- **Native Linux** (not WSL2) → `nsys` works normally.
- **OMP / managed backends** anywhere → standard .NET profiling tools (`dotnet-trace`, `BenchmarkDotNet`, etc.) work; this command focuses on CUDA-specific paths.

Ask the user which approach they want if it's not obvious from `$ARGUMENTS` and the platform.

## Approach A: in-process per-kernel profiler

Hook a `CudaInvoke.Profile(name, errCode)` helper at every kernel launch site. The helper wraps each launch with `cudaDeviceSynchronize` + a stopwatch and accumulates per-kernel totals. Gated by an env var (e.g. `LLAMA_KERNEL_PROFILE=1`) so it's off by default.

Pattern:

```csharp
public static class CudaInvoke {
    public static bool ProfilingEnabled = Environment.GetEnvironmentVariable("LLAMA_KERNEL_PROFILE") == "1";
    static readonly Dictionary<string, (long calls, double totalUs)> _stats = new();

    public static void Profile(string name, Action launch) {
        if (!ProfilingEnabled) { launch(); return; }
        var sw = Stopwatch.StartNew();
        launch();
        cuda.ERROR_CHECK(cuda.DeviceSynchronize());
        sw.Stop();
        var (c, t) = _stats.GetValueOrDefault(name);
        _stats[name] = (c + 1, t + sw.Elapsed.TotalMicroseconds);
    }

    public static void PrintProfile() {
        foreach (var (name, (c, t)) in _stats.OrderByDescending(kv => kv.Value.totalUs))
            Console.WriteLine($"{name,-40}  calls={c,8}  total={t,10:F1} µs  avg={t/c,8:F2} µs");
    }
}
```

Then wrap each launch:

```csharp
CudaInvoke.Profile("MatVecMulCoopRowShfl", () => _matVecMulCoopRowShfl(...));
```

Trade-off: forces a per-kernel sync, slowing throughput 2–3× during profiling. Gives correct **ranking** and **relative cost**, not absolute production cost.

**Mutual exclusion:** the per-kernel `cudaDeviceSynchronize` aborts CUDA stream capture. Gate any graph-capture path on `!ProfilingEnabled` and fall back to direct dispatch. See `skills/hybridizer-port/references/graph-capture.md`.

## Approach B: `nsys profile` (native only)

Standard nsys invocation:

```bash
nsys profile --stats=true -o profile.qdrep \
    dotnet run -c Release --project <project> -- <args>
```

Then inspect `profile_<run-id>.sqlite` with the nsys GUI or `nsys stats profile.qdrep`.

**Read the output carefully:** `nsys` instrumentation inflates per-call CUDA API cost 2–4× in tight-loop workloads. Trust the **counts** and the **proportions between rows**, not the absolute "Time (s)" or "Avg (ns)". Always re-measure native tok/s before declaring a perf win. See `skills/hybridizer-port/references/perf-tuning.md`.

## What to do

1. **Detect the platform** (`uname -a`, `lsb_release -a` or similar, and check for `/proc/sys/kernel/osrelease` containing `microsoft` for WSL2 detection).
2. **Ask which approach** the user wants if the platform allows both, or pick A (in-process) if WSL2 is detected.
3. **Find the kernel launch sites.** Grep for `wrapped.` or for `NativeLibrary.GetExport` (in the DllImport-bypass path).
4. **Wire in the profiler:**
   - Approach A: add `CudaInvoke.Profile` helper, wrap each launch, add a `PrintProfile` call at the end of the relevant entry point (e.g. after the generation loop completes).
   - Approach B: report the exact nsys command to run and what to look for in the output.
5. **Run the build** if approach A: `dotnet build -c Release`.
6. **Suggest the run command:**
   - A: `LLAMA_KERNEL_PROFILE=1 dotnet run -c Release -- <args>` (or whatever the project's main loop is).
   - B: the nsys command above, on the platform that supports it.

## Don't

- Try nsys on WSL2 — even after one apparent success, treat it as unsafe on the documented setups.
- Drop in a profiler without gating on an env var — always-on per-kernel sync ruins production runs.
- Quote nsys absolute times as if they were real. Always note the instrumentation overhead.
