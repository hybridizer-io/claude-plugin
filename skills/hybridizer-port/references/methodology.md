# Methodology — the 8-step Hybridizer port

The ordered workflow for bringing Hybridizer into an existing C# codebase. The ordering is load-bearing: tests before refactor (so refactors don't silently break behavior); refactor before kernel port (porting non-compliant code wastes a round-trip per discovery); memcpy-elimination before GPU profiling (otherwise the profile lies about where time goes); behavior-correctness invariant at every step (Hybridizer projects accumulate native/managed divergence very quickly otherwise).

## The steps

1. **Lock down behavior with tests first.** Both unit tests and at least one integration test that exercises the end-to-end pipeline. Without a baseline of green tests, nothing later in the workflow can be trusted.
2. **Profile the *managed* code** with non-intrusive instrumentation to get a raw, honest picture of where time is actually spent. Don't guess at hotspots; measure them. Profiling-as-information, not optimization.
3. **Audit and refactor for Hybridizer compliance** *before* writing any kernel. Eliminate heap allocations on hot paths, drop `throw`/`catch` inside what will become device code, remove `foreach`/`lock`/`string`/recursion/closure-captured-lambdas from the bodies you intend to port. See [device-code.md](device-code.md) for the no-go list. The codebase still runs on CPU at this stage — only the shape changes.
4. **Stand up the native DLL generation and loading infrastructure** before porting any real algorithm: `Directory.Build.props` with `$(HybridizerTool)`, `GenerateCUDA` + `CompileCUDA` MSBuild targets, `SatelliteLoader.Load()` wired up, a trivial smoke-test `[EntryPoint]` (e.g. add two vectors) that proves the round-trip works end to end. This is plumbing, not optimization. See [build-pipeline.md](build-pipeline.md) and [host-launch.md](host-launch.md).
5. **Port a single kernel to OMP and CUDA**, and unit-test both flavors. OMP first or in parallel — it's invaluable for catching transcoding/marshalling bugs without the GPU debug latency, and confirms the kernel body itself is correct independent of CUDA-specific issues.
6. **Iterate step 5 to drive down host↔device memcpies.** Promote frequently-reused buffers to `FloatResidentArray` / `ResidentArrayGeneric<T>`. Mark parameters `[In]` / `[Out]` to skip unnecessary transfers. Fuse kernels where the output of one is the input of the next on the same device. The goal here is *not* peak kernel speed — it's eliminating data motion, which usually dominates early-port profiles.
7. **Now profile the GPU execution** with `nvprof` / Nsight Systems / Nsight Compute (and `gprof` for the OMP/CPU side if relevant). Only at this point do GPU profiles give meaningful signal — before step 6, the memcpies drown everything else. See [perf-tuning.md](perf-tuning.md).
8. **Improve performance gradually**, guided by profiles. Shared-memory tiling, occupancy tuning, warp-level primitives, kernel fusion — case by case, profile-driven. See [kernel-patterns.md](kernel-patterns.md), [reductions.md](reductions.md), [graph-capture.md](graph-capture.md).

## The hard invariant

**After every change, rerun both unit tests and integration tests.** If anything breaks, **stop and investigate the root cause** — don't paper over it, don't move on. A broken test between two steps is the easiest place in the entire project to localise a bug; a broken test discovered three steps later is the worst.

## Performance regressions between steps 3 and 6 are expected

When porting, the pipeline necessarily replaces hand-tuned managed code (AVX2 intrinsics, fused inner loops, `Vector.Widen` tricks) with simpler scalar/`Parallel.For` shapes that Hybridizer can transcode cleanly to both OMP and CUDA. The "regression" is the price of getting the shape right for the transcoder.

Rules during steps 3–6:

- **The only metrics that matter are functional.** UT + integration tests must stay green. Tokens-per-second can drop and that's fine.
- **Don't raise alarms on perf regressions.** If managed throughput halves between step 3 and step 5 because a SIMD inner loop was replaced by scalar code that transcodes, log it and move on.
- **Do raise an alarm on functional regressions.** A broken test, a drifted golden output, NaNs appearing — those trigger immediate root-cause investigation per the hard invariant.

Performance comes back at steps 7 and 8. Investigating a slowdown mid-port wastes effort optimising code that's about to be replaced *and* risks distorting the kernel shape away from what transcodes well.

## Applying this in conversation

- When asked to "port X to Hybridizer" or "make X run on GPU", do not jump straight to writing a kernel. Position the work in this sequence and confirm which step we're at.
- Don't skip steps to look productive. Step 4 (smoke-test infrastructure) feels like throat-clearing but it's where most build-system bugs surface, before kernel logic is on the table.
- When a step is already done (tests exist, profiling has been run), say so and move to the next — but verify the prerequisite genuinely holds before advancing.
- If a test breaks mid-port, surface it immediately with the failing test name and last-known-green commit. Don't try to silently push through.
