# Kernel patterns

The shapes that perform. Use these instead of the default `Parallel.For(0, N, ...)` lowering for any kernel where N is small relative to the GPU's concurrent-block budget, or where the inner loop is non-trivial.

## Cooperative-block: one block per output row / head

The default `Parallel.For(0, N, ...)` lowering Hybridizer applies maps each iteration to **one** thread in the grid. For N much smaller than the launch grid (e.g. 256 output rows in a 32 K-thread launch = 0.8 % occupancy), the kernel barely runs.

**The fix** — one block per output element, threads in the block cooperate on the inner loop. Canonical pattern, lifted from `basic-samples/1.Simple/Reduction/Program.cs::ReduceAdd`:

```csharp
[EntryPoint]
[HybridizerIgnore("OMP")]
public static void MyCoopKernel(
    FloatResidentArray output,       // size = rows
    FloatResidentArray input,        // size = rows * cols
    int rows,
    int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;          // bounds check BEFORE the allocator
    int tid = threadIdx.x;
    int nThreads = blockDim.x;

    // Per-thread strided partial reduction.
    float partial = 0f;
    for (int i = tid; i < cols; i += nThreads)
        partial = partial + input[row * cols + i];

    // Shared-mem tree reduction (slower path, more __syncthreads).
    var cache = new SharedMemoryAllocator<float>().allocate(blockDim.x);
    cache[tid] = partial;
    CUDAIntrinsics.__syncthreads();
    int stride = nThreads >> 1;
    while (stride > 0)
    {
        if (tid < stride) cache[tid] = cache[tid] + cache[tid + stride];
        CUDAIntrinsics.__syncthreads();
        stride = stride >> 1;
    }
    if (tid == 0) output[row] = cache[0];
}
```

Host launch (using the `DllImport`-bypass pattern from [perf-tuning.md](perf-tuning.md)):

```csharp
int sharedBytes = CoopRowBlockDim * sizeof(float);   // e.g. 128 * 4 = 512
_myCoopKernel(
    rows, 1, 1, CoopRowBlockDim, 1, 1, sharedBytes, _stream,
    ResidentStructCache.Get(output),
    ResidentStructCache.Get(input),
    rows, cols);
```

### Tuning heuristics

- **`blockDim` should match the inner-loop natural width when possible.** For attention with `headDim = 64`, `blockDim = 64` is great because Phase 3 (V-weighted output) maps 1 thread per `d`. For matvec, `blockDim = 128` × 16 strides covers a 2048-col row well.
- **`gridDim.x = N` — under-saturates when N < ~192.** RTX 5070 ≈ 192 concurrent blocks (24 SMs × ~8 resident blocks/SM, dominated by thread-limit). The 1-row-per-block layout still works for any N because the wave count just becomes `ceil(N / 192)` — only really starves when both row count and bytes-per-row are tiny.
- **`SharedMemoryAllocator<T>()` is a fresh struct each kernel call.** Its destructor calls `__syncthreads()`. If the kernel has an early `return` (e.g. `if (row >= rows) return;`), put the bounds check **before** the `new SharedMemoryAllocator()` line — otherwise out-of-bounds blocks hit the allocator and produce malformed sync state.
- **Compute `shmem` size in two places** — `allocate(blockDim.x)` in the kernel and `sharedBytes = blockDim.x * sizeof(float)` in the launch. They must match; the wrapper adds 16 B for runtime bookkeeping so don't over-shrink.

### Replace the tree reduction with shuffle for the win

The shared-memory tree reduction in the example above is the slow path. Switch to warp shuffle (next section) for a real speedup.

## Warp-shuffle block reduction

For block-wide reductions in cooperative-block kernels, prefer warp-shuffle over shared-memory tree. Each warp collapses internally with zero `__syncthreads`, then one cross-warp pass via a tiny shared-mem scratch.

```csharp
using Hybridizer.Runtime.CUDAImports;
// ...

int laneId = tid & 31;        // = threadIdx.x % 32
int warpId = tid >> 5;        // = threadIdx.x / 32
int nWarps = blockDim.x >> 5; // = blockDim.x / 32

// Set up the warp tile once.
var tb   = cooperative_groups.this_thread_block();
var warp = cooperative_groups.tile_partition_32(tb);

// --- per-thread partial accumulation ---
float partial = /* ... */;

// Stage 1: within-warp shuffle reduce. Lane 0 of each warp holds the warp sum.
partial = partial + warp.shfl_down(partial, 16);
partial = partial + warp.shfl_down(partial,  8);
partial = partial + warp.shfl_down(partial,  4);
partial = partial + warp.shfl_down(partial,  2);
partial = partial + warp.shfl_down(partial,  1);

// Stage 2: cross-warp via tiny shared mem. Slot 31 reserved for broadcast.
var cw = new SharedMemoryAllocator<float>().allocate(32);
if (laneId == 0) cw[warpId] = partial;
CUDAIntrinsics.__syncthreads();

if (warpId == 0)
{
    float v = laneId < nWarps ? cw[laneId] : 0f;
    v = v + warp.shfl_down(v, 16);
    v = v + warp.shfl_down(v,  8);
    v = v + warp.shfl_down(v,  4);
    v = v + warp.shfl_down(v,  2);
    v = v + warp.shfl_down(v,  1);
    if (laneId == 0) cw[31] = v;       // broadcast slot
}
CUDAIntrinsics.__syncthreads();
float blockSum = cw[31];               // every thread sees the reduced value
```

### API call sites (all in `Hybridizer.Runtime.CUDAImports`)

- `cooperative_groups.this_thread_block()` — get the block's `thread_block`.
- `cooperative_groups.tile_partition_32(thread_block)` — get the 32-lane tile. **Static method**, not instance.
- `thread_block_tile_32.shfl_down(value, delta)` — per-thread call. Returns the value at lane `tid + delta`, or own value when out of range. Overloads for int/uint/long/ulong/float/double.
- Also available: `.shfl(v, rank)`, `.shfl_up(v, delta)`, `.shfl_xor(v, mask)`, `.sync()`, `.thread_rank()`, `.size()`.

### Max / min reductions

Replace `v + warp.shfl_down(v, k)` with:

```csharp
float o = warp.shfl_down(maxLocal, k);
if (o > maxLocal) maxLocal = o;
```

Seed lanes ≥ `nWarps` with the identity for your operator (`-FLT_MAX` for max, `0f` for sum). The `if`-guard in the cross-warp stage above handles this so out-of-range shuffles don't matter in practice.

### Why the broadcast slot at `cw[31]`?

If a downstream phase needs the reduced result on every thread (e.g. softmax `invSum` used by all lanes in the normalize step), the warp-0 lane-0 result needs to escape back to the other warps. One extra shared-mem slot + one extra `__syncthreads` is the cleanest path. `cw[31]` is safe for any `blockDim ≤ 1024` (max 32 warps).

### Gotchas

- `shfl_down` on a uniform (broadcast) value across lanes can degenerate — make sure the lane reads are actually different.
- For `blockDim` not a multiple of 32, the partial warp's behaviour is murkier. Stick to `blockDim ∈ {32, 64, 128, 256, 512, 1024}`.

## CUB `BlockReduce` via `[IntrinsicFunction]`

Functionally equivalent to a hand-rolled 2-stage warp shuffle (CUB uses the same `__shfl_*` primitives under the hood). The win is code quality and reusability.

Hybridizer can't lower `__shared__ typename cub::BlockReduce<...>::TempStorage` directly. Wrap each `(block-dim, op)` pair in a hand-written `__device__` helper in `intrinsics.cuh`, expose to C# via `[IntrinsicFunction]`:

```cpp
// intrinsics.cuh
#include <cub/cub.cuh>
#include <cuda/functional>   // for cuda::maximum<> — cub::Max is gone in CCCL 13.2

__device__ __forceinline__ float cub_block_reduce_sum_128(float v) {
    typedef cub::BlockReduce<float, 128> BR;
    __shared__ typename BR::TempStorage tmp;
    __shared__ float bcast;
    float r = BR(tmp).Sum(v);
    if (threadIdx.x == 0) bcast = r;
    __syncthreads();
    return bcast;
}
```

```csharp
[IntrinsicInclude("intrinsics.cuh")]
internal static class CubReduce
{
    [IntrinsicFunction("cub_block_reduce_sum_128")]
    public static float SumBlock128(float v) => v;   // managed fallback = identity
}
```

### Key constraints

1. **One helper per (block-dim, op).** No templates — `[IntrinsicFunction]` takes a flat C name. Add named helpers (`_sum_128`, `_sum_64`, `_max_64`) as needed.
2. **`cub::Max` is gone in CCCL 13.2.** Use `cuda::maximum<>{}` from `<cuda/functional>`.
3. **All helpers broadcast to every thread.** CUB returns the reduced value only on thread 0; the wrapper adds `__shared__ float bcast; if (tid==0) bcast=r; __syncthreads();` so callers don't have to.
4. **`TempStorage` reuse across calls is safe.** `__shared__` inside a `__device__ __forceinline__` function is a per-block static allocation, reused across every call site that inlines it. `BlockReduce::Sum/Reduce` syncs internally, so multiple reductions per kernel are fine.
5. **No dynamic shared memory needed.** Launch with `shmem = 0`; CUB's `TempStorage` is static `__shared__` and lives in a different namespace from any `extern __shared__` Hybridizer's dynamic-shared allocator might claim.
6. **CUDA-only.** Decorate the calling kernel with `[HybridizerIgnore("OMP")]` — the managed fallback is just `return v`, and OMP has no block concept anyway.

## Per-flavor `[EntryPoint]` split

When a kernel needs different bodies for CUDA vs OMP (typically reductions, atomic-driven ops, anything using `SharedMemoryAllocator` + `__syncthreads`), write **sibling `[EntryPoint]` methods** each carrying `[HybridizerIgnore]` for the other flavor:

```csharp
[EntryPoint]
[HybridizerIgnore("OMP")]
public static void ReduceCuda(...)
{
    // canonical shape-B with shared mem + __syncthreads
}

[EntryPoint]
[HybridizerIgnore("CUDA")]
public static void ReduceOmp(...)
{
    // Parallel.For + per-thread accumulator
}
```

`[HybridizerIgnore("FLAVOR")]` excludes that method from being transcoded for the named flavor. The `HybridizerIgnoreAttribute(string)` constructor splits on spaces/commas/semicolons; the `HybridizerFlavor` enum constants stringify to `"CUDA"` / `"OMP"` / `"HIP"` / etc., case-insensitive. The method still compiles in managed C# — only the *transcoded native* emission is suppressed.

The dispatcher routes by `_mode`:

```csharp
if (_mode == Cuda)
    _wrappedFloat!.ReduceCuda(...);  // CUDA satellite has only this symbol
else
    _wrappedFloat!.ReduceOmp(...);   // OMP satellite has only this symbol
```

`dynamic wrapped = runner.Wrap(...)` resolves method names via DLR; calling `wrapped.ReduceCuda(...)` when the OMP satellite is loaded throws `EntryPointNotFoundException`. So **the dispatch site must check `_mode` before picking the name.**

**When not to split.** Embarrassingly-parallel `Parallel.For` shapes (SAXPY, elementwise, accumulate) lower cleanly on both flavors with no flavor-specific code — keep them as one method. Only split when the kernel body would fail to compile on one flavor.

## Quantised-tensor split layout (Q8_0, Q4_0, Q5_0, K-quants)

Do **not** carry the interleaved GGUF/GGML byte blob (FP16 scale + packed quants) into the kernel. **Pre-decode the FP16 scales to FP32 once at host load** and pass them as a separate `float[]`:

```csharp
[EntryPoint]
public static void MatVecMul(
    [Out] float[] result,
    [In] byte[] values,    // packed quants: 32 sbytes per block for Q8_0
    [In] float[] scales,   // one FP32 per block, pre-decoded from FP16
    [In] float[] vector,
    int rows, int blocksPerRow) { ... }
```

**Why this matters specifically for Hybridizer:** the bundled `hybridizer.{cuda,c}.builtins` files do **not** map `BitConverter.UInt16BitsToHalf`, `System.Half`, or `(float)Half` to any native equivalent. Carrying FP16 into the kernel forces you to ship a custom `.builtins`, hand-roll bit manipulation, or decorate a helper with `[IntrinsicFunction("__half2float")]`. The split-layout pre-decode sidesteps the question entirely.

**Cost:** scales array is ~3 % the size of the original byte blob for Q8_0. Host-side decode runs once at model load.

**Generalises to other quants.** Q4_0, Q5_0, Q8_K, Q6_K, Q4_K — same principle: keep all FP16→FP32 conversion in C# host code, pass float scales to the kernel.

**Validation.** Test against `MatVecMul(dequantized_full_matrix, vector)`. That reference is order-independent quantization fidelity and catches both layout bugs and arithmetic bugs.

## Anti-pattern: don't widen to "K rows per block" as a first optimisation

Sharing an inner-loop load across K output rows by computing K partial sums per thread sounds like an obvious win. **It usually isn't.**

A real attempt on a Q8 matvec (K=4) **regressed 91 → 75 tok/s** because:

1. **Block-count collapse on small matrices.** TinyLlama Wk/Wv are 256 rows. K=4 means 64 blocks → only 1/3 of the GPU saturated. Adaptive 1-row-vs-K-row dispatch helped here.
2. **Register pressure on big matrices.** 4 partial-sum floats + 8 row-offset ints + 4 valid-bool flags + temporaries inflated past the per-thread register threshold. Occupancy halved on the big matrices too. Adaptive dispatch could **not** fix this.
3. **Bounded conditional compute kills vectorisation.** `if (rowKValid)` bounds-checks inside the inner loop generated 4 sequential branched blocks in the `.cu` output. The compiler couldn't fuse the 4 MACs into straight-line code.

**Prefer instead:**

1. **Warp-shuffle reduction** (no register pressure change, drops syncthreads).
2. **Kernel fusion across matvecs with the same input vector** (saves a launch + may improve L1/L2 hit rate).
3. **Persistent block / cooperative-groups graph-launch tricks** (much bigger commitment).

**Don't pursue:** N-rows-per-block without first profiling to confirm the kernel isn't already register-pressure-limited.

## Related

- The reduction skeleton in detail: [reductions.md](reductions.md)
- Device-code language subset: [device-code.md](device-code.md)
- `DllImport`-bypass launching, which goes with cooperative kernels: [perf-tuning.md](perf-tuning.md)
- CUDA graph capture (orthogonal but compounds): [graph-capture.md](graph-capture.md)
